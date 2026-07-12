"""
Evaluation utilities for trained RL agents.

Provides a reusable evaluation function that works with both MaskablePPO
and standard PPO models across all environment configurations.

Usage:
    from reinforcetactics.rl.evaluation import evaluate_model

    results = evaluate_model(model, env, n_episodes=50)
    print(f"Win rate: {results['win_rate']:.1%}")
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _model_accepts_action_masks(model: Any) -> bool:
    """Return True iff ``model.predict`` accepts an ``action_masks`` kwarg.

    ``MaskablePPO.predict`` declares ``action_masks`` explicitly; plain
    ``stable_baselines3.PPO.predict`` does not and raises ``TypeError`` if
    one is passed. This signature probe distinguishes the two without
    importing sb3-contrib (which is an optional dep) and also handles
    duck-typed test stubs whose ``predict`` accepts ``**kwargs``.
    """
    try:
        sig = inspect.signature(model.predict)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "action_masks" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# Action types are emitted as integers in info["action_type"]; this list maps
# them to human-readable names. Matches the encoding in
# StrategyGameEnv._encode_action (gym_env.py).
ACTION_TYPE_NAMES = (
    "create_unit",
    "move",
    "attack",
    "seize",
    "heal",
    "end_turn",
    "paralyze",
    "haste",
    "defence_buff",
    "attack_buff",
)

# Reward components emitted in info["reward_breakdown"] by StrategyGameEnv.
REWARD_COMPONENTS = ("action", "shaping_delta", "invalid_penalty", "terminal")

# Episode-end reasons emitted in info["end_reason"] by StrategyGameEnv.
# See gym_env.py for the classification rules.
END_REASONS = ("hq_capture", "elimination", "max_turns_draw", "max_steps_truncate")

# Unit types tracked by ``info["episode_stats"]["units_built"]``. Mirrors
# ``reinforcetactics.constants.ALL_UNIT_TYPES`` so this module stays
# importable without pulling the constants module.
UNIT_TYPE_LETTERS = ("W", "M", "C", "A", "K", "R", "S", "B")

# Combat / progression scalars surfaced via ``info["episode_stats"]``.
# Aggregated by summing across eval episodes so per-stage diagnostics
# can plot e.g. "captures per game" or "damage delta" curves.
COMBAT_STAT_KEYS = ("captures", "kills", "attacks", "seize_attempts", "damage_dealt", "damage_taken")

# Structure auto-heal economics surfaced via ``info["episode_stats"]``.
# Summed across eval episodes into the same ``combat_stats`` dict, but
# kept out of COMBAT_STAT_KEYS so viz.py's combat plot (which iterates
# that tuple) is unchanged. ``own_heal_gold`` = gold the agent silently
# spent auto-healing wounded units parked on its structures;
# ``opp_heal_hp`` = free durability the opponent's rebuild economy
# received -- the meat-wall / draw-machine probe.
HEALING_STAT_KEYS = ("own_heal_hp", "own_heal_gold", "opp_heal_hp", "opp_heal_gold")

# Structure types tracked under ``info["episode_stats"]["captures_by_type"]``.
# Populated by ``StrategyGameEnv._execute_action`` whenever a seize action
# captures a tile; aggregated here so eval_results.json shows per-structure
# capture counts (towers vs buildings vs HQ) rather than only the total.
CAPTURE_STRUCTURE_TYPES = ("tower", "building", "hq")


def evaluate_model(
    model: Any,
    env: Any,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Any = None,
    track_breakdown: bool = False,
    trace_dir: str | Path | None = None,
    trace_end_reasons: tuple | None = ("max_steps_truncate",),
) -> dict[str, Any]:
    """
    Evaluate a trained model and return summary statistics.

    Works with both MaskablePPO (using action_masks) and standard PPO.
    The environment can be a raw gym env or an ActionMaskedEnv wrapper.

    Args:
        model: Trained SB3 model (PPO, MaskablePPO, etc.)
        env: Gymnasium environment (single, not vectorized).
        n_episodes: Number of evaluation episodes.
        deterministic: Use deterministic actions (recommended for eval).
        seed: Optional integer seed. When provided, episode ``i`` is reset
            with ``seed + i`` so results are reproducible across runs.
        track_breakdown: When True, also accumulate per-step
            ``info["action_type"]`` counts and ``info["reward_breakdown"]``
            sums across the evaluation, returned under ``action_counts``
            and ``reward_components``. Disabled by default to keep the hot
            path lean; the per-step ``info`` dict is read either way.

    Returns:
        Dict with keys: win_rate, avg_reward, std_reward, avg_length,
        std_length, wins, losses, draws, episodes, rewards, lengths.

        ``rewards`` and ``lengths`` are the raw per-episode arrays (lists
        of plain floats / ints), exposed so callers can plot full
        distributions rather than only the mean ± std summary.

        When ``track_breakdown=True`` the dict also includes
        ``action_counts`` (dict keyed by ACTION_TYPE_NAMES, summed over
        every step of every episode) and ``reward_components`` (dict
        keyed by REWARD_COMPONENTS, summed analogously).

        Always includes ``seize_available_rate`` (fraction of decision
        points across all eval steps where a seize action was legal) and
        ``max_legal_actions`` (peak legal-action-set size over the eval) --
        action-space diagnostics for the capture bottleneck and the
        flat_discrete truncation guardrail respectively.

        When ``trace_dir`` is set, episodes whose ``info["end_reason"]``
        is in ``trace_end_reasons`` are dumped to JSON Lines at
        ``<trace_dir>/episode_<ep_idx>_<end_reason>.jsonl`` -- one line
        per env step with the chosen action, env info, and reward.
        Default trigger captures only ``max_steps_truncate`` episodes
        (the stalling failure mode); pass an empty tuple to disable.
        The returned dict gains a ``traces`` list of dumped file paths.
    """
    wins, losses, draws = 0, 0, 0
    rewards = []
    lengths = []
    turns = []
    # Only forward action masks to ``model.predict`` when the model itself
    # accepts them. Plain ``stable_baselines3.PPO.predict`` will raise
    # ``TypeError`` on an unexpected ``action_masks`` kwarg, which would
    # break eval for the ``ppo_baseline.yaml`` path (vanilla PPO behind a
    # mask-exposing env wrapper).
    has_action_masks = hasattr(env, "action_masks") and _model_accepts_action_masks(model)

    # Outcome × end-reason matrix accumulated per episode. Keys are
    # f"{outcome}_by_{reason}" e.g. "wins_by_hq_capture",
    # "losses_by_elimination". Always populated (cheap; one read of the
    # final info dict per episode).
    outcome_reasons = {f"{outcome}_by_{reason}": 0 for outcome in ("wins", "losses", "draws") for reason in END_REASONS}
    end_reasons = {reason: 0 for reason in END_REASONS}

    # Combat / build counters summed across eval episodes. Always
    # populated -- they're aggregated from ``info["episode_stats"]`` once
    # per episode (cheap) and the per-stage diagnostics in viz.py rely
    # on them.
    units_built = {ut: 0 for ut in UNIT_TYPE_LETTERS}
    combat_stats = {k: 0.0 for k in (*COMBAT_STAT_KEYS, *HEALING_STAT_KEYS)}
    captures_by_type = {k: 0 for k in CAPTURE_STRUCTURE_TYPES}

    # Action-space diagnostics aggregated from ``info["episode_stats"]``.
    # ``seize_available_steps`` / total steps -> the fraction of decision
    # points where a capture was legal (the "can-vs-won't seize" signal);
    # ``max_legal_actions`` tracks the peak legal-action-set size, a
    # guardrail for flat_discrete truncation and a proxy for army bloat.
    seize_available_steps_total = 0
    steps_total = 0
    max_legal_actions = 0

    # Army-economy diagnostics aggregated from ``info["episode_stats"]``.
    # Peaks take the max across the eval set; the *_sum totals are divided
    # by ``steps_total`` for per-decision means. Together they answer "does
    # the agent win by massing a big slow army or with a small precise
    # force" -- a high peak army + near-zero banked gold is the
    # economy-funds-mass signature.
    peak_own_units = 0
    own_units_sum_total = 0
    peak_gold_banked = 0.0
    gold_banked_sum_total = 0.0

    if track_breakdown:
        action_counts = {name: 0 for name in ACTION_TYPE_NAMES}
        reward_components = {name: 0.0 for name in REWARD_COMPONENTS}

    # Per-step trace capture, gated on ``trace_dir`` being set. Steps are
    # buffered in memory for the duration of each episode and flushed to
    # disk only if the terminal ``end_reason`` matches the trigger set.
    # That avoids spamming the filesystem with healthy-episode traces
    # while still giving full visibility on the failure modes we care
    # about (default: ``max_steps_truncate``, the stalling signature).
    # The directory is created lazily at first flush so eval blocks
    # without any trigger-matching episodes leave no empty folders
    # behind (relevant for Google-Drive-backed run dirs where empty
    # subfolders pile up across eval cadence).
    trace_dir_path: Path | None = Path(trace_dir) if trace_dir is not None else None
    trace_paths: list[str] = []
    trace_triggers = set(trace_end_reasons or ())
    if not trace_triggers:
        trace_dir_path = None

    for ep_idx in range(n_episodes):
        if seed is not None:
            obs, _ = env.reset(seed=int(seed) + ep_idx)
        else:
            obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_trace: list[dict] | None = [] if trace_dir_path is not None else None

        while not done:
            predict_kwargs: dict[str, Any] = {"deterministic": deterministic}
            if has_action_masks:
                masks = env.action_masks()
                # MaskablePPO expects a flat concatenated mask
                if isinstance(masks, tuple):
                    masks = np.concatenate([m.astype(np.bool_) for m in masks])
                predict_kwargs["action_masks"] = masks

            action, _ = model.predict(obs, **predict_kwargs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated

            if track_breakdown:
                at = info.get("action_type")
                if isinstance(at, (int, np.integer)) and 0 <= int(at) < len(ACTION_TYPE_NAMES):
                    action_counts[ACTION_TYPE_NAMES[int(at)]] += 1
                for k, v in info.get("reward_breakdown", {}).items():
                    if k in reward_components:
                        reward_components[k] += float(v)

            if ep_trace is not None:
                at_idx = info.get("action_type")
                at_name = (
                    ACTION_TYPE_NAMES[int(at_idx)]
                    if isinstance(at_idx, (int, np.integer)) and 0 <= int(at_idx) < len(ACTION_TYPE_NAMES)
                    else None
                )
                ep_trace.append(
                    {
                        "step": ep_len,
                        "action_index": int(np.asarray(action).flatten()[0]) if np.ndim(action) else int(action),
                        "action_type": int(at_idx) if isinstance(at_idx, (int, np.integer)) else None,
                        "action_type_name": at_name,
                        "unit_type": info.get("unit_type"),
                        "turn": int(info.get("turn", 0)),
                        "valid_action": bool(info.get("valid_action", False)),
                        "n_legal_actions": int(info.get("n_legal_actions", 0)),
                        "reward": float(reward),
                        "reward_breakdown": {k: float(v) for k, v in info.get("reward_breakdown", {}).items()},
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )

        rewards.append(ep_reward)
        lengths.append(ep_len)
        # Final game-turn count from the terminal info dict. ``ep_len`` is
        # env steps (one per agent action); a single turn comprises many
        # steps, so reporting both gives a clearer signal of game length
        # than steps alone -- particularly for stages with max_turns caps.
        turns.append(int(info.get("turn", 0)))

        # Determine outcome from info
        episode_stats = info.get("episode_stats", {})
        winner = episode_stats.get("winner", info.get("winner"))
        agent_player = getattr(env, "agent_player", None)
        if agent_player is None:
            agent_player = getattr(getattr(env, "unwrapped", None), "agent_player", 1)

        if winner == agent_player:
            wins += 1
            outcome = "wins"
        elif winner is not None:
            losses += 1
            outcome = "losses"
        else:
            draws += 1
            outcome = "draws"

        reason = info.get("end_reason")
        if reason in END_REASONS:
            end_reasons[reason] += 1
            outcome_reasons[f"{outcome}_by_{reason}"] += 1

        # Flush per-step trace if this episode tripped the configured
        # trigger (default: ``max_steps_truncate``, i.e. the stall mode
        # this dump is meant to diagnose). One JSONL file per matching
        # episode; healthy episodes are discarded without touching disk.
        # ``trace_dir_path`` is created lazily here so eval blocks with
        # no trigger-matching episodes leave no empty folder behind.
        if ep_trace is not None and reason in trace_triggers and trace_dir_path is not None:
            trace_dir_path.mkdir(parents=True, exist_ok=True)
            ep_seed = (int(seed) + ep_idx) if seed is not None else None
            trace_file = trace_dir_path / f"episode_{ep_idx:04d}_{reason}.jsonl"
            with trace_file.open("w") as fh:
                header = {
                    "episode_index": ep_idx,
                    "seed": ep_seed,
                    "end_reason": reason,
                    "outcome": outcome,
                    "winner": winner,
                    "agent_player": int(agent_player) if agent_player is not None else None,
                    "ep_length": ep_len,
                    "ep_reward": float(ep_reward),
                    "final_turn": int(info.get("turn", 0)),
                }
                fh.write(json.dumps({"_header": header}) + "\n")
                for record in ep_trace:
                    fh.write(json.dumps(record) + "\n")
            trace_paths.append(str(trace_file))

        # Aggregate combat / build counters from the env's per-episode
        # stats dict. Older envs that don't populate these keys just
        # contribute zeros, so this is safe with stale checkpoints.
        ep_units_built = episode_stats.get("units_built") or {}
        for ut, count in ep_units_built.items():
            if ut in units_built:
                units_built[ut] += int(count)
        for key in (*COMBAT_STAT_KEYS, *HEALING_STAT_KEYS):
            val = episode_stats.get(key)
            if val is not None:
                combat_stats[key] += float(val)

        # Per-structure capture breakdown. Older envs (pre-this change)
        # don't populate the key; older checkpoints just contribute zeros.
        ep_captures_by_type = episode_stats.get("captures_by_type") or {}
        for key in CAPTURE_STRUCTURE_TYPES:
            val = ep_captures_by_type.get(key)
            if val is not None:
                captures_by_type[key] += int(val)

        # Action-space diagnostics. ``ep_len`` is the per-episode step count
        # (one env step per agent action); older checkpoints that don't
        # populate the stats just contribute zeros.
        seize_available_steps_total += int(episode_stats.get("seize_available_steps", 0) or 0)
        steps_total += ep_len
        max_legal_actions = max(max_legal_actions, int(episode_stats.get("max_legal_actions", 0) or 0))

        peak_own_units = max(peak_own_units, int(episode_stats.get("peak_own_units", 0) or 0))
        own_units_sum_total += int(episode_stats.get("own_units_sum", 0) or 0)
        peak_gold_banked = max(peak_gold_banked, float(episode_stats.get("peak_gold_banked", 0.0) or 0.0))
        gold_banked_sum_total += float(episode_stats.get("gold_banked_sum", 0.0) or 0.0)

    rewards_arr = np.array(rewards)
    lengths_arr = np.array(lengths)
    turns_arr = np.array(turns)

    result: dict[str, Any] = {
        "win_rate": wins / n_episodes if n_episodes > 0 else 0.0,
        "avg_reward": float(rewards_arr.mean()) if n_episodes > 0 else 0.0,
        "std_reward": float(rewards_arr.std()) if n_episodes > 0 else 0.0,
        "avg_length": float(lengths_arr.mean()) if n_episodes > 0 else 0.0,
        "std_length": float(lengths_arr.std()) if n_episodes > 0 else 0.0,
        "avg_turns": float(turns_arr.mean()) if n_episodes > 0 else 0.0,
        "std_turns": float(turns_arr.std()) if n_episodes > 0 else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "episodes": n_episodes,
        "rewards": [float(r) for r in rewards],
        "lengths": [int(length) for length in lengths],
        "turns": [int(t) for t in turns],
        "end_reasons": end_reasons,
        "outcome_reasons": outcome_reasons,
        "units_built": units_built,
        "combat_stats": combat_stats,
        "captures_by_type": captures_by_type,
        "seize_available_rate": (seize_available_steps_total / steps_total) if steps_total > 0 else 0.0,
        "max_legal_actions": int(max_legal_actions),
        "peak_own_units": int(peak_own_units),
        "mean_own_units": (own_units_sum_total / steps_total) if steps_total > 0 else 0.0,
        "peak_gold_banked": float(peak_gold_banked),
        "mean_gold_banked": (gold_banked_sum_total / steps_total) if steps_total > 0 else 0.0,
    }
    if track_breakdown:
        result["action_counts"] = action_counts
        result["reward_components"] = reward_components
    if trace_dir_path is not None:
        result["traces"] = trace_paths
    return result
