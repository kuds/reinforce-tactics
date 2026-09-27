"""
Evaluation utilities for trained RL agents.

Provides a reusable evaluation function that works with both MaskablePPO
and standard PPO models across all environment configurations, a
vectorized variant that steps several envs per batched ``predict`` call
(:func:`evaluate_model_vec`), and the Wilson score bound the curriculum's
promotion gate can use (:func:`wilson_lower_bound`).

Usage:
    from reinforcetactics.rl.evaluation import evaluate_model

    results = evaluate_model(model, env, n_episodes=50)
    print(f"Win rate: {results['win_rate']:.1%}")

Both evaluators play the same episodes for the same arguments: episode
``i`` of seat ``s`` is reset with ``seed + i`` after the env's seat is set to
``s``, and every per-episode quantity is aggregated in episode order, so the
serial and the vectorized path return identical results whenever the
policy's actions are a deterministic function of the observation
(``deterministic=True``).

With sampled actions (``deterministic=False``) and a ``seed``, the policy
samples from its own torch stream, forked from the global generator and
restored afterwards: the evaluation neither consumes the random numbers
training samples its actions from nor depends on them, so it is a function
of the weights and the seed and can be re-run from a checkpoint. The serial
path reseeds that stream at every episode (:func:`policy_sampling_seed` of
the episode's seed and seat), so episode ``i`` draws the same numbers
whatever ``n_episodes`` is; the vectorized path seeds it once per call,
because one batched ``predict`` samples every env's action together, so its
stochastic results are reproducible but not equal to the serial path's.
Without a ``seed`` a stochastic evaluation samples from the global generator
as before.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import math
import multiprocessing as mp
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
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
# ``reinforcetactics.rules.ALL_UNIT_TYPES`` so this module stays
# importable without pulling the rules module.
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


# ---------------------------------------------------------------------------
# Wilson score bound (review rltrain-12 / prior-5)
# ---------------------------------------------------------------------------


def z_for_confidence(confidence: float) -> float:
    """The one-sided standard-normal quantile for ``confidence`` (0.95 -> 1.645)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(NormalDist().inv_cdf(confidence))


def wilson_lower_bound(successes: float, n: int, z: float) -> float:
    """Lower end of the Wilson score interval for ``successes`` out of ``n`` trials.

    ``successes`` may be fractional (a draw scored as half a win). ``z`` is
    the normal quantile: 1.96 gives the lower end of the usual two-sided
    95% interval, :func:`z_for_confidence` a one-sided bound. ``n == 0``
    returns 0.0 (nothing measured, nothing proven).
    """
    if n <= 0:
        return 0.0
    p = min(max(float(successes) / n, 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom)


# ---------------------------------------------------------------------------
# The policy's own random stream for a seeded stochastic evaluation
# ---------------------------------------------------------------------------

# Mixed into every policy-sampling seed so it never equals the episode's env
# seed (the two streams stay unrelated).
_POLICY_SEED_TAG = 0x5A3D_91C7


def policy_sampling_seed(seed: int, seat: int | None = None) -> int:
    """The torch seed a seeded stochastic evaluation samples episode ``seed`` (of ``seat``) with."""
    entropy = [int(seed) % (1 << 63), int(seat or 0), _POLICY_SEED_TAG]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


@contextlib.contextmanager
def _isolated_policy_rng(model: Any, enabled: bool) -> Iterator[Callable[[int], Any] | None]:
    """Give a seeded stochastic evaluation its own torch random stream.

    ``model.predict(deterministic=False)`` samples from torch's global
    generator, the stream SB3 also samples training actions from. Without
    this, an eval consumed training's random numbers (so ``n_eval_episodes``,
    ``eval_both_modes``, ``eval_seats`` or ``n_eval_envs`` changed the policy
    that was trained) and its own samples depended on whatever training had
    drawn before (so a row could not be reproduced from its checkpoint and
    eval seed). Inside the context the generator of the model's device is
    forked -- restored on exit -- and the yielded ``reseed(seed)`` sets it.
    Yields ``None`` (and changes nothing) when ``enabled`` is False.
    """
    if not enabled:
        yield None
        return
    import torch

    device = getattr(model, "device", None)
    device_type = getattr(device, "type", "cpu") if device is not None else "cpu"
    if device_type == "cpu":
        # Only the CPU generator: torch.manual_seed would also queue a seed
        # for a CUDA runtime this process may initialise later.
        with torch.random.fork_rng(devices=[]):
            yield torch.default_generator.manual_seed
        return
    count = torch.cuda.device_count() if device_type == "cuda" else 1
    with torch.random.fork_rng(devices=list(range(max(1, count))), device_type=device_type):
        yield torch.manual_seed


# ---------------------------------------------------------------------------
# Per-episode bookkeeping shared by the serial and the vectorized evaluators
# ---------------------------------------------------------------------------


def _flat_mask(masks: Any) -> np.ndarray:
    """MaskablePPO expects one flat mask; a multi_discrete env returns one per dimension."""
    if isinstance(masks, tuple):
        return np.concatenate([m.astype(np.bool_) for m in masks])
    return np.asarray(masks)


def _action_type_name(at: Any) -> str | None:
    if isinstance(at, (int, np.integer)) and 0 <= int(at) < len(ACTION_TYPE_NAMES):
        return ACTION_TYPE_NAMES[int(at)]
    return None


class _EpisodeTracker:
    """Accumulates one episode: its return, length, optional breakdown and trace."""

    def __init__(
        self,
        index: int,
        *,
        seed: int | None,
        seat: int | None,
        agent_player: Any,
        track_breakdown: bool,
        keep_trace: bool,
    ) -> None:
        self.index = index
        self.seed = seed
        self.seat = seat
        self.agent_player = agent_player
        self.reward = 0.0
        self.length = 0
        self.action_counts = {name: 0 for name in ACTION_TYPE_NAMES} if track_breakdown else None
        self.reward_components = {name: 0.0 for name in REWARD_COMPONENTS} if track_breakdown else None
        self.trace: list[dict] | None = [] if keep_trace else None

    def on_step(self, action: Any, reward: Any, terminated: bool, truncated: bool, info: Mapping[str, Any]) -> None:
        self.reward += float(reward)
        self.length += 1
        if self.action_counts is not None and self.reward_components is not None:
            name = _action_type_name(info.get("action_type"))
            if name is not None:
                self.action_counts[name] += 1
            for k, v in info.get("reward_breakdown", {}).items():
                if k in self.reward_components:
                    self.reward_components[k] += float(v)
        if self.trace is not None:
            at_idx = info.get("action_type")
            self.trace.append(
                {
                    "step": self.length,
                    "action_index": int(np.asarray(action).flatten()[0]) if np.ndim(action) else int(action),
                    "action_type": int(at_idx) if isinstance(at_idx, (int, np.integer)) else None,
                    "action_type_name": _action_type_name(at_idx),
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

    def finish(
        self,
        info: Mapping[str, Any],
        *,
        trace_dir: Path | None,
        trace_triggers: set,
        seat_tag: bool,
    ) -> dict[str, Any]:
        """The episode's record; dumps its trace when the end reason is a trigger."""
        episode_stats = info.get("episode_stats", {}) or {}
        winner = episode_stats.get("winner", info.get("winner"))
        agent_player = self.agent_player if self.agent_player is not None else 1
        if winner == agent_player:
            outcome = "wins"
        elif winner is not None:
            outcome = "losses"
        else:
            outcome = "draws"
        reason = info.get("end_reason")
        trace_path: str | None = None
        if self.trace is not None and reason in trace_triggers and trace_dir is not None:
            # Created lazily so eval blocks with no trigger-matching episode
            # leave no empty folder behind.
            trace_dir.mkdir(parents=True, exist_ok=True)
            seat_part = f"_seat{self.seat}" if seat_tag and self.seat is not None else ""
            trace_file = trace_dir / f"episode_{self.index:04d}{seat_part}_{reason}.jsonl"
            with trace_file.open("w") as fh:
                header = {
                    "episode_index": self.index,
                    "seed": self.seed,
                    "end_reason": reason,
                    "outcome": outcome,
                    "winner": winner,
                    "agent_player": int(agent_player) if agent_player is not None else None,
                    "ep_length": self.length,
                    "ep_reward": float(self.reward),
                    "final_turn": int(info.get("turn", 0)),
                }
                fh.write(json.dumps({"_header": header}) + "\n")
                for record in self.trace:
                    fh.write(json.dumps(record) + "\n")
            trace_path = str(trace_file)
        return {
            "index": self.index,
            "seat": int(agent_player) if isinstance(agent_player, (int, np.integer)) else agent_player,
            "reward": self.reward,
            "length": self.length,
            "turn": int(info.get("turn", 0)),
            "outcome": outcome,
            "end_reason": reason,
            "episode_stats": episode_stats,
            "action_counts": self.action_counts,
            "reward_components": self.reward_components,
            "trace_path": trace_path,
        }


def _outcome_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Win / loss / draw counts and rates (plus mean reward) of ``records``."""
    n = len(records)
    wins = sum(1 for r in records if r["outcome"] == "wins")
    losses = sum(1 for r in records if r["outcome"] == "losses")
    draws = n - wins - losses
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "episodes": n,
        "win_rate": wins / n if n else 0.0,
        "loss_rate": losses / n if n else 0.0,
        "draw_rate": draws / n if n else 0.0,
        "avg_reward": float(np.mean([r["reward"] for r in records])) if n else 0.0,
    }


def _aggregate(records: Sequence[Mapping[str, Any]], *, track_breakdown: bool, traced: bool) -> dict[str, Any]:
    """The evaluation result for per-episode ``records`` (in episode order)."""
    n_episodes = len(records)
    outcome_reasons = {f"{outcome}_by_{reason}": 0 for outcome in ("wins", "losses", "draws") for reason in END_REASONS}
    end_reasons = {reason: 0 for reason in END_REASONS}
    units_built = {ut: 0 for ut in UNIT_TYPE_LETTERS}
    combat_stats = {k: 0.0 for k in (*COMBAT_STAT_KEYS, *HEALING_STAT_KEYS)}
    captures_by_type = {k: 0 for k in CAPTURE_STRUCTURE_TYPES}
    seize_available_steps_total = 0
    steps_total = 0
    max_legal_actions = 0
    truncated_steps_total = 0
    peak_own_units = 0
    own_units_sum_total = 0
    peak_gold_banked = 0.0
    gold_banked_sum_total = 0.0
    action_counts = {name: 0 for name in ACTION_TYPE_NAMES}
    reward_components = {name: 0.0 for name in REWARD_COMPONENTS}

    for rec in records:
        reason = rec["end_reason"]
        if reason in END_REASONS:
            end_reasons[reason] += 1
            outcome_reasons[f"{rec['outcome']}_by_{reason}"] += 1
        episode_stats = rec["episode_stats"]
        # Older envs that don't populate these keys contribute zeros.
        for ut, count in (episode_stats.get("units_built") or {}).items():
            if ut in units_built:
                units_built[ut] += int(count)
        for key in (*COMBAT_STAT_KEYS, *HEALING_STAT_KEYS):
            val = episode_stats.get(key)
            if val is not None:
                combat_stats[key] += float(val)
        ep_captures_by_type = episode_stats.get("captures_by_type") or {}
        for key in CAPTURE_STRUCTURE_TYPES:
            val = ep_captures_by_type.get(key)
            if val is not None:
                captures_by_type[key] += int(val)
        # Action-space diagnostics: ``seize_available_steps`` / total steps is
        # the fraction of decision points where a capture was legal;
        # ``max_legal_actions`` is the peak legal-set size (flat_discrete
        # truncation guardrail); ``truncated_steps`` counts decision points
        # whose flat table was cut to max_flat_actions.
        seize_available_steps_total += int(episode_stats.get("seize_available_steps", 0) or 0)
        steps_total += rec["length"]
        max_legal_actions = max(max_legal_actions, int(episode_stats.get("max_legal_actions", 0) or 0))
        truncated_steps_total += int(episode_stats.get("truncated_steps", 0) or 0)
        # Army economy: a high peak army with near-zero banked gold is the
        # economy-funds-mass signature.
        peak_own_units = max(peak_own_units, int(episode_stats.get("peak_own_units", 0) or 0))
        own_units_sum_total += int(episode_stats.get("own_units_sum", 0) or 0)
        peak_gold_banked = max(peak_gold_banked, float(episode_stats.get("peak_gold_banked", 0.0) or 0.0))
        gold_banked_sum_total += float(episode_stats.get("gold_banked_sum", 0.0) or 0.0)
        if track_breakdown:
            for name, count in (rec["action_counts"] or {}).items():
                action_counts[name] += count
            for name, value in (rec["reward_components"] or {}).items():
                reward_components[name] += value

    rewards_arr = np.array([r["reward"] for r in records], dtype=float)
    lengths_arr = np.array([r["length"] for r in records], dtype=float)
    turns_arr = np.array([r["turn"] for r in records], dtype=float)
    summary = _outcome_summary(records)

    # Per-seat outcomes, keyed by the seat as a string (JSON object keys).
    by_seat: dict[str, dict[str, Any]] = {}
    for seat in sorted({r["seat"] for r in records if r["seat"] is not None}, key=str):
        by_seat[str(seat)] = _outcome_summary([r for r in records if r["seat"] == seat])

    result: dict[str, Any] = {
        "win_rate": summary["win_rate"],
        "avg_reward": float(rewards_arr.mean()) if n_episodes > 0 else 0.0,
        "std_reward": float(rewards_arr.std()) if n_episodes > 0 else 0.0,
        "avg_length": float(lengths_arr.mean()) if n_episodes > 0 else 0.0,
        "std_length": float(lengths_arr.std()) if n_episodes > 0 else 0.0,
        "avg_turns": float(turns_arr.mean()) if n_episodes > 0 else 0.0,
        "std_turns": float(turns_arr.std()) if n_episodes > 0 else 0.0,
        "wins": summary["wins"],
        "losses": summary["losses"],
        "draws": summary["draws"],
        "episodes": n_episodes,
        # Draws reported on their own, not folded into "not a win".
        "loss_rate": summary["loss_rate"],
        "draw_rate": summary["draw_rate"],
        "rewards": [float(r["reward"]) for r in records],
        "lengths": [int(r["length"]) for r in records],
        "turns": [int(r["turn"]) for r in records],
        "outcomes": [str(r["outcome"]) for r in records],
        "seats": [r["seat"] for r in records],
        "by_seat": by_seat,
        "end_reasons": end_reasons,
        "outcome_reasons": outcome_reasons,
        "units_built": units_built,
        "combat_stats": combat_stats,
        "captures_by_type": captures_by_type,
        "seize_available_rate": (seize_available_steps_total / steps_total) if steps_total > 0 else 0.0,
        "max_legal_actions": int(max_legal_actions),
        "flat_truncated_rate": (truncated_steps_total / steps_total) if steps_total > 0 else 0.0,
        "peak_own_units": int(peak_own_units),
        "mean_own_units": (own_units_sum_total / steps_total) if steps_total > 0 else 0.0,
        "peak_gold_banked": float(peak_gold_banked),
        "mean_gold_banked": (gold_banked_sum_total / steps_total) if steps_total > 0 else 0.0,
    }
    if track_breakdown:
        result["action_counts"] = action_counts
        result["reward_components"] = reward_components
    if traced:
        result["traces"] = [r["trace_path"] for r in records if r["trace_path"] is not None]
    return result


def _episode_plan(n_episodes: int, seats: Sequence[int] | None) -> list[tuple[int | None, int]]:
    """``(seat, i)`` per episode: ``n_episodes`` per seat, seat-major, the same ``i`` (seed) per seat."""
    if seats is None:
        return [(None, i) for i in range(n_episodes)]
    seats = [int(s) for s in seats]
    if not seats or any(s not in (1, 2) for s in seats) or len(set(seats)) != len(seats):
        raise ValueError(f"seats must be a non-empty list of distinct seats from (1, 2), got {seats!r}")
    return [(seat, i) for seat in seats for i in range(n_episodes)]


def _trace_settings(trace_dir: str | Path | None, trace_end_reasons: tuple | None) -> tuple[Path | None, set]:
    triggers = set(trace_end_reasons or ())
    if trace_dir is None or not triggers:
        return None, triggers
    return Path(trace_dir), triggers


def _agent_player_of(env: Any) -> Any:
    agent_player = getattr(env, "agent_player", None)
    if agent_player is None:
        agent_player = getattr(getattr(env, "unwrapped", None), "agent_player", 1)
    return agent_player


def evaluate_model(
    model: Any,
    env: Any,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Any = None,
    track_breakdown: bool = False,
    trace_dir: str | Path | None = None,
    trace_end_reasons: tuple | None = ("max_steps_truncate",),
    seats: Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a trained model and return summary statistics.

    Works with both MaskablePPO (using action_masks) and standard PPO.
    The environment can be a raw gym env or an ActionMaskedEnv wrapper.

    Args:
        model: Trained SB3 model (PPO, MaskablePPO, etc.)
        env: Gymnasium environment (single, not vectorized).
        n_episodes: Number of evaluation episodes (per seat when ``seats``
            is given).
        deterministic: Use deterministic actions. The curriculum's gate
            passes ``eval.eval_deterministic`` (False by default: the
            stochastic policy PPO trains).
        seed: Optional integer seed. When provided, episode ``i`` is reset
            with ``seed + i`` so results are reproducible across runs; with
            ``deterministic=False`` the policy's samples then also come from
            a stream seeded per episode (:func:`policy_sampling_seed`), and
            torch's global generator is left as it was.
        track_breakdown: When True, also accumulate per-step
            ``info["action_type"]`` counts and ``info["reward_breakdown"]``
            sums across the evaluation, returned under ``action_counts``
            and ``reward_components``. Disabled by default to keep the hot
            path lean; the per-step ``info`` dict is read either way.
        seats: Seats to evaluate, e.g. ``[1, 2]``: ``n_episodes`` per seat
            with the same seeds, the env's seat set with
            ``set_agent_seat`` before each reset (and restored after).
            ``None`` plays the env's own seat (the historical behaviour).

    Returns:
        Dict with keys: win_rate, avg_reward, std_reward, avg_length,
        std_length, wins, losses, draws, episodes, rewards, lengths.

        ``rewards`` and ``lengths`` are the raw per-episode arrays (lists
        of plain floats / ints), exposed so callers can plot full
        distributions rather than only the mean ± std summary;
        ``outcomes`` and ``seats`` are the per-episode outcome
        (wins / losses / draws) and agent seat. ``draw_rate`` /
        ``loss_rate`` report draws apart from losses, and ``by_seat`` maps
        each seat played (as a string) to its wins / losses / draws /
        episodes / rates / avg_reward.

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
        ``<trace_dir>/episode_<ep_idx>_<end_reason>.jsonl`` (with a
        ``_seat<k>`` tag when several seats are evaluated) -- one line
        per env step with the chosen action, env info, and reward.
        Default trigger captures only ``max_steps_truncate`` episodes
        (the stalling failure mode); pass an empty tuple to disable.
        The returned dict gains a ``traces`` list of dumped file paths.

    Raises:
        FlatActionVersionMismatch: A flat_discrete ``model`` whose decode
            table (``flat_action_version_of(model)``) differs from the
            env's: the evaluation would score actions the policy never
            chose. Build the env with the model's version.
    """
    # Imported here: gym_env pulls in the engine, and this module stays
    # importable on its own (see UNIT_TYPE_LETTERS).
    from reinforcetactics.rl.gym_env import check_flat_action_version

    check_flat_action_version(model, env, what="the evaluation env")
    plan = _episode_plan(n_episodes, seats)
    trace_dir_path, trace_triggers = _trace_settings(trace_dir, trace_end_reasons)
    seat_tag = seats is not None and len(seats) > 1
    # Only forward action masks to ``model.predict`` when the model itself
    # accepts them. Plain ``stable_baselines3.PPO.predict`` will raise
    # ``TypeError`` on an unexpected ``action_masks`` kwarg, which would
    # break eval for the ``ppo_baseline.yaml`` path (vanilla PPO behind a
    # mask-exposing env wrapper).
    has_action_masks = hasattr(env, "action_masks") and _model_accepts_action_masks(model)

    set_seat = getattr(env, "set_agent_seat", None) if seats is not None else None
    if seats is not None and set_seat is None and any(s != _agent_player_of(env) for s in seats):
        raise ValueError(f"evaluate_model(seats={list(seats)}): the env has no set_agent_seat() to change seats with")
    previous_seat = getattr(env, "agent_seat", None) if set_seat is not None else None

    records: list[dict[str, Any]] = []
    # A seeded stochastic eval samples from its own stream (module docstring).
    with _isolated_policy_rng(model, enabled=not deterministic and seed is not None) as reseed:
        _evaluate_serial(
            model,
            env,
            plan,
            records,
            deterministic=deterministic,
            seed=seed,
            reseed=reseed,
            set_seat=set_seat,
            previous_seat=previous_seat,
            has_action_masks=has_action_masks,
            track_breakdown=track_breakdown,
            trace_dir_path=trace_dir_path,
            trace_triggers=trace_triggers,
            seat_tag=seat_tag,
        )
    return _aggregate(records, track_breakdown=track_breakdown, traced=trace_dir_path is not None)


def _evaluate_serial(
    model: Any,
    env: Any,
    plan: Sequence[tuple[int | None, int]],
    records: list[dict[str, Any]],
    *,
    deterministic: bool,
    seed: Any,
    reseed: Callable[[int], Any] | None,
    set_seat: Callable[[Any], Any] | None,
    previous_seat: Any,
    has_action_masks: bool,
    track_breakdown: bool,
    trace_dir_path: Path | None,
    trace_triggers: set,
    seat_tag: bool,
) -> None:
    """:func:`evaluate_model`'s episode loop: plays ``plan`` and appends one record per episode."""
    try:
        for index, (seat, i) in enumerate(plan):
            if seat is not None and set_seat is not None:
                set_seat(seat)
            ep_seed = int(seed) + i if seed is not None else None
            if reseed is not None and ep_seed is not None:
                reseed(policy_sampling_seed(ep_seed, seat))
            obs, _ = env.reset(seed=ep_seed) if ep_seed is not None else env.reset()
            tracker = _EpisodeTracker(
                index,
                seed=ep_seed,
                seat=seat,
                agent_player=_agent_player_of(env),
                track_breakdown=track_breakdown,
                keep_trace=trace_dir_path is not None,
            )
            done = False
            info: dict[str, Any] = {}
            while not done:
                predict_kwargs: dict[str, Any] = {"deterministic": deterministic}
                if has_action_masks:
                    predict_kwargs["action_masks"] = _flat_mask(env.action_masks())
                action, _ = model.predict(obs, **predict_kwargs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                tracker.on_step(action, reward, terminated, truncated, info)
            records.append(tracker.finish(info, trace_dir=trace_dir_path, trace_triggers=trace_triggers, seat_tag=seat_tag))
    finally:
        if set_seat is not None and previous_seat is not None:
            set_seat(previous_seat)


# ---------------------------------------------------------------------------
# Vectorized evaluation (review rltrain-11)
# ---------------------------------------------------------------------------


def _env_reset(env: Any, seed: int | None, seat: int | None, want_masks: bool) -> tuple[Any, Any, Any]:
    if seat is not None:
        env.set_agent_seat(seat)
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    mask = _flat_mask(env.action_masks()) if want_masks else None
    return obs, _agent_player_of(env), mask


def _env_step(env: Any, action: Any, want_masks: bool) -> tuple[Any, float, bool, bool, dict, Any]:
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    mask = _flat_mask(env.action_masks()) if want_masks and not done else None
    return obs, reward, bool(terminated), bool(truncated), info, mask


def _env_probe(env: Any) -> dict[str, Any]:
    base = getattr(env, "unwrapped", env)
    return {
        "has_action_masks": hasattr(env, "action_masks"),
        "has_seats": hasattr(env, "set_agent_seat"),
        "agent_seat": getattr(env, "agent_seat", None),
        "action_space_type": getattr(base, "action_space_type", None),
        "flat_action_version": getattr(base, "flat_action_version", None),
    }


class _LocalEvalPool:
    """K eval envs stepped in this process; only the slots with an episode in play step."""

    def __init__(self, envs: Sequence[Any]) -> None:
        if not envs:
            raise ValueError("an eval pool needs at least one env")
        self.envs = list(envs)
        self.num_envs = len(self.envs)
        self.want_masks = False

    def probe(self) -> list[dict[str, Any]]:
        return [_env_probe(env) for env in self.envs]

    def reset(self, requests: Mapping[int, tuple[int | None, int | None]]) -> dict[int, tuple[Any, Any, Any]]:
        return {j: _env_reset(self.envs[j], seed, seat, self.want_masks) for j, (seed, seat) in requests.items()}

    def step(self, actions: Mapping[int, Any]) -> dict[int, tuple[Any, float, bool, bool, dict, Any]]:
        return {j: _env_step(self.envs[j], action, self.want_masks) for j, action in actions.items()}

    def set_seat(self, j: int, seat: Any) -> None:
        self.envs[j].set_agent_seat(seat)

    def close(self) -> None:
        for env in self.envs:
            close = getattr(env, "close", None)
            if close is not None:
                close()


def _pool_worker(remote: Any, parent_remote: Any, env_fn_wrapper: Any) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                seed, seat, want_masks = data
                remote.send(_env_reset(env, seed, seat, want_masks))
            elif cmd == "step":
                action, want_masks = data
                remote.send(_env_step(env, action, want_masks))
            elif cmd == "probe":
                remote.send(_env_probe(env))
            elif cmd == "set_seat":
                env.set_agent_seat(data)
                remote.send(None)
            elif cmd == "close":
                close = getattr(env, "close", None)
                if close is not None:
                    close()
                remote.close()
                break
            else:  # pragma: no cover - protocol error
                raise NotImplementedError(cmd)
    except (KeyboardInterrupt, EOFError):
        pass


class _ProcessEvalPool:
    """K eval envs, one worker process each; the parent batches ``predict``.

    Unlike SB3's ``SubprocVecEnv`` a worker never auto-resets and only the
    slots with an episode in play step, so an eval plays exactly its
    episodes, with the env's own (float64) rewards.
    """

    def __init__(self, env_fns: Sequence[Callable[[], Any]], start_method: str | None = None) -> None:
        from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

        if not env_fns:
            raise ValueError("an eval pool needs at least one env")
        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx: Any = mp.get_context(start_method)
        self.num_envs = len(env_fns)
        self.want_masks = False
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(work_remotes, self.remotes, env_fns, strict=True):
            process = ctx.Process(target=_pool_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)), daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self._closed = False

    def _call(self, commands: Mapping[int, tuple[str, Any]]) -> dict[int, Any]:
        # Send to every targeted worker first, then collect: the workers run
        # their env step in parallel.
        for j, command in commands.items():
            self.remotes[j].send(command)
        return {j: self.remotes[j].recv() for j in commands}

    def probe(self) -> list[dict[str, Any]]:
        out = self._call({j: ("probe", None) for j in range(self.num_envs)})
        return [out[j] for j in range(self.num_envs)]

    def reset(self, requests: Mapping[int, tuple[int | None, int | None]]) -> dict[int, tuple[Any, Any, Any]]:
        return self._call({j: ("reset", (seed, seat, self.want_masks)) for j, (seed, seat) in requests.items()})

    def step(self, actions: Mapping[int, Any]) -> dict[int, tuple[Any, float, bool, bool, dict, Any]]:
        return self._call({j: ("step", (action, self.want_masks)) for j, action in actions.items()})

    def set_seat(self, j: int, seat: Any) -> None:
        self._call({j: ("set_seat", seat)})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


class EvalEnvPool:
    """K copies of one eval env for :func:`evaluate_model_vec`.

    Build it from env factories (``use_subprocess`` runs each env in its
    own worker process) or wrap existing in-process envs with
    :meth:`from_envs`. Close it when done.

    It exposes enough of an env for the curriculum's checks
    (``action_space_type`` / ``flat_action_version`` of its envs, and
    ``close``).
    """

    def __init__(self, env_fns: Sequence[Callable[[], Any]], *, use_subprocess: bool = False) -> None:
        if use_subprocess:
            self._pool: Any = _ProcessEvalPool(env_fns)
        else:
            self._pool = _LocalEvalPool([fn() for fn in env_fns])
        self._init_probe()

    @classmethod
    def from_envs(cls, envs: Sequence[Any]) -> EvalEnvPool:
        pool = cls.__new__(cls)
        pool._pool = _LocalEvalPool(envs)
        pool._init_probe()
        return pool

    def _init_probe(self) -> None:
        self._probe = self._pool.probe()
        kinds = {p["action_space_type"] for p in self._probe}
        versions = {p["flat_action_version"] for p in self._probe}
        self.action_space_type = kinds.pop() if len(kinds) == 1 else None
        self.flat_action_version = versions.pop() if len(versions) == 1 else None

    @property
    def num_envs(self) -> int:
        return int(self._pool.num_envs)

    @property
    def unwrapped(self) -> EvalEnvPool:
        return self

    def close(self) -> None:
        self._pool.close()


def evaluate_model_vec(
    model: Any,
    pool: EvalEnvPool,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Any = None,
    track_breakdown: bool = False,
    trace_dir: str | Path | None = None,
    trace_end_reasons: tuple | None = ("max_steps_truncate",),
    seats: Sequence[int] | None = None,
) -> dict[str, Any]:
    """:func:`evaluate_model` over the K envs of ``pool``, one batched ``predict`` per step.

    Every episode of :func:`evaluate_model`'s plan (``n_episodes`` per seat,
    episode ``i`` reset with ``seed + i``) goes, in order, to whichever env
    is free next; the K in-play observations (and action masks) are stacked
    into one ``model.predict`` call per step. Results are aggregated in
    episode order, so the returned dict -- keys, per-episode lists and all
    -- equals :func:`evaluate_model`'s for the same arguments whenever the
    policy acts deterministically (``deterministic=True``). With sampled
    actions and a ``seed`` the samples come from a stream of their own,
    seeded once per call (the global generator is left as it was): the
    result is reproducible for the same pool size, but the draws differ
    from the serial path's (see the module docstring).

    The envs' seats are set per episode when ``seats`` is given and
    restored afterwards. ``pool`` stays open (the caller owns it).
    """
    from reinforcetactics.rl.gym_env import check_flat_action_version

    check_flat_action_version(model, pool, what="the evaluation envs")
    plan = _episode_plan(n_episodes, seats)
    trace_dir_path, trace_triggers = _trace_settings(trace_dir, trace_end_reasons)
    seat_tag = seats is not None and len(seats) > 1
    probe = pool._probe
    want_masks = all(p["has_action_masks"] for p in probe) and _model_accepts_action_masks(model)
    inner = pool._pool
    inner.want_masks = want_masks
    if seats is not None and not all(p["has_seats"] for p in probe):
        raise ValueError(f"evaluate_model_vec(seats={list(seats)}): the pool's envs have no set_agent_seat()")

    records: list[dict[str, Any] | None] = [None] * len(plan)
    trackers: list[_EpisodeTracker | None] = [None] * inner.num_envs
    obs: list[Any] = [None] * inner.num_envs
    masks: list[Any] = [None] * inner.num_envs
    next_item = 0

    def assign(slots: Sequence[int]) -> None:
        nonlocal next_item
        requests: dict[int, tuple[int | None, int | None]] = {}
        planned: dict[int, tuple[int, int | None, int | None]] = {}
        for j in slots:
            if next_item >= len(plan):
                trackers[j] = None
                continue
            seat, i = plan[next_item]
            ep_seed = int(seed) + i if seed is not None else None
            requests[j] = (ep_seed, seat)
            planned[j] = (next_item, ep_seed, seat)
            next_item += 1
        for j, (ob, agent_player, mask) in inner.reset(requests).items():
            index, ep_seed, seat = planned[j]
            obs[j], masks[j] = ob, mask
            trackers[j] = _EpisodeTracker(
                index,
                seed=ep_seed,
                seat=seat,
                agent_player=agent_player,
                track_breakdown=track_breakdown,
                keep_trace=trace_dir_path is not None,
            )

    # A seeded stochastic eval samples from its own stream, seeded once:
    # one batched predict draws every in-play env's action together.
    with _isolated_policy_rng(model, enabled=not deterministic and seed is not None) as reseed:
        if reseed is not None:
            reseed(policy_sampling_seed(int(seed)))
        try:
            assign(range(inner.num_envs))
            while True:
                active = [j for j in range(inner.num_envs) if trackers[j] is not None]
                if not active:
                    break
                batch_obs = _stack_obs([obs[j] for j in active])
                predict_kwargs: dict[str, Any] = {"deterministic": deterministic}
                if want_masks:
                    predict_kwargs["action_masks"] = np.stack([masks[j] for j in active])
                actions, _ = model.predict(batch_obs, **predict_kwargs)
                results = inner.step({j: actions[k] for k, j in enumerate(active)})
                finished = []
                for k, j in enumerate(active):
                    ob, reward, terminated, truncated, info, mask = results[j]
                    tracker = trackers[j]
                    assert tracker is not None
                    tracker.on_step(actions[k], reward, terminated, truncated, info)
                    if terminated or truncated:
                        records[tracker.index] = tracker.finish(
                            info, trace_dir=trace_dir_path, trace_triggers=trace_triggers, seat_tag=seat_tag
                        )
                        finished.append(j)
                    else:
                        obs[j], masks[j] = ob, mask
                if finished:
                    assign(finished)
        finally:
            if seats is not None:
                for j in range(inner.num_envs):
                    previous = probe[j]["agent_seat"]
                    if previous is not None:
                        inner.set_seat(j, previous)
    done = [r for r in records if r is not None]
    assert len(done) == len(plan), "every planned episode must finish"
    return _aggregate(done, track_breakdown=track_breakdown, traced=trace_dir_path is not None)


def _stack_obs(observations: Sequence[Any]) -> Any:
    """Stack single-env observations (dicts of arrays, or arrays) along a new batch axis."""
    first = observations[0]
    if isinstance(first, Mapping):
        return {key: np.stack([np.asarray(o[key]) for o in observations]) for key in first}
    return np.stack([np.asarray(o) for o in observations])
