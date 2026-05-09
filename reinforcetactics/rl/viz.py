"""
Plotting helpers for RL training notebooks.

These functions consume the per-eval ``results`` list produced by
``PeriodicEvalCallback`` and the per-rollout ``records`` list produced by
``TrainingMetricsCallback``. They return the matplotlib ``Figure`` so
callers can ``fig.savefig(...)`` or further customize before display.

Kept deliberately minimal — three plot families that together cover the
"why isn't PPO learning?" debugging surface:

- ``plot_eval_curves``: win rate / reward / episode length curves +
  PPO update health (approx_kl, explained_variance, value_loss) on
  shared axes. Supports ``stage_boundaries`` for the bootstrap-curriculum
  view where a single metrics timeline spans multiple stages.
- ``plot_reward_decomposition``: stacked area of action / shaping /
  invalid-penalty / terminal contributions per eval. Distinguishes
  "shaping-reward stalemate" from "actually winning".
- ``plot_action_distribution``: stacked area of action-type usage % per
  eval. Surfaces failure modes like "agent collapsed to spamming end_turn".
- ``plot_unit_build_distribution``: stacked area of unit-type build mix
  per eval. Distinguishes a diversified build order from one-trick army
  collapse.
- ``plot_combat_summary``: per-game averages of captures / kills /
  damage dealt vs taken / attack-seize activity per eval. The damage
  delta panel surfaces whether the agent is actually winning trades
  beyond what the win-rate scalar captures.
- ``plot_individual_game_stats``: 2x3 panel summarising a single
  recorded game (consumes ``record_evaluation_to_video``'s return
  dict). Independent of the per-eval aggregates above; surfaces
  what the policy *actually does* in one playthrough.
- ``plot_episode_length``: standalone mean-episode-length-per-eval
  chart. Small enough to be its own panel; useful for spotting the
  saturate-at-cap pattern that broader dashboards can hide.
- ``plot_curriculum_summary``: multi-stage win-rate timeline used by
  the bootstrap notebook to show stage progression on a single axis.
- ``plot_stage_diagnostics``: convenience wrapper that runs the full
  per-stage diagnostic suite (length / outcome / reward / action /
  unit build / combat) into one ``charts_dir``. Used by both
  notebooks so the produced filenames stay identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from reinforcetactics.rl.evaluation import (
    ACTION_TYPE_NAMES,
    COMBAT_STAT_KEYS,
    END_REASONS,
    REWARD_COMPONENTS,
    UNIT_TYPE_LETTERS,
)

# Stable colour assignments so cross-figure comparisons stay coherent.
_REWARD_COMPONENT_COLORS = {
    "action": "#4caf50",  # green — per-action shaping
    "shaping_delta": "#2196f3",  # blue  — potential-based ΔΦ
    "invalid_penalty": "#9e9e9e",  # grey  — penalty
    "terminal": "#ff9800",  # orange — win/loss/draw
}

_ACTION_COLORS = {
    "create_unit": "#2196f3",
    "move": "#4caf50",
    "attack": "#f44336",
    "seize": "#ff9800",
    "heal": "#e91e63",
    "end_turn": "#9e9e9e",
    "paralyze": "#9c27b0",
    "haste": "#00bcd4",
    "defence_buff": "#795548",
    "attack_buff": "#ff5722",
}

# Unit-type colours for the build-mix stack. Distinct hues per role
# (warrior=red, mage=purple, archer=orange, etc.) so the chart doubles
# as a quick "is the agent diversifying its army?" read.
_UNIT_TYPE_COLORS = {
    "W": "#d32f2f",  # Warrior
    "M": "#7b1fa2",  # Mage
    "C": "#388e3c",  # Cleric
    "A": "#f57c00",  # Archer
    "K": "#5d4037",  # Knight
    "R": "#1976d2",  # Rogue
    "S": "#00838f",  # Sorcerer
    "B": "#455a64",  # Berserker
}

# Combat-stat colours for the combat summary panel.
_COMBAT_STAT_COLORS = {
    "captures": "#ff9800",
    "kills": "#f44336",
    "damage_dealt": "#4caf50",
    "damage_taken": "#9e9e9e",
    "attacks": "#2196f3",
    "seize_attempts": "#ffb74d",
}


def _save_and_return(fig: plt.Figure, charts_dir: Optional[Any], name: str) -> plt.Figure:
    if charts_dir is not None:
        path = Path(charts_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig


def plot_eval_curves(
    results: Sequence[dict],
    train_records: Optional[Sequence[dict]] = None,
    *,
    opponent_label: str = "",
    charts_dir: Optional[Any] = None,
    stage_boundaries: Optional[Sequence[int]] = None,
) -> plt.Figure:
    """Win rate / avg reward / episode length / approx_kl / explained_variance / value_loss.

    Layout depends on whether ``train_records`` is provided:

    - ``train_records=None`` → 1x3 row of eval-only panels (win rate,
      avg reward, episode length).
    - ``train_records=[...]`` → 2x3 grid. Top row is the eval panels;
      bottom row is the PPO update health panels (approx_kl,
      explained_variance, value_loss). Read the bottom row for
      *whether the policy is updating at all* (approx_kl) and *whether
      the value head is fitting the returns* (explained_variance,
      value_loss). Both health panels matter when reward magnitudes
      are large (terminal bonuses in the thousands): explained_variance
      should climb toward 1.0; value_loss should plateau or decline.
      A value_loss curve that grows without bound while
      explained_variance stays near zero is the signature of a value
      head that can't keep up with the target distribution.

    Args:
        results: Per-eval dicts from ``PeriodicEvalCallback.results``.
        train_records: Per-rollout dicts from
            ``TrainingMetricsCallback.records``. Optional.
        opponent_label: Suffix for the figure suptitle (e.g. ``"random"``).
        charts_dir: Optional save directory; PNG named ``eval_curves.png``.
        stage_boundaries: Optional list of cumulative timesteps at which
            to draw vertical stage-transition lines on every panel.
            Useful for the bootstrap-curriculum view where one
            ``metrics_callback`` spans multiple stages.
    """
    if train_records:
        fig, axes_grid = plt.subplots(2, 3, figsize=(15, 8))
        axes = list(axes_grid.flatten())
    else:
        fig, axes_row = plt.subplots(1, 3, figsize=(15, 4))
        axes = list(axes_row)

    eval_ts = [r["timesteps"] for r in results]

    def _draw_stage_lines(ax: Any) -> None:
        if not stage_boundaries:
            return
        for ts in stage_boundaries:
            ax.axvline(ts, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    # Panel 1: win rate
    ax = axes[0]
    wr = [r["win_rate"] * 100 for r in results]
    ax.plot(eval_ts, wr, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title(f"Win Rate vs {opponent_label}" if opponent_label else "Win Rate")
    # Linear x-axis matching the other four panels in this row. The
    # previous log scale was added to surface the t=4 baseline eval but
    # made the panel hard to compare with the rest of the dashboard.
    ax.set_ylim(-5, 105)
    ax.axhline(y=70, color="green", linestyle="--", alpha=0.5, label="70% target")
    _draw_stage_lines(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: avg reward (with std band)
    ax = axes[1]
    avg_r = [r["avg_reward"] for r in results]
    std_r = [r["std_reward"] for r in results]
    ax.plot(eval_ts, avg_r, "o-", color="#FF9800", linewidth=2, markersize=6)
    ax.fill_between(
        eval_ts,
        [a - s for a, s in zip(avg_r, std_r)],
        [a + s for a, s in zip(avg_r, std_r)],
        alpha=0.2,
        color="#FF9800",
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Episode Reward")
    _draw_stage_lines(ax)
    ax.grid(True, alpha=0.3)

    # Panel 3: episode length
    ax = axes[2]
    avg_l = [r["avg_length"] for r in results]
    std_l = [r["std_length"] for r in results]
    ax.plot(eval_ts, avg_l, "o-", color="#4CAF50", linewidth=2, markersize=6)
    ax.fill_between(
        eval_ts,
        [a - s for a, s in zip(avg_l, std_l)],
        [a + s for a, s in zip(avg_l, std_l)],
        alpha=0.2,
        color="#4CAF50",
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Average Length (steps)")
    ax.set_title("Average Episode Length")
    _draw_stage_lines(ax)
    ax.grid(True, alpha=0.3)

    if train_records:
        train_ts = [r["timesteps"] for r in train_records]

        # Panel 4: approx_kl — near zero means policy is barely updating.
        ax = axes[3]
        kl = [r.get("train/approx_kl") for r in train_records]
        if any(v is not None for v in kl):
            ax.plot(train_ts, kl, color="#9c27b0", linewidth=1.5)
            ax.axhline(0.02, color="#4caf50", linestyle="--", alpha=0.5, label="target ~0.02")
            ax.axhline(0.05, color="#f44336", linestyle="--", alpha=0.5, label="danger >0.05")
            ax.legend(fontsize=7)
        ax.set_xlabel("Timesteps")
        ax.set_title("approx_kl (PPO update size)")
        _draw_stage_lines(ax)
        ax.grid(True, alpha=0.3)

        # Panel 5: explained_variance — near zero means value head isn't fitting.
        ax = axes[4]
        ev = [r.get("train/explained_variance") for r in train_records]
        if any(v is not None for v in ev):
            ax.plot(train_ts, ev, color="#00bcd4", linewidth=1.5)
            ax.axhline(1.0, color="#4caf50", linestyle="--", alpha=0.5, label="ideal 1.0")
            ax.axhline(0.1, color="#f44336", linestyle="--", alpha=0.5, label="danger <0.1")
            ax.legend(fontsize=7)
        ax.set_xlabel("Timesteps")
        ax.set_title("explained_variance (value fit)")
        _draw_stage_lines(ax)
        ax.grid(True, alpha=0.3)

        # Panel 6: value_loss — should plateau / decline once the value
        # head catches up to the return scale. A monotonically growing
        # curve while explained_variance stays near zero means the
        # value targets are too large for the head's capacity / lr;
        # consider scaling rewards down or adding clip_range_vf.
        # Plot on log scale because value_loss can span 4-5 orders of
        # magnitude across a run (huge at init, small once converged).
        ax = axes[5]
        vl = [r.get("train/value_loss") for r in train_records]
        if any(v is not None for v in vl):
            # Drop points with non-positive values so log scale doesn't
            # silently clip them; SB3's value_loss is non-negative
            # (MSE), but defensive against future loss formulations.
            vl_pairs = [(t, v) for t, v in zip(train_ts, vl) if v is not None and v > 0]
            if vl_pairs:
                xs, ys = zip(*vl_pairs)
                ax.plot(list(xs), list(ys), color="#e91e63", linewidth=1.5)
                ax.set_yscale("log")
        ax.set_xlabel("Timesteps")
        ax.set_title("value_loss (log scale)")
        _draw_stage_lines(ax)
        ax.grid(True, alpha=0.3, which="both")

    title = "PPO benchmarks" + (f"  |  vs {opponent_label}" if opponent_label else "")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "eval_curves.png")


def _stack_inputs(results: Sequence[dict], key: str, names: Iterable[str]) -> tuple[list[int], dict[str, list[float]]]:
    """Pivot results[i][key][name] -> {name: [values aligned with timesteps]}.

    Skips evals that don't have the breakdown dict (e.g. older runs from
    before track_breakdown was wired up).
    """
    timesteps: list[int] = []
    series: dict[str, list[float]] = {n: [] for n in names}
    for r in results:
        breakdown = r.get(key)
        if not breakdown:
            continue
        timesteps.append(int(r["timesteps"]))
        for name in series:
            series[name].append(float(breakdown.get(name, 0.0)))
    return timesteps, series


def plot_reward_decomposition(
    results: Sequence[dict],
    *,
    charts_dir: Optional[Any] = None,
) -> Optional[plt.Figure]:
    """Stacked area of summed reward components per eval.

    Reads from ``r["reward_components"]`` produced when ``evaluate_model``
    is called with ``track_breakdown=True``. Returns ``None`` if the
    breakdown is missing from every entry (older runs).

    Reading guide:
    - large green ``action`` band, ~0 ``terminal``: agent farms shaping but
      doesn't win.
    - large grey ``invalid_penalty`` band: action masks aren't being
      applied or are over-approximating.
    - growing orange ``terminal`` band over time: agent is learning to
      win.
    """
    timesteps, series = _stack_inputs(results, "reward_components", REWARD_COMPONENTS)
    if not timesteps:
        print("No reward_components recorded — re-run training with track_breakdown=True.")
        return None

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Split into positive and negative stacks so plot stays sensible when
    # invalid_penalty is large negative and terminal swings positive.
    arr = np.array([series[c] for c in REWARD_COMPONENTS], dtype=float)
    pos = np.where(arr > 0, arr, 0.0)
    neg = np.where(arr < 0, arr, 0.0)
    colors = [_REWARD_COMPONENT_COLORS[c] for c in REWARD_COMPONENTS]

    ax.stackplot(timesteps, pos, colors=colors, alpha=0.85, labels=REWARD_COMPONENTS)
    ax.stackplot(timesteps, neg, colors=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Summed reward across eval episodes")
    ax.set_title("Reward decomposition per eval")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "reward_decomposition.png")


def plot_action_distribution(
    results: Sequence[dict],
    *,
    charts_dir: Optional[Any] = None,
    drop_unused: bool = True,
) -> Optional[plt.Figure]:
    """Stacked area of action-type usage % per eval.

    Reads from ``r["action_counts"]`` produced when ``evaluate_model`` is
    called with ``track_breakdown=True``. Each bar normalizes to 100% so
    you see *which* actions the agent prefers, not how many steps it
    took. Returns ``None`` if no entry has the data.

    Reading guide:
    - dominant grey ``end_turn`` band: agent learned to skip turns to
      avoid penalties (a common collapse mode).
    - dominant blue ``create_unit`` band early, then green ``move`` /
      orange ``seize`` taking over: healthy progression.
    - large grey area at every checkpoint: agent never differentiates.
    """
    timesteps, series = _stack_inputs(results, "action_counts", ACTION_TYPE_NAMES)
    if not timesteps:
        print("No action_counts recorded — re-run training with track_breakdown=True.")
        return None

    arr = np.array([series[n] for n in ACTION_TYPE_NAMES], dtype=float)
    totals = arr.sum(axis=0)
    totals[totals == 0] = 1.0  # avoid div-by-zero when an eval somehow has no steps
    pct = 100.0 * arr / totals

    if drop_unused:
        keep = [i for i, n in enumerate(ACTION_TYPE_NAMES) if pct[i].max() > 0.5]
        if not keep:
            keep = list(range(len(ACTION_TYPE_NAMES)))
        names = [ACTION_TYPE_NAMES[i] for i in keep]
        pct = pct[keep]
    else:
        names = list(ACTION_TYPE_NAMES)

    colors = [_ACTION_COLORS.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(timesteps, pct, colors=colors, alpha=0.85, labels=names)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Action share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Action distribution per eval")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "action_distribution.png")


# Outcome × end-reason colour palette. Same hue per outcome (green=win,
# red=loss, grey=draw) with shading by reason so a stacked bar reads
# "this eval was 90% wins, almost all by HQ capture" at a glance.
_OUTCOME_REASON_COLORS = {
    "wins_by_hq_capture": "#1b5e20",
    "wins_by_elimination": "#4caf50",
    "wins_by_max_turns_draw": "#a5d6a7",  # shouldn't happen (draw isn't a win), kept for symmetry
    "wins_by_max_steps_truncate": "#c8e6c9",
    "losses_by_hq_capture": "#b71c1c",
    "losses_by_elimination": "#f44336",
    "losses_by_max_turns_draw": "#ef9a9a",
    "losses_by_max_steps_truncate": "#ffcdd2",
    "draws_by_hq_capture": "#424242",
    "draws_by_elimination": "#616161",
    "draws_by_max_turns_draw": "#9e9e9e",
    "draws_by_max_steps_truncate": "#e0e0e0",
}


def plot_outcome_breakdown(
    results: Sequence[dict],
    *,
    charts_dir: Optional[Any] = None,
    drop_unused: bool = True,
) -> Optional[plt.Figure]:
    """Stacked bar of outcome × end-reason per eval.

    Reads from ``r["outcome_reasons"]`` (always populated by
    ``evaluate_model``). Shows whether wins are coming from HQ captures
    (the intended goal) or from elimination, and whether losses are
    structural (HQ-captured by opponent) or symmetric (own units wiped
    out). Surfaces failure modes invisible in a single win-rate number.

    Reading guide:
    - mostly dark green ``wins_by_hq_capture``: agent has learned the
      win condition and converts material advantage into HQ pressure.
    - light green ``wins_by_elimination`` dominating: agent is winning
      via attrition; if the opponent never resigns this often means
      lucky outcomes rather than goal-directed play.
    - grey ``draws_by_max_turns_draw`` band: agent is timing out.
    """
    timesteps = []
    rows: list[dict] = []
    for r in results:
        if "outcome_reasons" not in r:
            continue
        timesteps.append(r["timesteps"])
        rows.append(r["outcome_reasons"])
    if not timesteps:
        print("No outcome_reasons recorded — re-run training with the updated evaluate_model.")
        return None

    keys = [f"{outcome}_by_{reason}" for outcome in ("wins", "losses", "draws") for reason in END_REASONS]
    arr = np.array([[row.get(k, 0) for row in rows] for k in keys], dtype=float)

    if drop_unused:
        keep = [i for i, _ in enumerate(keys) if arr[i].max() > 0]
        if not keep:
            keep = list(range(len(keys)))
        keys = [keys[i] for i in keep]
        arr = arr[keep]

    colors = [_OUTCOME_REASON_COLORS.get(k, "#888") for k in keys]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bottom = np.zeros(len(timesteps))
    width = max(1, int((max(timesteps) - min(timesteps)) / max(1, len(timesteps) - 1) * 0.8)) if len(timesteps) > 1 else 1
    for k, color, counts in zip(keys, colors, arr):
        ax.bar(timesteps, counts, bottom=bottom, color=color, label=k, width=width, edgecolor="white", linewidth=0.3)
        bottom = bottom + counts

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episodes")
    ax.set_title("Eval outcome × end-reason per eval")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "outcome_breakdown.png")


def plot_unit_build_distribution(
    results: Sequence[dict],
    *,
    charts_dir: Optional[Any] = None,
    drop_unused: bool = True,
) -> Optional[plt.Figure]:
    """Stacked area of unit-build mix per eval (% of units built per type).

    Reads ``r["units_built"]`` populated by ``evaluate_model`` from the
    env's per-episode ``episode_stats``. Helpful for spotting when the
    policy collapses to a one-trick army (e.g. 100% warriors) versus
    diversifying its build order.
    """
    timesteps, series = _stack_inputs(results, "units_built", UNIT_TYPE_LETTERS)
    if not timesteps:
        print("No units_built recorded — re-run training with the updated env to populate per-episode build stats.")
        return None

    arr = np.array([series[ut] for ut in UNIT_TYPE_LETTERS], dtype=float)
    totals = arr.sum(axis=0)
    if not np.any(totals > 0):
        print("All evals report zero units built — nothing to plot.")
        return None
    safe_totals = np.where(totals == 0, 1.0, totals)
    pct = 100.0 * arr / safe_totals

    if drop_unused:
        keep = [i for i, ut in enumerate(UNIT_TYPE_LETTERS) if pct[i].max() > 0.5]
        if not keep:
            keep = list(range(len(UNIT_TYPE_LETTERS)))
        names = [UNIT_TYPE_LETTERS[i] for i in keep]
        pct = pct[keep]
    else:
        names = list(UNIT_TYPE_LETTERS)

    colors = [_UNIT_TYPE_COLORS.get(n, "#888") for n in names]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(timesteps, pct, colors=colors, alpha=0.85, labels=names)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Build share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Unit-build distribution per eval")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "unit_build_distribution.png")


def plot_combat_summary(
    results: Sequence[dict],
    *,
    charts_dir: Optional[Any] = None,
) -> Optional[plt.Figure]:
    """Per-eval combat counters normalized to a per-game average.

    Reads ``r["combat_stats"]`` (totals over the eval episodes) and
    divides by ``r["episodes"]`` to give per-game averages: captures,
    kills, damage dealt vs damage taken, and attack/seize attempt
    counts. The damage delta panel is the most useful diagnostic — a
    consistently positive gap means the agent is winning trades.
    """
    timesteps, series = _stack_inputs(results, "combat_stats", COMBAT_STAT_KEYS)
    if not timesteps:
        print("No combat_stats recorded — re-run training with the updated env.")
        return None

    n_episodes = []
    for r in results:
        if not r.get("combat_stats"):
            continue
        n_episodes.append(max(1, int(r.get("episodes", 1))))

    per_game = {k: np.array(series[k], dtype=float) / np.array(n_episodes, dtype=float) for k in COMBAT_STAT_KEYS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(timesteps, per_game["captures"], "o-", color=_COMBAT_STAT_COLORS["captures"], label="captures")
    ax.plot(timesteps, per_game["kills"], "s-", color=_COMBAT_STAT_COLORS["kills"], label="kills")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Per game (avg)")
    ax.set_title("Captures and kills per game")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(timesteps, per_game["damage_dealt"], "o-", color=_COMBAT_STAT_COLORS["damage_dealt"], label="dealt")
    ax.plot(timesteps, per_game["damage_taken"], "s-", color=_COMBAT_STAT_COLORS["damage_taken"], label="taken")
    delta = per_game["damage_dealt"] - per_game["damage_taken"]
    ax.fill_between(timesteps, delta, 0, where=delta >= 0, color="#4caf50", alpha=0.15, label="net dealt")
    ax.fill_between(timesteps, delta, 0, where=delta < 0, color="#f44336", alpha=0.15, label="net taken")
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("HP per game (avg)")
    ax.set_title("Damage dealt vs taken")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(timesteps, per_game["attacks"], "o-", color=_COMBAT_STAT_COLORS["attacks"], label="attacks")
    ax.plot(timesteps, per_game["seize_attempts"], "s-", color=_COMBAT_STAT_COLORS["seize_attempts"], label="seize attempts")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Per game (avg)")
    ax.set_title("Attack / seize activity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "combat_summary.png")


def plot_individual_game_stats(
    result: Mapping[str, Any],
    *,
    charts_dir: Optional[Any] = None,
    title_suffix: str = "",
    filename: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Render a 2x3 panel summarising one recorded game.

    Reads the dict returned by
    :func:`reinforcetactics.utils.video.record_evaluation_to_video`
    (specifically ``step_stats`` for the per-step traces, plus
    ``winner`` / ``end_reason`` / ``steps`` / ``total_reward`` for the
    outcome summary) and produces:

    - **Top row (per player over time):** unit count, gold, structures
      owned (towers + buildings + HQ).
    - **Bottom-left:** action mix used in this game, with
      ``create_unit`` split by spawned unit type (``create_W``,
      ``create_M``, …) so build order is visible at a glance. Blue
      bars = creates, green = other actions.
    - **Bottom-middle:** cumulative reward by component (``action``,
      ``shaping_delta``, ``invalid_penalty``, ``terminal``) using the
      same colour palette as :func:`plot_reward_decomposition`.
    - **Bottom-right:** outcome summary text panel — winner, end
      reason, final unit/gold/structure counts, per-type creation
      counts.

    Args:
        result: Dict from ``record_evaluation_to_video``. Must contain
            ``step_stats``; missing terminal-summary fields just leave
            their slot as "n/a".
        charts_dir: Optional directory; the figure is saved as
            ``{filename}`` (default: ``individual_game_stats[_{title_suffix}].png``).
        title_suffix: Appended to the figure's suptitle, e.g. ``"best"``
            in self-play to distinguish best-by-WR from final
            checkpoint, or the stage name in the bootstrap notebook.
        filename: Override the saved filename (without leading
            directories). Defaults to the ``title_suffix``-aware name
            above so callers normally don't need this.

    Returns:
        The matplotlib ``Figure``, or ``None`` when ``step_stats`` is
        empty (caller should treat this as "no data, skip").
    """
    step_stats = list(result.get("step_stats") or [])
    if not step_stats:
        print(
            f"[{title_suffix or 'game'}] no step_stats — skipping "
            "(re-run the recording cell with the updated record_evaluation_to_video)."
        )
        return None

    steps = np.arange(len(step_stats))
    agent_units = [s["agent_units"] for s in step_stats]
    opp_units = [s["opponent_units"] for s in step_stats]
    agent_gold = [s["agent_gold"] for s in step_stats]
    opp_gold = [s["opponent_gold"] for s in step_stats]
    agent_struct = [s.get("agent_structures", 0) for s in step_stats]
    opp_struct = [s.get("opponent_structures", 0) for s in step_stats]

    # Action counts: skip the initial snapshot (action_type=None) so
    # only the agent's actual decisions land in the bar. ``create_unit``
    # is broken out per spawned unit type, so the bar shows e.g.
    # create_W / create_M / create_A separately. Other action types
    # stay aggregated at the ACTION_TYPE_NAMES granularity.
    action_counts: dict[str, int] = {}
    for s in step_stats:
        at = s.get("action_type")
        if at is None:
            continue
        at_int = int(at)
        if at_int == 0:
            ut = s.get("unit_type")
            label = f"create_{ut}" if ut else "create_unit"
        elif 0 <= at_int < len(ACTION_TYPE_NAMES):
            label = ACTION_TYPE_NAMES[at_int]
        else:
            label = str(at_int)
        action_counts[label] = action_counts.get(label, 0) + 1

    # Cumulative reward per component. Components match the breakdown
    # built in env._calculate_reward and the constants in evaluation.py.
    cum = {c: np.zeros(len(step_stats)) for c in REWARD_COMPONENTS}
    running = {c: 0.0 for c in REWARD_COMPONENTS}
    for i, s in enumerate(step_stats):
        rb = s.get("reward_breakdown") or {}
        for c in REWARD_COMPONENTS:
            running[c] += float(rb.get(c, 0.0) or 0.0)
            cum[c][i] = running[c]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    suffix = f" — {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Individual game statistics{suffix}", fontsize=13, fontweight="bold")

    # Use distinct blue / red for agent vs opponent across all three
    # top-row series; same hues as the plot_combat_summary palette so
    # cross-figure comparison stays coherent.
    AGENT_COLOR = "#1f77b4"
    OPP_COLOR = "#d62728"

    ax = axes[0, 0]
    ax.plot(steps, agent_units, label="agent", color=AGENT_COLOR)
    ax.plot(steps, opp_units, label="opponent", color=OPP_COLOR)
    ax.set_title("Unit count")
    ax.set_xlabel("Step")
    ax.set_ylabel("Units")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, agent_gold, label="agent", color=AGENT_COLOR)
    ax.plot(steps, opp_gold, label="opponent", color=OPP_COLOR)
    ax.set_title("Gold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gold")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(steps, agent_struct, label="agent", color=AGENT_COLOR)
    ax.plot(steps, opp_struct, label="opponent", color=OPP_COLOR)
    ax.set_title("Structures owned (towers + buildings + HQ)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Structures")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if action_counts:
        names = list(action_counts.keys())
        counts = [action_counts[n] for n in names]
        order = np.argsort(counts)[::-1]
        names = [names[i] for i in order]
        counts = [counts[i] for i in order]
        # Color creates differently from non-create actions so the
        # unit-type mix pops out of the bar.
        bar_colors = [
            _ACTION_COLORS["create_unit"] if n.startswith("create_") else _ACTION_COLORS.get(n, "#888") for n in names
        ]
        ax.bar(names, counts, color=bar_colors)
        ax.set_title(f"Action mix (n={sum(counts)}, blue=creates)")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No actions recorded", ha="center", va="center")
        ax.set_axis_off()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    for c in REWARD_COMPONENTS:
        ax.plot(steps, cum[c], label=c, color=_REWARD_COMPONENT_COLORS[c])
    total = float(result.get("total_reward", 0) or 0)
    ax.set_title(f"Cumulative reward by component (total={total:.0f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    # Outcome summary panel — pulls from the result dict so the chart
    # tells you immediately whether the recorded game was an HQ
    # capture, an elimination, or a timeout, without scrolling back to
    # the recording cell's print. Also lists per-unit-type creation
    # counts so you can see e.g. "agent only ever spawned Warriors" at
    # a glance.
    ax = axes[1, 2]
    ax.set_axis_off()
    winner_str = {1: "Agent wins", 2: "Opponent wins", None: "Draw"}.get(result.get("winner"), "Unknown")
    end_reason = result.get("end_reason") or "n/a"
    final_step = step_stats[-1]
    create_lines = [f"  {n}: {c}" for n, c in sorted(action_counts.items()) if n.startswith("create_")]
    summary_lines = [
        f"Result:        {winner_str}",
        f"End reason:    {end_reason}",
        f"Steps:         {result.get('steps', 0)}",
        f"Final turn:    {final_step.get('turn', 'n/a')}",
        f"Total reward:  {total:.0f}",
        "",
        f"Final units:   {final_step.get('agent_units', 0)} vs {final_step.get('opponent_units', 0)}",
        f"Final gold:    {final_step.get('agent_gold', 0)} vs {final_step.get('opponent_gold', 0)}",
        f"Final struct:  {final_step.get('agent_structures', 0)} vs {final_step.get('opponent_structures', 0)}",
        "",
        "Units created (agent):",
        *(create_lines or ["  (none)"]),
    ]
    ax.text(
        0.0,
        0.95,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.set_title("Outcome summary")

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if filename is None:
        suffix_part = f"_{title_suffix}" if title_suffix else ""
        filename = f"individual_game_stats{suffix_part}.png"
    return _save_and_return(fig, charts_dir, filename)


def plot_episode_length(
    results: Sequence[dict],
    *,
    charts_dir: Optional[Any] = None,
    title_suffix: str = "",
) -> Optional[plt.Figure]:
    """Mean episode length per eval — small standalone chart.

    A single-panel plot of ``avg_length`` vs ``timesteps``. The value
    also appears on ``plot_eval_curves`` panel 3, but breaking it out
    on its own makes the saturating-at-cap pattern (agent always
    timing out into a draw) unmistakable when scanning a stage's
    diagnostics directory.

    Args:
        results: Per-eval dicts from ``PeriodicEvalCallback.results``.
        charts_dir: Optional save directory; PNG named
            ``episode_length.png``.
        title_suffix: Prepended in brackets to the title — used by the
            bootstrap notebook to tag each stage's chart.
    """
    if not results:
        return None
    timesteps = [r["timesteps"] for r in results]
    lengths = [r["avg_length"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(timesteps, lengths, marker="o", color="#4caf50")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean episode length")
    title = "Episode length per eval"
    if title_suffix:
        title = f"[{title_suffix}] {title}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "episode_length.png")


def plot_curriculum_summary(
    history: Sequence[Mapping[str, Any]],
    stages: Sequence[Any],
    *,
    charts_dir: Optional[Any] = None,
) -> Optional[plt.Figure]:
    """Concatenated per-stage win-rate timeline with promotion thresholds.

    Each stage from ``history`` is drawn as its own line on a shared
    cumulative-timestep axis, with a dotted horizontal segment at the
    stage's ``promotion_win_rate`` and a dashed vertical line at the
    stage's last eval timestep marking the transition to the next
    stage. Returns ``None`` if no stage has any eval results.

    Args:
        history: ``run_curriculum`` result's ``"history"`` list — each
            entry has ``"stage"`` (name) and ``"results"`` (list of
            eval dicts with ``timesteps`` and ``win_rate``).
        stages: ``cfg.curriculum.stages`` — used to look up each
            stage's ``promotion_win_rate`` for the threshold line.
        charts_dir: Optional save directory; PNG named
            ``curriculum_winrate.png``.
    """
    if not any(h.get("results") for h in history):
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    stage_lookup = {s.name: s for s in stages}
    for i, h in enumerate(history):
        if not h["results"]:
            continue
        xs = [r["timesteps"] for r in h["results"]]
        ys = [r["win_rate"] for r in h["results"]]
        color = cmap(i % 10)
        ax.plot(xs, ys, marker="o", label=h["stage"], color=color)
        stage_cfg = stage_lookup.get(h["stage"])
        if stage_cfg is not None:
            ax.hlines(
                stage_cfg.promotion_win_rate,
                xmin=xs[0],
                xmax=xs[-1],
                colors=[color],
                linestyles=":",
                alpha=0.5,
            )
        ax.axvline(xs[-1], color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Cumulative env timesteps")
    ax.set_ylabel("Eval win rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Curriculum win rate (dotted = stage threshold, dashed = transition)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "curriculum_winrate.png")


def plot_stage_diagnostics(
    results: Sequence[dict],
    *,
    charts_dir: Optional[Any] = None,
    title_suffix: str = "",
) -> dict[str, Optional[plt.Figure]]:
    """Render the full per-stage diagnostic suite into ``charts_dir``.

    Calls, in order: ``plot_episode_length``, ``plot_outcome_breakdown``,
    ``plot_reward_decomposition``, ``plot_action_distribution``,
    ``plot_unit_build_distribution``, ``plot_combat_summary`` — each
    saving its canonical filename under ``charts_dir``. Used by both
    the bootstrap notebook (per-stage subdirectory) and the self-play
    training notebook (single run-level call) so the produced
    filenames stay identical.

    Returns a dict keyed by chart name with the matplotlib ``Figure``
    (or ``None`` when the underlying data was missing). Callers can
    iterate the returned figures to call ``plt.show()`` inline.
    """
    return {
        "episode_length": plot_episode_length(results, charts_dir=charts_dir, title_suffix=title_suffix),
        "outcome_breakdown": plot_outcome_breakdown(results, charts_dir=charts_dir),
        "reward_decomposition": plot_reward_decomposition(results, charts_dir=charts_dir),
        "action_distribution": plot_action_distribution(results, charts_dir=charts_dir),
        "unit_build_distribution": plot_unit_build_distribution(results, charts_dir=charts_dir),
        "combat_summary": plot_combat_summary(results, charts_dir=charts_dir),
    }
