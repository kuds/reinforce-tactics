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
- ``plot_curriculum_metrics``: curriculum-wide companion to
  ``plot_curriculum_summary`` — concatenates every stage's eval
  snapshots and renders the full per-eval diagnostic suite (length,
  outcome, reward, action, unit build, combat) on one timestep axis
  with stage-transition lines.
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


def _draw_stage_boundaries(ax: Any, stage_boundaries: Optional[Sequence[int]]) -> None:
    """Dashed vertical lines at each cumulative-timestep stage transition."""
    if not stage_boundaries:
        return
    for ts in stage_boundaries:
        ax.axvline(ts, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)


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
        _draw_stage_boundaries(ax, stage_boundaries)

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
    stage_boundaries: Optional[Sequence[int]] = None,
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
    _draw_stage_boundaries(ax, stage_boundaries)

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
    stage_boundaries: Optional[Sequence[int]] = None,
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
    _draw_stage_boundaries(ax, stage_boundaries)
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
    stage_boundaries: Optional[Sequence[int]] = None,
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
    _draw_stage_boundaries(ax, stage_boundaries)

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
    stage_boundaries: Optional[Sequence[int]] = None,
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
    _draw_stage_boundaries(ax, stage_boundaries)
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
    stage_boundaries: Optional[Sequence[int]] = None,
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
    _draw_stage_boundaries(ax, stage_boundaries)
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
    _draw_stage_boundaries(ax, stage_boundaries)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("HP per game (avg)")
    ax.set_title("Damage dealt vs taken")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(timesteps, per_game["attacks"], "o-", color=_COMBAT_STAT_COLORS["attacks"], label="attacks")
    ax.plot(timesteps, per_game["seize_attempts"], "s-", color=_COMBAT_STAT_COLORS["seize_attempts"], label="seize attempts")
    _draw_stage_boundaries(ax, stage_boundaries)
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
    stage_boundaries: Optional[Sequence[int]] = None,
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
    _draw_stage_boundaries(ax, stage_boundaries)
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


def plot_curriculum_metrics(
    history: Sequence[Mapping[str, Any]],
    *,
    charts_dir: Optional[Any] = None,
) -> dict[str, Optional[plt.Figure]]:
    """Curriculum-wide diagnostic timelines on a single timestep axis.

    Flattens every stage's eval snapshots into one list and renders the
    full per-eval diagnostic suite (episode length, outcome breakdown,
    reward decomposition, action distribution, unit-build distribution,
    combat summary) across the entire run. Vertical dashed lines mark
    stage transitions so the impact of map / opponent shifts on each
    metric is visible at a glance — companion to
    :func:`plot_curriculum_summary` (which covers win rate only).

    Saves each PNG under ``charts_dir/curriculum/`` with the
    ``curriculum_<metric>.png`` naming scheme so the curriculum-wide
    views don't collide with the per-stage diagnostics written by
    :func:`plot_stage_diagnostics`.

    Args:
        history: ``run_curriculum`` result's ``"history"`` list — each
            entry has ``"stage"`` (name) and ``"results"`` (list of
            eval dicts). Stage order in the list defines the timeline.
        charts_dir: Optional save directory. PNGs go under
            ``{charts_dir}/curriculum/``.

    Returns:
        Dict keyed by chart name with the matplotlib ``Figure``
        (or ``None`` when the underlying data was missing). Callers
        can iterate the returned figures to call ``plt.show()`` inline.
    """
    all_results: list[dict] = []
    stage_boundaries: list[int] = []
    for h in history[:-1]:
        all_results.extend(h.get("results") or [])
        if h.get("results"):
            stage_boundaries.append(int(h["results"][-1]["timesteps"]))
    if history:
        all_results.extend(history[-1].get("results") or [])

    if not all_results:
        return {
            "episode_length": None,
            "outcome_breakdown": None,
            "reward_decomposition": None,
            "action_distribution": None,
            "unit_build_distribution": None,
            "combat_summary": None,
        }

    sub_dir = None
    if charts_dir is not None:
        sub_dir = Path(charts_dir) / "curriculum"
        sub_dir.mkdir(parents=True, exist_ok=True)

    def _renamed(fig: Optional[plt.Figure], default: str, new: str) -> Optional[plt.Figure]:
        # ``_save_and_return`` writes the helpers' canonical filenames;
        # rename in place under the ``curriculum/`` subdir so the
        # curriculum-wide PNGs don't shadow the per-stage ones.
        if fig is None or sub_dir is None:
            return fig
        src = sub_dir / default
        if src.exists():
            src.rename(sub_dir / new)
        return fig

    return {
        "episode_length": _renamed(
            plot_episode_length(
                all_results,
                charts_dir=sub_dir,
                title_suffix="curriculum",
                stage_boundaries=stage_boundaries,
            ),
            "episode_length.png",
            "curriculum_episode_length.png",
        ),
        "outcome_breakdown": _renamed(
            plot_outcome_breakdown(
                all_results,
                charts_dir=sub_dir,
                stage_boundaries=stage_boundaries,
            ),
            "outcome_breakdown.png",
            "curriculum_outcome_breakdown.png",
        ),
        "reward_decomposition": _renamed(
            plot_reward_decomposition(
                all_results,
                charts_dir=sub_dir,
                stage_boundaries=stage_boundaries,
            ),
            "reward_decomposition.png",
            "curriculum_reward_decomposition.png",
        ),
        "action_distribution": _renamed(
            plot_action_distribution(
                all_results,
                charts_dir=sub_dir,
                stage_boundaries=stage_boundaries,
            ),
            "action_distribution.png",
            "curriculum_action_distribution.png",
        ),
        "unit_build_distribution": _renamed(
            plot_unit_build_distribution(
                all_results,
                charts_dir=sub_dir,
                stage_boundaries=stage_boundaries,
            ),
            "unit_build_distribution.png",
            "curriculum_unit_build_distribution.png",
        ),
        "combat_summary": _renamed(
            plot_combat_summary(
                all_results,
                charts_dir=sub_dir,
                stage_boundaries=stage_boundaries,
            ),
            "combat_summary.png",
            "curriculum_combat_summary.png",
        ),
    }


def plot_curriculum_composition_summary(
    history: Sequence[Mapping[str, Any]],
    *,
    charts_dir: Optional[Any] = None,
    final_evals_to_average: int = 3,
) -> Optional[plt.Figure]:
    """One-row-per-stage summary of *what the policy actually did*.

    Per-stage diagnostics (``plot_stage_diagnostics``) and curriculum-
    wide stacks (``plot_curriculum_metrics``) both surface unit-build
    composition, but reading them requires opening one figure per
    stage. This summary collapses the cleared / stalled curriculum
    into a single horizontal stacked bar per stage, annotated with the
    strategic signals that differentiated v34's mono-Warrior cleared-
    14-stages run from v35's diverse-comp stalled-at-stage-6 run:

      * unit-build composition (Warrior% / Mage% / etc.) on the bar
      * Warrior% printed inline on the W segment (the dominant-unit
        red flag -- 100% on every v34 stage, oscillating in v35)
      * HQ captures + building captures printed on the right
        (zero on every v34 stage; non-zero on v35 late evals)
      * Mage paralyze + Sorcerer haste counts printed on the right
        (zero on every v34 stage; the status-effect channels added in
        PR #383 have nothing to observe if the policy never builds
        the casters)
      * Best WR + promoted flag printed on the left

    Averaging window: by default the LAST ``final_evals_to_average=3``
    evals per stage are pooled to smooth single-eval noise (compositions
    on the last eval can be highly variable during recovery, as seen at
    v35's beginner_random_10 stall: 7% W on eval 30 jumped to 10% W on
    eval 31). Set to 1 to read off only the terminal eval.

    Args:
        history: ``run_curriculum`` result's ``"history"`` list. Each
            entry must have ``"stage"`` and ``"results"`` (list of
            eval dicts with ``units_built`` / ``combat_stats`` /
            ``action_counts`` / ``win_rate`` / ``captures_by_type``).
        charts_dir: Optional save directory.
        final_evals_to_average: Number of trailing evals per stage to
            pool when computing the composition. Default 3.

    Returns:
        Matplotlib ``Figure`` with the horizontal stacked bar chart,
        or ``None`` if no stage has any units_built data.
    """
    rows: list[dict] = []
    for h in history:
        results = h.get("results") or []
        if not results:
            continue
        tail = results[-max(1, int(final_evals_to_average)) :]
        # Pool counters across the trailing evals.
        ub_totals = {ut: 0.0 for ut in UNIT_TYPE_LETTERS}
        kills = 0.0
        para = haste = atk_buf = def_buf = 0.0
        hq_caps = building_caps = tower_caps = 0.0
        for r in tail:
            for ut, v in (r.get("units_built") or {}).items():
                if ut in ub_totals:
                    ub_totals[ut] += float(v or 0)
            cs = r.get("combat_stats") or {}
            kills += float(cs.get("kills") or 0)
            ac = r.get("action_counts") or {}
            para += float(ac.get("paralyze") or 0)
            haste += float(ac.get("haste") or 0)
            atk_buf += float(ac.get("attack_buff") or 0)
            def_buf += float(ac.get("defence_buff") or 0)
            cbt = r.get("captures_by_type") or {}
            hq_caps += float(cbt.get("hq") or 0)
            building_caps += float(cbt.get("building") or 0)
            tower_caps += float(cbt.get("tower") or 0)
        total_units = sum(ub_totals.values())
        if total_units == 0:
            # Stage has no build data at all -- skip to keep the
            # chart focused on stages where the question is meaningful.
            continue
        pct = {ut: 100.0 * ub_totals[ut] / total_units for ut in UNIT_TYPE_LETTERS}
        rows.append(
            {
                "stage": h.get("stage", "?"),
                "promoted": bool(h.get("promoted", False)),
                "best_wr": float(h.get("best_win_rate", 0.0)),
                "pct": pct,
                "kills": kills,
                "abilities": int(para + haste + atk_buf + def_buf),
                "paralyze": int(para),
                "hq_caps": int(hq_caps),
                "building_caps": int(building_caps),
                "tower_caps": int(tower_caps),
                "n_evals_pooled": len(tail),
            }
        )

    if not rows:
        print("plot_curriculum_composition_summary: no stage has units_built data; nothing to plot.")
        return None

    n_stages = len(rows)
    # Height scales with stage count so the chart stays legible from
    # the validation slice (6 stages) up to the full production
    # curriculum (33-35 stages).
    fig_h = max(3.0, 0.45 * n_stages + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    y_positions = list(range(n_stages))
    # Iterate in REVERSE so the first stage lands at the TOP of the
    # chart (matplotlib's y axis grows upward). Easier to read as a
    # curriculum timeline top-to-bottom.
    rows_plot = list(reversed(rows))
    y_labels = [row["stage"] for row in rows_plot]

    # Stacked bar across unit types.
    left = np.zeros(n_stages, dtype=float)
    for ut in UNIT_TYPE_LETTERS:
        values = np.array([row["pct"][ut] for row in rows_plot])
        color = _UNIT_TYPE_COLORS.get(ut, "#888")
        ax.barh(y_positions, values, left=left, color=color, edgecolor="white", linewidth=0.5, label=ut)
        # Print "W:NN%" on the W segment when it's >= 8% wide enough
        # to fit; this is the single most diagnostic readout for mono-
        # build collapse. Other segments stay unannotated to avoid
        # clutter.
        if ut == "W":
            for i, v in enumerate(values):
                if v >= 8.0:
                    ax.text(
                        left[i] + v / 2,
                        y_positions[i],
                        f"W:{int(round(v))}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )
        left += values

    # Left margin: best WR + promoted flag.
    for i, row in enumerate(rows_plot):
        wr_pct = row["best_wr"] * 100
        flag = "OK" if row["promoted"] else "X"
        ax.text(
            -2.0,
            y_positions[i],
            f"{flag} {wr_pct:5.1f}%",
            ha="right",
            va="center",
            fontsize=9,
            color="#2e7d32" if row["promoted"] else "#c62828",
            fontfamily="monospace",
        )

    # Right margin: strategic signals (HQ caps / building caps /
    # paralyze / total ability use). These are the metrics that
    # showed mono-Warrior elimination (hq=0, abilities=0) vs diverse
    # comp (hq>0, abilities>0) in the v34 vs v35 comparison.
    for i, row in enumerate(rows_plot):
        annot = (
            f"  HQ:{row['hq_caps']:>2}  B:{row['building_caps']:>3}  T:{row['tower_caps']:>3}  "
            f"|  para:{row['paralyze']:>4}  abil:{row['abilities']:>4}  "
            f"|  kills:{int(row['kills']):>5}"
        )
        ax.text(102, y_positions[i], annot, ha="left", va="center", fontsize=8, fontfamily="monospace")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlim(0, 100)
    # Make room for left WR annotation and right strategic signals.
    ax.set_xlim(left=-20, right=100)
    ax.set_xlabel("Build share (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    n_pooled_note = f" (avg of last {final_evals_to_average} evals/stage)" if final_evals_to_average > 1 else ""
    ax.set_title(f"Per-stage composition + strategic signals{n_pooled_note}", fontsize=11)
    ax.legend(loc="lower right", bbox_to_anchor=(1.45, 0.0), fontsize=8, title="Unit", framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, "curriculum_composition_summary.png")


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


# ---------------------------------------------------------------------------
# Behavior-cloning diagnostics
#
# Separate from the PPO/eval plotters above because BC has a very different
# data shape: per-epoch supervised metrics (loss + accuracy) and per-
# scenario demonstrator outcomes (W/L/D, avg turns). No timestep axis.
# ---------------------------------------------------------------------------


def plot_bc_training_curves(
    bc_stats: Sequence[Any],
    *,
    charts_dir: Optional[Any] = None,
    name: str = "bc_training_curves.png",
) -> plt.Figure:
    """Per-epoch BC supervised metrics: loss + two accuracy curves.

    Consumes the ``List[BCStats]`` returned by ``behavior_clone`` /
    ``make_warm_started_model``. Loss on the left axis, accuracies on the
    right. action_type_acc is the easier metric (10-way); full_action_acc
    requires all 6 dims to match and is structurally much lower -- both
    are plotted so trend (rising) and ceiling are visible.
    """
    if not bc_stats:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No BC stats", ha="center", va="center", transform=ax.transAxes)
        return _save_and_return(fig, charts_dir, name)

    epochs = [s.epoch for s in bc_stats]
    loss = [s.loss for s in bc_stats]
    acc_type = [s.accuracy_action_type for s in bc_stats]
    acc_full = [s.accuracy_full for s in bc_stats]

    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_loss.plot(epochs, loss, marker="o", color="#d32f2f", label="loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("masked cross-entropy loss", color="#d32f2f")
    ax_loss.tick_params(axis="y", labelcolor="#d32f2f")
    ax_loss.grid(True, alpha=0.3)

    ax_acc = ax_loss.twinx()
    ax_acc.plot(epochs, acc_type, marker="s", color="#2e7d32", label="action_type acc")
    ax_acc.plot(epochs, acc_full, marker="^", color="#1565c0", label="full_action acc")
    ax_acc.set_ylabel("accuracy", color="#1565c0")
    ax_acc.tick_params(axis="y", labelcolor="#1565c0")
    ax_acc.set_ylim(0.0, 1.0)

    # Combined legend across both axes so the panel reads cleanly.
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines_loss + lines_acc, labels_loss + labels_acc, loc="center right")

    ax_loss.set_title(f"BC training -- {len(bc_stats)} epochs")
    fig.tight_layout()
    return _save_and_return(fig, charts_dir, name)


def plot_bc_demo_outcomes(
    scenario_stats: Sequence[Any],
    *,
    charts_dir: Optional[Any] = None,
    name: str = "bc_demo_outcomes.png",
) -> plt.Figure:
    """Per-scenario demonstrator outcomes: stacked W/L/D + avg turn count.

    Consumes ``DemonstrationDataset.scenario_stats``. Top panel: stacked
    bar of demonstrator wins / losses / draws per scenario, with the
    demonstrator win-rate annotated. Bottom panel: average game length
    (turns) per scenario.

    Read together, these surface BC-label quality issues *before* PPO
    fine-tuning: a scenario with WR < 50% means BC is being taught
    losing trajectories at the majority rate; long avg_turns (near the
    max_turns cap) means games are timing out as draws rather than
    resolving, so the demonstrator's "winning" trajectories are mostly
    stalemates.
    """
    if not scenario_stats:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No scenario stats", ha="center", va="center", transform=ax.transAxes)
        return _save_and_return(fig, charts_dir, name)

    names = [s.name for s in scenario_stats]
    wins = np.array([s.demo_wins for s in scenario_stats])
    losses = np.array([s.demo_losses for s in scenario_stats])
    draws = np.array([s.draws for s in scenario_stats])
    win_rates = np.array([s.demo_win_rate for s in scenario_stats])
    avg_turns = np.array([s.avg_turns for s in scenario_stats])

    n = len(names)
    fig, (ax_wld, ax_turns) = plt.subplots(2, 1, figsize=(max(8, 1.2 * n + 4), 7), sharex=True)

    x = np.arange(n)
    ax_wld.bar(x, wins, color="#2e7d32", label="demo wins")
    ax_wld.bar(x, losses, bottom=wins, color="#c62828", label="demo losses")
    ax_wld.bar(x, draws, bottom=wins + losses, color="#9e9e9e", label="draws")
    ax_wld.set_ylabel("games")
    ax_wld.set_title("Demonstrator W/L/D per scenario")
    ax_wld.legend(loc="upper right")
    ax_wld.grid(True, alpha=0.3, axis="y")
    # Annotate demonstrator WR on top of each bar -- the key BC-quality
    # signal at a glance.
    totals = wins + losses + draws
    for xi, total, wr in zip(x, totals, win_rates):
        ax_wld.text(xi, total + max(totals) * 0.02, f"{100.0 * wr:.0f}%", ha="center", va="bottom", fontsize=9)

    ax_turns.bar(x, avg_turns, color="#1565c0")
    ax_turns.set_ylabel("avg turns / game")
    ax_turns.set_title("Average game length per scenario")
    ax_turns.grid(True, alpha=0.3, axis="y")
    ax_turns.set_xticks(x)
    ax_turns.set_xticklabels(names, rotation=30, ha="right")

    fig.tight_layout()
    return _save_and_return(fig, charts_dir, name)
