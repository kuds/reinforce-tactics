"""
Plotting helpers for RL training notebooks.

These functions consume the per-eval ``results`` list produced by
``PeriodicEvalCallback`` and the per-rollout ``records`` list produced by
``TrainingMetricsCallback``. They return the matplotlib ``Figure`` so
callers can ``fig.savefig(...)`` or further customize before display.

Kept deliberately minimal — three plot families that together cover the
"why isn't PPO learning?" debugging surface:

- ``plot_eval_curves``: win rate / reward / episode length curves +
  PPO update health (approx_kl, explained_variance) on shared axes.
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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

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
) -> plt.Figure:
    """Win rate / avg reward / episode length / approx_kl / explained_variance.

    First three panels read from ``results`` (per-eval). Last two read
    from ``train_records`` (per-rollout, finer grid) and are skipped if
    that argument is None or empty.
    """
    n_panels = 5 if train_records else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    eval_ts = [r["timesteps"] for r in results]

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
        ax.grid(True, alpha=0.3)

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
