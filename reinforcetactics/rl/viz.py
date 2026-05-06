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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from reinforcetactics.rl.evaluation import ACTION_TYPE_NAMES, REWARD_COMPONENTS

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
    have_train = bool(train_records)
    n_panels = 5 if have_train else 3
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
    ax.set_xscale("log") if eval_ts and eval_ts[0] > 0 else None
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

    if have_train:
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
