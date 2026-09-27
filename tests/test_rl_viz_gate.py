"""Curriculum charts after the eval-gate changes (review rltrain-20, critic-gaps-2).

* Stage-entry (carry-in, ``best_eligible`` False) evals are hatched or dropped.
* Reward components and outcome counts are normalised by the eval's episodes.
* A best win rate of None (or the old -1.0) renders as n/a.
* No fixed 70% target line; the winner label compares against the agent's seat.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from reinforcetactics.rl.viz import (  # noqa: E402
    format_win_rate,
    plot_curriculum_composition_summary,
    plot_eval_curves,
    plot_individual_game_stats,
    plot_outcome_breakdown,
    plot_reward_decomposition,
)


def _row(ts: int, episodes: int, wins: int, *, eligible: bool = True, **extra):
    outcome = {"wins_by_hq_capture": wins, "losses_by_elimination": episodes - wins}
    return {
        "timesteps": ts,
        "episodes": episodes,
        "win_rate": wins / episodes,
        "avg_reward": 0.0,
        "std_reward": 0.0,
        "avg_length": 1.0,
        "std_length": 0.0,
        "outcome_reasons": outcome,
        "reward_components": {"action": 2.0 * episodes, "shaping_delta": 0.0, "invalid_penalty": 0.0, "terminal": 0.0},
        "best_eligible": eligible,
        **extra,
    }


def _bar_heights(fig):
    ax = fig.axes[0]
    by_x: dict[float, float] = {}
    for patch in ax.patches:
        x = round(patch.get_x() + patch.get_width() / 2, 3)
        by_x[x] = by_x.get(x, 0.0) + patch.get_height()
    return by_x


class TestOutcomeBreakdown:
    def test_normalised_to_episode_share(self):
        fig = plot_outcome_breakdown([_row(100, 10, 5), _row(200, 40, 20)])
        heights = _bar_heights(fig)
        assert sorted(heights.values()) == [pytest.approx(100.0), pytest.approx(100.0)]
        assert fig.axes[0].get_ylabel() == "Share of eval episodes (%)"
        plt.close(fig)

    def test_carry_in_bars_are_hatched_or_dropped(self):
        rows = [_row(100, 10, 5, eligible=False), _row(200, 10, 8), _row(300, 10, 9)]
        fig = plot_outcome_breakdown(rows)
        hatched = {round(p.get_x() + p.get_width() / 2) for p in fig.axes[0].patches if p.get_hatch()}
        assert hatched == {100}
        assert "carry-in" in fig.axes[0].get_title()
        plt.close(fig)
        fig = plot_outcome_breakdown(rows, carry_in="drop")
        assert set(round(x) for x in _bar_heights(fig)) == {200, 300}
        plt.close(fig)
        with pytest.raises(ValueError):
            plot_outcome_breakdown(rows, carry_in="fade")


def test_reward_decomposition_is_per_episode():
    fig = plot_reward_decomposition([_row(100, 10, 5), _row(200, 40, 20)])
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Mean reward per eval episode"
    # 2.0 * episodes summed -> 2.0 per episode for both evals.
    tops = [collection.get_paths()[0].vertices[:, 1].max() for collection in ax.collections if collection.get_paths()]
    assert max(tops) == pytest.approx(2.0)
    plt.close(fig)


def test_best_win_rate_none_renders_as_na():
    assert format_win_rate(None) == "n/a" and format_win_rate(-1.0) == "n/a"
    assert format_win_rate(0.5).strip() == "50.0%"
    history = [
        {
            "stage": "s",
            "promoted": True,
            "best_win_rate": None,
            "results": [{"units_built": {"W": 3}, "win_rate": 1.0}],
        }
    ]
    fig = plot_curriculum_composition_summary(history)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("n/a" in t for t in texts)
    plt.close(fig)


class TestEvalCurves:
    def test_no_fixed_target_line_by_default(self):
        rows = [_row(100, 10, 5), _row(200, 10, 7)]
        fig = plot_eval_curves(rows)
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert not any("target" in label for label in labels)
        plt.close(fig)
        fig = plot_eval_curves(rows, target_win_rate=0.85)
        assert "85% target" in fig.axes[0].get_legend_handles_labels()[1]
        plt.close(fig)

    def test_both_modes_and_carry_in_markers(self):
        rows = [
            _row(100, 10, 5, eligible=False, win_rate_greedy=0.9, win_rate_stochastic=0.5),
            _row(200, 10, 7, win_rate_greedy=0.8, win_rate_stochastic=0.7),
        ]
        fig = plot_eval_curves(rows)
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert {"greedy", "stochastic", "carry-in eval"} <= set(labels)
        plt.close(fig)


@pytest.mark.parametrize(
    ("winner", "agent_player", "expected"),
    [(2, 2, "Agent wins (P2)"), (1, 2, "Opponent wins (P1)"), (1, 1, "Agent wins (P1)"), (None, 2, "Draw")],
)
def test_game_summary_compares_the_winner_with_the_agents_seat(winner, agent_player, expected):
    steps = [
        {
            "turn": 0,
            "action_type": None,
            "agent_units": 1,
            "opponent_units": 1,
            "agent_gold": 0,
            "opponent_gold": 0,
            "reward_breakdown": None,
        },
        {
            "turn": 1,
            "action_type": 5,
            "agent_units": 1,
            "opponent_units": 1,
            "agent_gold": 0,
            "opponent_gold": 0,
            "reward_breakdown": {"action": 1.0},
        },
    ]
    fig = plot_individual_game_stats({"step_stats": steps, "winner": winner, "agent_player": agent_player})
    summary = fig.axes[5].texts[0].get_text()
    assert f"Result:        {expected}" in summary
    plt.close(fig)
