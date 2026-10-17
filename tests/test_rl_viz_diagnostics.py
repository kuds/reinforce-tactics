"""Numerical semantics of the training diagnosis plots, using headless figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from reinforcetactics.rl.viz import (  # noqa: E402
    _wilson_bounds,
    plot_action_space_diagnostics,
    plot_curriculum_metrics,
    plot_episode_distributions,
    plot_eval_curves,
    plot_optimization_diagnostics,
    plot_reward_by_outcome,
    plot_stage_diagnostics,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _eval_row(ts=100, **extra):
    return {
        "timesteps": ts,
        "episodes": 30,
        "win_rate": 0.5,
        "avg_reward": 0.0,
        "std_reward": 100.0,
        "avg_length": 10.0,
        "std_length": 5.0,
        **extra,
    }


def _line(ax, label):
    return next(line for line in ax.lines if line.get_label() == label)


def test_wilson_has_nonzero_uncertainty_at_extreme_results_and_no_fake_legacy_count():
    assert _wilson_bounds(1.0, 30) == pytest.approx((0.886486606826, 1.0))
    assert _wilson_bounds(0.0, 30) == pytest.approx((0.0, 0.113513393174))
    assert np.isnan(_wilson_bounds(0.5, 0)).all()
    rows = [_eval_row(win_rate=1.0, gate_win_rate=0.8, seat_aggregate="min")]
    fig = plot_eval_curves(rows)
    band = fig.axes[0].collections[0].get_paths()[0].vertices[:, 1]
    assert band.min() == pytest.approx(88.6486606826)
    assert band.max() == pytest.approx(100.0)
    assert _line(fig.axes[0], "pooled win rate (eval mode)").get_ydata()[0] == 100.0
    row = _eval_row()
    del row["episodes"]
    assert not plot_eval_curves([row]).axes[0].collections


def test_paired_seats_get_separate_intervals_without_a_pooled_independence_assumption():
    row = _eval_row(
        episodes=60,
        win_rate=0.75,
        gate_win_rate=0.5,
        by_seat={
            "1": {"episodes": 30, "win_rate": 1.0},
            "2": {"episodes": 30, "win_rate": 0.5},
        },
    )
    fig = plot_eval_curves([row])
    ax = fig.axes[0]
    assert _line(ax, "pooled win rate (eval mode)").get_ydata()[0] == 75.0
    assert _line(ax, "seat 1 win rate").get_ydata()[0] == 100.0
    assert _line(ax, "seat 2 win rate").get_ydata()[0] == 50.0
    bands = {collection.get_label(): collection for collection in ax.collections}
    assert set(bands) == {
        "seat 1: 95% Wilson interval (pointwise)",
        "seat 2: 95% Wilson interval (pointwise)",
    }
    seat_one_bounds = bands["seat 1: 95% Wilson interval (pointwise)"].get_paths()[0].vertices[:, 1]
    assert seat_one_bounds.min() == pytest.approx(88.6486606826)
    assert seat_one_bounds.max() == pytest.approx(100.0)


def test_eval_sd_and_configured_kl_are_labeled_without_universal_failure_thresholds():
    train = [{"timesteps": 100, "train/approx_kl": 0.03, "train/explained_variance": 0.0}]
    fig = plot_eval_curves([_eval_row()], train, target_kl=0.01)
    assert "±1 episode SD" in fig.axes[1].get_legend_handles_labels()[1]
    assert "±1 episode SD" in fig.axes[2].get_legend_handles_labels()[1]
    assert _line(fig.axes[3], "configured target_kl").get_ydata() == [0.01, 0.01]
    all_labels = [line.get_label() for ax in fig.axes for line in ax.lines]
    assert not any("danger" in label for label in all_labels)
    default = plot_eval_curves([_eval_row()], train)
    assert not any("target" in line.get_label() for line in default.axes[3].lines)


def test_reward_outcome_denominators_and_empty_groups_are_not_pooled_or_zero_filled():
    first = _eval_row(
        reward_components_by_outcome={
            "wins": {"episodes": 4, "action": 400, "terminal": 20_000},
            "draws": {"episodes": 2, "action": 1000, "terminal": -10_000},
            "losses": {"episodes": 0},
        }
    )
    rows = [
        first,
        _eval_row(200),
        _eval_row(
            300,
            reward_components_by_outcome={
                "wins": {"episodes": 1, "action": 200, "terminal": 5000},
            },
        ),
    ]
    fig = plot_reward_by_outcome(rows)
    assert _line(fig.axes[0], "action").get_ydata() == pytest.approx([100, np.nan, 200], nan_ok=True)
    assert _line(fig.axes[0], "terminal").get_ydata()[0] == 5000
    assert _line(fig.axes[1], "action").get_ydata()[0] == 500
    assert _line(fig.axes[1], "terminal").get_ydata()[0] == -5000
    assert np.isnan(_line(fig.axes[2], "net reward").get_ydata()).all()
    assert any("No losses" in text.get_text() for text in fig.axes[2].texts)
    assert plot_reward_by_outcome([_eval_row()]) is None


def test_episode_distributions_condition_on_outcomes_and_keep_legacy_gaps():
    row = _eval_row(outcomes=["wins", "wins", "draws"], turns=[2, 8, 20], lengths=[10, 30, 150])
    malformed = _eval_row(300, outcomes=["wins", "wins"], turns=[1], lengths=[3])
    fig = plot_episode_distributions([row, _eval_row(200), malformed], max_turns=20, max_steps=400)
    assert _line(fig.axes[0], "wins").get_ydata() == pytest.approx([5, np.nan, np.nan], nan_ok=True)
    assert _line(fig.axes[1], "wins").get_ydata()[0] == 20
    assert _line(fig.axes[0], "draws").get_ydata()[0] == 20
    assert np.isnan(_line(fig.axes[0], "losses").get_ydata()).all()
    # The wins turn IQR is [3.5, 6.5], not the pooled episodes' IQR.
    band = fig.axes[0].collections[0].get_paths()[0].vertices[:, 1]
    assert band.min() == 3.5 and band.max() == 6.5
    assert _line(fig.axes[0], "configured cap (20)").get_ydata() == [20, 20]
    assert _line(fig.axes[1], "configured cap (400)").get_ydata() == [400, 400]
    assert plot_episode_distributions([_eval_row()]) is None


def test_optimization_sign_and_normalized_measurement_with_forced_only_gap():
    records = [
        {
            "timesteps": 100,
            "train/entropy_loss": -2.0,
            "train/clip_fraction": 0.3,
            "train/learning_rate": 3e-4,
            "train/ent_coef": 0.05,
            "rollout/normalized_policy_entropy": 0.7,
            "rollout/forced_action_fraction": 0.2,
            "rollout/legal_actions_mean": 12,
        },
        {"timesteps": 200, "train/entropy_loss": 0.0, "rollout/forced_action_fraction": 1.0, "rollout/legal_actions_mean": 1},
    ]
    fig = plot_optimization_diagnostics(records)
    assert len(fig.axes) == 6
    assert fig.axes[0].lines[0].get_ydata() == pytest.approx([2.0, 0.0])
    assert fig.axes[1].lines[0].get_ydata()[0] == 0.3
    assert _line(fig.axes[4], "H/log(legal choices), choices > 1").get_ydata() == pytest.approx([0.7, np.nan], nan_ok=True)
    assert _line(fig.axes[4], "forced decisions").get_ydata() == pytest.approx([0.2, 1.0])
    assert fig.axes[5].lines[0].get_ydata() == pytest.approx([12, 1])
    assert plot_optimization_diagnostics([]) is None
    assert len(plot_optimization_diagnostics([{"timesteps": 1, "train/entropy_loss": -1}]).axes) == 4


def test_action_choices_without_opportunities_are_gaps():
    rows = [
        _eval_row(seize_available_rate=0.0, seize_choice_rate=None, forced_action_rate=1.0),
        _eval_row(200, seize_available_rate=0.4, seize_choice_rate=0.75, forced_action_rate=0.1),
    ]
    fig = plot_action_space_diagnostics(rows)
    assert _line(fig.axes[2], "seize chosen / offered").get_ydata() == pytest.approx([np.nan, 0.75], nan_ok=True)
    assert _line(fig.axes[2], "seize offered / all steps").get_ydata() == pytest.approx([0.0, 0.4])
    assert plot_action_space_diagnostics([_eval_row()]) is None


def test_suite_handles_legacy_rows_and_curriculum_saves_new_canonical_names(tmp_path):
    legacy = plot_stage_diagnostics([_eval_row()])
    for key in ("reward_by_outcome", "episode_distributions", "optimization_diagnostics", "action_space_diagnostics"):
        assert legacy[key] is None
    row = _eval_row(
        outcomes=["wins"],
        lengths=[4],
        turns=[2],
        forced_action_rate=0.1,
        reward_components_by_outcome={"wins": {"episodes": 1, "terminal": 5000}},
    )
    figures = plot_curriculum_metrics(
        [{"stage": "a", "results": [row]}],
        charts_dir=tmp_path,
        train_records=[{"timesteps": 100, "train/entropy_loss": -1}],
    )
    for key in ("reward_by_outcome", "episode_distributions", "optimization_diagnostics", "action_space_diagnostics"):
        assert figures[key] is not None
        assert (tmp_path / "curriculum" / f"curriculum_{key}.png").is_file()
        assert not (tmp_path / "curriculum" / f"{key}.png").exists()
    assert set(plot_curriculum_metrics([])) == set(figures)
