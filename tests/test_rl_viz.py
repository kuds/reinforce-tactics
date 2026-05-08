"""Tests for ``reinforcetactics.rl.viz`` plot helpers.

These tests exercise the rendering path with a headless matplotlib
backend and synthetic input dicts shaped like the real callers.
We don't validate pixel content -- the value is catching regressions
where the helper raises (key errors, dtype mismatches, division by
zero on empty input) before someone re-runs an 8-stage notebook and
discovers the failure at hour 6.
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib

# Force a non-interactive backend so the tests work in headless CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from reinforcetactics.rl.viz import plot_eval_curves, plot_individual_game_stats  # noqa: E402


def _synthetic_step_stats(n: int = 30) -> List[Dict[str, Any]]:
    """Build a step_stats list shaped like ``record_evaluation_to_video``.

    First entry is the pre-action initial snapshot (``action_type=None``);
    subsequent entries cycle through action types so the action-mix bar
    has every category to draw.
    """
    # The dict literals mix int/str/float/None/dict values, so type the
    # list explicitly to keep mypy from inferring a too-narrow value type
    # from the first entry.
    steps: List[Dict[str, Any]] = [
        {
            "turn": 0,
            "current_player": 1,
            "action_type": None,
            "unit_type": None,
            "agent_units": 3,
            "opponent_units": 3,
            "agent_gold": 100,
            "opponent_gold": 100,
            "agent_hp_total": 30,
            "opponent_hp_total": 30,
            "agent_structures": 1,
            "opponent_structures": 1,
            "reward_breakdown": None,
        }
    ]
    for i in range(1, n):
        at = i % 6  # 0..5 covers create / move / attack / seize / heal / end_turn
        ut = "W" if at == 0 else None
        steps.append(
            {
                "turn": i // 5,
                "current_player": 1,
                "action_type": at,
                "unit_type": ut,
                "agent_units": 3 + (i // 10),
                "opponent_units": max(0, 3 - i // 12),
                "agent_gold": 100 + i * 2,
                "opponent_gold": 100 + i,
                "agent_hp_total": 30,
                "opponent_hp_total": max(0, 30 - i),
                "agent_structures": 1 + (i // 15),
                "opponent_structures": 1,
                "reward": float(i * 0.5),
                "reward_breakdown": {
                    "action": float(i * 0.5),
                    "shaping_delta": float(i * 0.1),
                    "invalid_penalty": -1.0 if i % 7 == 0 else 0.0,
                    "terminal": 100.0 if i == n - 1 else 0.0,
                },
                "valid_action": True,
            }
        )
    return steps


def _make_result(steps, *, winner=1, end_reason="hq_capture", total_reward=1234.5):
    return {
        "step_stats": steps,
        "winner": winner,
        "end_reason": end_reason,
        "steps": len(steps) - 1,
        "total_reward": total_reward,
    }


class TestPlotIndividualGameStats:
    def test_writes_default_filename_when_no_suffix(self, tmp_path):
        result = _make_result(_synthetic_step_stats())
        fig = plot_individual_game_stats(result, charts_dir=tmp_path)
        assert fig is not None
        out = tmp_path / "individual_game_stats.png"
        assert out.is_file()
        assert out.stat().st_size > 0
        plt.close(fig)

    def test_appends_title_suffix_to_default_filename(self, tmp_path):
        result = _make_result(_synthetic_step_stats())
        plot_individual_game_stats(result, charts_dir=tmp_path, title_suffix="best")
        # The library's default naming pattern matches ppo_training 9e:
        # ``individual_game_stats_<suffix>.png`` so two recordings (best
        # vs final) save to distinct files in the same charts_dir.
        assert (tmp_path / "individual_game_stats_best.png").is_file()

    def test_explicit_filename_overrides_default(self, tmp_path):
        result = _make_result(_synthetic_step_stats())
        plot_individual_game_stats(
            result,
            charts_dir=tmp_path,
            title_suffix="starter_random",
            filename="starter_random.png",
        )
        # Bootstrap notebook overrides the filename so per-stage panels
        # land flat in an ``individual_game_stats/`` directory without
        # the redundant prefix.
        assert (tmp_path / "starter_random.png").is_file()
        # Defaulted name should NOT also be written when an override is
        # provided -- otherwise the bootstrap loop would double-save.
        assert not (tmp_path / "individual_game_stats_starter_random.png").exists()

    def test_returns_none_for_empty_step_stats(self, tmp_path):
        # Empty step_stats can happen if the recorder bails out before
        # the first action; helper should skip silently rather than
        # raising and breaking a multi-stage diagnostics loop.
        result = _make_result([], winner=None, end_reason=None, total_reward=0.0)
        fig = plot_individual_game_stats(result, charts_dir=tmp_path)
        assert fig is None
        # And no chart should land on disk.
        assert not list(tmp_path.iterdir())

    def test_handles_draw_outcome(self, tmp_path):
        # winner=None + max_turns_draw is the failure mode the bootstrap
        # diagnostics need to flag; helper must render the outcome
        # summary without raising on the missing winner.
        steps = _synthetic_step_stats()
        # Strip the terminal bonus to mirror a draw.
        steps[-1]["reward_breakdown"]["terminal"] = 0.0
        result = _make_result(steps, winner=None, end_reason="max_turns_draw", total_reward=-200.0)
        fig = plot_individual_game_stats(result, charts_dir=tmp_path, title_suffix="draw")
        assert fig is not None
        plt.close(fig)

    def test_handles_unknown_action_type_gracefully(self, tmp_path):
        # If gym_env grows new action types the helper shouldn't crash
        # on a code outside ACTION_TYPE_NAMES; it should fall back to
        # the integer label.
        steps = _synthetic_step_stats(n=10)
        steps[3]["action_type"] = 99
        result = _make_result(steps)
        fig = plot_individual_game_stats(result, charts_dir=tmp_path)
        assert fig is not None
        plt.close(fig)

    def test_returns_figure_when_charts_dir_is_none(self):
        # Notebook callers usually save then ``plt.show()``; the
        # programmatic path may want the Figure without a save side
        # effect. ``charts_dir=None`` should still return a Figure.
        result = _make_result(_synthetic_step_stats())
        fig = plot_individual_game_stats(result, charts_dir=None)
        assert fig is not None
        plt.close(fig)


def _synthetic_eval_results(n: int = 6, base_ts: int = 50_000) -> List[Dict[str, Any]]:
    """Build per-eval result dicts shaped like ``PeriodicEvalCallback.results``."""
    out = []
    for i in range(n):
        out.append(
            {
                "timesteps": base_ts * (i + 1),
                "win_rate": min(1.0, 0.2 + 0.1 * i),
                "avg_reward": 1000.0 * i,
                "std_reward": 500.0,
                "avg_length": 200.0 + 20.0 * i,
                "std_length": 50.0,
                "wins": 6 * i,
                "losses": max(0, 30 - 6 * i),
                "draws": 0,
            }
        )
    return out


def _synthetic_train_records(n: int = 30, base_ts: int = 10_000) -> List[Dict[str, Any]]:
    """Build per-rollout train_records shaped like ``TrainingMetricsCallback.records``."""
    out = []
    for i in range(n):
        # ``value_loss`` decays exponentially so the log-scale panel
        # has both the high-init and the converged regime to draw.
        out.append(
            {
                "timesteps": base_ts * (i + 1),
                "rollout/ep_rew_mean": 100.0 * i,
                "rollout/ep_len_mean": 200.0,
                "train/approx_kl": 0.01 + 0.001 * i,
                "train/clip_fraction": 0.1,
                "train/entropy_loss": -0.5 - 0.01 * i,
                "train/explained_variance": min(0.95, 0.0 + 0.03 * i),
                "train/learning_rate": 3e-4,
                "train/loss": 100.0,
                "train/policy_gradient_loss": -0.05,
                "train/value_loss": 1e6 * (0.9**i),
            }
        )
    return out


class TestPlotEvalCurves:
    def test_eval_only_layout_is_1x3(self, tmp_path):
        # Without train_records the figure should be a 1x3 row of eval
        # panels (win rate / avg reward / episode length). Height-of-1
        # keeps notebook output compact when the PPO internals aren't
        # being captured (e.g. older runs).
        results = _synthetic_eval_results()
        fig = plot_eval_curves(results, charts_dir=tmp_path)
        assert fig is not None
        # Match the documented 1x3 row layout.
        assert len(fig.axes) == 3
        assert (tmp_path / "eval_curves.png").is_file()
        plt.close(fig)

    def test_with_train_records_layout_is_2x3(self, tmp_path):
        # When PPO internals are present, switch to 2x3: top row eval,
        # bottom row update health (approx_kl, EV, value_loss). Six
        # axes total.
        results = _synthetic_eval_results()
        train_records = _synthetic_train_records()
        fig = plot_eval_curves(results, train_records=train_records, charts_dir=tmp_path, opponent_label="random")
        assert fig is not None
        assert len(fig.axes) == 6
        # value_loss panel renders log-scale because magnitudes can
        # span 4-5 orders. Confirm the helper actually flips the axis
        # rather than letting matplotlib auto-pick a linear one and
        # silently squashing the curve.
        value_loss_ax = fig.axes[5]
        assert value_loss_ax.get_yscale() == "log"
        plt.close(fig)

    def test_stage_boundaries_draw_vertical_lines(self, tmp_path):
        # Bootstrap-curriculum view passes stage transition timesteps;
        # the helper should draw a dashed vline on each panel so the
        # reader can see whether a metric jumps at a transition.
        results = _synthetic_eval_results()
        train_records = _synthetic_train_records()
        boundaries = [100_000, 200_000]
        fig = plot_eval_curves(
            results,
            train_records=train_records,
            charts_dir=None,
            stage_boundaries=boundaries,
        )
        assert fig is not None
        # Each axis should have at least len(boundaries) vertical lines
        # in addition to whatever target/threshold hlines the panel
        # itself draws. axvline returns a Line2D in ax.lines; count by
        # vertical-line predicate (xdata is a 2-element array of equal
        # values).
        for ax in fig.axes:
            verticals = [
                line for line in ax.lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
            ]
            # >= because some panels also render approx_kl / EV
            # threshold hlines, which are horizontal not vertical, so
            # they shouldn't count -- the inequality just guards against
            # off-by-one.
            assert len(verticals) >= len(boundaries)
        plt.close(fig)

    def test_handles_missing_value_loss_gracefully(self, tmp_path):
        # Older callbacks may not capture value_loss; the panel should
        # still render (empty plot) without raising.
        results = _synthetic_eval_results()
        train_records = _synthetic_train_records()
        for r in train_records:
            r.pop("train/value_loss", None)
        fig = plot_eval_curves(results, train_records=train_records, charts_dir=None)
        assert fig is not None
        plt.close(fig)

    def test_charts_dir_none_returns_figure_without_saving(self, tmp_path):
        results = _synthetic_eval_results()
        fig = plot_eval_curves(results, charts_dir=None)
        assert fig is not None
        # Helper should NOT have written to tmp_path even though we
        # passed it for the fixture (we only passed charts_dir=None).
        assert not list(tmp_path.iterdir())
        plt.close(fig)
