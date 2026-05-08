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

from reinforcetactics.rl.viz import plot_individual_game_stats  # noqa: E402


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
