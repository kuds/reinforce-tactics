"""The seed aggregator (reinforcetactics/experiments/run_summary.py, scripts/eval/summarize_seeds.py).

Synthetic run directories (tests/helpers/run_dirs.py) in the layout the
curriculum runner writes today and in the 2026-06 archive's legacy layout;
the report is checked against a golden file (regenerate with
``UPDATE_GOLDEN=1``).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from reinforcetactics.experiments import run_summary as rs
from reinforcetactics.experiments.stats import describe, t_quantile_975, wilson_interval, z_two_sided
from tests.helpers.run_dirs import MAP, PER_EPISODE, eval_row, make_config, stage, write_run

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "data" / "seed_report_golden.md"
T0 = 1_800_000_000.0


@pytest.fixture
def summarize_seeds(monkeypatch):
    monkeypatch.setattr(sys, "path", [*sys.path])
    spec = importlib.util.spec_from_file_location(
        "summarize_seeds_under_test", REPO_ROOT / "scripts" / "eval" / "summarize_seeds.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(start: int, results, *, other=None, wall0: float | None = T0, **kwargs):
    """Rows every 100 steps after ``start``; ``results`` / ``other`` are (W, D, L) per row."""
    out = []
    for i, wdl in enumerate(results):
        ts = start + 100 * (i + 1)
        extra = dict(kwargs)
        if wall0 is not None and extra.get("mode", "stochastic") != "legacy":
            extra.update(wall_time=wall0 + ts * 0.6, eval_seconds=12.0)
        out.append(eval_row(ts, *wdl, other=other[i] if other else None, **extra))
    return out


def _group_stages(seed: int) -> tuple[list[dict], str]:
    """Stages A-D of one seed: all clear A and B; 42 and 1042 clear C (2042 stalls); 42 clears D (1042 is in it)."""
    a = stage(
        "A",
        _rows(0, [(5, 4, 1), (8, 1, 1), (9, 1, 0)], other=[(7, 2, 1), (9, 1, 0), (10, 0, 0)]),
        promoted=True,
        start=0,
        end=300,
        created_at="2026-09-28T01:00:00+00:00",
    )
    b = stage("B", _rows(300, [(8, 2, 0), (9, 1, 0)], other=[(8, 2, 0), (8, 2, 0)]), promoted=True, start=300, end=500)
    if seed == 2042:
        c_rows = _rows(500, [(2, 7, 1)] * 10 + [(3, 6, 1)] * 10, other=[(1, 8, 1)] * 20, truncated=2)
        c = stage("C", c_rows, promoted=False, start=500, end=2500, retries=1)
        return [a, b, c, stage("D", [], promoted=None)], "stalled"
    c_rows = _rows(500, [(4, 5, 1), (6, 3, 1), (7, 2, 1), (8, 1, 1), (8, 2, 0), (9, 1, 0)], other=[(6, 3, 1)] * 6)
    c = stage("C", c_rows, promoted=True, start=500, end=1100 if seed == 42 else 1300)
    if seed == 42:
        d = stage("D", _rows(1100, [(7, 3, 0), (8, 2, 0)], other=[(9, 1, 0), (9, 1, 0)]), promoted=True, start=1100, end=1300)
        return [a, b, c, d], "completed"
    d = stage("D", _rows(1300, [(3, 7, 0)], other=[(2, 8, 0)]), promoted=None)
    return [a, b, c, d], "interrupted"


def _write_group(root: Path) -> list[Path]:
    runs = []
    for seed in (42, 1042, 2042):
        stages, status = _group_stages(seed)
        runs.append(
            write_run(
                root, f"20260928_120000_val_s{seed}", stages, seed=seed, status=status, resume_count=1 if seed == 1042 else 0
            )
        )
    return runs


def _write_legacy(root: Path) -> Path:
    """A v52a-like archive run: greedy rows, resampled eval seeds, 'mixed' on stage A, killed in D."""
    legacy_rows = lambda start, results: [  # noqa: E731
        eval_row(start + 100 * (i + 1), *wdl, mode="legacy", eval_seed=1_000_042 + 1000 * (start // 100 + i + 1))
        for i, wdl in enumerate(results)
    ]
    stages = [
        stage(
            "A",
            legacy_rows(0, [(6, 3, 1), (10, 0, 0), (10, 0, 0)]),
            promoted=True,
            opponent="mixed",
            created_at="2026-06-01T18:00:00+00:00",
        ),
        stage("B", legacy_rows(300, [(9, 1, 0), (10, 0, 0)]), promoted=True, created_at="2026-06-01T19:00:00+00:00"),
        stage(
            "C",
            legacy_rows(500, [(1, 9, 0), (10, 0, 0), (10, 0, 0)]),
            promoted=True,
            created_at="2026-06-01T21:00:00+00:00",
        ),
        stage("D", legacy_rows(800, [(2, 8, 0)]), promoted=None),
    ]
    return write_run(root, "20260601_172412", stages, seed=42, legacy=True)


# ---------------------------------------------------------------------------
# Readers and per-stage metrics
# ---------------------------------------------------------------------------


class TestNewLayout:
    def test_stochastic_gate_row_and_greedy_other_mode(self, tmp_path):
        run = rs.read_run(_write_group(tmp_path)[0])
        assert (run.layout, run.status, run.seed, run.group) == ("new", "completed", 42, "20260928_120000_val")
        metrics = rs.run_metrics(run)
        a = metrics["stages"][0]
        # The final row is the promoting eval (9/1/0 stochastic, greedy 10/0/0 from other_mode).
        assert (a["stoch_wins"], a["stoch_draws"], a["stoch_losses"], a["stoch_episodes"]) == (9, 1, 0, 10)
        assert (a["greedy_wins"], a["greedy_draws"], a["greedy_losses"]) == (10, 0, 0)
        assert a["gate_mode"] == "stochastic" and a["label"] == "at gate"
        lo, hi = wilson_interval(9, 10, 0.95)
        assert (a["stoch_wr_lo"], a["stoch_wr_hi"]) == (pytest.approx(lo), pytest.approx(hi))
        # Exact steps from the stage record, and the patience window of the last attempt.
        assert (a["steps_to_promotion"], a["steps_exact"], a["censored"], a["cum_steps_end"]) == (300, True, False, 300)
        assert a["window_stoch_win_rate"] == pytest.approx(17 / 20)
        assert a["retries"] == 0 and "retries_note" not in a
        assert a["eval_resampled"] is False
        assert a["steps_per_hour"] == pytest.approx(100 / 60 * 3600)
        assert a["eval_share"] == pytest.approx(12 / 60)

    def test_greedy_gate_takes_stochastic_from_other_mode(self, tmp_path):
        rows = [eval_row(100, 9, 1, 0, mode="greedy", other=(6, 3, 1))]
        path = write_run(tmp_path, "run_g", [stage("A", rows, promoted=True, start=0, end=100)], seed=1)
        a = rs.run_metrics(rs.read_run(path))["stages"][0]
        assert a["gate_mode"] == "greedy"
        assert (a["greedy_wins"], a["greedy_draws"]) == (9, 1)
        assert (a["stoch_wins"], a["stoch_draws"], a["stoch_losses"]) == (6, 3, 1)

    def test_stall_is_censored_with_retries_and_run_level_fields(self, tmp_path):
        run = rs.run_metrics(rs.read_run(_write_group(tmp_path)[2]))
        c = run["stages"][2]
        assert c["outcome"] == "stalled" and c["censored"] is True
        assert c["steps_to_promotion"] is None and c["trained_steps"] == 2000
        assert c["retries"] == 1
        assert c["end_reason_rate_max_steps_truncate"] == pytest.approx(0.2)
        assert run["stages"][3]["outcome"] == "not_reached"
        assert (run["status"], run["stages_cleared"], run["stalled_stage"], run["deepest_stage"]) == ("stalled", 2, "C", "C")
        assert run["retries_used"] == 1 and run["metadata_write_failures"] == 0

    def test_interrupted_run_reads_the_stage_in_progress_from_its_jsonl(self, tmp_path):
        run = rs.run_metrics(rs.read_run(_write_group(tmp_path)[1]))
        d = run["stages"][3]
        assert (run["status"], d["outcome"], d["rows_source"], d["n_evals"]) == (
            "interrupted",
            "interrupted",
            "eval_results.jsonl",
            1,
        )
        assert run["resume_count"] == 1

    def test_reward_shares_by_outcome_and_captures(self, tmp_path):
        a = rs.run_metrics(rs.read_run(_write_group(tmp_path)[0]))["stages"][0]
        # 9 wins (10, -2, 0, 50) and 1 draw (6, -1, -0.5, -10) per episode.
        a_sum, s_sum, i_sum, t_sum = 9 * 10 + 6, 9 * -2 - 1, -0.5, 9 * 50 - 10
        nt_abs = abs(a_sum) + abs(s_sum) + abs(i_sum)
        assert a["shaping_share_abs"] == pytest.approx(nt_abs / (nt_abs + abs(t_sum)))
        assert a["shaping_share_signed"] == pytest.approx((a_sum + s_sum + i_sum) / (a_sum + s_sum + i_sum + t_sum))
        ep_abs = 9 * 12 + 7.5
        assert a["episode_abs_share"] == pytest.approx(ep_abs / (ep_abs + 9 * 50 + 10))
        assert a["by_outcome"]["draws"] == {
            "episodes": 1,
            "non_terminal_per_ep": 4.5,
            "terminal_per_ep": -10.0,
            "return_per_ep": -5.5,
        }
        # Without the potential term (6 - 0.5 - 10), and with it (-1 more).
        assert a["draw_return_per_ep"] == pytest.approx(-4.5) and a["draw_breakeven"] is False
        assert a["draw_return_raw_per_ep"] == pytest.approx(-5.5) and a["draw_return_has_potential"] is False
        assert a["reward_per_ep_terminal"] == pytest.approx(t_sum / 10) and a["reward_sum_mismatch"] is False
        assert (a["captures_per_ep_tower"], a["captures_per_ep_building"], a["captures_per_ep_hq"]) == (1.0, 0.5, 0.0)
        assert a["opponent_captures_per_ep_neutral"] == 1.0 and a["opponent_captures_per_ep_owned"] == 0.0

    def test_dense_share_takes_the_turn_penalty_out_of_the_action_stream(self, tmp_path):
        a = rs.run_metrics(rs.read_run(_write_group(tmp_path)[0]))["stages"][0]
        # 20 end_turns per episode at the config's turn_penalty -0.5: -10 per episode inside `action`.
        assert a["reward_per_ep_turn_penalty"] == pytest.approx(-10.0)
        dense, t_sum = abs(9 * 10 + 6 - 10 * -10.0), 9 * 50 - 10
        assert a["dense_share_abs"] == pytest.approx(dense / (dense + abs(t_sum)))
        # A stage's own reward_config wins over env.reward_config.
        rows = [eval_row(100, 9, 1, 0, end_turns=4)]
        path = write_run(
            tmp_path, "run_tp", [stage("A", rows, promoted=True, start=0, end=100, reward_config={"turn_penalty": -2.0})]
        )
        a = rs.run_metrics(rs.read_run(path))["stages"][0]
        assert a["reward_per_ep_turn_penalty"] == pytest.approx(-8.0)
        # An archive row without action_counts has no dense share.
        legacy = rs.run_metrics(rs.read_run(_write_legacy(tmp_path)))["stages"][0]
        assert legacy["dense_share_abs"] is None and legacy["reward_per_ep_turn_penalty"] is None
        assert rs._dense_share({"action": -30.0, "terminal": 50.0}, -40.0) == pytest.approx(10 / 60)
        assert rs._dense_share({"action": 1.0, "terminal": 1.0}, None) is None

    def test_per_seat_gate_win_rates(self, tmp_path):
        a = rs.run_metrics(rs.read_run(_write_group(tmp_path)[0]))["stages"][0]
        assert (a["seat1_win_rate"], a["seat2_win_rate"]) == (pytest.approx(0.9), None)
        row = eval_row(100, 6, 2, 2)
        row["by_seat"] = {"1": {"win_rate": 0.9, "episodes": 5}, "2": {"win_rate": 0.3, "episodes": 5}}
        path = write_run(tmp_path, "run_seats", [stage("A", [row], promoted=True, start=0, end=100)])
        a = rs.run_metrics(rs.read_run(path))["stages"][0]
        assert (a["seat1_win_rate"], a["seat2_win_rate"]) == (0.9, 0.3)
        assert rs.run_metrics(rs.read_run(_write_legacy(tmp_path)))["stages"][0]["seat1_win_rate"] is None

    def test_a_row_whose_components_do_not_add_up_is_flagged(self, tmp_path):
        rows = [eval_row(100, 9, 1, 0), eval_row(200, 9, 1, 0)]
        rows[0]["avg_reward"] += 5.0
        path = write_run(tmp_path, "run_m", [stage("A", rows, promoted=True, start=0, end=200)], seed=5)
        summary = rs.build_summary([rs.read_run(path)])
        a = summary["runs"][0]["stages"][0]
        assert a["reward_sum_mismatch"] is True and a["reward_sum_mismatch_at"] == [100]
        assert any(f["kind"] == "reward_sum_mismatch" and f["stage"] == "A" for f in summary["flags"])

    def test_a_stage_record_without_rows(self, tmp_path):
        path = write_run(tmp_path, "run_n", [stage("A", [], promoted=True, start=0, end=100)], seed=6)
        a = rs.run_metrics(rs.read_run(path))["stages"][0]
        assert (a["outcome"], a["gate_mode"], a["stoch_win_rate"], a["steps_to_promotion"]) == ("cleared", None, None, 100)

    def test_signed_share_is_none_when_the_return_nearly_cancels(self):
        comps = {"action": 5.0, "shaping_delta": 0.0, "invalid_penalty": 0.0, "terminal": -5.5}
        assert rs._share_signed(comps, episodes=1) is None
        assert rs._share_signed(comps, episodes=0) is None
        assert rs._share_abs(comps) == pytest.approx(5 / 10.5)
        assert rs._share_abs({c: 0.0 for c in rs.REWARD_COMPONENTS}) is None

    def test_draw_breakeven_counts_every_row(self, tmp_path):
        farming = {"draws": {"action": 12.0, "shaping_delta": 0.0, "invalid_penalty": 0.0, "terminal": -10.0}}
        per = {**PER_EPISODE, **farming}
        rows = [eval_row(100, 2, 8, 0, per_episode=per), eval_row(200, 9, 1, 0)]
        path = write_run(tmp_path, "run_d", [stage("A", rows, promoted=True, start=0, end=200)], seed=3)
        a = rs.run_metrics(rs.read_run(path))["stages"][0]
        assert a["draw_breakeven_evals"] == 1 and a["draw_breakeven_at"] == [100]
        assert a["draw_breakeven"] is False  # the final row's draws lose

    def test_draw_breakeven_leaves_the_potential_term_out(self, tmp_path):
        """The smoke test's corner_points draws: -258 action, -50 terminal, +393 potential term per draw.

        With the potential term a draw 'returns' +85 and was flagged break-even; its undiscounted eval sum
        is drift, not reward the policy can farm, so the flag reads the -308 without it.
        """
        drift = {"draws": {"action": -258.0, "shaping_delta": 393.0, "invalid_penalty": 0.0, "terminal": -50.0}}
        rows = [eval_row(100, 0, 7, 0, per_episode={**PER_EPISODE, **drift})]
        path = write_run(tmp_path, "run_phi", [stage("A", rows, promoted=True, start=0, end=100)], seed=8)
        summary = rs.build_summary([rs.read_run(path)])
        a = summary["runs"][0]["stages"][0]
        assert a["draw_return_per_ep"] == pytest.approx(-308.0) and a["draw_return_raw_per_ep"] == pytest.approx(85.0)
        assert a["draw_breakeven"] is False and a["draw_breakeven_evals"] == 0
        assert not any(f["kind"] == "draw_breakeven" for f in summary["flags"])
        # The other way round: a potential drain must not hide a draw that pays.
        drain = {"draws": {"action": 60.0, "shaping_delta": -1580.0, "invalid_penalty": 0.0, "terminal": -50.0}}
        rows = [eval_row(100, 0, 5, 0, per_episode={**PER_EPISODE, **drain})]
        path = write_run(tmp_path, "run_drain", [stage("A", rows, promoted=True, start=0, end=100)], seed=9)
        a = rs.run_metrics(rs.read_run(path))["stages"][0]
        assert a["draw_return_per_ep"] == pytest.approx(10.0) and a["draw_breakeven"] is True
        # A row with whole-episode returns only cannot take it out, and says so.
        row = {"draws": 2, "wins": 0, "losses": 0, "rewards": [3.0, 5.0], "outcomes": ["draws", "draws"]}
        assert rs.draw_return(row) == (4.0, True) and rs.draw_return_has_potential(row) is True

    def test_eval_and_training_throughput(self, tmp_path):
        """Eval: agent steps / eval seconds; training: steps between rows / (wall time - the later row's eval)."""
        rows = [
            eval_row(1000, 9, 1, 0, wall_time=T0, eval_seconds=10.0),
            eval_row(2000, 9, 1, 0, wall_time=T0 + 30.0, eval_seconds=10.0),  # 1000 steps in 20 s of training
            eval_row(3000, 9, 1, 0, wall_time=T0 + 55.0, eval_seconds=5.0),  # 1000 steps in 20 s
            eval_row(4000, 9, 1, 0, wall_time=T0 + 9000.0, eval_seconds=5.0),  # across a pause: skipped
        ]
        path = write_run(tmp_path, "run_tp", [stage("A", rows, promoted=True, start=0, end=4000)], seed=4)
        run = rs.run_metrics(rs.read_run(path))
        a = run["stages"][0]
        # 10 episodes of 100 agent steps per row: 4000 steps over 30 s of eval.
        assert a["eval_agent_steps_per_s"] == pytest.approx(4000 / 30.0)
        assert a["train_steps_per_s"] == pytest.approx(2000 / 40.0)
        assert run["throughput_by_map"][MAP] == {
            "eval_agent_steps_per_s": pytest.approx(4000 / 30.0),
            "train_steps_per_s": pytest.approx(50.0),
        }
        # A both-modes row's lengths cover one mode only: left out of the eval rate.
        both = eval_row(100, 9, 1, 0, other=(9, 1, 0), wall_time=T0, eval_seconds=10.0)
        assert rs._eval_throughput([both]) is None


class TestLegacyLayout:
    def test_greedy_only_approximate_steps_aborted(self, tmp_path):
        run = rs.read_run(_write_legacy(tmp_path))
        assert (run.layout, run.status, run.seed, run.config_source) == (
            "legacy",
            "aborted",
            42,
            "v52a_maxturn_scaled_draw.yaml",
        )
        metrics = rs.run_metrics(run)
        a, _, c, d = metrics["stages"]
        assert a["gate_mode"] == "greedy"
        assert a["greedy_win_rate"] == 1.0 and a["stoch_win_rate"] is None and a["stoch_wins"] is None
        # No step bounds in a legacy record: last row - first row, marked approximate.
        assert (a["steps_to_promotion"], a["steps_exact"]) == (200, False)
        assert a["retries"] == 0 and a["retries_note"] == "no retry feature"
        assert a["eval_resampled"] is True
        assert d["outcome"] == "interrupted" and d["rows_source"] == "eval_results.jsonl"
        assert metrics["git"] == "078313e" and metrics["wall_clock_source"].startswith("stage records")
        assert c["draw_breakeven_evals"] == 0

    def test_legacy_draw_return_lower_bound(self):
        # The archive row at 5.75M: 11/0/69 with a positive average; which
        # episodes drew is not recorded, so the lowest possible draw total is used.
        row = {
            "wins": 11,
            "losses": 0,
            "draws": 69,
            "episodes": 80,
            "avg_reward": 97.0,
            "rewards": [200.0] * 11 + [80.0] * 69,
        }
        value, exact = rs.draw_return(row)
        assert exact is False and value == pytest.approx(80.0)
        all_draw = {"wins": 0, "losses": 0, "draws": 10, "episodes": 10, "avg_reward": 3.0}
        assert rs.draw_return(all_draw) == (3.0, True)

    def test_csv_only_input(self, tmp_path):
        run = rs.read_run(_write_legacy(tmp_path) / "bootstrap_results.csv")
        assert run.layout == "csv" and run.status == "unknown"
        metrics = rs.run_metrics(run)
        assert [s["outcome"] for s in metrics["stages"]] == ["cleared", "cleared", "unknown"]
        assert metrics["stages"][0]["greedy_win_rate"] == 1.0 and metrics["stages"][0]["steps_to_promotion"] == 200

    def test_runs_per_stage_rows(self, tmp_path):
        path = tmp_path / "runs_per_stage.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "run_id",
                    "stage_name",
                    "map_file",
                    "opponent",
                    "opponent_max_actions",
                    "promotion_win_rate",
                    "patience",
                    "max_turns",
                    "steps_in_stage",
                    "final_win_rate",
                    "peak_win_rate",
                    "outcome",
                ]
            )
            w.writerow(["r1", "A", MAP, "random", 10, 0.7, 2, 30, 250, 0.9, 0.95, "cleared"])
            w.writerow(["r1", "B", MAP, "simple", "", 0.7, 2, 30, "", "", "", "not_started"])
            w.writerow(["r2", "A", MAP, "random", 10, 0.7, 2, 30, 900, 0.5, 0.6, "stalled"])
        run = rs.read_run(f"{path}:r1")
        metrics = rs.run_metrics(run)
        a, b = metrics["stages"]
        assert (a["outcome"], a["greedy_win_rate"], a["steps_to_promotion"], a["peak_gate_wr"]) == ("cleared", 0.9, 250, 0.95)
        assert run.stages[0].settings["opponent_kwargs"] == {"max_actions": 10}
        assert b["outcome"] == "not_reached"


# ---------------------------------------------------------------------------
# Across seeds, replicates, comparison
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_three_seeds_and_censoring(self, tmp_path):
        runs = [rs.run_metrics(rs.read_run(p)) for p in _write_group(tmp_path)]
        agg = rs.aggregate(runs)
        assert agg["stage_order"] == ["A", "B", "C", "D"]
        by = {s["stage"]: s for s in agg["stages"]}
        assert (by["C"]["n_reached"], by["C"]["n_cleared"], by["C"]["n_stalled"]) == (3, 2, 1)
        assert by["C"]["seed_sensitive"] is True and by["A"]["seed_sensitive"] is False
        steps = by["C"]["metrics"]["steps_to_promotion"]
        # Only the seeds that cleared it: 600 and 800 env steps.
        assert (steps["n"], steps["median"], steps["min"], steps["max"]) == (2, 700, 600, 800)
        assert by["C"]["metrics"]["trained_steps"]["values"] == {"s42": 600, "s1042": 800, "s2042": 2000}
        wr = by["A"]["metrics"]["stoch_win_rate"]
        assert wr["n"] == 3 and wr["sd"] == 0.0 and wr["ci95_lo"] == wr["ci95_hi"] == pytest.approx(0.9)
        assert by["D"]["n_reached"] == 2 and by["D"]["outcomes"]["s2042"] == "not_reached"

    def test_runs_of_the_same_seed_are_kept_apart(self, tmp_path):
        """Two different runs of seed 42 (the whole archive is seed 42): neither overwrites the other."""
        done_rows = _rows(0, [(9, 1, 0), (9, 1, 0)])
        stalled_rows = _rows(0, [(1, 9, 0)] * 3)
        done = write_run(tmp_path, "repa_s42", [stage("s2", done_rows, promoted=True, start=0, end=200)], seed=42)
        stalled = write_run(
            tmp_path, "stl_s42", [stage("s2", stalled_rows, promoted=False, start=0, end=300)], seed=42, status="stalled"
        )
        other = write_run(tmp_path, "repa_s1042", [stage("s2", done_rows, promoted=True, start=0, end=200)], seed=1042)
        summary = rs.build_summary([rs.read_run(p) for p in (done, stalled, other)], allow_mixed=True)
        labels = [r["label"] for r in summary["runs"]]
        assert labels == ["s42@repa_s42", "s42@stl_s42", "s1042"]
        (s2,) = summary["stages"]
        assert (s2["n_reached"], s2["n_cleared"], s2["n_stalled"]) == (3, 2, 1)
        assert s2["outcomes"] == {"s42@repa_s42": "cleared", "s42@stl_s42": "stalled", "s1042": "cleared"}
        wr = s2["metrics"]["stoch_win_rate"]
        assert wr["n"] == 3 and wr["values"] == {"s42@repa_s42": 0.9, "s42@stl_s42": 0.1, "s1042": 0.9}
        assert rs.render_report(summary).count("s42@stl_s42") >= 2
        assert rs.unique_labels(["s1", "s1", "s2"], ["a", "a", "b"]) == ["s1@a", "s1@a#2", "s2"]

    def test_active_hours_leave_out_the_time_between_sessions(self, tmp_path):
        # Rows every 60 s, then a 50-minute outage, then rows every 60 s again.
        walls = [T0 + 60.0 * i for i in range(5)] + [T0 + 240.0 + 3000.0 + 60.0 * i for i in range(5)]
        rows = [eval_row(100 * (i + 1), 9, 1, 0, wall_time=w, eval_seconds=5.0) for i, w in enumerate(walls)]
        path = write_run(tmp_path, "20260928_120000_val_s42", [stage("A", rows, promoted=True, start=0, end=1000)])
        record = rs.read_run(path)
        run = rs.run_metrics(record)
        assert run["wall_clock_h"] == pytest.approx((240.0 + 3000.0 + 240.0) / 3600.0)
        # The outage counts as 3x the median gap (180 s), not 3000 s.
        assert run["active_h"] == pytest.approx((8 * 60.0 + 180.0) / 3600.0) and run["active_source"].startswith("eval")
        # The launcher's sessions, when given, are the process time itself.
        stamp = lambda t: datetime.fromtimestamp(t, UTC).isoformat()  # noqa: E731
        sessions = [
            {"started_at": stamp(T0 - 20.0), "ended_at": stamp(T0 + 250.0)},
            {"started_at": stamp(T0 + 3400.0), "ended_at": None},  # its launcher died: to its last row
        ]
        run = rs.run_metrics(record, sessions=sessions)
        assert run["active_h"] == pytest.approx((270.0 + (walls[-1] - (T0 + 3400.0))) / 3600.0)
        assert run["active_source"] == "launcher sessions"
        summary = rs.build_summary([record], sessions={record.run_id: sessions})
        assert summary["runs"][0]["active_h"] == pytest.approx(run["active_h"])
        assert "| active h |" in rs.render_report(summary)

    def test_single_run(self, tmp_path):
        run = rs.run_metrics(rs.read_run(_write_group(tmp_path)[0]))
        agg = rs.aggregate([run])
        stats = agg["stages"][0]["metrics"]["stoch_win_rate"]
        assert (stats["n"], stats["mean"], stats["sd"], stats["ci95_lo"]) == (1, 0.9, None, None)

    def test_t_table_and_describe(self):
        assert t_quantile_975(2) == pytest.approx(4.3027)
        assert t_quantile_975(31) == pytest.approx(1.95996, abs=1e-4)
        d = describe([1.0, 2.0, 3.0, None, float("nan")])
        assert (d["n"], d["mean"], d["sd"], d["median"]) == (3, 2.0, 1.0, 2.0)
        assert d["ci95_hi"] - d["mean"] == pytest.approx(4.3027 / 3**0.5)

    def test_wilson_matches_the_gate_formula(self):
        from reinforcetactics.rl.evaluation import wilson_lower_bound

        for wins, n in ((0, 10), (7, 10), (80, 80), (33, 80)):
            lo, hi = wilson_interval(wins, n, 0.95)
            assert lo == pytest.approx(wilson_lower_bound(wins, n, z_two_sided(0.95)))
            assert hi == pytest.approx(1 - wilson_lower_bound(n - wins, n, z_two_sided(0.95)))


class TestReplicateCheck:
    def test_only_seed_or_device_differs(self, tmp_path):
        paths = _write_group(tmp_path)
        records = [rs.read_run(p) for p in paths]
        records[1].config["ppo"]["device"] = "cuda"  # type: ignore[index]
        diffs, notes = rs.replicate_differences(records)
        assert diffs == [] and notes == []

    def test_a_different_curriculum_fails_the_check(self, tmp_path, summarize_seeds, capsys):
        paths = _write_group(tmp_path / "g")
        cfg = make_config([stage("A", [], promoted=True), stage("B", [], promoted=True, patience=3)], seed=1042)
        import yaml

        raw = yaml.safe_load((paths[1] / "resolved_config.yaml").read_text())
        raw["curriculum"]["stages"][1]["patience"] = cfg["curriculum"]["stages"][1]["patience"]
        (paths[1] / "resolved_config.yaml").write_text(yaml.safe_dump(raw))
        diffs, _ = rs.replicate_differences([rs.read_run(p) for p in paths])
        assert diffs == ["s42 vs s1042: curriculum.stages[1].patience"]
        argv = [str(p) for p in paths] + ["--out-dir", str(tmp_path / "out"), "--quiet"]
        assert summarize_seeds.main(argv) == 1
        assert summarize_seeds.main([*argv, "--allow-mixed"]) == 0
        assert "FAILED" in (tmp_path / "out" / "report.md").read_text()


class TestComparison:
    def test_mixed_vs_random_and_reward_deltas(self, tmp_path):
        group = [rs.read_run(p) for p in _write_group(tmp_path / "g")]
        summary = rs.build_summary(group, baselines=[rs.baseline_side("v52a", str(_write_legacy(tmp_path / "b")))])
        comp = summary["comparisons"][0]
        rows = {r["stage"]: r for r in comp["stages"]}
        assert rows["A"]["comparable"] is False and rows["A"]["differing"] == ["opponent: 'mixed' vs 'random'"]
        assert rows["B"]["comparable"] is True and rows["B"]["unverified"] == []
        assert rows["B"]["baseline_greedy_wr"] == 1.0 and rows["B"]["group_greedy_wr"] == pytest.approx(0.8)
        assert rows["B"]["delta_greedy_wr"] == pytest.approx(-0.2)
        assert rows["C"]["steps_ratio"] == pytest.approx(700 / 200)
        deltas = {d["field"]: (d["baseline"], d["group"]) for d in comp["config_deltas"]}
        assert deltas["env.reward_config.turn_penalty"] == (0.0, -0.5)
        assert deltas["env.max_actions_per_turn"] == (None, 40)
        assert deltas["gate mode"] == ("greedy", "stochastic")
        assert deltas["eval.resample_eval_seeds"] == (True, False)
        assert comp["deepest_shared_stage_reached"] == "D"
        assert comp["cleared_among_shared"]["baseline"] == {"s42": 3}
        assert comp["cleared_among_shared"]["group"] == {"s42": 4, "s1042": 3, "s2042": 2}
        text = " ".join(comp["caveats"])
        assert "078313e" in text and "opponent-freeze" in text and "resampled" in text and "greedy" in text

    def test_group_against_a_summary_json(self, tmp_path):
        group = [rs.read_run(p) for p in _write_group(tmp_path / "g")]
        first = rs.build_summary(group, group_id="20260928_120000_val")
        out = tmp_path / "first"
        rs.write_outputs(first, out)
        side = rs.baseline_side("prev", str(out / "summary.json"))
        comp = rs.compare(side, rs.side_from_runs("g", group, first["runs"], source="x"))
        assert all(r["comparable"] for r in comp["stages"])
        assert all(r["delta_greedy_wr"] in (None, pytest.approx(0.0)) for r in comp["stages"])
        assert comp["config_deltas"] == []


# ---------------------------------------------------------------------------
# The CLI and its files
# ---------------------------------------------------------------------------


class TestCli:
    def test_no_runs_is_exit_2(self, summarize_seeds, tmp_path):
        assert summarize_seeds.main(["--out-dir", str(tmp_path / "o")]) == 2
        assert summarize_seeds.main(["--glob", str(tmp_path / "nothing*"), "--out-dir", str(tmp_path / "o")]) == 2

    def test_unreadable_input_is_exit_1(self, summarize_seeds, tmp_path):
        assert summarize_seeds.main([str(tmp_path / "missing"), "--out-dir", str(tmp_path / "o")]) == 1
        run = _write_group(tmp_path)[0]
        assert summarize_seeds.main([str(run), "--compare", "x=" + str(tmp_path / "nope"), "--quiet"]) == 1
        assert summarize_seeds.main([str(run), "--compare", "no-equals", "--quiet"]) == 2

    def test_group_manifest_input_and_files(self, summarize_seeds, tmp_path, monkeypatch):
        root = tmp_path / "root"
        _write_group(root)
        manifest = root / "_groups" / "20260928_120000_val" / "seed_group.json"
        manifest.parent.mkdir(parents=True)
        runs = {
            str(s): {"run_dir": f"20260928_120000_val_s{s}", "sessions": [], "last_state": "pending"}
            for s in (42, 1042, 2042, 3042)
        }
        manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "group": "20260928_120000_val",
                    "seeds": [42, 1042, 2042, 3042],
                    "runs": runs,
                    "config": "configs/ppo/bootstrap.yaml",
                    "config_digest": "d" * 64,
                }
            )
        )
        monkeypatch.chdir(tmp_path)
        assert summarize_seeds.main(["--group-manifest", str(manifest), "--quiet"]) == 0
        out = manifest.parent / "report"
        assert sorted(p.name for p in out.iterdir()) == sorted(
            ["report.md", "runs.csv", "per_seed_stage.csv", "per_stage.csv", "comparison.csv", "summary.json"]
        )
        summary = json.loads((out / "summary.json").read_text())
        assert summary["schema_version"] == 1
        assert set(summary) == {
            "schema_version",
            "generated_at",
            "inputs",
            "group",
            "definitions",
            "runs",
            "stages",
            "comparisons",
            "flags",
        }
        assert [r["seed"] for r in summary["runs"]] == [42, 1042, 2042]  # 3042 was never started
        assert summary["group"]["id"] == "20260928_120000_val" and summary["group"]["config_digest"] == "d" * 64
        with (out / "per_seed_stage.csv").open() as fh:
            assert len(list(csv.DictReader(fh))) == 12
        kinds = {f["kind"] for f in summary["flags"]}
        assert {"seed_sensitive", "skip_ahead", "resumed", "max_steps_truncate"} <= kinds

    def test_golden_report(self, summarize_seeds, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_DATE_EPOCH", "1790000000")
        paths = _write_group(tmp_path / "g")
        legacy = _write_legacy(tmp_path / "b")
        out = tmp_path / "out"
        argv = [*map(str, paths), "--compare", f"v52a={legacy}", "--out-dir", str(out), "--quiet"]
        assert summarize_seeds.main(argv) == 0
        report = (out / "report.md").read_text()
        if os.environ.get("UPDATE_GOLDEN"):
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_text(report)
        assert report == GOLDEN.read_text(), "report.md changed; rerun with UPDATE_GOLDEN=1 if intended"
        # Deterministic: a second run writes the same files.
        again = tmp_path / "again"
        assert summarize_seeds.main([*argv[:-3], "--out-dir", str(again), "--quiet"]) == 0
        for name in ("report.md", "per_stage.csv", "per_seed_stage.csv", "comparison.csv", "runs.csv"):
            assert (again / name).read_text() == (out / name).read_text()


def test_experiments_package_does_not_import_torch():
    code = (
        "import sys; import reinforcetactics.experiments.run_summary, reinforcetactics.experiments.seed_runs; "
        "print('torch' in sys.modules, 'stable_baselines3' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert out.stdout.strip() == "False False"
