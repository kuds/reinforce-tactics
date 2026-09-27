"""Review fixes on the eval-gate / lifecycle package: stall verdicts, run records and --resume hardening.

* Stall verdicts and records are worded from what the promotion criterion
  compared (Wilson bound, rolling mean, win+draw/2 score, the weaker seat
  with the pooled win_rate peak next to it), not from the win-only point
  peak; the steps a stage trained over its attempts are reported. A rolling
  window still filling records no gate value.
* bootstrap_results.csv says which mode ``win_rate`` measured and what the gate
  compared.
* --resume: a promoted stage whose config.json write failed is not
  retrained; a checkpoint taken on the promoting eval finishes the stage; a
  run killed after its last stage promoted gets its final_model.zip and
  run_status.json; an interrupt mid-eval leaves the eval pending and the best
  record true to best_model.zip; torn records are handled; a retry killed
  before its first checkpoint restarts from the checkpoint it began from; the
  config check lives in run_curriculum; a pre-change record keeps its meaning
  (including a ``ppo.lr_schedule`` that had no effect then).
* Write failures (rolling checkpoint, eval JSONL, train-metrics CSV, the
  interrupt-path checkpoint) are reported and counted, over every session of
  a run.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from reinforcetactics.rl import bootstrap, callbacks
from reinforcetactics.rl.bootstrap import CurriculumStalled, ResumeError, _plan_resume, run_curriculum
from reinforcetactics.rl.callbacks import TrainingMetricsCallback
from reinforcetactics.rl.config import TrainingConfig, save_config
from reinforcetactics.rl.evaluation import wilson_lower_bound, z_for_confidence
from tests.test_curriculum_recovery import (
    MAP,
    _cfg,
    _config_json,
    _Env,
    _fake_evaluate,
    _Interrupted,
    _ScriptedModel,
    _stage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(
    cfg: TrainingConfig,
    out: Path,
    programs: dict[str, list[list[float]]],
    monkeypatch,
    *,
    evaluate=_fake_evaluate,
    interrupt_at: tuple[str, int, int] | None = None,
    env_factory=lambda s, c: _Env(),
):
    monkeypatch.setattr("reinforcetactics.rl.callbacks.evaluate_model", evaluate)
    names = [s.name for s in cfg.curriculum.stages]
    holder: dict[str, _ScriptedModel] = {}

    def factory(vec_env, cfg_arg, output_dir):
        holder["model"] = _ScriptedModel(programs, names)
        holder["model"].interrupt_at = interrupt_at
        return holder["model"]

    result = run_curriculum(
        cfg, out, train_env_factory=env_factory, eval_env_factory=lambda s, c: _Env(), model_factory=factory
    )
    return result, holder["model"]


def _resume(
    cfg: TrainingConfig,
    out: Path,
    programs: dict[str, list[list[float]]],
    *,
    stage_offset: int = 0,
    attempts: dict[str, int] | None = None,
    force: bool = False,
):
    """``run_curriculum(resume=True)`` with a scripted model loaded from the checkpoint it names."""
    loaded: dict[str, Any] = {}

    def loader(path, vec_env, cfg_arg, output_dir):
        model = _ScriptedModel(programs, [s.name for s in cfg_arg.curriculum.stages])
        model.stage_offset = stage_offset
        model.num_timesteps = json.loads(Path(path).read_text())["num_timesteps"]
        for name, count in (attempts or {}).items():
            model.attempts[name] = count
        loaded["path"], loaded["model"] = Path(path), model
        return model

    def must_not_build(*args):
        raise AssertionError("resume must load a checkpoint, not build a fresh model")

    result = run_curriculum(
        cfg,
        out,
        train_env_factory=lambda s, c: _Env(),
        eval_env_factory=lambda s, c: _Env(),
        model_factory=must_not_build,
        model_loader=loader,
        resume=True,
        **({"force": True} if force else {}),
    )
    return result, loaded


def _status(out: Path) -> dict[str, Any]:
    return json.loads((out / "run_status.json").read_text())


def _manifest(out: Path) -> dict[str, Any]:
    return json.loads((out / "run_manifest.json").read_text())


def _two_seat_evaluate(model, env, **kwargs):
    """Seat 1 wins at the scripted rate, seat 2 at half of it; 20 episodes per seat."""
    wr = float(model.current_wr)
    w1, w2, n = round(wr * 20), round(wr * 10), 40
    wins = w1 + w2
    return {
        "win_rate": wins / n,
        "avg_reward": wr,
        "std_reward": 0.0,
        "avg_length": 1.0,
        "avg_turns": 1.0,
        "wins": wins,
        "losses": n - wins,
        "draws": 0,
        "episodes": n,
        "draw_rate": 0.0,
        "loss_rate": (n - wins) / n,
        "by_seat": {
            "1": {"wins": w1, "losses": 20 - w1, "draws": 0, "episodes": 20, "win_rate": w1 / 20},
            "2": {"wins": w2, "losses": 20 - w2, "draws": 0, "episodes": 20, "win_rate": w2 / 20},
        },
    }


def _draw_evaluate(model, env, **kwargs):
    """Like ``_fake_evaluate``, but every game the agent does not win is a draw."""
    m = _fake_evaluate(model, env, **kwargs)
    draws = m["losses"]
    n = m["episodes"]
    m.update(losses=0, draws=draws, draw_rate=draws / n, loss_rate=0.0)
    m["by_seat"] = {"1": {"wins": m["wins"], "losses": 0, "draws": draws, "episodes": n, "win_rate": m["win_rate"]}}
    return m


# ---------------------------------------------------------------------------
# Stall verdicts in the gate's own terms
# ---------------------------------------------------------------------------


class TestStallVerdict:
    def test_wilson_stall_reports_the_bound_not_the_point_peak(self, tmp_path, monkeypatch):
        # 19/20 on every eval: the point estimate (95%) clears 90% four times
        # in a row, but the Wilson bound never does.
        cfg = _cfg(_stage("a", patience=1, max_retries=0, promotion_criterion="wilson"))
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[0.95, 0.95, 0.95, 0.95]]}, monkeypatch)
        bound = wilson_lower_bound(19, 20, z_for_confidence(0.95))
        message = str(excinfo.value)
        assert f"peak Wilson 95% lower bound of win_rate {bound:.1%} did not reach threshold 90.0%" in message
        assert "(win_rate peaked at 95.0%)" in message
        assert "never held" not in message and "patience" not in message
        status = _status(tmp_path)
        assert status["peak_gate_value"] == pytest.approx(bound)
        assert status["gate"]["criterion"] == "wilson" and status["gate"]["passes"] == 0
        assert status["achieved_win_rate"] == pytest.approx(0.95)
        assert _config_json(tmp_path, "a")["extra"]["gate"]["peak_gate_value"] == pytest.approx(bound)

    def test_rolling_stall_reports_the_rolling_mean(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1, max_retries=0, promotion_criterion="rolling", promotion_rolling_k=3))
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[0.5, 0.95, 0.5, 0.5]]}, monkeypatch)
        assert "peak rolling-3 mean win_rate 65.0% did not reach threshold 90.0% (win_rate peaked at 95.0%)" in str(
            excinfo.value
        )
        # Each row records what the gate compared and its verdict; nothing is
        # compared until the window holds 3 evals, so the first two rows have
        # no gate value (a partial mean would chart as a comparison).
        rows = json.loads((tmp_path / "a" / "eval_results.json").read_text())
        assert [r["gate_passed"] for r in rows] == [False] * 4
        assert [r["gate_value"] for r in rows] == [None, None, pytest.approx(0.65), pytest.approx(0.65)]
        with (tmp_path / "bootstrap_results.csv").open(encoding="utf-8") as fh:
            assert [r["gate_value"] for r in csv.DictReader(fh)][:2] == ["", ""]

    def test_filling_rolling_window_has_no_gate_value(self, tmp_path, monkeypatch):
        # The first eval (95%) is above the threshold on its own, but the
        # rolling-3 gate compared nothing there: no gate value to chart.
        cfg = _cfg(_stage("a", patience=1, max_retries=0, promotion_criterion="rolling", promotion_rolling_k=3))
        with pytest.raises(CurriculumStalled):
            _run(cfg, tmp_path, {"a": [[0.95, 0.5, 0.5, 0.5]]}, monkeypatch)
        rows = json.loads((tmp_path / "a" / "eval_results.json").read_text())
        assert [r["gate_value"] for r in rows] == [None, None, pytest.approx(0.65), pytest.approx(0.5)]
        assert _status(tmp_path)["peak_gate_value"] == pytest.approx(0.65)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from reinforcetactics.rl.viz import plot_curriculum_summary

        fig = plot_curriculum_summary([{"stage": "a", "results": rows}], cfg.curriculum.stages)
        drawn = [tuple(round(float(v), 4) for v in line.get_ydata()) for line in fig.axes[0].lines]
        plt.close(fig)
        # The thin dashed gate line starts where the gate first compared.
        assert (0.65, 0.5) in drawn
        assert (0.95, 0.725, 0.65, 0.5) not in drawn

    def test_weaker_seat_gate_names_the_seat_and_the_pooled_peak(self, tmp_path, monkeypatch):
        # Pooled over both seats the first evals win 75% / 72.5% (above 70%),
        # but the gate takes the weaker seat, which peaks at 50%.
        cfg = _cfg(
            _stage("a", promotion_win_rate=0.7, patience=2, max_retries=0),
            eval={"eval_seats": [1, 2], "seat_aggregate": "min"},
        )
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[1.0, 0.95, 0.6, 0.5]]}, monkeypatch, evaluate=_two_seat_evaluate)
        assert str(excinfo.value).endswith(
            "peak weaker-seat win_rate 50.0% did not reach threshold 70.0% (pooled win_rate peaked at 75.0%)"
        )
        gate = _status(tmp_path)["gate"]
        assert gate["seat_aggregate"] == "min" and gate["seats"] == [1, 2]
        assert gate["peak_pooled_win_rate"] == pytest.approx(0.75)
        assert _config_json(tmp_path, "a")["extra"]["gate"]["peak_pooled_win_rate"] == pytest.approx(0.75)

    def test_pooled_two_seat_gate_keeps_the_plain_wording(self, tmp_path, monkeypatch):
        cfg = _cfg(
            _stage("a", promotion_win_rate=0.8, patience=2, max_retries=0),
            eval={"eval_seats": [1, 2], "seat_aggregate": "mean"},
        )
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[1.0, 0.95, 0.6, 0.5]]}, monkeypatch, evaluate=_two_seat_evaluate)
        assert str(excinfo.value).endswith("peak win_rate 75.0% did not reach threshold 80.0%")

    def test_weaker_seat_wording_for_other_criteria(self):
        record = {
            "criterion": "wilson",
            "score": "win_rate",
            "confidence": 0.95,
            "seat_aggregate": "min",
            "seats": [1, 2],
            "peak_gate_value": 0.31,
            "passes": 0,
            "evals_judged": 4,
            "peak_pooled_win_rate": 0.75,
        }
        assert bootstrap.stall_verdict(0.5, 0.7, 2, record) == (
            "peak Wilson 95% lower bound of weaker-seat win_rate 31.0% did not reach threshold 70.0% "
            "(weaker-seat win_rate peaked at 50.0%; pooled win_rate peaked at 75.0%)"
        )
        # One seat: 'min' is the pooled rate, nothing to name.
        single = {**record, "criterion": "point", "seats": [1], "peak_gate_value": 0.5}
        assert bootstrap.stall_verdict(0.5, 0.7, 2, single) == "peak win_rate 50.0% did not reach threshold 70.0%"

    def test_half_draw_score_that_passed_but_never_held(self, tmp_path, monkeypatch):
        # 17 wins + 3 draws scores 92.5% (>= 90%) while the win rate is 85%.
        cfg = _cfg(_stage("a", max_retries=0, promotion_score="win_plus_half_draw"))
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[0.85, 0.5, 0.85, 0.5]]}, monkeypatch, evaluate=_draw_evaluate)
        assert (
            "win+draw/2 score peaked at 92.5% (>= threshold 90.0%) but never held it for patience=2 consecutive "
            "evals (passed on 2 of 4 evals, longest run 1) (win_rate peaked at 85.0%)"
        ) in str(excinfo.value)

    def test_passes_inside_the_min_timesteps_window_are_named(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1, max_retries=0, min_timesteps_before_promotion=25))
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[0.95, 0.95, 0.5, 0.5]]}, monkeypatch)
        message = str(excinfo.value)
        assert "peak win_rate 50.0% did not reach threshold 90.0%" in message
        assert "2 eval(s) inside the first 25 stage steps (min_timesteps_before_promotion) passed but do not count" in (
            message
        )

    def test_record_spans_attempts_and_reports_steps_trained(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a"))  # max_retries 1, patience 2
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[0.95, 0.5], [0.95, 0.5]]}, monkeypatch)
        message = str(excinfo.value)
        assert "stalled (after 1 retry; 40 timesteps trained, budget 1,000 per attempt)" in message
        assert "passed on 2 of 4 evals, longest run 1" in message
        assert excinfo.value.trained_timesteps == 40
        status = _status(tmp_path)
        assert status["trained_timesteps"] == 40 and status["gate"]["passes"] == 2


# ---------------------------------------------------------------------------
# bootstrap_results.csv says what win_rate measured
# ---------------------------------------------------------------------------


def test_results_csv_carries_mode_and_gate_columns(tmp_path, monkeypatch):
    cfg = _cfg(_stage("a", patience=1))
    _run(cfg, tmp_path, {"a": [[0.5, 0.95]]}, monkeypatch)
    with (tmp_path / "bootstrap_results.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for column in ("deterministic", "win_rate_stochastic", "win_rate_greedy", "gate_win_rate", "gate_value", "attempt"):
        assert column in rows[0]
    assert [r["deterministic"] for r in rows] == ["False", "False"]
    assert [r["gate_passed"] for r in rows] == ["False", "True"]
    assert rows[1]["gate_criterion"] == "point" and float(rows[1]["gate_value"]) == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# --resume: records that failed, promotions already won, interrupted evals
# ---------------------------------------------------------------------------


class TestResumeRecords:
    def test_promoted_stage_whose_config_write_failed_is_not_retrained(self, tmp_path, monkeypatch, capsys):
        real = bootstrap._write_stage_config

        def flaky(**kwargs):
            if kwargs["stage"].name == "a":
                raise OSError("EIO")
            return real(**kwargs)

        monkeypatch.setattr(bootstrap, "_write_stage_config", flaky)
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=3))
        programs = {"a": [[0.95]], "b": [[0.95] * 5]}
        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch, interrupt_at=("b", 0, 2))
        assert not (tmp_path / "a" / "config.json").exists()

        result, loaded = _resume(cfg, tmp_path, programs, stage_offset=1)
        # Stage a is taken from the manifest's record, not retrained from scratch.
        assert loaded["path"] == tmp_path / "b" / "latest.zip"
        assert [e["stage"] for e in loaded["model"].learns] == ["b"]
        first = result["history"][0]
        assert first["stage"] == "a" and first["promoted"] is True and first["from_previous_session"] is True
        assert [r["timesteps"] for r in first["results"]] == [10]
        assert "using run_manifest.json's record" in capsys.readouterr().out
        # The first session's failure is still counted after the resume.
        status = _status(tmp_path)
        assert status["status"] == "completed_curriculum" and status["resume_count"] == 1
        assert status["metadata_write_failures"] == 1
        assert status["metadata_write_failure_log"] == ["write a/config.json"]
        assert [e["stage"] for e in _manifest(tmp_path)["completed"]] == ["a", "b"]

    def test_refuses_to_overwrite_later_stages_unless_forced(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1))
        _run(cfg, tmp_path, {"a": [[0.95]], "b": [[0.95]]}, monkeypatch)
        # Stage a's record is lost everywhere, but stage b has output.
        (tmp_path / "a" / "config.json").unlink()
        (tmp_path / "run_status.json").unlink()
        manifest = _manifest(tmp_path)
        manifest["completed"] = [e for e in manifest["completed"] if e["stage"] != "a"]
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ResumeError, match=r"later stage\(s\) \['b'\] already have output"):
            _plan_resume(cfg, tmp_path)
        assert _plan_resume(cfg, tmp_path, force=True).start_index == 0

    @pytest.mark.parametrize(
        ("checkpoint_freq", "max_timesteps"),
        [(20, 1_000), (1_000, 1_000), (20, 20)],
        ids=["checkpoint-on-the-promoting-eval", "no-periodic-checkpoint", "promoted-on-the-last-budget-step"],
    )
    def test_kill_after_promotion_finishes_the_stage_on_resume(self, tmp_path, monkeypatch, checkpoint_freq, max_timesteps):
        cfg = _cfg(
            _stage("a", patience=1, max_retries=0, max_timesteps=max_timesteps),
            _stage("b", patience=1),
            eval={"checkpoint_freq": checkpoint_freq},
        )
        programs = {"a": [[0.5, 0.95]], "b": [[0.95]]}
        real = callbacks.save_model_atomically
        fired: list[bool] = []

        def killed_writing_stage_final(model, path):
            if Path(path).name == "stage_final.zip" and not fired:
                fired.append(True)
                raise _Interrupted
            return real(model, path)

        monkeypatch.setattr(callbacks, "save_model_atomically", killed_writing_stage_final)
        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch)
        current = _manifest(tmp_path)["current"]
        assert current["stage"] == "a" and current["promotion_state"]["promoted"] is True
        assert current["latest_timesteps"] == 20

        result, loaded = _resume(cfg, tmp_path, programs)
        # a is finished, not trained again (and not declared stalled); b trains.
        assert [e["stage"] for e in loaded["model"].learns] == ["b"]
        extra = _config_json(tmp_path, "a")["extra"]
        assert extra["promoted"] is True and [a["promoted"] for a in extra["attempts"]] == [True]
        assert [r["timesteps"] for r in result["history"][0]["results"]] == [10, 20]
        assert _status(tmp_path)["status"] == "completed_curriculum"

    def test_kill_after_the_last_stage_is_written_up_on_resume(self, tmp_path, monkeypatch, capsys):
        # Killed while closing the last stage's envs: every stage's config.json
        # says promoted, but there is no final_model.zip or run_status.json,
        # and the manifest still names 'b' in progress.
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=2))
        programs = {"a": [[0.95]], "b": [[0.5, 0.95, 0.95]]}

        class _KilledOnClose(_Env):
            def close(self) -> None:
                raise _Interrupted

        def envs(stage, c):
            return _KilledOnClose() if stage.name == "b" else _Env()

        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch, env_factory=envs)
        assert _config_json(tmp_path, "b")["extra"]["promoted"] is True
        assert not (tmp_path / "final_model.zip").exists() and not (tmp_path / "run_status.json").exists()
        assert _manifest(tmp_path)["current"]["stage"] == "b"
        assert (tmp_path / "b" / "latest.zip").exists()

        result, loaded = _resume(cfg, tmp_path, programs)
        assert loaded == {}  # nothing is trained
        final = tmp_path / "final_model.zip"
        assert result["final_model_path"] == str(final)
        # The policy the run would have saved: b's best (restore_best_checkpoint_between_stages).
        assert final.read_bytes() == (tmp_path / "b" / "best_model.zip").read_bytes()
        status = _status(tmp_path)
        assert status["status"] == "completed_curriculum" and status["finished_on_resume"] is True
        assert status["resume_count"] == 1 and status["retries"] == {"a": 0, "b": 0}
        assert status["stages_completed"] == 2 and status["metadata_write_failures"] == 0
        manifest = _manifest(tmp_path)
        assert manifest["current"] is None and [e["stage"] for e in manifest["completed"]] == ["a", "b"]
        assert manifest["resume_count"] == 1
        assert not (tmp_path / "b" / "latest.zip").exists()
        with (tmp_path / "bootstrap_results.csv").open(encoding="utf-8") as fh:
            assert [r["stage"] for r in csv.DictReader(fh)] == ["a", "b", "b", "b"]
        assert "stopped before writing final_model.zip and run_status.json" in capsys.readouterr().out

        # Written up once: a second resume has nothing to do.
        written_at = status["written_at"]
        again, _ = _resume(cfg, tmp_path, programs)
        assert again["final_model_path"] == str(final) and _status(tmp_path)["written_at"] == written_at
        assert "nothing to resume" in capsys.readouterr().out

    def test_completed_run_missing_its_final_records(self, tmp_path, monkeypatch):
        cfg = _cfg(
            _stage("a", patience=1), _stage("b", patience=1), curriculum={"restore_best_checkpoint_between_stages": False}
        )
        programs = {"a": [[0.95]], "b": [[0.5, 0.95]]}
        _run(cfg, tmp_path, programs, monkeypatch)
        final = tmp_path / "final_model.zip"

        # Killed between final_model.zip and run_status.json.
        (tmp_path / "run_status.json").unlink()
        _resume(cfg, tmp_path, programs)
        assert _status(tmp_path)["status"] == "completed_curriculum"

        # Both gone: without the best-checkpoint restore the final model is
        # the last stage's end-of-stage policy.
        (tmp_path / "run_status.json").unlink()
        final.unlink()
        result, _ = _resume(cfg, tmp_path, programs)
        assert result["final_model_path"] == str(final)
        assert final.read_bytes() == (tmp_path / "b" / "stage_final.zip").read_bytes()
        assert _status(tmp_path)["resume_count"] == 2

        # A finished run whose final_model.zip was lost gets it back; its
        # run_status.json is left as it was.
        written_at = _status(tmp_path)["written_at"]
        final.unlink()
        _resume(cfg, tmp_path, programs)
        assert final.is_file() and _status(tmp_path)["written_at"] == written_at

        # Nothing to rebuild it from: a clear refusal, not a run without a final model.
        final.unlink()
        (tmp_path / "run_status.json").unlink()
        (tmp_path / "b" / "stage_final.zip").unlink()
        with pytest.raises(ResumeError, match="cannot write .*final_model.zip"):
            _resume(cfg, tmp_path, programs)
        assert not (tmp_path / "run_status.json").exists()

    def test_interrupt_mid_eval_leaves_the_eval_pending(self, tmp_path, monkeypatch):
        # eval_freq 20 with 10-step increments: b evaluates at 20, 40, 60, ...
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1), eval={"eval_freq": 20})
        # One entry per 10-step env step; b's evaluated entries are every other one.
        programs = {"a": [[0.95]], "b": [[0.5, 0.5, 0.5, 0.5, 0.95, 0.95]]}
        fired: list[bool] = []

        def interrupted_at_40(model, env, **kwargs):
            if model.num_timesteps == 40 and not fired:
                fired.append(True)
                raise _Interrupted
            return _fake_evaluate(model, env, **kwargs)

        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch, evaluate=interrupted_at_40)
        current = _manifest(tmp_path)["current"]
        # The checkpoint names the step its zip holds, and the eval at 40 (block
        # 2) is not marked done.
        assert current["latest_timesteps"] == 40
        assert json.loads((tmp_path / "b" / "latest.zip").read_text())["num_timesteps"] == 40
        assert current["last_eval_block"] == 1
        # Had the kill come after the eval's row reached the JSONL but before
        # the eval was committed, that row is not part of the checkpoint either.
        jsonl = tmp_path / "b" / "eval_results.jsonl"
        uncommitted = {**json.loads(jsonl.read_text().splitlines()[0]), "timesteps": 40}
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(uncommitted) + "\n")

        result, _ = _resume(cfg, tmp_path, programs, stage_offset=1)
        timeline = [r["timesteps"] for r in result["history"][1]["results"]]
        # Block 2 is evaluated after the resume (at 50), not skipped; every
        # block once.
        assert timeline == [20, 50, 60, 80, 100]

    def test_interrupt_while_saving_a_new_best_keeps_the_record_true_to_the_file(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1), eval={"eval_freq": 20})
        # b: 20 is a carry-in eval (not best-eligible); 40 is b's first best.
        programs = {"a": [[0.95]], "b": [[0.3, 0.3, 0.6, 0.6, 0.95, 0.95]]}
        real = callbacks.save_model_atomically
        fired: list[bool] = []

        def killed_saving_best(model, path):
            if Path(path).parent.name == "b" and Path(path).name == "best_model.zip" and not fired:
                fired.append(True)
                raise _Interrupted
            return real(model, path)

        monkeypatch.setattr(callbacks, "save_model_atomically", killed_saving_best)
        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch)
        state = _manifest(tmp_path)["current"]["eval_state"]
        # No best was committed, as none reached the disk.
        assert not (tmp_path / "b" / "best_model.zip").exists()
        assert state["best_win_rate"] is None and state["best_timestep"] == -1

        _resume(cfg, tmp_path, programs, stage_offset=1)
        extra = _config_json(tmp_path, "b")["extra"]
        saved_at = json.loads((tmp_path / "b" / "best_model.zip").read_text())["num_timesteps"]
        assert extra["best_checkpoint_timestep"] == saved_at


class TestTornRecords:
    def test_write_run_config_never_leaves_a_torn_file(self, tmp_path, monkeypatch):
        from reinforcetactics.utils.run_config import write_run_config

        target = tmp_path / "config.json"
        write_run_config({"extra": {"promoted": True}}, target)
        real_write_text = Path.write_text

        def torn(self, data, *args, **kwargs):
            real_write_text(self, data[: len(data) // 2], *args, **kwargs)
            raise OSError("killed mid-write")

        monkeypatch.setattr(Path, "write_text", torn)
        with pytest.raises(OSError):
            write_run_config({"extra": {"promoted": False, "pad": "x" * 200}}, target)
        monkeypatch.undo()
        assert json.loads(target.read_text()) == {"extra": {"promoted": True}}
        assert list(tmp_path.iterdir()) == [target]

    def test_torn_eval_results_json_falls_back_to_the_jsonl(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1))
        _run(cfg, tmp_path, {"a": [[0.5, 0.95]], "b": [[0.95]]}, monkeypatch)
        path = tmp_path / "a" / "eval_results.json"
        path.write_text(path.read_text()[:40])
        plan = _plan_resume(cfg, tmp_path)
        assert [r["timesteps"] for r in plan.history[0]["results"]] == [10, 20]

    def test_torn_config_json(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1))
        _run(cfg, tmp_path, {"a": [[0.95]], "b": [[0.95]]}, monkeypatch)
        (tmp_path / "run_status.json").unlink()
        config = tmp_path / "b" / "config.json"
        config.write_text(config.read_text()[:100])
        # The manifest recorded b's promotion: the stage stands.
        assert _plan_resume(cfg, tmp_path).completed
        # Without that record: a clear error naming the file.
        manifest = _manifest(tmp_path)
        manifest["completed"] = [e for e in manifest["completed"] if e["stage"] != "b"]
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ResumeError, match=r"b/config.json is unreadable"):
            _plan_resume(cfg, tmp_path)


class TestRetryKilledBeforeItsFirstCheckpoint:
    def test_resumes_the_retry_not_the_stage(self, tmp_path, monkeypatch):
        real = bootstrap._RunManifest.begin_attempt
        fired: list[bool] = []

        def begin_then_die(self, **fields):
            real(self, **fields)
            if fields["attempt"] == 1 and not fired:
                fired.append(True)
                raise _Interrupted

        monkeypatch.setattr(bootstrap._RunManifest, "begin_attempt", begin_then_die)
        cfg = _cfg(_stage("a"))  # patience 2, max_retries 1
        programs = {"a": [[0.95, 0.3], [0.95, 0.95]]}
        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch)
        current = _manifest(tmp_path)["current"]
        assert current["attempt"] == 1 and current["latest_timesteps"] is None

        result, loaded = _resume(cfg, tmp_path, programs, attempts={"a": 1})
        # The retry restarts from the checkpoint it began from, at its own start.
        assert loaded["path"] == tmp_path / "a" / "best_model.zip"
        assert [(e["stage"], e["attempt"], e["start"]) for e in loaded["model"].learns] == [("a", 1, 20)]
        extra = _config_json(tmp_path, "a")["extra"]
        assert extra["promoted"] is True and [a["promoted"] for a in extra["attempts"]] == [False, True]
        assert [r["attempt"] for r in result["history"][0]["results"]] == [0, 0, 1, 1]
        # The best found before the kill carried over.
        assert extra["best_win_rate"] == pytest.approx(0.95) and extra["best_checkpoint_timestep"] == 10


class TestResumeConfigCheck:
    def _killed_in_b(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=3, opponent="noop"))
        programs = {"a": [[0.95]], "b": [[0.95] * 5]}
        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch, interrupt_at=("b", 0, 2))
        return programs

    def test_run_curriculum_refuses_another_curriculum_unless_forced(self, tmp_path, monkeypatch):
        programs = self._killed_in_b(tmp_path, monkeypatch)
        changed = _cfg(_stage("a", patience=1), _stage("b", patience=3, opponent="simple", max_timesteps=1_536))
        with pytest.raises(ResumeError, match="curriculum_hash"):
            _resume(changed, tmp_path, programs, stage_offset=1)
        result, _ = _resume(changed, tmp_path, programs, stage_offset=1, force=True)
        assert [h["stage"] for h in result["history"]] == ["a", "b"]

    def test_run_curriculum_compares_with_resolved_config(self, tmp_path, monkeypatch):
        programs = self._killed_in_b(tmp_path, monkeypatch)
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=3, opponent="noop"))
        save_config(bootstrap.resolve_config(cfg), tmp_path / "resolved_config.yaml")
        other = _cfg(_stage("a", patience=1), _stage("b", patience=3, opponent="noop"), eval={"n_eval_episodes": 40})
        with pytest.raises(ResumeError, match=r"eval\.n_eval_episodes"):
            _resume(other, tmp_path, programs, stage_offset=1)
        result, _ = _resume(cfg, tmp_path, programs, stage_offset=1)
        assert result["history"][1]["promoted"] is True


def _old_record(cfg: TrainingConfig, path: Path) -> None:
    """A resolved_config.yaml as written before the eval-gate change."""
    data = bootstrap.resolve_config(cfg).to_dict()
    for section, key in bootstrap.LEGACY_RECORD_DEFAULTS:
        data[section].pop(key)
    data["eval"]["eval_seats"] = None
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


class TestPreChangeRecord:
    def test_keeps_its_greedy_gate_and_reports_no_spurious_difference(self, tmp_path):
        cfg = _cfg(_stage("a"), eval={"eval_both_modes": True})
        _old_record(cfg, tmp_path / "resolved_config.yaml")
        recorded = bootstrap.load_recorded_config(tmp_path / "resolved_config.yaml")
        assert recorded.eval.eval_deterministic is True
        assert recorded.eval.eval_both_modes is False
        assert recorded.curriculum.max_retries == 0
        # Its own record: nothing differs (eval_seats null resolves to [1]).
        assert bootstrap.resume_config_differences(recorded, tmp_path) == []
        # Today's defaults gate on something else, and that is reported.
        differences = bootstrap.resume_config_differences(cfg, tmp_path)
        assert {"eval.eval_deterministic", "eval.eval_both_modes", "curriculum.max_retries"} <= set(differences)
        assert "eval.eval_seats" not in differences

    def test_lr_schedule_it_ignored_stays_ignored(self, tmp_path):
        # Before the change ppo.lr_schedule was reported as not implemented
        # and the LR stayed constant; today 'linear' anneals every stage to 0.
        cfg = _cfg(_stage("a"), ppo={"lr_schedule": "linear"})
        _old_record(cfg, tmp_path / "resolved_config.yaml")
        assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text())["ppo"]["lr_schedule"] == "linear"
        recorded = bootstrap.load_recorded_config(tmp_path / "resolved_config.yaml")
        assert recorded.ppo.lr_schedule == "constant"
        assert recorded.curriculum.stages[0].resolve_learning_rate_schedule(recorded.ppo) is None
        assert bootstrap.resume_config_differences(recorded, tmp_path) == []
        # Resuming it with the linear schedule would change what is trained.
        assert "ppo.lr_schedule" in bootstrap.resume_config_differences(cfg, tmp_path)

    def test_record_written_after_the_change_keeps_its_lr_schedule(self, tmp_path):
        cfg = _cfg(_stage("a"), ppo={"lr_schedule": "linear"})
        save_config(bootstrap.resolve_config(cfg), tmp_path / "resolved_config.yaml")
        assert bootstrap.load_recorded_config(tmp_path / "resolved_config.yaml").ppo.lr_schedule == "linear"
        assert bootstrap.resume_config_differences(cfg, tmp_path) == []


# ---------------------------------------------------------------------------
# Write failures: reported, counted, never fatal
# ---------------------------------------------------------------------------


class TestWriteFailures:
    def test_failed_rolling_checkpoint_does_not_end_the_run(self, tmp_path, monkeypatch):
        real = callbacks.save_model_atomically

        def flaky(model, path):
            if Path(path).name == "latest.zip" and model.num_timesteps == 20:
                raise OSError(5, "Input/output error")
            return real(model, path)

        monkeypatch.setattr(callbacks, "save_model_atomically", flaky)
        cfg = _cfg(_stage("a", patience=4))
        _run(cfg, tmp_path, {"a": [[0.95] * 4]}, monkeypatch)
        status = _status(tmp_path)
        assert status["status"] == "completed_curriculum"
        assert status["metadata_write_failures"] == 1
        assert "rolling checkpoint" in status["metadata_write_failure_log"][0]

    def test_failed_interrupt_save_is_carried_into_the_resumed_run(self, tmp_path, monkeypatch):
        # The last checkpoint on the way out fails. A successful save rewrites
        # the manifest (on_save); a failed one used to leave the manifest
        # without the failure, so the resumed run reported 0.
        real = callbacks.save_model_atomically

        def fails_while_interrupted(model, path):
            if Path(path).name == "latest.zip" and isinstance(sys.exc_info()[1], KeyboardInterrupt):
                raise OSError(5, "EIO on the interrupt save")
            return real(model, path)

        monkeypatch.setattr(callbacks, "save_model_atomically", fails_while_interrupted)
        cfg = _cfg(_stage("a", patience=1), _stage("b"))
        programs = {"a": [[0.95]], "b": [[0.1, 0.2, 0.3, 0.95, 0.95]]}
        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch, interrupt_at=("b", 0, 3))
        manifest = _manifest(tmp_path)
        assert manifest["metadata_write_failures"] == 1
        assert manifest["metadata_write_failure_log"] == ["save latest.zip on interrupt"]
        # The manifest still describes the last good checkpoint.
        assert manifest["current"]["stage"] == "b" and manifest["current"]["latest_timesteps"] == 30

        _resume(cfg, tmp_path, programs, stage_offset=1)
        status = _status(tmp_path)
        assert status["status"] == "completed_curriculum"
        assert status["metadata_write_failures"] == 1
        assert status["metadata_write_failure_log"] == ["save latest.zip on interrupt"]

    def test_eval_jsonl_append_failures_are_counted(self, tmp_path, monkeypatch):
        def broken(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(callbacks, "json", SimpleNamespace(dumps=broken))
        cfg = _cfg(_stage("a", patience=2))
        _run(cfg, tmp_path, {"a": [[0.95, 0.95]]}, monkeypatch)
        status = _status(tmp_path)
        assert status["metadata_write_failures"] == 2
        assert all("eval_results.jsonl" in w for w in status["metadata_write_failure_log"])

    def test_train_metrics_csv_failure_is_reported(self, tmp_path, caplog):
        cb = TrainingMetricsCallback(csv_path=tmp_path / "is_a_directory")
        (tmp_path / "is_a_directory").mkdir()
        with caplog.at_level(logging.WARNING, logger="reinforcetactics.rl.callbacks"):
            cb._append_csv_row({"timesteps": 1})
        assert any("is_a_directory" in r.getMessage() and r.exc_info for r in caplog.records)
        seen: list[str] = []
        cb.on_write_failure = seen.append
        cb._append_csv_row({"timesteps": 2})
        assert seen and "is_a_directory" in seen[0]


# ---------------------------------------------------------------------------
# The CLI: clean refusals
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    import importlib.util

    monkeypatch.setattr(sys, "path", [*sys.path])
    spec = importlib.util.spec_from_file_location("train_bootstrap_hardening", REPO_ROOT / "scripts/train/train_bootstrap.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_COMMON = ["--device", "cpu", "--no-gcs", "--skip-plots", "--skip-videos", "--sanity-episodes", "0"]
_BASE = {
    "env": {"n_envs": 1, "use_subprocess": False, "action_space_type": "multi_discrete"},
    "curriculum": {"stages": [{"name": "s", "map_file": MAP, "opponent": "noop", "max_timesteps": 100}]},
}


class TestCli:
    def test_unresumable_run_is_a_message_not_a_traceback(self, cli, tmp_path):
        from reinforcetactics.rl.config import config_from_dict

        out = tmp_path / "run"
        out.mkdir()
        cfg = config_from_dict(_BASE)
        cfg.ppo.device = "cpu"
        save_config(bootstrap.resolve_config(cfg), out / "resolved_config.yaml")
        (out / "run_status.json").write_text(json.dumps({"status": "curriculum_stalled", "stalled_stage": "s"}))
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--resume", str(out), *_COMMON])
        assert isinstance(excinfo.value.code, str) and "stalled at stage 's'" in excinfo.value.code

    def test_build_bc_run_stopped_before_the_warm_start(self, cli, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []

        def fake_run(cfg, output_dir, **kwargs):
            calls.append({"cfg": cfg, **kwargs})
            return {"history": [], "final_model_path": None}

        def bc_dies(cfg, output_dir, args):
            raise RuntimeError("preempted while building the BC warm start")

        monkeypatch.setattr(bootstrap, "run_curriculum", fake_run)
        monkeypatch.setattr(cli, "_bc_build", bc_dies)
        config = tmp_path / "c.yaml"
        config.write_text(yaml.safe_dump(_BASE), encoding="utf-8")
        out = tmp_path / "run"
        with pytest.raises(RuntimeError, match="preempted"):
            cli.main(["--config", str(config), "--output-dir", str(out), "--build-bc", *_COMMON])
        assert (out / cli.BC_PENDING_MARKER).exists()
        with pytest.raises(SystemExit, match="--build-bc"):
            cli.main(["--resume", str(out), *_COMMON])
        assert calls == []
        assert cli.main(["--resume", str(out), "--force", *_COMMON]) == 0
        assert calls[-1]["resume"] is True and calls[-1]["force"] is True

    def test_resuming_a_pre_change_run_keeps_its_gate(self, cli, tmp_path, monkeypatch):
        calls: list[dict[str, Any]] = []

        def fake_run(cfg, output_dir, **kwargs):
            calls.append({"cfg": cfg, **kwargs})
            return {"history": [], "final_model_path": None}

        monkeypatch.setattr(bootstrap, "run_curriculum", fake_run)
        from reinforcetactics.rl.config import config_from_dict

        out = tmp_path / "run"
        out.mkdir()
        cfg = config_from_dict(_BASE)
        cfg.ppo.device = "cpu"
        _old_record(cfg, out / "resolved_config.yaml")
        assert cli.main(["--resume", str(out), *_COMMON]) == 0
        resumed = calls[-1]["cfg"]
        assert resumed.eval.eval_deterministic is True and resumed.eval.eval_both_modes is False
        assert resumed.curriculum.max_retries == 0


# ---------------------------------------------------------------------------
# The chart compares the threshold with what the gate compared
# ---------------------------------------------------------------------------


def test_curriculum_summary_draws_the_gate_value(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reinforcetactics.rl.viz import plot_curriculum_summary

    stage = SimpleNamespace(name="a", promotion_win_rate=0.9)
    plain = [{"timesteps": t, "win_rate": 0.95, "gate_value": 0.95, "best_eligible": True} for t in (10, 20, 30)]
    wilson = [{**r, "gate_value": 0.78} for r in plain]
    lines = []
    for rows in (plain, wilson):
        fig = plot_curriculum_summary([{"stage": "a", "results": rows}], [stage])
        lines.append([tuple(line.get_ydata()) for line in fig.axes[0].lines])
        plt.close(fig)
    assert (0.78, 0.78, 0.78) not in lines[0]
    assert (0.78, 0.78, 0.78) in lines[1]


def test_bootstrap_yaml_wilson_advice_matches_the_bound():
    text = (REPO_ROOT / "configs" / "ppo" / "bootstrap.yaml").read_text(encoding="utf-8")
    pairs = [(float(a), float(b)) for a, b in re.findall(r"(\d\.\d\d) -> (\d\.\d\d)", text)]
    assert len(pairs) == 10
    z = z_for_confidence(0.95)
    for (threshold, advised), n in zip(pairs, [80] * 5 + [160] * 5, strict=True):
        assert advised == pytest.approx(round(wilson_lower_bound(math.ceil(n * threshold - 1e-9), n, z), 2))


def test_a_stage_restarted_fresh_on_resume_drops_the_aborted_sessions_best(tmp_path, monkeypatch):
    """A stage with no checkpoint to resume from starts over; its old best_model.zip must go with it.

    The aborted session's best_model.zip stayed on disk, so the between-stages
    restore (and a retry) loaded weights no eval of the restarted stage chose.
    """
    real = callbacks.save_model_atomically

    def no_latest_for_b(model, path):
        if Path(path).name == "latest.zip" and Path(path).parent.name == "b":
            raise OSError(5, "EIO")
        return real(model, path)

    monkeypatch.setattr(callbacks, "save_model_atomically", no_latest_for_b)
    cfg = _cfg(_stage("a", patience=1), _stage("b", patience=2))
    programs = {"a": [[0.95]], "b": [[0.5, 0.6, 0.7, 0.7, 0.7]]}
    with pytest.raises(KeyboardInterrupt):
        _run(cfg, tmp_path, programs, monkeypatch, interrupt_at=("b", 0, 2))
    stale = tmp_path / "b" / "best_model.zip"
    assert stale.is_file() and not (tmp_path / "b" / "latest.zip").exists()

    # The resumed session's evals of b are never best-eligible, so nothing it
    # does saves a new best_model.zip for b.
    resumed_cfg = _cfg(_stage("a", patience=1), _stage("b", patience=2), eval={"best_eligible_after": 10_000})
    resumed = {"a": [[0.95]], "b": [[0.95, 0.95]]}
    _, loaded = _resume(resumed_cfg, tmp_path, resumed, stage_offset=1, force=True)

    assert _status(tmp_path)["status"] == "completed_curriculum"
    assert str(stale) not in loaded["model"].set_parameters_calls
    assert not stale.exists()
