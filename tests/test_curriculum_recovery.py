"""The curriculum runner: stall retries, regression guard, LR schedules, resume, run records.

Review findings covered here:

* rltrain-7 / prior-3: a stall ended the run; no retry from the stage's best
  checkpoint, no within-stage regression guard, and a message that said
  "did not reach threshold" for stages that had peaked above it.
* rltrain-8 / prior-6: ``ppo.lr_schedule`` was dropped and a stage could not
  set its learning rate; an SB3-native schedule would anneal over the
  cumulative counter.
* rltrain-5 / prior-3: a killed run could not resume.
* rltrain-18: ``ep_info_buffer`` carried the previous stage's episodes into
  the next stage's first rollout metrics.
* rltrain-21 / prior-16: metadata write failures were swallowed.
* rltrain-6: config.json said promoted with best_win_rate -1.0.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import pytest
import yaml

from reinforcetactics.rl import bootstrap
from reinforcetactics.rl.bootstrap import CurriculumStalled, ResumeError, _plan_resume, run_curriculum
from reinforcetactics.rl.callbacks import (
    EntropyScheduleCallback,
    LRScheduleCallback,
    PeriodicEvalCallback,
    PromotionCallback,
    RegressionGuardCallback,
    RollingCheckpointCallback,
)
from reinforcetactics.rl.config import TrainingConfig, config_from_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP = "maps/1v1/beginner.csv"


# ---------------------------------------------------------------------------
# A scripted stand-in for MaskablePPO that drives the real callbacks
# ---------------------------------------------------------------------------


class _Logger:
    def __init__(self) -> None:
        self.name_to_value: dict[str, Any] = {}
        self.dumps: list[int] = []

    def record(self, key: str, value: Any) -> None:
        self.name_to_value[key] = value

    def dump(self, step: int = 0) -> None:
        self.dumps.append(step)


class _Interrupted(KeyboardInterrupt):
    pass


class _ScriptedModel:
    """Plays scripted eval win rates through the runner's real callbacks.

    ``programs[stage]`` is one list of win rates per attempt; each entry is
    one eval, ``step`` env steps after the previous one. ``interrupt_at``
    (stage, attempt, eval index) raises KeyboardInterrupt there, as a
    SIGTERM would.
    """

    def __init__(self, programs: dict[str, list[list[float]]], stage_names: list[str], *, step: int = 10) -> None:
        self.programs = programs
        self.stage_names = stage_names
        self.step = step
        self.num_timesteps = 0
        self.logger = _Logger()
        self.ep_info_buffer: deque = deque([{"r": 1.0, "l": 3}], maxlen=100)
        self.ep_success_buffer: deque = deque([True], maxlen=100)
        self.ent_coef = 0.0
        self.learning_rate = 3e-4
        self.current_wr = 0.0
        self.stage_idx = 0
        self.attempts: Counter = Counter()
        self.learns: list[dict[str, Any]] = []
        self.saved: list[str] = []
        self.set_parameters_calls: list[str] = []
        self.interrupt_at: tuple[str, int, int] | None = None
        self.stage_offset = 0

    # SB3 surface -------------------------------------------------------
    def set_env(self, env: Any) -> None:
        self.stage_idx += 1

    def save(self, path: str) -> None:
        self.saved.append(str(path))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"num_timesteps": self.num_timesteps}), encoding="utf-8")

    def set_parameters(self, path: str, exact_match: bool = True) -> None:
        self.set_parameters_calls.append(str(path))

    def learn(self, total_timesteps, callback, reset_num_timesteps=True, progress_bar=False):
        name = self.stage_names[self.stage_idx + self.stage_offset]
        attempt = self.attempts[name]
        self.attempts[name] += 1
        self.learns.append(
            {
                "stage": name,
                "attempt": attempt,
                "budget": total_timesteps,
                "start": self.num_timesteps,
                "ep_info": len(self.ep_info_buffer),
                "ep_success": len(self.ep_success_buffer),
                "ent_coef": self.ent_coef,
                "learning_rate": self.learning_rate,
                "callbacks": list(callback),
            }
        )
        for cb in callback:
            cb.init_callback(self)
            cb.on_training_start({}, {})
        program = self.programs[name][min(attempt, len(self.programs[name]) - 1)]
        for i, wr in enumerate(program):
            if self.interrupt_at == (name, attempt, i):
                raise _Interrupted
            self.current_wr = wr
            self.num_timesteps += self.step
            keep_going = True
            for cb in callback:
                keep_going = cb.on_step() and keep_going
            for cb in callback:
                cb.on_rollout_start()
            # A fresh rollout fills the buffer again.
            self.ep_info_buffer.append({"r": wr, "l": 1})
            if not keep_going:
                break
        for cb in callback:
            cb.on_training_end()


def _fake_evaluate(model, env, **kwargs):
    wr = float(model.current_wr)
    n = 20
    wins = round(wr * n)
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
        "by_seat": {"1": {"wins": wins, "losses": n - wins, "draws": 0, "episodes": n, "win_rate": wins / n}},
    }


class _Env:
    def close(self) -> None:
        pass


def _stage(name: str, **kwargs: Any) -> dict[str, Any]:
    base = {"name": name, "map_file": MAP, "opponent": "random", "promotion_win_rate": 0.9, "patience": 2}
    return {**base, "max_timesteps": 1_000, **kwargs}


def _cfg(*stages: dict[str, Any], **sections: Any) -> TrainingConfig:
    raw: dict[str, Any] = {
        "env": {"n_envs": 1, "enabled_units": ["W"]},
        "eval": {"eval_freq": 10, "checkpoint_freq": 20, "n_eval_episodes": 20, "eval_both_modes": False},
        "curriculum": {"stages": list(stages)},
    }
    for key, value in sections.items():
        raw[key] = {**raw.get(key, {}), **value}
    return config_from_dict(raw)


def _run(cfg: TrainingConfig, out: Path, programs: dict[str, list[list[float]]], monkeypatch, **kwargs: Any):
    monkeypatch.setattr("reinforcetactics.rl.callbacks.evaluate_model", _fake_evaluate)
    names = [s.name for s in cfg.curriculum.stages]
    holder: dict[str, _ScriptedModel] = {}

    def factory(vec_env, cfg_arg, output_dir):
        holder["model"] = _ScriptedModel(programs, names)
        if "interrupt_at" in kwargs:
            holder["model"].interrupt_at = kwargs["interrupt_at"]
        return holder["model"]

    try:
        result = run_curriculum(
            cfg,
            out,
            train_env_factory=lambda stage, c: _Env(),
            eval_env_factory=lambda stage, c: _Env(),
            model_factory=factory,
        )
    finally:
        pass
    return result, holder["model"]


def _config_json(out: Path, stage: str) -> dict[str, Any]:
    return json.loads((out / stage / "config.json").read_text())


# ---------------------------------------------------------------------------
# rltrain-7: stall retry and message
# ---------------------------------------------------------------------------


class TestStallRetry:
    def test_retry_restores_the_best_checkpoint_and_can_promote(self, tmp_path, monkeypatch):
        # Attempt 0 peaks at 95% once (patience 2 needs two in a row), then
        # collapses; attempt 1 holds it.
        cfg = _cfg(_stage("a", ent_coef={"start": 0.2, "end": 0.02}))
        result, model = _run(cfg, tmp_path, {"a": [[0.4, 0.95, 0.3, 0.2], [0.95, 0.95]]}, monkeypatch)

        assert [(entry["stage"], entry["attempt"]) for entry in model.learns] == [("a", 0), ("a", 1)]
        # The retry loaded the stage's best checkpoint, re-warmed entropy and got a full budget.
        assert model.set_parameters_calls[0] == str(tmp_path / "a" / "best_model.zip")
        assert [entry["ent_coef"] for entry in model.learns] == [0.2, 0.2]
        assert [entry["budget"] for entry in model.learns] == [1_000, 1_000]
        # Fresh callbacks per attempt.
        first_eval = next(c for c in model.learns[0]["callbacks"] if isinstance(c, PeriodicEvalCallback))
        second_eval = next(c for c in model.learns[1]["callbacks"] if isinstance(c, PeriodicEvalCallback))
        assert first_eval is not second_eval
        # The retry keeps the best found so far (it only replaces it with a better eval).
        assert second_eval.best_state()["best_win_rate"] == pytest.approx(0.95)

        extra = _config_json(tmp_path, "a")["extra"]
        assert extra["promoted"] is True and extra["retries_used"] == 1
        assert [a["promoted"] for a in extra["attempts"]] == [False, True]
        assert extra["attempts"][1]["restored_from"] == str(tmp_path / "a" / "best_model.zip")
        rows = json.loads((tmp_path / "a" / "eval_results.json").read_text())
        assert [r["attempt"] for r in rows] == [0, 0, 0, 0, 1, 1]
        status = json.loads((tmp_path / "run_status.json").read_text())
        assert status["status"] == "completed_curriculum" and status["retries"] == {"a": 1}
        assert result["history"][0]["retries"] == 1
        # The stage's resume point and retry fallback are cleaned up.
        assert not (tmp_path / "a" / "latest.zip").exists() and not (tmp_path / "a" / "stage_start.zip").exists()

    def test_final_stall_says_the_stage_peaked_above_the_threshold(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a"))
        with pytest.raises(CurriculumStalled) as excinfo:
            _run(cfg, tmp_path, {"a": [[0.4, 0.95, 0.3], [0.92, 0.5]]}, monkeypatch)
        exc = excinfo.value
        assert exc.retries == 1 and exc.achieved_win_rate == pytest.approx(0.95)
        message = str(exc)
        assert "after 1 retry" in message
        assert "peaked at 95.0% (>= threshold 90.0%) but never held it for patience=2" in message
        assert "did not reach" not in message
        status = json.loads((tmp_path / "run_status.json").read_text())
        assert status["status"] == "curriculum_stalled" and status["retries_used"] == 1
        assert status["peak_win_rate"] == pytest.approx(0.95)

    def test_below_threshold_message_and_no_retries(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", max_retries=0))
        with pytest.raises(CurriculumStalled, match=r"peak win_rate 50.0% did not reach threshold 90.0%") as excinfo:
            _run(cfg, tmp_path, {"a": [[0.4, 0.5]]}, monkeypatch)
        assert excinfo.value.retries == 0

    def test_retry_without_a_best_falls_back_to_the_stage_start(self, tmp_path, monkeypatch):
        # best_eligible_after keeps every eval out of the best race.
        cfg = _cfg(_stage("a"), eval={"best_eligible_after": 10_000})
        with pytest.raises(CurriculumStalled):
            _run(cfg, tmp_path, {"a": [[0.1], [0.1]]}, monkeypatch)
        # The stall path read the model from stage_start.zip.
        assert (tmp_path / "a" / "config.json").exists()
        extra = _config_json(tmp_path, "a")["extra"]
        assert extra["attempts"][1]["restored_from"] == str(tmp_path / "a" / "stage_start.zip")
        assert extra["best_win_rate"] is None  # rltrain-6: no eligible eval -> None, not -1.0


class TestRegressionGuard:
    def test_restores_best_after_n_evals_below_best_minus_drop(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", regression_guard_evals=2, regression_guard_drop=0.3, max_retries=0, patience=3))
        # Best 0.8; then 0.45 and 0.4 (both > 0.3 below) -> restore; 0.9 x3 promotes.
        _, model = _run(cfg, tmp_path, {"a": [[0.8, 0.45, 0.4, 0.9, 0.9, 0.9]]}, monkeypatch)
        assert model.set_parameters_calls[0] == str(tmp_path / "a" / "best_model.zip")
        restores = _config_json(tmp_path, "a")["extra"]["regression_restores"]
        assert len(restores) == 1 and restores[0]["best_win_rate"] == pytest.approx(0.8)

    def test_unit_counting_resets_on_a_recovered_eval(self, tmp_path):
        stub = PeriodicEvalCallback.__new__(PeriodicEvalCallback)
        stub.results = []  # type: ignore[attr-defined]
        stub.best_win_rate = 0.9  # type: ignore[attr-defined]
        stub.best_timestep = 5  # type: ignore[attr-defined]
        best = tmp_path / "best_model.zip"
        best.write_text("x")
        guard = RegressionGuardCallback(stub, n_evals=2, drop=0.2, best_path=best, verbose=0)
        model = _ScriptedModel({}, [])
        guard.init_callback(model)  # type: ignore[arg-type]
        for wr in (0.5, 0.85, 0.5):  # the recovered 0.85 resets the count
            stub.results.append({"win_rate": wr})
            guard.on_step()
            guard.on_rollout_start()
        assert model.set_parameters_calls == []
        stub.results.append({"win_rate": 0.6})
        guard.on_step()
        assert model.set_parameters_calls == []  # applied at the next rollout, not mid-rollout
        guard.on_rollout_start()
        assert model.set_parameters_calls == [str(best)]


# ---------------------------------------------------------------------------
# rltrain-8: learning-rate schedules
# ---------------------------------------------------------------------------


def _tiny_maskable_ppo():
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.logger import configure

    from reinforcetactics.rl.masking import make_maskable_env

    env = make_maskable_env(map_file=MAP, opponent="noop", action_space_type="flat_discrete", max_flat_actions=32)
    model = MaskablePPO("MultiInputPolicy", env, policy_kwargs={"net_arch": [8]}, n_steps=16, batch_size=16, seed=0)
    model.set_logger(configure(None, []))  # learn() would set one up
    return model


class TestLRSchedule:
    def _drive(self, model, cb, timesteps):
        model.num_timesteps = timesteps
        cb.num_timesteps = timesteps
        cb._on_step()
        model._update_learning_rate(model.policy.optimizer)
        return [g["lr"] for g in model.policy.optimizer.param_groups]

    def test_optimizer_lr_follows_a_stage_relative_schedule_across_stages(self):
        model = _tiny_maskable_ppo()
        # Stage 1 starts at cumulative step 0, stage 2 at 5_000 (reset_num_timesteps=False).
        for stage_start in (0, 5_000):
            cb = LRScheduleCallback(start=1e-3, end=1e-4, total_timesteps=1_000, schedule="linear")
            cb.init_callback(model)
            model.num_timesteps = stage_start
            cb.on_training_start({}, {})
            assert self._drive(model, cb, stage_start) == [pytest.approx(1e-3)]
            assert self._drive(model, cb, stage_start + 500) == [pytest.approx(5.5e-4)]
            assert self._drive(model, cb, stage_start + 1_000) == [pytest.approx(1e-4)]
            # Past the horizon it holds the end value.
            assert self._drive(model, cb, stage_start + 4_000) == [pytest.approx(1e-4)]
            assert model.learning_rate == pytest.approx(1e-4)

    def test_cosine_constant_and_resumed_position(self):
        model = _tiny_maskable_ppo()
        cb = LRScheduleCallback(start=1e-3, end=0.0, total_timesteps=100, schedule="cosine", start_step=200)
        cb.init_callback(model)
        model.num_timesteps = 250  # resumed mid-stage: progress counts from 200
        cb.on_training_start({}, {})
        assert self._drive(model, cb, 250) == [pytest.approx(0.5e-3)]
        const = LRScheduleCallback(start=2e-4, end=0.0, total_timesteps=10, schedule="constant")
        const.init_callback(model)
        const.on_training_start({}, {})
        assert self._drive(model, const, 999) == [pytest.approx(2e-4)]
        with pytest.raises(ValueError):
            LRScheduleCallback(start=0.0, end=0.0, total_timesteps=10)

    def test_saved_checkpoint_reloads_the_current_rate(self, tmp_path):
        from sb3_contrib import MaskablePPO

        from reinforcetactics.rl.callbacks import set_learning_rate

        model = _tiny_maskable_ppo()
        set_learning_rate(model, 7e-5)
        model.save(str(tmp_path / "m.zip"))
        loaded = MaskablePPO.load(str(tmp_path / "m.zip"))
        assert loaded.lr_schedule(1.0) == pytest.approx(7e-5)

    def test_runner_applies_stage_and_ppo_schedules(self, tmp_path, monkeypatch):
        cfg = _cfg(
            _stage("a", patience=1, learning_rate={"start": 1e-3, "end": 1e-5, "schedule": "cosine"}, anneal_horizon=400),
            _stage("b", patience=1),
            _stage("c", patience=1, learning_rate=5e-5),
        )
        _, model = _run(cfg, tmp_path, {"a": [[0.95]], "b": [[0.95]], "c": [[0.95]]}, monkeypatch)
        lr_cbs = [[c for c in entry["callbacks"] if isinstance(c, LRScheduleCallback)] for entry in model.learns]
        assert [len(cbs) for cbs in lr_cbs] == [1, 0, 0]
        assert (lr_cbs[0][0].start, lr_cbs[0][0].end, lr_cbs[0][0].total_timesteps) == (1e-3, 1e-5, 400)
        # Stage b configures nothing: back to ppo.learning_rate, not stage a's schedule.
        assert [entry["learning_rate"] for entry in model.learns] == [1e-3, 3e-4, 5e-5]
        hyper = json.loads((tmp_path / "a" / "config.json").read_text())["hyperparams"]
        assert hyper["learning_rate_schedule"] == {"start": 1e-3, "end": 1e-5, "schedule": "cosine", "horizon": 400}

    def test_ppo_linear_lr_schedule_anneals_every_stage(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1, max_timesteps=500), ppo={"lr_schedule": "linear"})
        _, model = _run(cfg, tmp_path, {"a": [[0.95]], "b": [[0.95]]}, monkeypatch)
        for entry, budget in zip(model.learns, (1_000, 500), strict=True):
            (cb,) = [c for c in entry["callbacks"] if isinstance(c, LRScheduleCallback)]
            assert (cb.start, cb.end, cb.schedule, cb.total_timesteps) == (3e-4, 0.0, "linear", budget)

    def test_entropy_anneal_horizon(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1, anneal_horizon=250, ent_coef={"start": 0.1, "end": 0.0}))
        _, model = _run(cfg, tmp_path, {"a": [[0.95]]}, monkeypatch)
        (cb,) = [c for c in model.learns[0]["callbacks"] if isinstance(c, EntropyScheduleCallback)]
        assert cb.total_timesteps == 250


# ---------------------------------------------------------------------------
# rltrain-18 / rltrain-21 / rltrain-19: run hygiene
# ---------------------------------------------------------------------------


class TestRunHygiene:
    def test_episode_buffers_are_cleared_before_each_learn(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1))
        _, model = _run(cfg, tmp_path, {"a": [[0.95]], "b": [[0.95]]}, monkeypatch)
        assert [(entry["ep_info"], entry["ep_success"]) for entry in model.learns] == [(0, 0), (0, 0)]

    def test_metadata_write_failures_are_logged_and_counted(self, tmp_path, monkeypatch, caplog):
        def broken(history, path):
            raise OSError("disk full")

        monkeypatch.setattr(bootstrap, "_write_results_csv", broken)
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1))
        with caplog.at_level(logging.WARNING, logger="reinforcetactics.rl.bootstrap"):
            _run(cfg, tmp_path, {"a": [[0.95]], "b": [[0.95]]}, monkeypatch)
        status = json.loads((tmp_path / "run_status.json").read_text())
        assert status["metadata_write_failures"] == 2
        assert any("bootstrap_results.csv" in r.getMessage() and r.exc_info for r in caplog.records)

    def test_eval_rows_carry_wall_time_and_eval_seconds(self, tmp_path, monkeypatch):
        before = time.time()
        _run(_cfg(_stage("a", patience=2)), tmp_path, {"a": [[0.5, 0.95, 0.95]]}, monkeypatch)
        rows = json.loads((tmp_path / "a" / "eval_results.json").read_text())
        walls = [r["wall_time"] for r in rows]
        assert len(rows) == 3 and walls == sorted(walls) and before <= walls[0] <= time.time()
        assert all(isinstance(r["eval_seconds"], float) and r["eval_seconds"] >= 0 for r in rows)
        # bootstrap_results.csv keeps its fixed schema.
        header = (tmp_path / "bootstrap_results.csv").read_text().splitlines()[0].split(",")
        assert header == list(bootstrap._RESULTS_CSV_COLUMNS)

    def test_promoting_eval_reaches_the_logger(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1))
        _, model = _run(cfg, tmp_path, {"a": [[0.95]]}, monkeypatch)
        assert 10 in model.logger.dumps

    def test_stage_record_has_gate_eval_and_peak(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1, promotion_criterion="rolling", promotion_rolling_k=2))
        _run(cfg, tmp_path, {"a": [[0.95, 0.95]]}, monkeypatch)
        extra = _config_json(tmp_path, "a")["extra"]
        assert extra["promotion"]["criterion"] == "rolling" and extra["promotion"]["rolling_k"] == 2
        assert extra["eval_deterministic"] is False and extra["eval_seats"] == [1]
        assert extra["peak_win_rate"] == pytest.approx(0.95)
        assert extra["last_eval"]["win_rate"] == pytest.approx(0.95)
        assert "draws" in extra["last_eval"]


# ---------------------------------------------------------------------------
# rltrain-5: resume
# ---------------------------------------------------------------------------


class TestResume:
    def test_rolling_checkpoint_callback_cadence(self, tmp_path):
        saves: list[int] = []
        cb = RollingCheckpointCallback(tmp_path / "latest.zip", 25, on_save=saves.append)
        model = _ScriptedModel({}, [])
        cb.init_callback(model)  # type: ignore[arg-type]
        model.num_timesteps = 100
        cb.on_training_start({}, {})
        for ts in (110, 120, 125, 130, 150, 151):
            model.num_timesteps = ts
            cb.on_step()
        assert saves == [125, 150] and (tmp_path / "latest.zip").exists()

    def test_interrupted_run_resumes_the_interrupted_stage(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=3, promotion_criterion="rolling", promotion_rolling_k=2))
        programs = {"a": [[0.95]], "b": [[0.95, 0.95, 0.95, 0.95, 0.95]]}
        # Killed at b's third eval (two evals in; the rolling window is full).
        with pytest.raises(KeyboardInterrupt):
            _run(cfg, tmp_path, programs, monkeypatch, interrupt_at=("b", 0, 2))
        assert not (tmp_path / "run_status.json").exists()  # absent = aborted, as before
        manifest = json.loads((tmp_path / "run_manifest.json").read_text())
        current = manifest["current"]
        assert current["stage"] == "b" and current["latest_timesteps"] == 30
        assert current["promotion_state"]["streak"] == 1 and len(current["promotion_state"]["window"]) == 2
        killed_rows = [json.loads(line) for line in (tmp_path / "b" / "eval_results.jsonl").read_text().splitlines()]
        wall_before = {r["timesteps"]: r["wall_time"] for r in killed_rows}

        loaded: dict[str, Any] = {}

        def loader(path, vec_env, cfg_arg, out):
            model = _ScriptedModel(programs, [s.name for s in cfg_arg.curriculum.stages])
            model.stage_offset = 1  # the first set_env-free learn is stage b
            model.num_timesteps = json.loads(Path(path).read_text())["num_timesteps"]
            loaded["path"], loaded["model"] = path, model
            return model

        def must_not_build(*args):
            raise AssertionError("resume must load the checkpoint, not build a model")

        result = run_curriculum(
            cfg,
            tmp_path,
            train_env_factory=lambda stage, c: _Env(),
            eval_env_factory=lambda stage, c: _Env(),
            model_factory=must_not_build,
            model_loader=loader,
            resume=True,
        )
        model = loaded["model"]
        assert loaded["path"] == tmp_path / "b" / "latest.zip"
        # Stage a is not re-run; b continues from 30 with the rest of its budget.
        assert [(e["stage"], e["start"], e["budget"]) for e in model.learns] == [("b", 30, 1_000 - 20)]
        eval_cb = next(c for c in model.learns[0]["callbacks"] if isinstance(c, PeriodicEvalCallback))
        promote_cb = next(c for c in model.learns[0]["callbacks"] if isinstance(c, PromotionCallback))
        # The pre-kill evals are back in the timeline and the streak continues:
        # patience 3 is reached after two more passing evals, not three.
        assert [r["timesteps"] for r in result["history"][1]["results"]] == [20, 30, 40, 50]
        assert promote_cb.promoted
        assert eval_cb._stage_start_step == 10  # min-timesteps / best-eligible stay stage-relative
        assert [h["stage"] for h in result["history"]] == ["a", "b"]
        assert result["history"][0].get("from_previous_session") is True
        status = json.loads((tmp_path / "run_status.json").read_text())
        assert status["status"] == "completed_curriculum" and status["resume_count"] == 1
        assert _config_json(tmp_path, "b")["extra"]["resumed"] is True
        # The rewritten JSONL holds each eval once.
        rows = [json.loads(line) for line in (tmp_path / "b" / "eval_results.jsonl").read_text().splitlines()]
        assert [r["timesteps"] for r in rows] == [20, 30, 40, 50]
        # Rows kept from the killed session keep their own wall_time; the
        # resumed session's rows are later (summarize_seeds' steps/h skips the gap).
        assert {r["timesteps"]: r["wall_time"] for r in rows[:2]} == wall_before
        assert rows[2]["wall_time"] >= rows[1]["wall_time"] and all(r["eval_seconds"] >= 0 for r in rows)

    def test_plan_between_stages_and_completed_and_stalled(self, tmp_path, monkeypatch):
        cfg = _cfg(_stage("a", patience=1), _stage("b", patience=1))
        _run(cfg, tmp_path, {"a": [[0.95]], "b": [[0.95]]}, monkeypatch)
        # Completed: nothing to do, no model built.
        plan = _plan_resume(cfg, tmp_path)
        assert plan.completed and [h["stage"] for h in plan.history] == ["a", "b"]
        result = run_curriculum(
            cfg,
            tmp_path,
            train_env_factory=lambda s, c: _Env(),
            eval_env_factory=lambda s, c: _Env(),
            model_factory=lambda *a: pytest.fail("no training on a completed run"),
            resume=True,
        )
        assert result["resumed"] is True and len(result["history"]) == 2

        # Killed between stages: b has no config.json, and the manifest names no
        # stage in progress and has not recorded b as finished.
        (tmp_path / "b" / "config.json").unlink()
        (tmp_path / "run_status.json").unlink()
        manifest = json.loads((tmp_path / "run_manifest.json").read_text())
        manifest["current"] = None
        manifest["completed"] = [e for e in manifest["completed"] if e["stage"] != "b"]
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
        plan = _plan_resume(cfg, tmp_path)
        assert plan.start_index == 1 and plan.model_path == tmp_path / "a" / "stage_final.zip"
        assert plan.restore_best_path == tmp_path / "a" / "best_model.zip"

        # A stalled run is refused.
        (tmp_path / "run_status.json").write_text(json.dumps({"status": "curriculum_stalled", "stalled_stage": "b"}))
        with pytest.raises(ResumeError, match="stalled"):
            _plan_resume(cfg, tmp_path)

    def test_resume_config_check(self, tmp_path):
        cfg = _cfg(_stage("a"))
        resolved = bootstrap.resolve_config(cfg)
        from reinforcetactics.rl.config import save_config

        save_config(resolved, tmp_path / "resolved_config.yaml")
        assert bootstrap.resume_config_differences(cfg, tmp_path) == []
        other = _cfg(_stage("a", max_timesteps=2_000), ppo={"device": "cuda"})
        assert bootstrap.resume_config_differences(other, tmp_path) == ["curriculum.stages[0].max_timesteps"]
        with pytest.raises(ResumeError, match="resolved_config"):
            bootstrap.resume_config_differences(cfg, tmp_path / "nope")


@pytest.fixture
def train_bootstrap(monkeypatch):
    import importlib.util

    monkeypatch.setattr(sys, "path", [*sys.path])
    spec = importlib.util.spec_from_file_location(
        "train_bootstrap_resume", REPO_ROOT / "scripts" / "train" / "train_bootstrap.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestResumeCli:
    def _write(self, path: Path, data: dict[str, Any]) -> Path:
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def test_refuses_a_different_config_unless_forced(self, train_bootstrap, monkeypatch, tmp_path):
        calls: list[dict[str, Any]] = []

        def fake_run(cfg, output_dir, **kwargs):
            calls.append({"cfg": cfg, "output_dir": output_dir, **kwargs})
            return {"history": [], "final_model_path": None}

        monkeypatch.setattr(bootstrap, "run_curriculum", fake_run)
        base = {
            "env": {"n_envs": 1, "use_subprocess": False},
            "curriculum": {"stages": [{"name": "s", "map_file": MAP, "opponent": "noop", "max_timesteps": 100}]},
        }
        config = self._write(tmp_path / "c.yaml", base)
        out = tmp_path / "run"
        common = ["--device", "cpu", "--no-gcs", "--skip-plots", "--skip-videos", "--sanity-episodes", "0"]
        assert train_bootstrap.main(["--config", str(config), "--output-dir", str(out), *common]) == 0
        assert "resume" not in calls[-1]

        # Its own config (no --config): resumes.
        assert train_bootstrap.main(["--resume", str(out), *common]) == 0
        assert calls[-1]["resume"] is True and Path(calls[-1]["output_dir"]) == out

        changed = self._write(tmp_path / "d.yaml", {**base, "seed": 7})
        with pytest.raises(SystemExit, match="seed"):
            train_bootstrap.main(["--resume", str(out), "--config", str(changed), *common])
        assert train_bootstrap.main(["--resume", str(out), "--config", str(changed), "--force", *common]) == 0
        assert calls[-1]["cfg"].seed == 7
        assert list(out.glob("resolved_config.before_resume_*.yaml"))

        with pytest.raises(SystemExit, match="resolved_config"):
            train_bootstrap.main(["--resume", str(tmp_path / "missing"), *common])
        with pytest.raises(SystemExit, match="--force only applies"):
            train_bootstrap.main(["--config", str(config), "--force", *common])


# ---------------------------------------------------------------------------
# A real tiny curriculum: LR across stages (fast) and SIGTERM + --resume (slow)
# ---------------------------------------------------------------------------


def _tiny_config(tmp_path: Path, stage2_budget: int, **extra: Any) -> Path:
    data: dict[str, Any] = {
        "seed": 0,
        "env": {
            "n_envs": 1,
            "use_subprocess": False,
            "action_space_type": "flat_discrete",
            "max_flat_actions": 64,
            "max_steps": 40,
            "max_turns": 5,
            "enabled_units": ["W"],
        },
        "ppo": {"n_steps": 64, "batch_size": 32, "n_epochs": 1, "policy_kwargs": {"net_arch": [16]}},
        "eval": {"eval_freq": 64, "n_eval_episodes": 1, "checkpoint_freq": 64, "eval_both_modes": False},
        "curriculum": {
            "stages": [
                {
                    "name": "s1",
                    "map_file": MAP,
                    "opponent": "noop",
                    "promotion_win_rate": 0.0,
                    "patience": 1,
                    "min_timesteps_before_promotion": 128,
                    "max_timesteps": 320,
                    "learning_rate": {"start": 1e-3, "end": 1e-4, "schedule": "linear"},
                },
                {
                    "name": "s2",
                    "map_file": MAP,
                    "opponent": "noop",
                    "promotion_win_rate": 0.0,
                    "patience": 1,
                    "min_timesteps_before_promotion": stage2_budget - 128,
                    "max_timesteps": stage2_budget,
                },
            ]
        },
    }
    for key, value in extra.items():
        data[key] = {**data.get(key, {}), **value}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_real_curriculum_learning_rate_across_stages(tmp_path):
    from reinforcetactics.rl.config import load_config

    cfg = load_config(_tiny_config(tmp_path, 256))
    cfg.ppo.device = "cpu"
    result = run_curriculum(cfg, tmp_path / "run")
    rows = [r for r in _read_csv(tmp_path / "run" / "train_metrics.csv") if r["train/learning_rate"]]
    by_stage: dict[str, list[float]] = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(float(r["train/learning_rate"]))
    # s1 anneals linearly from 1e-3 over its 320 steps (64 in: 1e-3 - 0.2 * 9e-4);
    # s2 configures none and trains at ppo.learning_rate.
    assert by_stage["s1"][0] == pytest.approx(1e-3 - 0.2 * 9e-4)
    assert by_stage["s1"] == sorted(by_stage["s1"], reverse=True)
    assert by_stage["s2"] and all(lr == pytest.approx(3e-4) for lr in by_stage["s2"])
    model = result["model"]
    assert [g["lr"] for g in model.policy.optimizer.param_groups] == [pytest.approx(3e-4)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_sigterm_mid_stage_then_resume_completes(tmp_path):
    config = _tiny_config(tmp_path, 3_200)
    out = tmp_path / "run"
    env = {k: v for k, v in os.environ.items() if k not in ("GCS_OUTPUT_URI", "AIP_MODEL_DIR", "GCS_WRAPPER_SYNC")}
    env.update(PYTHONPATH=str(REPO_ROOT), SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", MPLBACKEND="Agg")
    common = ["--device", "cpu", "--no-gcs", "--skip-plots", "--skip-videos", "--sanity-episodes", "0"]
    script = str(REPO_ROOT / "scripts" / "train" / "train_bootstrap.py")
    proc = subprocess.Popen(
        [sys.executable, script, "--config", str(config), "--output-dir", str(out), "--strict", *common],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    jsonl = out / "s2" / "eval_results.jsonl"
    try:
        deadline = time.monotonic() + 300
        while not (jsonl.exists() and len(jsonl.read_text().splitlines()) >= 3):
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                pytest.fail(f"never reached stage 2:\n{proc.communicate()[0][-4000:]}")
            time.sleep(0.05)
        proc.send_signal(signal.SIGTERM)
        first_output = proc.communicate(timeout=120)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 143, first_output[-4000:]
    assert not (out / "run_status.json").exists()
    killed_at = json.loads((out / "run_manifest.json").read_text())["current"]["latest_timesteps"]
    assert killed_at > 128  # inside stage 2
    s1_record = (out / "s1" / "config.json").read_text()

    resumed = subprocess.run(
        [sys.executable, script, "--resume", str(out), *common],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout[-4000:] + resumed.stderr[-4000:]
    # Stage 1 was skipped (its record untouched, no new banner); stage 2 continued.
    assert (out / "s1" / "config.json").read_text() == s1_record
    assert "=== Stage 's1'" not in resumed.stdout
    assert f"resuming 's2' attempt 1 at {killed_at:,}" in resumed.stdout
    status = json.loads((out / "run_status.json").read_text())
    assert status["status"] == "completed_curriculum" and status["resume_count"] == 1
    timeline = [json.loads(line)["timesteps"] for line in (out / "s2" / "eval_results.jsonl").read_text().splitlines()]
    # num_timesteps continued: one eval per 64 steps, no restart from 0, no duplicates.
    assert timeline == sorted(set(timeline)) and timeline[0] <= killed_at < timeline[-1]
    assert all(b - a <= 64 for a, b in zip(timeline, timeline[1:]))
    s2 = json.loads((out / "s2" / "config.json").read_text())["extra"]
    assert s2["promoted"] is True and s2["resumed"] is True
    assert s2["stage_end_timesteps"] >= 3_200 - 128 + 128  # stage 2 trained its window after the resume
    assert not math.isnan(float(s2["stage_start_timesteps"]))
