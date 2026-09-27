"""The new gate / recovery / seat config fields: defaults, coercion, ranges, reporting, records.

Every field added for the eval-gate package goes through ``validate()``
(type coercion and range checks), is declared read by the entry points that
honour it (so ``--strict`` accepts it), and lands in ``resolved_config.yaml``
and each stage's ``config.json``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from reinforcetactics.rl import bootstrap
from reinforcetactics.rl.config import (
    CurriculumConfig,
    CurriculumStage,
    EnvConfig,
    EvalConfig,
    PPOConfig,
    TrainingConfig,
    config_from_dict,
    ignored_config_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP = "maps/1v1/beginner.csv"


def _stage(**kwargs: Any) -> dict[str, Any]:
    return {"name": "s", "map_file": MAP, "opponent": "random", "max_timesteps": 1_000, **kwargs}


def _cfg(stage: dict[str, Any] | None = None, **sections: Any) -> TrainingConfig:
    return config_from_dict({"curriculum": {"stages": [stage or _stage()], **sections.pop("curriculum", {})}, **sections})


class TestDefaultsKeepTodaysBehaviour:
    def test_defaults(self):
        env, ev, cur, stage = EnvConfig(), EvalConfig(), CurriculumConfig(), CurriculumStage()
        assert env.agent_seat == 1
        # The one deliberate change: the gate measures the stochastic policy.
        assert ev.eval_deterministic is False and ev.eval_both_modes is True
        assert (ev.eval_seats, ev.seat_aggregate, ev.n_eval_envs, ev.eval_use_subprocess) == (None, "mean", 1, False)
        assert (cur.promotion_criterion, cur.promotion_rolling_k, cur.promotion_confidence, cur.promotion_score) == (
            "point",
            3,
            0.95,
            "win_rate",
        )
        assert (cur.max_retries, cur.regression_guard_evals, cur.regression_guard_drop) == (1, 0, 0.2)
        for name in (
            "learning_rate",
            "anneal_horizon",
            "promotion_criterion",
            "promotion_rolling_k",
            "promotion_confidence",
            "promotion_score",
            "max_retries",
            "regression_guard_evals",
            "regression_guard_drop",
        ):
            assert getattr(stage, name) is None, name
        assert PPOConfig().lr_schedule == "constant"

    def test_default_stage_resolves_to_the_historical_gate_and_no_lr_schedule(self):
        cfg = _cfg()
        stage = cfg.curriculum.stages[0]
        assert stage.resolve_promotion(cfg.curriculum) == {
            "criterion": "point",
            "rolling_k": 3,
            "confidence": 0.95,
            "score": "win_rate",
            "threshold": 0.9,
            "patience": 2,
        }
        assert stage.resolve_learning_rate_schedule(cfg.ppo) is None
        assert stage.resolve_regression_guard(cfg.curriculum) == {"evals": 0, "drop": 0.2}
        assert stage.resolve_horizon() == 1_000
        assert cfg.eval.resolve_eval_seats(cfg.env) == [1]


class TestCoercion:
    def test_yaml_strings_are_coerced(self):
        cfg = _cfg(
            _stage(
                learning_rate={"start": "1e-3", "end": "0", "schedule": "cosine", "horizon": "1e3"},
                promotion_confidence="0.9",
                promotion_rolling_k="5",
                max_retries="2",
                anneal_horizon="500",
            ),
            env={"agent_seat": "2"},
            eval={"eval_seats": ["1", "2"], "eval_deterministic": "true", "n_eval_envs": "4.0"},
            curriculum={"regression_guard_drop": "0.25"},
        )
        stage = cfg.curriculum.stages[0]
        assert cfg.env.agent_seat == 2
        assert cfg.eval.eval_seats == [1, 2] and cfg.eval.eval_deterministic is True and cfg.eval.n_eval_envs == 4
        assert stage.learning_rate == {"start": 1e-3, "end": 0.0, "schedule": "cosine", "horizon": 1000}
        assert (stage.promotion_confidence, stage.promotion_rolling_k, stage.max_retries) == (0.9, 5, 2)
        assert stage.anneal_horizon == 500 and cfg.curriculum.regression_guard_drop == 0.25
        assert stage.resolve_learning_rate_schedule(cfg.ppo) == {
            "start": 1e-3,
            "end": 0.0,
            "schedule": "cosine",
            "horizon": 1000,
        }

    def test_random_seat(self):
        cfg = _cfg(env={"agent_seat": "random"})
        assert cfg.env.agent_seat == "random"
        assert cfg.eval.resolve_eval_seats(cfg.env) == [1, 2]


class TestRanges:
    @pytest.mark.parametrize(
        ("sections", "match"),
        [
            ({"env": {"agent_seat": 3}}, "env.agent_seat must"),
            ({"env": {"agent_seat": "left"}}, "env.agent_seat must"),
            ({"eval": {"eval_seats": [3]}}, "eval.eval_seats must"),
            ({"eval": {"eval_seats": [1, 1]}}, "eval.eval_seats must"),
            ({"eval": {"eval_seats": []}}, "eval.eval_seats must"),
            ({"eval": {"seat_aggregate": "max"}}, "eval.seat_aggregate must"),
            ({"eval": {"n_eval_envs": 0}}, "eval.n_eval_envs must"),
            ({"curriculum": {"promotion_criterion": "bayes"}}, "promotion_criterion must"),
            ({"curriculum": {"promotion_score": "elo"}}, "promotion_score must"),
            ({"curriculum": {"promotion_rolling_k": 0}}, "promotion_rolling_k must"),
            ({"curriculum": {"promotion_confidence": 1.0}}, "promotion_confidence must"),
            ({"curriculum": {"promotion_confidence": 0.4}}, "promotion_confidence must"),
            ({"curriculum": {"max_retries": -1}}, "max_retries must"),
            ({"curriculum": {"regression_guard_evals": -1}}, "regression_guard_evals must"),
            ({"curriculum": {"regression_guard_drop": 0.0}}, "regression_guard_drop must"),
            ({"curriculum": {"regression_guard_drop": 1.5}}, "regression_guard_drop must"),
        ],
    )
    def test_run_level_fields(self, sections, match):
        with pytest.raises(ValueError, match=match):
            _cfg(**sections)

    @pytest.mark.parametrize(
        ("stage", "match"),
        [
            ({"learning_rate": 0.0}, "learning_rate override must be > 0"),
            ({"learning_rate": {"start": 0.0, "end": 0.0}}, "learning_rate.start must be > 0"),
            ({"learning_rate": {"start": 1e-3, "end": -1.0}}, r"learning_rate.end must be a number >= 0"),
            ({"learning_rate": {"start": 1e-3, "end": 0.0, "schedule": "exp"}}, "learning_rate.schedule must be"),
            ({"learning_rate": {"start": 1e-3, "end": 0.0, "warmup": 3}}, "unknown keys"),
            ({"learning_rate": {"start": 1e-3, "end": 0.0, "horizon": 0}}, "horizon must be a positive"),
            ({"ent_coef": {"start": 0.1, "end": 0.0, "horizon": -5}}, "horizon must be a positive"),
            ({"anneal_horizon": 0}, "anneal_horizon must"),
            ({"promotion_criterion": "vote"}, "promotion_criterion must"),
            ({"promotion_score": "draws"}, "promotion_score must"),
            ({"promotion_rolling_k": 0}, "promotion_rolling_k must"),
            ({"promotion_confidence": 1.2}, "promotion_confidence must"),
            ({"max_retries": -2}, "max_retries must"),
            ({"regression_guard_evals": -1}, "regression_guard_evals must"),
            ({"regression_guard_drop": 2.0}, "regression_guard_drop must"),
        ],
    )
    def test_stage_fields(self, stage, match):
        with pytest.raises(ValueError, match=match):
            _cfg(_stage(**stage))

    def test_stage_overrides_win(self):
        cfg = _cfg(
            _stage(
                promotion_criterion="wilson", promotion_score="win_plus_half_draw", max_retries=0, regression_guard_evals=3
            ),
            curriculum={"promotion_criterion": "rolling", "max_retries": 2, "regression_guard_drop": 0.3},
        )
        stage = cfg.curriculum.stages[0]
        promo = stage.resolve_promotion(cfg.curriculum)
        assert (promo["criterion"], promo["score"]) == ("wilson", "win_plus_half_draw")
        assert stage.resolve_max_retries(cfg.curriculum) == 0
        assert stage.resolve_regression_guard(cfg.curriculum) == {"evals": 3, "drop": 0.3}


FULL = {
    "env": {"agent_seat": "random"},
    "ppo": {"lr_schedule": "linear"},
    "eval": {
        "checkpoint_freq": 1234,
        "eval_deterministic": True,
        "eval_both_modes": False,
        "eval_seats": [1, 2],
        "seat_aggregate": "min",
        "n_eval_envs": 2,
        "eval_use_subprocess": True,
    },
    "curriculum": {
        "promotion_criterion": "wilson",
        "promotion_rolling_k": 4,
        "promotion_confidence": 0.9,
        "promotion_score": "win_plus_half_draw",
        "max_retries": 2,
        "regression_guard_evals": 2,
        "regression_guard_drop": 0.3,
        "stages": [
            _stage(
                learning_rate={"start": 1e-3, "end": 1e-5, "schedule": "cosine"},
                anneal_horizon=500,
                promotion_criterion="rolling",
                max_retries=0,
            )
        ],
    },
}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_gate_config", REPO_ROOT / "scripts" / "train" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReportingAndRecords:
    def test_the_curriculum_reads_every_new_field(self):
        cfg = config_from_dict(FULL)
        assert ignored_config_fields(cfg, bootstrap.CONSUMED_CONFIG_FIELDS, algorithms=bootstrap.CONSUMED_ALGORITHMS) == []

    def test_other_entry_points_read_the_seat(self, monkeypatch):
        monkeypatch.setattr(sys, "path", [*sys.path])
        self_play = _script("train_self_play")
        assert self_play._ARG_TO_CONFIG_PATH["agent_seat"] == "env.agent_seat"
        assert self_play.parse_args(["--agent-seat", "random"]).agent_seat == "random"
        assert self_play.parse_args(["--agent-seat", "2"]).agent_seat == 2
        feudal = _script("train_feudal_rl")
        assert "env.agent_seat" in feudal.consumed_config_fields("flat")
        assert "env.agent_seat" in feudal.consumed_config_fields("feudal")

    def test_resolved_config_and_stage_record(self, monkeypatch, tmp_path):
        import json

        monkeypatch.setattr(sys, "path", [*sys.path])
        train_bootstrap = _script("train_bootstrap")
        seen: list[TrainingConfig] = []
        monkeypatch.setattr(
            bootstrap, "run_curriculum", lambda cfg, output_dir: seen.append(cfg) or {"history": [], "final_model_path": None}
        )
        data = {**FULL, "env": {**FULL["env"], "n_envs": 1, "use_subprocess": False}}
        config = tmp_path / "c.yaml"
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        argv = ["--config", str(config), "--output-dir", str(tmp_path / "out"), "--device", "cpu", "--no-gcs"]
        argv += ["--skip-plots", "--skip-videos", "--sanity-episodes", "0", "--strict"]
        assert train_bootstrap.main(argv) == 0
        record = yaml.safe_load((tmp_path / "out" / "resolved_config.yaml").read_text())
        for section in ("eval", "curriculum"):
            for key, value in FULL[section].items():
                if key != "stages":
                    assert record[section][key] == value, (section, key)
        assert record["env"]["agent_seat"] == "random"
        assert record["curriculum"]["stages"][0]["learning_rate"]["schedule"] == "cosine"

        cfg = bootstrap.resolve_config(config_from_dict(FULL))
        stage = cfg.curriculum.stages[0]
        bootstrap._write_stage_config(
            stage=stage, cfg=cfg, stage_dir=tmp_path, output_dir=tmp_path, promoted=True, best_win_rate=None
        )
        stage_record = json.loads((tmp_path / "config.json").read_text())
        extra, hyper = stage_record["extra"], stage_record["hyperparams"]
        assert extra["promotion"]["criterion"] == "rolling" and extra["promotion"]["score"] == "win_plus_half_draw"
        assert (extra["eval_deterministic"], extra["eval_both_modes"], extra["eval_seats"]) == (True, False, [1, 2])
        assert (extra["seat_aggregate"], extra["n_eval_envs"], extra["checkpoint_freq"]) == ("min", 2, 1234)
        assert (extra["max_retries"], extra["regression_guard"], extra["anneal_horizon"]) == (
            0,
            {"evals": 2, "drop": 0.3},
            500,
        )
        assert hyper["learning_rate_schedule"] == {"start": 1e-3, "end": 1e-5, "schedule": "cosine", "horizon": 500}
        assert stage_record["env_config"]["agent_seat"] == "random"
