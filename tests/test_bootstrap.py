"""Tests for reinforcetactics.rl.bootstrap and the PromotionCallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml  # type: ignore[import-untyped]

from reinforcetactics.rl.bootstrap import (
    BootstrapConfig,
    BootstrapEnvDefaults,
    CurriculumStage,
    CurriculumStalled,
    config_from_dict,
    load_bootstrap_config,
    run_curriculum,
)
from reinforcetactics.rl.callbacks import PeriodicEvalCallback, PromotionCallback
from reinforcetactics.rl.config import PPOConfig

# ---------------------------------------------------------------------------
# PromotionCallback unit tests (no env / no real model needed)
# ---------------------------------------------------------------------------


class _StubLogger:
    """Minimal stand-in for sb3 Logger; only tracks .record calls."""

    def __init__(self) -> None:
        self.records: Dict[str, Any] = {}
        # TrainingMetricsCallback reads ``self.model.logger.name_to_value``.
        self.name_to_value: Dict[str, Any] = {}

    def record(self, key: str, value: Any) -> None:
        self.records[key] = value


def _make_eval_stub() -> PeriodicEvalCallback:
    """Build a PeriodicEvalCallback shell whose ``results`` we can append to.

    We bypass __init__ so we don't need a real env / save_dir; the
    PromotionCallback only ever reads ``eval_callback.results``.
    """
    stub = PeriodicEvalCallback.__new__(PeriodicEvalCallback)
    stub.results = []  # type: ignore[attr-defined]
    return stub


def _step_promote(cb: PromotionCallback, num_timesteps: int = 0) -> bool:
    """Drive the callback's ``_on_step`` without standing up a full SB3 model."""
    cb.num_timesteps = num_timesteps  # type: ignore[attr-defined]
    return cb._on_step()


def _append_eval(eval_cb: PeriodicEvalCallback, win_rate: float) -> None:
    eval_cb.results.append({"win_rate": win_rate, "avg_reward": 0.0})


class TestPromotionCallback:
    def test_validates_threshold_and_patience(self):
        eval_cb = _make_eval_stub()
        with pytest.raises(ValueError):
            PromotionCallback(eval_cb, threshold=1.5, patience=1)
        with pytest.raises(ValueError):
            PromotionCallback(eval_cb, threshold=0.9, patience=0)

    def test_does_not_promote_before_any_eval(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)
        for _ in range(5):
            assert _step_promote(cb) is True
        assert cb.promoted is False

    def test_single_passing_eval_is_not_enough(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)

        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb) is True
        assert cb.promoted is False
        assert cb._streak == 1

    def test_promotes_after_patience_consecutive_passes(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=3, verbose=0)

        _append_eval(eval_cb, 0.92)
        assert _step_promote(cb) is True
        _append_eval(eval_cb, 0.91)
        assert _step_promote(cb) is True
        _append_eval(eval_cb, 0.95)
        assert _step_promote(cb) is False
        assert cb.promoted is True

    def test_failing_eval_resets_streak(self):
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)

        _append_eval(eval_cb, 0.95)
        _step_promote(cb)
        assert cb._streak == 1

        _append_eval(eval_cb, 0.50)  # regression
        _step_promote(cb)
        assert cb._streak == 0
        assert cb.promoted is False

        _append_eval(eval_cb, 0.95)
        _step_promote(cb)
        _append_eval(eval_cb, 0.95)
        result = _step_promote(cb)
        assert result is False
        assert cb.promoted is True

    def test_consumes_multiple_new_results_in_one_step(self):
        # Defensive: if two evals somehow land between callback ticks,
        # the streak accounting must still be correct.
        eval_cb = _make_eval_stub()
        cb = PromotionCallback(eval_cb, threshold=0.9, patience=2, verbose=0)
        _append_eval(eval_cb, 0.95)
        _append_eval(eval_cb, 0.95)
        result = _step_promote(cb)
        assert result is False
        assert cb.promoted is True


# ---------------------------------------------------------------------------
# BootstrapConfig loading and validation
# ---------------------------------------------------------------------------


VALID_DICT: Dict[str, Any] = {
    "seed": 7,
    "n_envs": 2,
    "eval_freq": 1000,
    "env": {
        "max_steps": 100,
        "max_turns": 20,
        "enabled_units": ["W"],
        "action_space_type": "flat_discrete",
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 256,
    },
    "stages": [
        {
            "name": "stage_a",
            "map_file": "maps/1v1/starter.csv",
            "opponent": "random",
            "promotion_win_rate": 0.9,
            "patience": 2,
            "max_timesteps": 5000,
            "n_eval_episodes": 5,
        },
        {
            "name": "stage_b",
            "map_file": "maps/1v1/starter.csv",
            "opponent": "simple",
            "promotion_win_rate": 0.8,
            "patience": 1,
            "max_timesteps": 5000,
            "n_eval_episodes": 5,
        },
    ],
}


class TestBootstrapConfig:
    def test_round_trips_through_dict(self):
        cfg = config_from_dict(VALID_DICT)
        assert cfg.seed == 7
        assert cfg.n_envs == 2
        assert cfg.eval_freq == 1000
        assert len(cfg.stages) == 2
        assert cfg.stages[0].name == "stage_a"
        assert cfg.stages[1].opponent == "simple"
        assert isinstance(cfg.ppo, PPOConfig)
        assert isinstance(cfg.env, BootstrapEnvDefaults)

    def test_round_trips_through_yaml_file(self, tmp_path):
        path = tmp_path / "bootstrap.yaml"
        path.write_text(yaml.safe_dump(VALID_DICT), encoding="utf-8")
        cfg = load_bootstrap_config(path)
        assert len(cfg.stages) == 2
        assert cfg.stages[0].promotion_win_rate == pytest.approx(0.9)

    def test_rejects_unknown_top_level_key(self):
        bad = dict(VALID_DICT)
        bad["unknown"] = 1
        with pytest.raises(ValueError, match="Unknown top-level keys"):
            config_from_dict(bad)

    def test_rejects_unknown_stage_field(self):
        bad = {**VALID_DICT, "stages": [{**VALID_DICT["stages"][0], "bogus": 1}]}
        with pytest.raises(ValueError, match="Unknown keys for CurriculumStage"):
            config_from_dict(bad)

    def test_rejects_empty_stages(self):
        bad = {**VALID_DICT, "stages": []}
        with pytest.raises(ValueError, match="stages must be non-empty"):
            config_from_dict(bad)

    def test_rejects_duplicate_stage_names(self):
        bad = {
            **VALID_DICT,
            "stages": [VALID_DICT["stages"][0], VALID_DICT["stages"][0]],
        }
        with pytest.raises(ValueError, match="duplicate stage name"):
            config_from_dict(bad)

    def test_rejects_unknown_opponent(self):
        bad = {
            **VALID_DICT,
            "stages": [{**VALID_DICT["stages"][0], "opponent": "godlike"}],
        }
        with pytest.raises(ValueError, match="unknown opponent"):
            config_from_dict(bad)

    def test_shipped_config_loads(self):
        # The repo's configs/bootstrap.yaml should always be valid.
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bootstrap_config(repo_root / "configs" / "bootstrap.yaml")
        names = [s.name for s in cfg.stages]
        # Ensure the expected curriculum is intact (catches accidental edits
        # that drop stages or reorder them).
        assert names == [
            "starter_random",
            "starter_simple",
            "starter_medium",
            "beginner_random",
            "beginner_simple",
            "beginner_medium",
        ]
        # Regression: PyYAML 1.1 parses ``3e-4`` (no decimal) as a string,
        # which then fails deep inside SB3's lr-schedule check. Every
        # numeric PPO field must round-trip as a number.
        assert isinstance(cfg.ppo.learning_rate, (int, float))
        assert isinstance(cfg.ppo.ent_coef, (int, float))
        assert isinstance(cfg.ppo.clip_range, (int, float))


# ---------------------------------------------------------------------------
# run_curriculum integration with fakes (no SB3, no real env)
# ---------------------------------------------------------------------------


@dataclass
class _FakeEnv:
    """Just enough of a Gym env for the callbacks not to complain."""

    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeModel:
    """In-memory stand-in for MaskablePPO. Records the order of learn()
    calls and feeds preprogrammed eval win_rates into the callback list
    so PromotionCallback drives the loop the same way it would in real
    training."""

    win_rate_program: Dict[str, List[float]] = field(default_factory=dict)
    learn_calls: List[Dict[str, Any]] = field(default_factory=list)
    save_calls: List[str] = field(default_factory=list)
    set_env_calls: List[Any] = field(default_factory=list)
    num_timesteps: int = 0
    # SB3's BaseCallback exposes ``self.logger`` as a read-only property
    # that delegates to ``self.model.logger``; we satisfy that by exposing
    # the stub here and (below) setting ``cb.model = self``.
    logger: _StubLogger = field(default_factory=_StubLogger)
    ep_info_buffer: list = field(default_factory=list)
    _current_stage_idx: int = 0
    _stage_names: List[str] = field(default_factory=list)

    def set_env(self, env: Any) -> None:
        self.set_env_calls.append(env)
        self._current_stage_idx += 1

    def learn(
        self,
        total_timesteps: int,
        callback: List[Any],
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> None:
        stage_name = self._stage_names[self._current_stage_idx]
        program = list(self.win_rate_program.get(stage_name, []))

        # SB3 wires up callbacks before _on_step; mimic the parts we use.
        # Don't set ``cb.logger`` directly: BaseCallback.logger is a
        # read-only property that delegates to ``self.model.logger``.
        for cb in callback:
            cb.model = self

        # Each programmed entry simulates one eval block. We append directly
        # to PeriodicEvalCallback.results (bypassing the real evaluate_model
        # call, which would need a working env) and tick PromotionCallback
        # so it observes the new result and decides whether to short-circuit.
        # We skip PeriodicEvalCallback._on_step deliberately: its real path
        # would re-run evaluate_model against the stub env. The unit tests
        # for PromotionCallback already cover the streak logic; here we
        # verify the runner's loop / promotion / stall behaviour.
        eval_cb = next(c for c in callback if isinstance(c, PeriodicEvalCallback))
        promote_cb = next(c for c in callback if isinstance(c, PromotionCallback))
        for win_rate in program:
            eval_cb.results.append({"win_rate": win_rate, "avg_reward": 0.0})
            self.num_timesteps += 1
            if not promote_cb._on_step():
                self.learn_calls.append(
                    {
                        "stage": stage_name,
                        "total_timesteps": total_timesteps,
                        "exited_early": True,
                    }
                )
                return
        # No early exit -> stage exhausted its budget without promoting.
        self.learn_calls.append({"stage": stage_name, "total_timesteps": total_timesteps, "exited_early": False})

    def save(self, path: str) -> None:
        self.save_calls.append(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake-checkpoint", encoding="utf-8")


def _make_cfg(stages: List[CurriculumStage]) -> BootstrapConfig:
    cfg = BootstrapConfig(
        stages=stages,
        ppo=PPOConfig(),
        env=BootstrapEnvDefaults(enabled_units=["W"]),
        eval_freq=1,
        n_envs=1,
        seed=0,
    )
    cfg.validate()
    return cfg


def _stage(name: str, opp: str = "random", patience: int = 2, threshold: float = 0.9) -> CurriculumStage:
    return CurriculumStage(
        name=name,
        map_file="maps/1v1/starter.csv",
        opponent=opp,
        promotion_win_rate=threshold,
        patience=patience,
        max_timesteps=10_000,
        n_eval_episodes=2,
    )


def _setup_run(stages, win_rate_program, tmp_path):
    cfg = _make_cfg(stages)

    fake_model: Optional[_FakeModel] = None

    def model_factory(vec_env, cfg_arg, output_dir):
        nonlocal fake_model
        fake_model = _FakeModel(
            win_rate_program=win_rate_program,
            _stage_names=[s.name for s in cfg_arg.stages],
        )
        return fake_model

    train_envs: List[_FakeEnv] = []
    eval_envs: List[_FakeEnv] = []

    def train_env_factory(stage, cfg_arg):
        env = _FakeEnv()
        train_envs.append(env)
        return env

    def eval_env_factory(stage, cfg_arg):
        env = _FakeEnv()
        eval_envs.append(env)
        return env

    return cfg, model_factory, train_env_factory, eval_env_factory, train_envs, eval_envs, lambda: fake_model


class TestRunCurriculum:
    def test_advances_through_all_stages_when_each_promotes(self, tmp_path):
        stages = [_stage("a", patience=2), _stage("b", "simple", patience=2)]
        program = {
            # Two consecutive >= 0.9 evals trigger promotion.
            "a": [0.95, 0.95],
            "b": [0.92, 0.93],
        }
        cfg, mf, tef, eef, train_envs, eval_envs, get_model = _setup_run(stages, program, tmp_path)

        result = run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=tef,
            eval_env_factory=eef,
            model_factory=mf,
        )

        model = get_model()
        assert model is not None
        # Every stage's learn() was called and exited early via promotion.
        assert [c["stage"] for c in model.learn_calls] == ["a", "b"]
        assert all(c["exited_early"] for c in model.learn_calls)
        # set_env called for every stage after the first.
        assert len(model.set_env_calls) == len(stages) - 1
        # History reports both stages promoted.
        assert [h["stage"] for h in result["history"]] == ["a", "b"]
        assert all(h["promoted"] for h in result["history"])
        # Final checkpoint was written.
        assert Path(result["final_model_path"]).is_file()
        # All envs were closed.
        assert all(e.closed for e in train_envs)
        assert all(e.closed for e in eval_envs)

    def test_raises_when_stage_stalls(self, tmp_path):
        stages = [_stage("a", patience=2), _stage("b", "simple", patience=2)]
        program = {
            # Fewer-than-patience successful evals -> the runner should
            # see no promotion and raise CurriculumStalled before reaching
            # stage 'b'.
            "a": [0.95, 0.40, 0.80],
            "b": [0.95, 0.95],
        }
        cfg, mf, tef, eef, _, _, get_model = _setup_run(stages, program, tmp_path)

        with pytest.raises(CurriculumStalled) as excinfo:
            run_curriculum(
                cfg,
                output_dir=tmp_path,
                train_env_factory=tef,
                eval_env_factory=eef,
                model_factory=mf,
            )

        assert excinfo.value.stage_name == "a"
        assert excinfo.value.threshold == pytest.approx(0.9)
        # We never advance to stage 'b'.
        model = get_model()
        assert [c["stage"] for c in model.learn_calls] == ["a"]
