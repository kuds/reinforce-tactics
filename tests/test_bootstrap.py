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
from reinforcetactics.rl.callbacks import (
    EntropyScheduleCallback,
    PeriodicEvalCallback,
    PromotionCallback,
)
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

    def test_accepts_balanced_random_opponent(self):
        # BalancedRandomBot is a curriculum stepping stone between `noop`
        # and `random`; ensure validation accepts the opponent string.
        cfg = config_from_dict(
            {
                **VALID_DICT,
                "stages": [{**VALID_DICT["stages"][0], "opponent": "balanced_random"}],
            }
        )
        assert cfg.stages[0].opponent == "balanced_random"

    def test_shipped_config_loads(self):
        # The repo's configs/bootstrap.yaml should always be valid.
        repo_root = Path(__file__).resolve().parents[1]
        cfg = load_bootstrap_config(repo_root / "configs" / "bootstrap.yaml")
        names = [s.name for s in cfg.stages]
        # Earlier iterations included `noop` stages on each map as a
        # stage-0 sanity check; they actively prevented PPO from
        # learning (no opponent variance -> constant returns ->
        # advantages collapse to ~0 -> policy never updates). Running
        # logs showed 250k+ steps with std=0.0 every eval. Reverting
        # to the original 6-stage layout (now 7, with the
        # `beginner_balanced_random` bridge for the map shift) lets
        # opponent randomness drive exploration the way PPO needs.
        # Regression: catch any future re-introduction of noop stages
        # without acknowledgement.
        assert names == [
            "starter_random",
            "starter_simple",
            "starter_medium",
            "beginner_balanced_random",
            "beginner_random_10",
            "beginner_random_20",
            "beginner_simple",
            "beginner_medium",
        ]
        assert "starter_noop" not in names, "noop stages broke PPO learning in earlier runs -- removing them was deliberate"
        assert "beginner_noop" not in names
        # Regression: PyYAML 1.1 parses ``3e-4`` (no decimal) as a string,
        # which then fails deep inside SB3's lr-schedule check. Every
        # numeric PPO field must round-trip as a number.
        assert isinstance(cfg.ppo.learning_rate, (int, float))
        assert isinstance(cfg.ppo.ent_coef, (int, float))
        assert isinstance(cfg.ppo.clip_range, (int, float))
        by_name = {s.name: s for s in cfg.stages}
        # Starter stages inherit env defaults (no per-stage overrides --
        # the original starter map config worked with the global env
        # defaults; we don't need to special-case it).
        assert by_name["starter_random"].max_turns is None
        assert by_name["starter_random"].ent_coef is None
        # Beginner stages bump max_turns / max_steps for the bigger map.
        first_beginner = by_name["beginner_balanced_random"]
        assert first_beginner.max_turns is not None
        assert first_beginner.max_turns >= 30
        # Entropy bump on the FIRST beginner stage (map-shift exploration
        # shock). Cooled on later stages. The shipped config now drives
        # the random-opponent stages with a {start, end, schedule}
        # mapping so the coefficient anneals down as the policy
        # approaches the threshold; resolving against the PPO default
        # gives the *initial* value (= schedule['start']) which still
        # has to clear the global default to count as a bump.
        assert first_beginner.ent_coef is not None
        assert first_beginner.resolve_ent_coef(cfg.ppo) > cfg.ppo.ent_coef
        # And: the schedule must actually anneal (start > end), otherwise
        # we paid for the schedule machinery without getting the
        # commitment-phase cooling that justified introducing it.
        sched = first_beginner.resolve_ent_coef_schedule()
        assert sched is not None, "first beginner stage should drive an entropy schedule"
        assert sched["start"] > sched["end"]
        # Reward-shape override on beginner stages: HQ capture is much
        # harder than elimination on the bigger map, so the two terminal
        # rewards must be equalized (or capture <= elimination).
        beginner_random = by_name["beginner_random_20"]
        assert beginner_random.reward_config is not None
        assert beginner_random.reward_config["win_by_hq_capture"] <= beginner_random.reward_config["win_by_elimination"]
        # Policy MLP capacity: SB3 defaults net_arch to [64, 64] which is
        # undersized for a Dict obs (~734 input dims) feeding a flat-
        # discrete head with up to 512 logits. The shipped config bumps
        # both pi and vf to at least [128, 128]. Catches accidental
        # removal of the policy_kwargs block.
        pk = cfg.ppo.policy_kwargs or {}
        net_arch = pk.get("net_arch")
        assert net_arch is not None, "expected ppo.policy_kwargs.net_arch in shipped config"
        assert isinstance(net_arch, dict)
        assert min(net_arch["pi"]) >= 128
        assert min(net_arch["vf"]) >= 128

    def test_reward_config_override_merges_with_defaults(self):
        # Stage override should *merge* over BootstrapEnvDefaults.reward_config,
        # not replace it. So a stage that only overrides one key still gets
        # the rest of the env defaults.
        defaults = BootstrapEnvDefaults(reward_config={"win": 5000.0, "loss": -5000.0, "draw": -5000.0})
        stage = CurriculumStage(
            name="s",
            map_file="m.csv",
            opponent="random",
            reward_config={"win": 3000.0, "win_by_elimination": 3000.0},
        )
        resolved = stage.resolve_reward_config(defaults)
        assert resolved == {
            "win": 3000.0,  # overridden
            "loss": -5000.0,  # inherited
            "draw": -5000.0,  # inherited
            "win_by_elimination": 3000.0,  # added by stage
        }

    def test_reward_config_resolves_to_defaults_when_unset(self):
        defaults = BootstrapEnvDefaults(reward_config={"win": 5000.0, "loss": -5000.0})
        stage = CurriculumStage(name="s", map_file="m.csv", opponent="random")
        assert stage.resolve_reward_config(defaults) == {"win": 5000.0, "loss": -5000.0}

    def test_reward_config_returns_none_when_nothing_specified(self):
        defaults = BootstrapEnvDefaults(reward_config=None)
        stage = CurriculumStage(name="s", map_file="m.csv", opponent="random")
        assert stage.resolve_reward_config(defaults) is None

    def test_rejects_non_mapping_reward_config(self):
        common = dict(name="s", map_file="m.csv", opponent="random")
        with pytest.raises(TypeError, match="reward_config"):
            CurriculumStage(**common, reward_config=[1, 2, 3]).validate()


class TestCurriculumStageResolution:
    def test_resolves_to_defaults_when_unset(self):
        env = BootstrapEnvDefaults(max_steps=400, max_turns=20)
        ppo = PPOConfig(ent_coef=0.05)
        stage = CurriculumStage(name="s", map_file="m.csv", opponent="random")
        assert stage.resolve_max_steps(env) == 400
        assert stage.resolve_max_turns(env) == 20
        assert stage.resolve_ent_coef(ppo) == pytest.approx(0.05)

    def test_resolves_to_override_when_set(self):
        env = BootstrapEnvDefaults(max_steps=400, max_turns=20)
        ppo = PPOConfig(ent_coef=0.05)
        stage = CurriculumStage(
            name="s",
            map_file="m.csv",
            opponent="random",
            max_steps=800,
            max_turns=40,
            ent_coef=0.10,
        )
        assert stage.resolve_max_steps(env) == 800
        assert stage.resolve_max_turns(env) == 40
        assert stage.resolve_ent_coef(ppo) == pytest.approx(0.10)

    def test_rejects_invalid_overrides(self):
        common = dict(name="s", map_file="m.csv", opponent="random")
        with pytest.raises(ValueError, match="max_steps"):
            CurriculumStage(**common, max_steps=0).validate()
        with pytest.raises(ValueError, match="max_turns"):
            CurriculumStage(**common, max_turns=-1).validate()
        with pytest.raises(ValueError, match="ent_coef"):
            CurriculumStage(**common, ent_coef=-0.01).validate()


class TestCurriculumStageEntropySchedule:
    """``ent_coef`` accepts either a constant or a ``{start, end, schedule}``
    mapping. The mapping form drives :class:`EntropyScheduleCallback` so
    exploration noise can be cooled across a stage."""

    common = dict(name="s", map_file="m.csv", opponent="random")

    def test_resolve_initial_value_uses_schedule_start(self):
        ppo = PPOConfig(ent_coef=0.05)
        stage = CurriculumStage(
            **self.common,
            ent_coef={"start": 0.10, "end": 0.03, "schedule": "linear"},
        )
        # Initial seed before the callback takes over: the schedule's
        # start, not the ppo default.
        assert stage.resolve_ent_coef(ppo) == pytest.approx(0.10)

    def test_resolve_schedule_returns_descriptor(self):
        stage = CurriculumStage(
            **self.common,
            ent_coef={"start": 0.10, "end": 0.03, "schedule": "linear"},
        )
        sched = stage.resolve_ent_coef_schedule()
        assert sched == {"start": 0.10, "end": 0.03, "schedule": "linear"}

    def test_resolve_schedule_defaults_to_linear(self):
        stage = CurriculumStage(**self.common, ent_coef={"start": 0.10, "end": 0.03})
        sched = stage.resolve_ent_coef_schedule()
        assert sched is not None
        assert sched["schedule"] == "linear"

    def test_resolve_schedule_is_none_for_constant(self):
        stage = CurriculumStage(**self.common, ent_coef=0.10)
        assert stage.resolve_ent_coef_schedule() is None

    def test_resolve_schedule_is_none_when_unset(self):
        stage = CurriculumStage(**self.common)
        assert stage.resolve_ent_coef_schedule() is None

    def test_validate_rejects_unknown_schedule_keys(self):
        with pytest.raises(ValueError, match="unknown keys"):
            CurriculumStage(
                **self.common,
                ent_coef={"start": 0.1, "end": 0.0, "extra": 1},
            ).validate()

    def test_validate_rejects_missing_required_key(self):
        with pytest.raises(ValueError, match="missing required key 'end'"):
            CurriculumStage(**self.common, ent_coef={"start": 0.1}).validate()
        with pytest.raises(ValueError, match="missing required key 'start'"):
            CurriculumStage(**self.common, ent_coef={"end": 0.03}).validate()

    def test_validate_rejects_negative_endpoint(self):
        with pytest.raises(ValueError, match="ent_coef.end"):
            CurriculumStage(
                **self.common,
                ent_coef={"start": 0.1, "end": -0.01},
            ).validate()

    def test_validate_rejects_unknown_schedule_kind(self):
        with pytest.raises(ValueError, match="schedule must be 'linear' or 'cosine'"):
            CurriculumStage(
                **self.common,
                ent_coef={"start": 0.1, "end": 0.03, "schedule": "exponential"},
            ).validate()


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
    # Tracks the value of ``ent_coef`` that the runner set immediately
    # before each ``learn()`` call, parallel to ``learn_calls``.
    ent_coef: float = 0.0
    ent_coef_at_learn: List[float] = field(default_factory=list)
    # The full callback list passed to each ``learn()`` invocation,
    # so tests can assert which callbacks the runner installed for a
    # given stage (notably whether ``EntropyScheduleCallback`` was
    # added when the stage's ``ent_coef`` is a schedule mapping).
    callbacks_at_learn: List[List[Any]] = field(default_factory=list)
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
        # Snapshot whatever ``ent_coef`` the runner mutated us to prior
        # to this learn() call.
        self.ent_coef_at_learn.append(float(self.ent_coef))
        self.callbacks_at_learn.append(list(callback))

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

    def test_applies_per_stage_ent_coef_and_env_overrides(self, tmp_path):
        # First stage inherits everything from cfg defaults; second stage
        # provides explicit overrides for max_steps, max_turns, and
        # ent_coef.
        stages = [
            _stage("a", patience=2),
            CurriculumStage(
                name="b",
                map_file="maps/1v1/beginner.csv",
                opponent="simple",
                promotion_win_rate=0.9,
                patience=2,
                max_timesteps=10_000,
                n_eval_episodes=2,
                max_steps=800,
                max_turns=40,
                ent_coef=0.10,
            ),
        ]
        program = {"a": [0.95, 0.95], "b": [0.95, 0.95]}

        train_calls: List[Dict[str, Any]] = []

        def train_env_factory(stage, cfg_arg):
            # Mirror what the default factory does: resolve via the
            # stage's helpers so the test exercises the same code path.
            train_calls.append(
                {
                    "name": stage.name,
                    "max_steps": stage.resolve_max_steps(cfg_arg.env),
                    "max_turns": stage.resolve_max_turns(cfg_arg.env),
                }
            )
            return _FakeEnv()

        def eval_env_factory(stage, cfg_arg):
            return _FakeEnv()

        cfg = _make_cfg(stages)
        fake_model = _FakeModel(
            win_rate_program=program,
            _stage_names=[s.name for s in stages],
        )

        def model_factory(vec_env, cfg_arg, output_dir):
            return fake_model

        run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=train_env_factory,
            eval_env_factory=eval_env_factory,
            model_factory=model_factory,
        )

        # Stage 'a' inherits cfg.env / cfg.ppo defaults.
        assert train_calls[0]["max_steps"] == cfg.env.max_steps
        assert train_calls[0]["max_turns"] == cfg.env.max_turns
        assert fake_model.ent_coef_at_learn[0] == pytest.approx(cfg.ppo.ent_coef)
        # Stage 'b' uses overrides.
        assert train_calls[1]["max_steps"] == 800
        assert train_calls[1]["max_turns"] == 40
        assert fake_model.ent_coef_at_learn[1] == pytest.approx(0.10)

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

        # The exception carries a partial result so notebook
        # diagnostics / replay-video helpers can run after a stall
        # instead of discarding everything. ``history`` should
        # contain one entry (the stalled stage 'a'), and the saved
        # ``final_model.zip`` should exist on disk so callers can
        # ``MaskablePPO.load(...)`` it for replay generation.
        partial = excinfo.value.partial_result()
        assert partial["stalled"] is True
        assert partial["stalled_stage"] == "a"
        assert len(partial["history"]) == 1
        assert partial["history"][0]["stage"] == "a"
        assert partial["history"][0]["promoted"] is False
        assert partial["final_model_path"] is not None
        assert (tmp_path / "final_model.zip").exists()

    def test_installs_entropy_schedule_callback_for_dict_ent_coef(self, tmp_path):
        """When ``stage.ent_coef`` is a ``{start, end, schedule}`` mapping,
        the runner should attach :class:`EntropyScheduleCallback` configured
        against the stage's budget so the coefficient anneals across the
        stage. A constant ``ent_coef`` must NOT install one (we want the
        commitment-phase stages to keep a fixed coefficient).
        """
        stages = [
            # Constant: no schedule callback expected.
            CurriculumStage(
                name="constant",
                map_file="maps/1v1/starter.csv",
                opponent="random",
                promotion_win_rate=0.9,
                patience=2,
                max_timesteps=10_000,
                n_eval_episodes=2,
                ent_coef=0.05,
            ),
            # Schedule mapping: schedule callback expected.
            CurriculumStage(
                name="annealed",
                map_file="maps/1v1/beginner.csv",
                opponent="simple",
                promotion_win_rate=0.9,
                patience=2,
                max_timesteps=20_000,
                n_eval_episodes=2,
                ent_coef={"start": 0.10, "end": 0.03, "schedule": "linear"},
            ),
        ]
        program = {"constant": [0.95, 0.95], "annealed": [0.95, 0.95]}
        cfg, mf, tef, eef, _, _, get_model = _setup_run(stages, program, tmp_path)

        run_curriculum(
            cfg,
            output_dir=tmp_path,
            train_env_factory=tef,
            eval_env_factory=eef,
            model_factory=mf,
        )

        model = get_model()
        # First stage: constant ent_coef -> no schedule callback.
        constant_cbs = model.callbacks_at_learn[0]
        assert not any(isinstance(c, EntropyScheduleCallback) for c in constant_cbs)
        # Second stage: schedule callback installed and configured to span
        # the stage's own budget (not the cumulative timestep counter).
        annealed_cbs = model.callbacks_at_learn[1]
        sched_cb = next((c for c in annealed_cbs if isinstance(c, EntropyScheduleCallback)), None)
        assert sched_cb is not None, "expected EntropyScheduleCallback for dict ent_coef"
        assert sched_cb.start == pytest.approx(0.10)
        assert sched_cb.end == pytest.approx(0.03)
        assert sched_cb.schedule == "linear"
        assert sched_cb.total_timesteps == 20_000
        # The model's ent_coef seeded before learn() should be the
        # schedule's *start* value, so the callback has a sensible
        # initial position to anneal from.
        assert model.ent_coef_at_learn[1] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# EntropyScheduleCallback unit tests
# ---------------------------------------------------------------------------


class _EntropyScheduleHarness:
    """Minimal SB3-compatible model stub for driving EntropyScheduleCallback.

    The real SB3 training loop pokes ``cb.num_timesteps`` directly on the
    callback (``BaseCallback`` keeps it as a plain attribute, not a
    property), and the callback writes ``model.ent_coef`` back. Tests
    set ``cb.num_timesteps`` to drive progress and read
    ``harness.ent_coef`` to verify the schedule.
    """

    def __init__(self) -> None:
        self.ent_coef = 0.0
        self.logger = _StubLogger()


class TestEntropyScheduleCallback:
    def test_validates_inputs(self):
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=-0.1, end=0.0, total_timesteps=100)
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=0.1, end=-0.1, total_timesteps=100)
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=0.1, end=0.0, total_timesteps=0)
        with pytest.raises(ValueError):
            EntropyScheduleCallback(start=0.1, end=0.0, total_timesteps=100, schedule="unknown")

    def test_linear_schedule_endpoints_and_midpoint(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000)
        harness = _EntropyScheduleHarness()
        cb.model = harness
        # _on_training_start anchors the stage start step against the
        # cumulative ``num_timesteps`` so progress is per-stage. SB3
        # writes this attribute on the callback during real training;
        # in tests we mutate it directly.
        cb.num_timesteps = 5000  # already non-zero from prior stages
        cb._on_training_start()
        # Step 0 of this stage -> start.
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.10)
        # Halfway through the stage -> linear midpoint.
        cb.num_timesteps = 5000 + 500
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.06)
        # End of stage -> end value.
        cb.num_timesteps = 5000 + 1000
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.02)

    def test_clamps_progress_beyond_budget(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000)
        harness = _EntropyScheduleHarness()
        cb.model = harness
        cb.num_timesteps = 0
        cb._on_training_start()
        # Overshoot the budget (can happen if the stage's actual
        # ``learn()`` runs slightly past the announced budget): the
        # coefficient should clamp at ``end``, not extrapolate.
        cb.num_timesteps = 10_000
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.02)

    def test_cosine_schedule_starts_high_ends_low(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000, schedule="cosine")
        harness = _EntropyScheduleHarness()
        cb.model = harness
        cb.num_timesteps = 0
        cb._on_training_start()
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.10)
        # Cosine half-cycle midpoint = (start + end) / 2.
        cb.num_timesteps = 500
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.06)
        cb.num_timesteps = 1000
        cb._on_step()
        assert harness.ent_coef == pytest.approx(0.02)

    def test_records_value_to_logger(self):
        cb = EntropyScheduleCallback(start=0.10, end=0.02, total_timesteps=1000)
        harness = _EntropyScheduleHarness()
        cb.model = harness
        cb.num_timesteps = 0
        cb._on_training_start()
        cb.num_timesteps = 250
        cb._on_step()
        # Tensorboard plumbing: the live coefficient lands under
        # ``train/ent_coef`` so it shows up alongside SB3's own
        # train/* curves. Lets us see the schedule actually firing
        # in TB without parsing logs.
        assert "train/ent_coef" in harness.logger.records
        assert harness.logger.records["train/ent_coef"] == pytest.approx(harness.ent_coef)
