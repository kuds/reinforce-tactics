"""Config fields take effect, or the entry point says they do not.

Review findings covered here:

* rltrain-9: a stage without ``n_eval_episodes`` ran a hidden 30 episodes
  whatever ``eval.n_eval_episodes`` said; ``env.fog_of_war`` was dropped by
  ``bootstrap._stage_env_kwargs``; ``eval.checkpoint_freq``, ``logging.*``,
  ``ppo.lr_schedule`` and friends were ignored without a word. Entry points
  now declare what they read and report the rest (``--strict``: an error).
* rltrain-13: ``resolved_config.yaml`` was written before ``pad_to_size`` was
  resolved, and each stage's ``config.json`` hand-copied a subset of the env
  kwargs (no pad_to_size, no observation scales).
* flat_action_version (from the env package's hand-off): a config can pin
  the flat_discrete table version; left unset, a warm start keeps the table
  its checkpoint was trained on.
"""

import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml
from gymnasium import spaces

from reinforcetactics.rl import bootstrap
from reinforcetactics.rl.bootstrap import (
    CONSUMED_CONFIG_FIELDS,
    _stage_env_kwargs,
    _write_stage_config,
    make_stage_env,
    resolve_config,
    run_curriculum,
)
from reinforcetactics.rl.callbacks import PeriodicEvalCallback
from reinforcetactics.rl.config import (
    CurriculumStage,
    EnvConfig,
    EvalConfig,
    IgnoredConfigFieldError,
    IgnoredConfigFieldWarning,
    TrainingConfig,
    check_ignored_config_fields,
    config_from_dict,
    ignored_config_fields,
)
from reinforcetactics.rl.gym_env import (
    FLAT_ACTION_VERSION_LATEST,
    FLAT_ACTION_VERSION_LEGACY,
    checkpoint_flat_action_version,
    resolve_flat_action_version,
    stamp_flat_action_version,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SMALL_MAP = "maps/1v1/beginner.csv"  # 6x6
BIG_MAP = "maps/1v1/corner_points.csv"  # 10x12


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / "train" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage(name="s1", map_file=SMALL_MAP, **kwargs) -> dict[str, Any]:
    return {"name": name, "map_file": map_file, "opponent": "random", "max_timesteps": 1_000, **kwargs}


def _curriculum_cfg(*stages, **sections) -> TrainingConfig:
    return config_from_dict({"curriculum": {"stages": list(stages) or [_stage()]}, **sections})


# ---------------------------------------------------------------------------
# rltrain-9: n_eval_episodes resolves against cfg.eval
# ---------------------------------------------------------------------------


class _Abort(Exception):
    pass


class _RecordingModel:
    """Stands in for MaskablePPO: records the first stage's callbacks, then aborts the run."""

    def __init__(self) -> None:
        self.num_timesteps = 0
        self.ent_coef = 0.0
        self.callbacks: list[Any] = []

    def learn(self, total_timesteps, callback, reset_num_timesteps=True, progress_bar=False):
        self.callbacks = list(callback)
        raise _Abort


class _NullEnv:
    def close(self) -> None:
        pass


def _run_first_stage(cfg: TrainingConfig, tmp_path: Path) -> _RecordingModel:
    model = _RecordingModel()
    with pytest.raises(_Abort):
        run_curriculum(
            cfg,
            tmp_path / "run",
            train_env_factory=lambda stage, c: _NullEnv(),
            eval_env_factory=lambda stage, c: _NullEnv(),
            model_factory=lambda vec_env, c, out: model,
        )
    return model


class TestStageEvalEpisodes:
    def test_stage_default_inherits_eval_section(self):
        stage = CurriculumStage(name="s", map_file=SMALL_MAP, opponent="random")
        assert stage.n_eval_episodes is None
        assert stage.resolve_n_eval_episodes(EvalConfig(n_eval_episodes=80)) == 80

    def test_stage_override_wins(self):
        stage = CurriculumStage(name="s", map_file=SMALL_MAP, opponent="random", n_eval_episodes=5)
        assert stage.resolve_n_eval_episodes(EvalConfig(n_eval_episodes=80)) == 5

    def test_runner_evaluates_with_the_eval_section_value(self, tmp_path):
        cfg = _curriculum_cfg(_stage(), eval={"n_eval_episodes": 7})
        model = _run_first_stage(cfg, tmp_path)
        eval_cb = next(cb for cb in model.callbacks if isinstance(cb, PeriodicEvalCallback))
        assert eval_cb.n_eval_episodes == 7

    def test_stage_config_records_the_resolved_count(self, tmp_path):
        cfg = resolve_config(_curriculum_cfg(_stage(), eval={"n_eval_episodes": 7}))
        stage = cfg.curriculum.stages[0]
        _write_stage_config(stage=stage, cfg=cfg, stage_dir=tmp_path, output_dir=tmp_path, promoted=True, best_win_rate=1.0)
        record = json.loads((tmp_path / "config.json").read_text())
        assert record["extra"]["n_eval_episodes"] == 7


# ---------------------------------------------------------------------------
# rltrain-9: every EnvConfig field reaches the curriculum's envs
# ---------------------------------------------------------------------------


class TestStageEnvKwargs:
    def test_fog_of_war_reaches_the_stage_env(self):
        stage = CurriculumStage(name="s", map_file=SMALL_MAP, opponent="random")
        env = make_stage_env(stage, EnvConfig(fog_of_war=True), seed=0)
        try:
            assert env.unwrapped.fog_of_war is True
        finally:
            env.close()

    def test_fog_of_war_reaches_the_training_vec_env(self):
        cfg = _curriculum_cfg(_stage(), env={"fog_of_war": True, "n_envs": 1, "use_subprocess": False})
        vec_env = bootstrap._default_train_env_factory(cfg.curriculum.stages[0], cfg)
        try:
            assert vec_env.envs[0].unwrapped.fog_of_war is True
        finally:
            vec_env.close()

    def test_flat_action_version_reaches_the_stage_env(self):
        stage = CurriculumStage(name="s", map_file=SMALL_MAP, opponent="random")
        env = make_stage_env(stage, EnvConfig(action_space_type="flat_discrete", flat_action_version=1), seed=0)
        try:
            assert env.unwrapped.flat_action_version == 1
        finally:
            env.close()

    def test_every_env_field_is_forwarded_or_declared_stage_sourced(self):
        """A new EnvConfig field must be forwarded by _stage_env_kwargs (or consciously listed here)."""
        env_fields = {f.name for f in fields(EnvConfig)}
        forwarded = set(_stage_env_kwargs(CurriculumStage(name="s", map_file=SMALL_MAP, opponent="random"), EnvConfig()))
        from_stage = {"map_file", "opponent", "opponent_kwargs"}
        vec_env_knobs = {"n_envs", "use_subprocess"}
        assert env_fields - forwarded - vec_env_knobs == from_stage - forwarded
        consumed_env = {p.split(".", 1)[1] for p in CONSUMED_CONFIG_FIELDS if p.startswith("env.")}
        assert consumed_env == (env_fields & forwarded) - from_stage | vec_env_knobs


# ---------------------------------------------------------------------------
# rltrain-13: resolved record
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def _mixed(self, **env) -> TrainingConfig:
        return _curriculum_cfg(
            _stage("small", SMALL_MAP),
            _stage("big", BIG_MAP),
            env={"action_space_type": "flat_discrete", **env},
        )

    def test_resolves_pad_and_version_without_touching_the_input(self):
        cfg = self._mixed()
        resolved = resolve_config(cfg)
        assert resolved.env.pad_to_size == (10, 12)
        assert resolved.env.flat_action_version == FLAT_ACTION_VERSION_LATEST
        assert cfg.env.pad_to_size is None and cfg.env.flat_action_version is None
        assert resolve_config(resolved).to_dict() == resolved.to_dict()  # idempotent

    def test_explicit_version_is_kept(self):
        assert resolve_config(self._mixed(flat_action_version=1)).env.flat_action_version == 1

    def test_run_curriculum_writes_the_resolution_back(self, tmp_path):
        cfg = self._mixed()
        _run_first_stage(cfg, tmp_path)
        assert cfg.env.pad_to_size == (10, 12)
        assert cfg.env.flat_action_version == FLAT_ACTION_VERSION_LATEST

    def test_stage_config_env_is_exactly_the_env_kwargs(self, tmp_path):
        cfg = resolve_config(self._mixed(gold_scale=500.0, fog_of_war=True))
        stage = cfg.curriculum.stages[0]
        _write_stage_config(stage=stage, cfg=cfg, stage_dir=tmp_path, output_dir=tmp_path, promoted=True, best_win_rate=1.0)
        env_config = json.loads((tmp_path / "config.json").read_text())["env_config"]
        expected = json.loads(json.dumps(_stage_env_kwargs(stage, cfg.env)))
        assert env_config == expected
        assert env_config["pad_to_size"] == [10, 12]
        assert (env_config["gold_scale"], env_config["fog_of_war"], env_config["flat_action_version"]) == (500.0, True, 2)
        # Enough to rebuild the stage env's observation space from the record alone.
        rebuilt = make_stage_env(stage, EnvConfig(**{k: v for k, v in env_config.items() if hasattr(EnvConfig, k)}), seed=0)
        original = make_stage_env(stage, cfg.env, seed=0)
        try:
            assert rebuilt.observation_space == original.observation_space
        finally:
            rebuilt.close()
            original.close()


class TestFlatActionVersionFromCheckpoint:
    @staticmethod
    def _checkpoint(path: Path, version: int | None) -> Path:
        from stable_baselines3.common.save_util import save_to_zip_file

        space: spaces.Discrete = spaces.Discrete(16)
        if version is not None:
            stamp_flat_action_version(space, version)
        save_to_zip_file(path, data={"action_space": space})
        return path

    def test_reads_the_stamp_or_legacy(self, tmp_path):
        assert checkpoint_flat_action_version(self._checkpoint(tmp_path / "old.zip", None)) == FLAT_ACTION_VERSION_LEGACY
        assert checkpoint_flat_action_version(self._checkpoint(tmp_path / "new.zip", 2)) == 2

    def test_not_a_checkpoint(self, tmp_path):
        bogus = tmp_path / "bogus.zip"
        bogus.write_text("not a zip")
        with pytest.raises(ValueError, match="not a Stable-Baselines3 checkpoint"):
            checkpoint_flat_action_version(bogus)

    def test_resolution_order(self, tmp_path):
        old = self._checkpoint(tmp_path / "old.zip", None)
        assert resolve_flat_action_version(None, action_space_type="flat_discrete") == FLAT_ACTION_VERSION_LATEST
        assert resolve_flat_action_version(None, action_space_type="flat_discrete", checkpoint_path=str(old)) == 1
        assert resolve_flat_action_version(2, action_space_type="flat_discrete", checkpoint_path=str(old)) == 2
        # multi_discrete never reads the checkpoint (it need not even be one).
        assert resolve_flat_action_version(None, action_space_type="multi_discrete", checkpoint_path="nope") == 2

    def test_warm_start_keeps_the_checkpoint_table(self, tmp_path):
        old = self._checkpoint(tmp_path / "old.zip", None)
        cfg = _curriculum_cfg(_stage(), env={"action_space_type": "flat_discrete"}, warm_start_path=str(old))
        assert resolve_config(cfg).env.flat_action_version == FLAT_ACTION_VERSION_LEGACY


# ---------------------------------------------------------------------------
# rltrain-9: ignored fields are reported
# ---------------------------------------------------------------------------


class TestIgnoredFieldReport:
    def test_defaults_report_nothing(self):
        assert ignored_config_fields(TrainingConfig(), CONSUMED_CONFIG_FIELDS) == []

    def test_non_default_unread_fields_are_reported(self):
        cfg = config_from_dict(
            {"eval": {"checkpoint_freq": 5}, "logging": {"wandb": True}, "ppo": {"lr_schedule": "linear", "gamma": 0.9}}
        )
        assert ignored_config_fields(cfg, CONSUMED_CONFIG_FIELDS) == [
            "ppo.lr_schedule",
            "eval.checkpoint_freq",
            "logging.wandb",
        ]

    def test_section_wildcards_and_algorithm_labels(self):
        cfg = config_from_dict({"algorithm": "feudal", "feudal": {"manager_horizon": 3}})
        assert ignored_config_fields(cfg, ["feudal.*"]) == ["algorithm"]
        assert ignored_config_fields(cfg, ["feudal.*"], algorithms=["feudal"]) == []

    def test_warns_by_default_and_raises_when_strict(self):
        cfg = config_from_dict({"eval": {"checkpoint_freq": 5}})
        with pytest.warns(IgnoredConfigFieldWarning, match=r"eval.checkpoint_freq = 5  \(saved per stage\)"):
            check_ignored_config_fields(
                cfg, CONSUMED_CONFIG_FIELDS, entry_point="x", hints={"eval.checkpoint_freq": "saved per stage"}
            )
        with pytest.raises(IgnoredConfigFieldError, match="eval.checkpoint_freq"):
            check_ignored_config_fields(cfg, CONSUMED_CONFIG_FIELDS, entry_point="x", strict=True)


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


MINIMAL_BOOTSTRAP = {
    "seed": 1,
    "env": {"n_envs": 1, "use_subprocess": False, "action_space_type": "flat_discrete"},
    "curriculum": {"stages": [_stage("small", SMALL_MAP), _stage("big", BIG_MAP)]},
}


@pytest.fixture
def train_bootstrap(monkeypatch):
    monkeypatch.setattr(sys, "path", [*sys.path])
    return _load_script("train_bootstrap")


class TestTrainBootstrapEntryPoint:
    def _argv(self, config: Path, out: Path, *extra: str) -> list[str]:
        return [
            "--config", str(config), "--output-dir", str(out), "--device", "cpu", "--no-gcs",
            "--skip-plots", "--skip-videos", "--sanity-episodes", "0", *extra,
        ]  # fmt: skip

    def _stub_curriculum(self, monkeypatch):
        seen: list[TrainingConfig] = []

        def fake_run(cfg, output_dir):
            seen.append(cfg)
            return {"history": [], "final_model_path": None}

        monkeypatch.setattr(bootstrap, "run_curriculum", fake_run)
        return seen

    def test_resolved_config_records_pad_and_version(self, train_bootstrap, monkeypatch, tmp_path):
        seen = self._stub_curriculum(monkeypatch)
        config = _write_yaml(tmp_path / "c.yaml", MINIMAL_BOOTSTRAP)
        assert train_bootstrap.main(self._argv(config, tmp_path / "out")) == 0
        record = yaml.safe_load((tmp_path / "out" / "resolved_config.yaml").read_text())
        assert record["env"]["pad_to_size"] == [10, 12]
        assert record["env"]["flat_action_version"] == FLAT_ACTION_VERSION_LATEST
        # The curriculum gets the same resolved config.
        assert seen[0].env.pad_to_size == (10, 12)

    def test_ignored_field_warns_and_the_run_proceeds(self, train_bootstrap, monkeypatch, tmp_path):
        seen = self._stub_curriculum(monkeypatch)
        config = _write_yaml(tmp_path / "c.yaml", {**MINIMAL_BOOTSTRAP, "logging": {"log_dir": "elsewhere"}})
        with pytest.warns(IgnoredConfigFieldWarning, match="logging.log_dir"):
            assert train_bootstrap.main(self._argv(config, tmp_path / "out")) == 0
        assert len(seen) == 1

    def test_strict_rejects_before_any_output(self, train_bootstrap, monkeypatch, tmp_path):
        seen = self._stub_curriculum(monkeypatch)
        config = _write_yaml(tmp_path / "c.yaml", {**MINIMAL_BOOTSTRAP, "total_timesteps": 5_000})
        with pytest.raises(IgnoredConfigFieldError, match="total_timesteps.*sum of curriculum.stages"):
            train_bootstrap.main(self._argv(config, tmp_path / "out", "--strict"))
        assert not seen and not (tmp_path / "out").exists()

    def test_missing_warm_start_fails_before_any_output(self, train_bootstrap, monkeypatch, tmp_path):
        seen = self._stub_curriculum(monkeypatch)
        config = _write_yaml(tmp_path / "c.yaml", {**MINIMAL_BOOTSTRAP, "warm_start_path": str(tmp_path / "nope.zip")})
        with pytest.raises(FileNotFoundError, match="warm_start_path"):
            train_bootstrap.main(self._argv(config, tmp_path / "out"))
        assert not seen and not (tmp_path / "out").exists()

    def test_device_defaults_to_the_config(self, train_bootstrap, monkeypatch, tmp_path):
        # --device used to default to "auto" and overwrite ppo.device.
        seen = self._stub_curriculum(monkeypatch)
        config = _write_yaml(tmp_path / "c.yaml", {**MINIMAL_BOOTSTRAP, "ppo": {"device": "cuda:1"}})
        argv = self._argv(config, tmp_path / "out")
        del argv[argv.index("--device") : argv.index("--device") + 2]
        assert train_bootstrap.main(argv) == 0
        assert seen[0].ppo.device == "cuda:1"
