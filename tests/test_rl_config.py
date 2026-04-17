"""Tests for reinforcetactics.rl.config."""

import json
from pathlib import Path

import pytest
import yaml

from reinforcetactics.rl.config import (
    AlphaZeroConfig,
    EnvConfig,
    EvalConfig,
    FeudalConfig,
    LoggingConfig,
    PPOConfig,
    SelfPlayConfig,
    TrainingConfig,
    apply_overrides,
    config_from_dict,
    config_to_argparse_defaults,
    load_config,
    save_config,
)

# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_training_config_defaults_validate(self):
        cfg = TrainingConfig()
        cfg.validate()  # should not raise
        assert cfg.algorithm == "maskable_ppo"
        assert isinstance(cfg.env, EnvConfig)
        assert isinstance(cfg.ppo, PPOConfig)
        assert isinstance(cfg.feudal, FeudalConfig)
        assert isinstance(cfg.self_play, SelfPlayConfig)
        assert isinstance(cfg.alphazero, AlphaZeroConfig)
        assert isinstance(cfg.eval, EvalConfig)
        assert isinstance(cfg.logging, LoggingConfig)

    def test_ppo_as_sb3_kwargs_excludes_non_sb3_fields(self):
        cfg = PPOConfig()
        kwargs = cfg.as_sb3_kwargs()
        assert "use_action_masking" not in kwargs
        assert "learning_rate" in kwargs
        assert "device" in kwargs

    def test_to_dict_roundtrip(self):
        cfg = TrainingConfig()
        data = cfg.to_dict()
        assert data["algorithm"] == "maskable_ppo"
        assert "env" in data and "ppo" in data
        cfg2 = config_from_dict(data)
        assert cfg2.to_dict() == data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_algorithm_rejected(self):
        cfg = TrainingConfig(algorithm="dqn")
        with pytest.raises(ValueError, match="Unknown algorithm"):
            cfg.validate()

    def test_non_positive_total_timesteps_rejected(self):
        cfg = TrainingConfig(total_timesteps=0)
        with pytest.raises(ValueError, match="total_timesteps"):
            cfg.validate()

    def test_non_positive_n_envs_rejected(self):
        cfg = TrainingConfig()
        cfg.env.n_envs = 0
        with pytest.raises(ValueError, match="n_envs"):
            cfg.validate()

    def test_non_positive_max_steps_rejected(self):
        cfg = TrainingConfig()
        cfg.env.max_steps = 0
        with pytest.raises(ValueError, match="max_steps"):
            cfg.validate()

    def test_gamma_out_of_range_rejected(self):
        cfg = TrainingConfig()
        cfg.ppo.gamma = 1.5
        with pytest.raises(ValueError, match="gamma"):
            cfg.validate()

    def test_gae_lambda_out_of_range_rejected(self):
        cfg = TrainingConfig()
        cfg.ppo.gae_lambda = -0.1
        with pytest.raises(ValueError, match="gae_lambda"):
            cfg.validate()

    def test_non_positive_batch_size_rejected(self):
        cfg = TrainingConfig()
        cfg.ppo.batch_size = 0
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()

    def test_non_positive_n_steps_rejected(self):
        cfg = TrainingConfig()
        cfg.ppo.n_steps = -1
        with pytest.raises(ValueError, match="n_steps"):
            cfg.validate()

    def test_invalid_action_space_type_rejected(self):
        cfg = TrainingConfig()
        cfg.env.action_space_type = "bogus"
        with pytest.raises(ValueError, match="action_space_type"):
            cfg.validate()

    def test_invalid_pool_strategy_rejected(self):
        cfg = TrainingConfig()
        cfg.self_play.pool_strategy = "bogus"
        with pytest.raises(ValueError, match="pool_strategy"):
            cfg.validate()

    def test_win_rate_out_of_range_rejected(self):
        cfg = TrainingConfig()
        cfg.self_play.min_win_rate_for_pool = 1.5
        with pytest.raises(ValueError, match="min_win_rate"):
            cfg.validate()


# ---------------------------------------------------------------------------
# config_from_dict
# ---------------------------------------------------------------------------


class TestConfigFromDict:
    def test_minimal_dict(self):
        cfg = config_from_dict({"algorithm": "ppo", "total_timesteps": 1000})
        assert cfg.algorithm == "ppo"
        assert cfg.total_timesteps == 1000

    def test_nested_sections(self):
        cfg = config_from_dict(
            {
                "env": {"opponent": "random", "n_envs": 2},
                "ppo": {"learning_rate": 1e-4, "gamma": 0.95},
            }
        )
        assert cfg.env.opponent == "random"
        assert cfg.env.n_envs == 2
        assert cfg.ppo.learning_rate == 1e-4
        assert cfg.ppo.gamma == 0.95

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown top-level keys"):
            config_from_dict({"bogus": 42})

    def test_unknown_section_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown keys in section 'ppo'"):
            config_from_dict({"ppo": {"bogus_field": 1.0}})

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            config_from_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_non_mapping_section_rejected(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            config_from_dict({"env": "not a dict"})

    def test_none_section_uses_defaults(self):
        cfg = config_from_dict({"env": None})
        assert cfg.env.opponent == "bot"  # default


# ---------------------------------------------------------------------------
# File loading (YAML / JSON)
# ---------------------------------------------------------------------------


class TestFileLoading:
    def test_load_yaml(self, tmp_path: Path):
        path = tmp_path / "c.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "algorithm": "feudal",
                    "total_timesteps": 5000,
                    "env": {"opponent": "random", "max_steps": 100},
                    "ppo": {"learning_rate": 1e-5},
                }
            )
        )
        cfg = load_config(path)
        assert cfg.algorithm == "feudal"
        assert cfg.env.opponent == "random"
        assert cfg.env.max_steps == 100
        assert cfg.ppo.learning_rate == 1e-5
        assert cfg.total_timesteps == 5000

    def test_load_yml_extension(self, tmp_path: Path):
        path = tmp_path / "c.yml"
        path.write_text(yaml.safe_dump({"algorithm": "ppo"}))
        cfg = load_config(path)
        assert cfg.algorithm == "ppo"

    def test_load_json(self, tmp_path: Path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"algorithm": "ppo", "seed": 7}))
        cfg = load_config(path)
        assert cfg.algorithm == "ppo"
        assert cfg.seed == 7

    def test_load_empty_yaml_uses_defaults(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_config(path)
        assert cfg.algorithm == "maskable_ppo"

    def test_unsupported_extension_rejected(self, tmp_path: Path):
        path = tmp_path / "c.toml"
        path.write_text("algorithm = 'ppo'\n")
        with pytest.raises(ValueError, match="Unsupported config extension"):
            load_config(path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_non_mapping_top_level_rejected(self, tmp_path: Path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump([1, 2, 3]))
        with pytest.raises(TypeError, match="must contain a mapping"):
            load_config(path)

    def test_roundtrip_yaml(self, tmp_path: Path):
        path = tmp_path / "c.yaml"
        cfg = TrainingConfig(algorithm="ppo", seed=42)
        cfg.ppo.learning_rate = 7e-4
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.algorithm == "ppo"
        assert loaded.seed == 42
        assert loaded.ppo.learning_rate == 7e-4

    def test_roundtrip_json(self, tmp_path: Path):
        path = tmp_path / "c.json"
        cfg = TrainingConfig(algorithm="feudal")
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.algorithm == "feudal"

    def test_save_unsupported_extension_rejected(self, tmp_path: Path):
        cfg = TrainingConfig()
        with pytest.raises(ValueError, match="Unsupported config extension"):
            save_config(cfg, tmp_path / "c.txt")


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    def test_apply_top_level_override(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, {"seed": 99})
        assert new_cfg.seed == 99
        assert cfg.seed == 0  # original untouched

    def test_apply_nested_override(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, {"ppo.learning_rate": 1e-5})
        assert new_cfg.ppo.learning_rate == 1e-5

    def test_deeply_nested_unchanged_on_copy(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, {"env.n_envs": 16})
        assert new_cfg.env.n_envs == 16
        assert cfg.env.n_envs == 4  # unchanged

    def test_none_override_ignored(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, {"seed": None})
        assert new_cfg.seed == 0  # ignored

    def test_null_sentinel_sets_none(self):
        cfg = TrainingConfig()
        cfg.logging.wandb_entity = "team"
        new_cfg = apply_overrides(cfg, {"logging.wandb_entity": "null"})
        assert new_cfg.logging.wandb_entity is None

    def test_string_coerced_to_int(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, {"seed": "42"})
        assert new_cfg.seed == 42

    def test_string_coerced_to_float(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, {"ppo.learning_rate": "1e-5"})
        assert new_cfg.ppo.learning_rate == 1e-5

    def test_string_coerced_to_bool(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, {"ppo.use_action_masking": "false"})
        assert new_cfg.ppo.use_action_masking is False

    def test_invalid_bool_string_raises(self):
        cfg = TrainingConfig()
        with pytest.raises(ValueError, match="Cannot parse"):
            apply_overrides(cfg, {"ppo.use_action_masking": "maybe"})

    def test_unknown_key_raises(self):
        cfg = TrainingConfig()
        with pytest.raises(KeyError, match="Unknown config key"):
            apply_overrides(cfg, {"ppo.bogus": 42})

    def test_unknown_section_segment_raises(self):
        cfg = TrainingConfig()
        with pytest.raises(KeyError, match="Unknown config key segment"):
            apply_overrides(cfg, {"bogus.learning_rate": 42})

    def test_path_through_non_dataclass_raises(self):
        cfg = TrainingConfig()
        with pytest.raises(KeyError, match="does not point to a config section"):
            apply_overrides(cfg, {"algorithm.foo": "ppo"})

    def test_empty_overrides_returns_copy(self):
        cfg = TrainingConfig()
        new_cfg = apply_overrides(cfg, None)
        assert new_cfg is not cfg
        assert new_cfg.to_dict() == cfg.to_dict()

    def test_override_triggers_validation(self):
        cfg = TrainingConfig()
        with pytest.raises(ValueError, match="gamma"):
            apply_overrides(cfg, {"ppo.gamma": 2.0})


# ---------------------------------------------------------------------------
# config_to_argparse_defaults
# ---------------------------------------------------------------------------


class TestArgparseDefaults:
    def test_basic_mapping(self):
        cfg = TrainingConfig()
        cfg.ppo.learning_rate = 5e-5
        cfg.seed = 13
        out = config_to_argparse_defaults(
            cfg,
            {"learning_rate": "ppo.learning_rate", "seed": "seed"},
        )
        assert out == {"learning_rate": 5e-5, "seed": 13}

    def test_missing_path_silently_skipped(self):
        cfg = TrainingConfig()
        out = config_to_argparse_defaults(
            cfg,
            {"learning_rate": "ppo.learning_rate", "ghost": "ppo.does_not_exist"},
        )
        assert "learning_rate" in out
        assert "ghost" not in out


# ---------------------------------------------------------------------------
# Shipped example configs
# ---------------------------------------------------------------------------


class TestShippedConfigs:
    @pytest.mark.parametrize(
        "path",
        [
            "configs/maskable_ppo.yaml",
            "configs/ppo_baseline.yaml",
            "configs/feudal_rl.yaml",
            "configs/self_play.yaml",
            "configs/alphazero.yaml",
            "configs/curriculum/easy.yaml",
            "configs/curriculum/medium.yaml",
            "configs/curriculum/hard.yaml",
        ],
    )
    def test_shipped_config_loads(self, path: str):
        cfg = load_config(path)
        cfg.validate()
