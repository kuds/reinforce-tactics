"""Config values are typed, range-checked and checked against what the env accepts.

Review findings covered here:

* rltrain-10: raw YAML values reached the trainers unconverted (PyYAML reads
  ``3e-4`` and ``1.0e6`` as strings), nothing was range-checked
  (``eval_freq: 0`` loaded), reward_config keys were unchecked,
  ``purchase_explore_eps`` was accepted with flat_discrete (where the hook
  does nothing), and a missing ``warm_start_path`` surfaced only after the
  first stage's envs were built.
* rltrain-22 / rulebots-7 / rlenv-10 (config side): the curriculum opponent
  list was a hand-written copy of the bot registry that rejected ``master``,
  and ``opponent_kwargs`` were never checked against the bot they go to.
"""

from pathlib import Path

import pytest
import yaml

from reinforcetactics.game.bot_registry import SCRIPTED_BOTS, accepted_names
from reinforcetactics.rl.config import (
    _CURRICULUM_OPPONENTS,
    CurriculumStage,
    TrainingConfig,
    apply_overrides,
    config_from_dict,
    load_config,
    save_config,
)
from reinforcetactics.rl.gym_env import KNOWN_REWARD_KEYS

STAGE = {
    "name": "s1",
    "map_file": "maps/1v1/beginner.csv",
    "opponent": "random",
    "max_timesteps": 10_000,
}


def _cfg(**sections):
    return config_from_dict(sections)


def _stage_cfg(**stage_overrides):
    return _cfg(curriculum={"stages": [{**STAGE, **stage_overrides}]})


# ---------------------------------------------------------------------------
# rltrain-10: type coercion
# ---------------------------------------------------------------------------


class TestTypeCoercion:
    def test_yaml_scientific_notation_strings_become_numbers(self, tmp_path: Path):
        # PyYAML (YAML 1.1) reads both of these as *strings*.
        path = tmp_path / "c.yaml"
        path.write_text("ppo:\n  learning_rate: 3e-4\nalphazero:\n  buffer_size: 1.0e6\n", encoding="utf-8")
        assert yaml.safe_load(path.read_text())["ppo"]["learning_rate"] == "3e-4"

        cfg = load_config(path)
        assert cfg.ppo.learning_rate == pytest.approx(3e-4) and isinstance(cfg.ppo.learning_rate, float)
        assert cfg.alphazero.buffer_size == 1_000_000 and isinstance(cfg.alphazero.buffer_size, int)

    def test_integral_float_becomes_int_and_int_becomes_float(self):
        cfg = _cfg(env={"n_envs": 4.0}, ppo={"learning_rate": 1})
        assert cfg.env.n_envs == 4 and isinstance(cfg.env.n_envs, int)
        assert cfg.ppo.learning_rate == 1.0 and isinstance(cfg.ppo.learning_rate, float)

    def test_non_integral_float_for_int_field_rejected(self):
        with pytest.raises(ValueError, match="env.n_envs must be an integer"):
            _cfg(env={"n_envs": 4.5})

    def test_unparseable_number_rejected_with_its_path(self):
        with pytest.raises(ValueError, match="ppo.learning_rate must be a number"):
            _cfg(ppo={"learning_rate": "fast"})

    def test_bool_is_not_a_number(self):
        with pytest.raises(TypeError, match="env.max_steps"):
            _cfg(env={"max_steps": True})

    def test_string_booleans_parse_instead_of_being_truthy(self):
        # bool("false") is True: the old loader turned this string into True.
        cfg = _cfg(env={"fog_of_war": "false"}, curriculum={"restore_best_checkpoint_between_stages": "false"})
        assert cfg.env.fog_of_war is False
        assert cfg.curriculum.restore_best_checkpoint_between_stages is False

    def test_non_finite_number_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            _cfg(ppo={"ent_coef": float("nan")})

    def test_pad_to_size_becomes_a_pair_of_ints(self):
        cfg = _cfg(env={"pad_to_size": [10, "12"]})
        assert cfg.env.pad_to_size == (10, 12)
        with pytest.raises(ValueError, match="must have 2 entries"):
            _cfg(env={"pad_to_size": [10, 12, 14]})

    def test_reward_config_values_become_floats(self):
        cfg = _cfg(env={"reward_config": {"win": 1000, "damage_scale": "5e-2"}})
        assert cfg.env.reward_config == {"win": 1000.0, "damage_scale": 0.05}

    def test_reward_config_bool_value_rejected(self):
        with pytest.raises(TypeError, match="reward_config"):
            _cfg(env={"reward_config": {"win": True}})

    def test_wrong_shape_rejected(self):
        with pytest.raises(TypeError, match="env.enabled_units must be a list"):
            _cfg(env={"enabled_units": "W"})
        with pytest.raises(TypeError, match="env.engine_overrides must be a mapping"):
            _cfg(env={"engine_overrides": [1, 2]})

    def test_stage_fields_are_coerced(self):
        cfg = _stage_cfg(max_timesteps="1e4", promotion_win_rate="0.75", ent_coef={"start": "1e-1", "end": 0.01})
        stage = cfg.curriculum.stages[0]
        assert stage.max_timesteps == 10_000 and isinstance(stage.max_timesteps, int)
        assert stage.promotion_win_rate == 0.75
        assert stage.ent_coef == {"start": 0.1, "end": 0.01}

    def test_set_override_is_coerced_to_the_field_type(self):
        cfg = apply_overrides(TrainingConfig(), {"env.pad_to_size": [8, 8], "eval.n_eval_episodes": "20"})
        assert cfg.env.pad_to_size == (8, 8)
        assert cfg.eval.n_eval_episodes == 20

    def test_validate_normalizes_programmatic_values(self):
        cfg = TrainingConfig()
        cfg.ppo.learning_rate = "1e-4"  # type: ignore[assignment]
        cfg.validate()
        assert cfg.ppo.learning_rate == pytest.approx(1e-4)

    def test_tuple_pad_to_size_round_trips_through_yaml(self, tmp_path: Path):
        cfg = _cfg(env={"pad_to_size": [10, 12]})
        save_config(cfg, tmp_path / "c.yaml")
        assert "!!python" not in (tmp_path / "c.yaml").read_text()
        assert load_config(tmp_path / "c.yaml").env.pad_to_size == (10, 12)


# ---------------------------------------------------------------------------
# rltrain-10: range checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("eval", "eval_freq", 0, "eval.eval_freq"),
        ("eval", "n_eval_episodes", 0, "eval.n_eval_episodes"),
        ("eval", "checkpoint_freq", -1, "eval.checkpoint_freq"),
        ("env", "max_flat_actions", 0, "env.max_flat_actions"),
        ("env", "max_turns", 0, "env.max_turns"),
        ("env", "pad_to_size", [0, 12], "env.pad_to_size"),
        ("env", "gold_scale", 0.0, "env.gold_scale"),
        ("env", "flat_action_version", 3, "env.flat_action_version"),
        ("ppo", "learning_rate", 0.0, "ppo.learning_rate"),
        ("ppo", "clip_range", 0.0, "ppo.clip_range"),
        ("ppo", "n_epochs", 0, "ppo.n_epochs"),
        ("ppo", "ent_coef", -0.1, "ppo.ent_coef"),
        ("ppo", "max_grad_norm", 0.0, "ppo.max_grad_norm"),
        ("ppo", "purchase_explore_eps", 1.5, "ppo.purchase_explore_eps"),
        ("ppo", "lr_schedule", "cosine", "ppo.lr_schedule"),
        ("self_play", "bot_ratio", 1.0, "self_play.bot_ratio"),
        ("self_play", "pool_size", 0, "self_play.pool_size"),
        ("self_play", "latest_opponent_prob", 1.5, "self_play.latest_opponent_prob"),
        ("feudal", "worker_reward_alpha", 2.0, "feudal.worker_reward_alpha"),
        ("alphazero", "lr", 0.0, "alphazero.lr"),
        ("alphazero", "eval_threshold", 1.5, "alphazero.eval_threshold"),
    ],
)
def test_out_of_range_value_rejected(section, key, value, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**{section: {key: value}})


def test_negative_seed_rejected():
    with pytest.raises(ValueError, match="seed"):
        config_from_dict({"seed": -1})


class TestPurchaseExplorationNeedsMultiDiscrete:
    def test_ppo_level_eps_with_flat_discrete_rejected(self):
        with pytest.raises(ValueError, match="purchase_explore_eps.*multi_discrete"):
            _cfg(env={"action_space_type": "flat_discrete"}, ppo={"purchase_explore_eps": 0.1})

    def test_stage_schedule_with_flat_discrete_rejected(self):
        with pytest.raises(ValueError, match="stage 's1'"):
            _cfg(
                env={"action_space_type": "flat_discrete"},
                curriculum={"stages": [{**STAGE, "purchase_explore_eps": {"start": 0.2, "end": 0.0}}]},
            )

    def test_zero_eps_with_flat_and_positive_eps_with_multi_discrete_accepted(self):
        _cfg(env={"action_space_type": "flat_discrete"}, curriculum={"stages": [{**STAGE, "purchase_explore_eps": 0.0}]})
        _cfg(env={"action_space_type": "multi_discrete"}, ppo={"purchase_explore_eps": 0.1})


class TestWarmStartPathChecked:
    def test_missing_checkpoint_fails_validation_with_check_files(self, tmp_path: Path):
        cfg = config_from_dict({"warm_start_path": str(tmp_path / "missing.zip")})  # loads: no file check
        with pytest.raises(FileNotFoundError, match="warm_start_path"):
            cfg.validate(check_files=True)

    def test_existing_checkpoint_passes(self, tmp_path: Path):
        ckpt = tmp_path / "ckpt.zip"
        ckpt.write_bytes(b"x")
        config_from_dict({"warm_start_path": str(ckpt)}).validate(check_files=True)

    def test_run_curriculum_checks_before_building_anything(self, tmp_path: Path):
        from reinforcetactics.rl.bootstrap import run_curriculum

        cfg = _stage_cfg()
        cfg.warm_start_path = str(tmp_path / "missing.zip")

        def must_not_run(*_args):
            raise AssertionError("an env or model was built before the warm_start_path check")

        with pytest.raises(FileNotFoundError, match="warm_start_path"):
            run_curriculum(
                cfg,
                tmp_path / "out",
                train_env_factory=must_not_run,
                eval_env_factory=must_not_run,
                model_factory=must_not_run,
            )


# ---------------------------------------------------------------------------
# Reward keys (rlenv-16, config side)
# ---------------------------------------------------------------------------


class TestRewardKeys:
    def test_unknown_env_reward_key_rejected(self):
        with pytest.raises(ValueError, match="env.reward_config.*Unknown reward_config keys \\['wins'\\]"):
            _cfg(env={"reward_config": {"wins": 1.0}})

    def test_unknown_stage_reward_key_rejected(self):
        with pytest.raises(ValueError, match="stage 's1'.*Unknown reward_config keys \\['captur'\\]"):
            _stage_cfg(reward_config={"captur": 5.0})

    def test_every_known_key_accepted(self):
        cfg = _cfg(env={"reward_config": dict.fromkeys(KNOWN_REWARD_KEYS, 1.0)})
        assert set(cfg.env.reward_config) == set(KNOWN_REWARD_KEYS)

    def test_standalone_stage_validate_checks_keys(self):
        with pytest.raises(ValueError, match="Unknown reward_config keys"):
            CurriculumStage(**STAGE, reward_config={"bogus": 1.0}).validate()


# ---------------------------------------------------------------------------
# rltrain-22 / rulebots-7: opponents come from the registry
# ---------------------------------------------------------------------------


class TestOpponentsFromRegistry:
    def test_curriculum_opponents_are_the_registry_names(self):
        assert _CURRICULUM_OPPONENTS == accepted_names()
        assert set(SCRIPTED_BOTS) <= set(_CURRICULUM_OPPONENTS)

    def test_master_is_a_valid_stage_opponent(self):
        assert _stage_cfg(opponent="master").curriculum.stages[0].opponent == "master"

    def test_self_is_not_a_curriculum_opponent(self):
        with pytest.raises(ValueError, match="unknown opponent 'self'"):
            _stage_cfg(opponent="self")

    def test_env_opponent_validated(self):
        _cfg(env={"opponent": "master"})
        _cfg(env={"opponent": "self"})
        with pytest.raises(ValueError, match="env.opponent.*Unknown opponent 'mastr'"):
            _cfg(env={"opponent": "mastr"})

    def test_eval_opponent_must_be_a_scripted_bot(self):
        _cfg(self_play={"eval_opponent": "master"})
        with pytest.raises(ValueError, match="self_play.eval_opponent"):
            _cfg(self_play={"eval_opponent": "self"})

    def test_kwargs_for_a_deterministic_bot_rejected(self):
        with pytest.raises(ValueError, match="stage 's1'.*unknown kwargs \\['max_actions'\\] for scripted bot 'medium'"):
            _stage_cfg(opponent="medium", opponent_kwargs={"max_actions": 10})

    def test_random_bot_kwargs_accepted_and_typos_rejected(self):
        _stage_cfg(opponent="random", opponent_kwargs={"max_actions": 10})
        with pytest.raises(ValueError, match="unknown kwargs \\['max_action'\\]"):
            _stage_cfg(opponent="random", opponent_kwargs={"max_action": 10})

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"easy": "simple", "hard": "godlike"}, "godlike"),
            ({"p_hard": 1.5}, "p_hard"),
            ({"easy": "random", "easy_kwargs": {"max_actons": 5}}, "max_actons"),
            ({"hard": "medium", "hard_kwargs": {"max_actions": 5}}, "max_actions"),
        ],
    )
    def test_mixed_bot_kwargs_checked_in_depth(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _stage_cfg(opponent="mixed", opponent_kwargs=kwargs)

    def test_env_opponent_kwargs_checked_against_env_opponent(self):
        _cfg(env={"opponent": "random", "opponent_kwargs": {"max_actions": 5}})
        with pytest.raises(ValueError, match="env.opponent"):
            _cfg(env={"opponent": "bot", "opponent_kwargs": {"max_actions": 5}})
