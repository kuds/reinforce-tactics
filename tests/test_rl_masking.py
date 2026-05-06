"""Tests for reinforcetactics.rl.masking targeting previously uncovered paths."""

import numpy as np
import pytest

from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.masking import (
    _CURRICULUM_PRESETS,
    ActionMaskedEnv,
    make_curriculum_env,
    make_maskable_env,
    make_maskable_vec_env,
    validate_action_mask,
)


class TestActionMaskedEnvBasics:
    def test_delegates_attribute_access(self):
        base = StrategyGameEnv(opponent="random", render_mode=None)
        wrapped = ActionMaskedEnv(base)
        # grid_width is on the base env; wrapper should pass through
        assert wrapped.grid_width == base.grid_width
        assert wrapped.grid_height == base.grid_height
        wrapped.close()

    def test_stats_disabled_returns_empty_dict(self):
        base = StrategyGameEnv(opponent="random", render_mode=None)
        wrapped = ActionMaskedEnv(base, track_stats=False)
        wrapped.reset()
        assert wrapped.get_masking_stats() == {}
        wrapped.close()

    def test_stats_track_percentages(self):
        base = StrategyGameEnv(opponent="random", render_mode=None)
        wrapped = ActionMaskedEnv(base, track_stats=True)
        wrapped.reset()
        # Send two end_turn actions (action_type=5)
        for _ in range(2):
            wrapped.step(np.array([5, 0, 0, 0, 0, 0]))
        stats = wrapped.get_masking_stats()
        assert stats["total_actions"] == 2
        assert "action_type_percentages" in stats
        # index 5 = end_turn => 100%
        assert stats["action_type_percentages"][5] == pytest.approx(100.0)
        wrapped.close()


class TestMakeMaskableEnv:
    def test_default_construction(self):
        env = make_maskable_env(opponent="random")
        env.reset()
        masks = env.action_masks()
        assert isinstance(masks, np.ndarray)
        assert masks.dtype == np.bool_
        env.close()

    def test_flat_discrete_mode(self):
        env = make_maskable_env(
            opponent="random",
            action_space_type="flat_discrete",
            max_flat_actions=128,
        )
        env.reset()
        # In flat mode, action_masks() still returns a 1D bool array
        masks = env.action_masks()
        assert masks.ndim == 1
        env.close()

    def test_max_turns_threaded_to_underlying_env(self):
        """make_maskable_env should forward max_turns to StrategyGameEnv
        and onto the underlying GameState."""
        wrapped = make_maskable_env(opponent="random", max_turns=11)
        # ActionMaskedEnv delegates attr access to the wrapped StrategyGameEnv
        assert wrapped.max_turns == 11
        assert wrapped.game_state.max_turns == 11
        wrapped.close()

    def test_max_turns_default_none(self):
        wrapped = make_maskable_env(opponent="random")
        assert wrapped.max_turns is None
        assert wrapped.game_state.max_turns is None
        wrapped.close()


class TestMakeMaskableVecEnv:
    def test_dummy_vec_env_single(self):
        vec = make_maskable_vec_env(n_envs=1, opponent="random", use_subprocess=False)
        obs = vec.reset()
        assert obs is not None
        vec.close()

    def test_dummy_vec_env_multiple(self):
        vec = make_maskable_vec_env(n_envs=2, opponent="random", use_subprocess=False)
        obs = vec.reset()
        assert obs is not None
        vec.close()

    def test_max_turns_threaded_to_each_subenv(self):
        """make_maskable_vec_env should forward max_turns to every sub-env."""
        vec = make_maskable_vec_env(
            n_envs=2,
            opponent="random",
            use_subprocess=False,
            max_turns=9,
        )
        # DummyVecEnv exposes attrs across sub-envs via env_method / get_attr.
        # Use a direct attribute fetch via the .envs list (DummyVecEnv only).
        for sub in vec.envs:
            # sub is the ActionMaskedEnv wrapper around StrategyGameEnv
            assert sub.max_turns == 9
            assert sub.game_state.max_turns == 9
        vec.close()


class TestValidateActionMask:
    def test_fresh_game_is_valid(self):
        env = StrategyGameEnv(opponent="random", render_mode=None)
        env.reset()
        result = validate_action_mask(env)
        assert result["valid"] is True
        assert result["errors"] == []
        assert "mask_summary" in result
        # All named action types should appear
        expected = {
            "create",
            "move",
            "attack",
            "seize",
            "heal",
            "end_turn",
            "paralyze",
            "haste",
            "defence_buff",
            "attack_buff",
        }
        assert set(result["mask_summary"].keys()) == expected
        env.close()

    def test_end_turn_always_has_legal_actions(self):
        env = StrategyGameEnv(opponent="random", render_mode=None)
        env.reset()
        result = validate_action_mask(env)
        assert result["mask_summary"]["end_turn"]["has_legal_actions"] is True
        env.close()


class TestCurriculum:
    @pytest.mark.parametrize("difficulty", _CURRICULUM_PRESETS)
    def test_builtin_presets_load(self, difficulty):
        env = make_curriculum_env(difficulty=difficulty)
        env.reset()
        env.close()

    def test_unknown_difficulty_raises(self):
        with pytest.raises(ValueError, match="Unknown curriculum difficulty"):
            make_curriculum_env(difficulty="does_not_exist")

    def test_kwargs_override_preset(self):
        # max_steps from kwargs should override whatever is in the YAML preset.
        env = make_curriculum_env(difficulty="easy", max_steps=77)
        assert env.env.max_steps == 77
        env.close()
