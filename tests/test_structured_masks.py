"""
Tests for AlphaStar-style structured action masks.

Validates that ``StrategyGameEnv.structured_action_masks()`` is internally
consistent and matches the flat / per-dimension masks already produced by the
environment.
"""

import numpy as np
import pytest

from reinforcetactics.rl.gym_env import StrategyGameEnv, StructuredActionMasks


@pytest.fixture
def env():
    e = StrategyGameEnv(map_file=None, opponent="random", render_mode=None)
    e.reset(seed=0)
    yield e
    e.close()


def _flat_legal_tuples(env: StrategyGameEnv):
    """Return the set of legal (atype, ut, sx, sy, tx, ty) tuples from
    ``_build_flat_actions``, which is the existing source of truth."""
    env._build_flat_actions()
    return {tuple(int(v) for v in a) for a in env._current_actions}


class TestStructuredMasksShape:
    def test_returns_dataclass(self, env):
        masks = env.structured_action_masks()
        assert isinstance(masks, StructuredActionMasks)

    def test_atype_shape(self, env):
        masks = env.structured_action_masks()
        assert masks.atype.shape == (10,)
        assert masks.atype.dtype == np.bool_

    def test_source_shape(self, env):
        masks = env.structured_action_masks()
        assert masks.source.shape == (10, env.grid_height, env.grid_width)
        assert masks.source.dtype == np.bool_

    def test_target_arrays_are_grid_shaped(self, env):
        masks = env.structured_action_masks()
        for arr in masks.target.values():
            assert arr.shape == (env.grid_height, env.grid_width)
            assert arr.dtype == np.bool_

    def test_unit_type_arrays_are_unit_sized(self, env):
        masks = env.structured_action_masks()
        for arr in masks.unit_type.values():
            assert arr.shape == (8,)
            assert arr.dtype == np.bool_


class TestStructuredMasksInvariants:
    def test_end_turn_always_legal(self, env):
        masks = env.structured_action_masks()
        assert masks.atype[5]
        assert masks.source[5, 0, 0]
        assert masks.target[(5, 0, 0)][0, 0]

    def test_atype_implies_source(self, env):
        masks = env.structured_action_masks()
        for at in range(10):
            if masks.atype[at]:
                assert masks.source[at].any(), f"atype {at} legal but no legal source"

    def test_source_implies_target(self, env):
        masks = env.structured_action_masks()
        for at in range(10):
            ys, xs = np.where(masks.source[at])
            for y, x in zip(ys.tolist(), xs.tolist()):
                key = (at, x, y)
                assert key in masks.target, f"missing target mask for {key}"
                assert masks.target[key].any()

    def test_target_keys_match_source_bits(self, env):
        masks = env.structured_action_masks()
        for at, sx, sy in masks.target.keys():
            assert masks.source[at, sy, sx], f"target key {(at, sx, sy)} has no source bit"


class TestStructuredVsFlatConsistency:
    def test_every_flat_action_appears_in_structured(self, env):
        masks = env.structured_action_masks()
        for at, ut, sx, sy, tx, ty in _flat_legal_tuples(env):
            assert masks.atype[at]
            assert masks.source[at, sy, sx]
            assert masks.target[(at, sx, sy)][ty, tx]
            if at == 0:  # create_unit -> unit_type must be legal
                assert masks.unit_type[(sx, sy)][ut]

    def test_structured_does_not_overgenerate(self, env):
        """Every (atype, sx, sy, tx, ty) implied by the structured masks must
        correspond to at least one legal flat action (modulo unit_type, which
        only matters for create_unit)."""
        masks = env.structured_action_masks()
        flat = _flat_legal_tuples(env)
        flat_no_ut = {(at, sx, sy, tx, ty) for (at, _, sx, sy, tx, ty) in flat}
        for (at, sx, sy), tmask in masks.target.items():
            ys, xs = np.where(tmask)
            for y, x in zip(ys.tolist(), xs.tolist()):
                assert (at, sx, sy, x, y) in flat_no_ut, f"structured implies illegal action {(at, sx, sy, x, y)}"

    def test_unit_type_keys_subset_of_create_sources(self, env):
        masks = env.structured_action_masks()
        create_sources = set()
        ys, xs = np.where(masks.source[0])
        for y, x in zip(ys.tolist(), xs.tolist()):
            create_sources.add((x, y))
        for key in masks.unit_type.keys():
            assert key in create_sources


class TestEncodeStructuredAction:
    def test_round_trip_through_step(self, env):
        # End turn is always available; verify the encoded action is accepted.
        action = StrategyGameEnv.encode_structured_action(atype=5, sx=0, sy=0, tx=0, ty=0, unit_type_idx=0)
        assert action.shape == (6,)
        assert action.dtype == np.int32
        obs, reward, terminated, truncated, info = env.step(action)
        # The opponent's reply or game end can change state; the only thing we
        # assert is that the step function processed our action as valid
        # (end_turn is unconditionally legal).
        assert info["valid_action"] is True
        assert info["action_type"] == 5
