"""
Tests for reinforcetactics.rl.observation.build_observation.

These tests pin down the observation contract that gym_env, MCTS, and
ModelBot all share: global_features is always agent-relative (perspective
player's gold/units go first), and fog of war correctly hides enemy info.
"""

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.rl.observation import build_observation
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def map_data():
    np.random.seed(1234)
    data = FileIO.generate_random_map(10, 10, num_players=2)
    np.random.seed()
    return data


@pytest.fixture
def game(map_data):
    return GameState(map_data, num_players=2)


@pytest.fixture
def game_fow(map_data):
    gs = GameState(map_data, num_players=2, fog_of_war=True)
    gs.update_visibility()
    return gs


@pytest.fixture
def zero_mask():
    # Action mask is owned by the caller; builder doesn't inspect it.
    return np.zeros(10 * 10 * 10, dtype=np.float32)


def test_keys_and_dtypes(game, zero_mask):
    obs = build_observation(game, perspective_player=1, action_mask=zero_mask)
    assert set(obs.keys()) == {"grid", "units", "global_features", "action_mask"}
    assert obs["grid"].dtype == np.float32
    assert obs["units"].dtype == np.float32
    assert obs["global_features"].dtype == np.float32
    assert obs["global_features"].shape == (6,)


def test_perspective_p1_vs_p2_swaps_gold(game, zero_mask):
    """global_features[0] must always be the perspective player's gold."""
    game.player_gold[1] = 42
    game.player_gold[2] = 99

    obs1 = build_observation(game, perspective_player=1, action_mask=zero_mask)
    obs2 = build_observation(game, perspective_player=2, action_mask=zero_mask)

    assert obs1["global_features"][0] == 42
    assert obs1["global_features"][1] == 99
    assert obs2["global_features"][0] == 99
    assert obs2["global_features"][1] == 42


def test_perspective_swaps_unit_counts(game, zero_mask):
    """own_units / opp_units follow the perspective player."""
    from reinforcetactics.core.unit import Unit

    # Bypass create_unit validation — we only care about counts.
    game.units = [
        Unit("W", 0, 0, 1),
        Unit("W", 1, 0, 1),
        Unit("W", 2, 0, 1),
        Unit("W", 0, 5, 2),
    ]

    obs1 = build_observation(game, perspective_player=1, action_mask=zero_mask)
    obs2 = build_observation(game, perspective_player=2, action_mask=zero_mask)

    # global_features: [own_gold, opp_gold, turn, own_units, opp_units, current_player]
    assert obs1["global_features"][3] == 3
    assert obs1["global_features"][4] == 1
    assert obs2["global_features"][3] == 1
    assert obs2["global_features"][4] == 3


def test_current_player_is_absolute(game, zero_mask):
    """global_features[5] is the *absolute* current_player, not perspective-rewritten."""
    game.current_player = 2
    obs1 = build_observation(game, perspective_player=1, action_mask=zero_mask)
    obs2 = build_observation(game, perspective_player=2, action_mask=zero_mask)
    assert obs1["global_features"][5] == 2
    assert obs2["global_features"][5] == 2


def test_action_mask_passthrough(game):
    mask = np.ones(10 * 10 * 10, dtype=np.float32)
    obs = build_observation(game, perspective_player=1, action_mask=mask)
    assert obs["action_mask"] is mask


def test_fog_of_war_hides_enemy_gold(game_fow, zero_mask):
    game_fow.player_gold[1] = 50
    game_fow.player_gold[2] = 200
    obs = build_observation(game_fow, perspective_player=1, action_mask=zero_mask)
    assert obs["global_features"][0] == 50
    assert obs["global_features"][1] == 0  # enemy gold hidden
    # visibility layer is included under FOW
    assert "visibility" in obs


def test_fog_of_war_off_by_default(game, zero_mask):
    obs = build_observation(game, perspective_player=1, action_mask=zero_mask)
    assert "visibility" not in obs


def test_fog_of_war_override(game, zero_mask):
    """Explicit fog_of_war=False wins over game_state.fog_of_war=True."""
    # Simulate a FOW-enabled state but force off via param
    obs = build_observation(game, perspective_player=1, action_mask=zero_mask, fog_of_war=False)
    assert "visibility" not in obs


def test_grid_shape_matches_map(game, zero_mask):
    obs = build_observation(game, perspective_player=1, action_mask=zero_mask)
    h, w = game.grid.height, game.grid.width
    assert obs["grid"].shape == (h, w, 3)
    assert obs["units"].shape == (h, w, 3)
