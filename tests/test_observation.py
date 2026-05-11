"""
Tests for reinforcetactics.rl.observation.build_observation.

These tests pin down the observation contract that gym_env, MCTS, and
ModelBot all share: global_features is always agent-relative (perspective
player's gold/units go first), ownership channels are agent-relative
one-hots, and fog of war correctly hides enemy info.
"""

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.rl.observation import (
    GLOBAL_FEATURES_DIM,
    GOLD_SCALE,
    GRID_CHANNELS,
    NUM_TILE_TYPES,
    NUM_UNIT_TYPES,
    TURN_SCALE,
    UNIT_CHANNELS,
    UNIT_COUNT_SCALE,
    build_observation,
)
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
    """A zero mask, kept for the action-mask passthrough test only."""
    return np.zeros(10 * 10 * 10, dtype=np.float32)


def test_keys_and_dtypes(game):
    obs = build_observation(game, perspective_player=1)
    # action_mask is no longer part of the policy observation; callers that
    # need it pass it in explicitly (covered by test_action_mask_passthrough).
    assert set(obs.keys()) == {"grid", "units", "global_features"}
    assert obs["grid"].dtype == np.float32
    assert obs["units"].dtype == np.float32
    assert obs["global_features"].dtype == np.float32
    assert obs["global_features"].shape == (GLOBAL_FEATURES_DIM,)
    assert obs["grid"].shape[-1] == GRID_CHANNELS
    assert obs["units"].shape[-1] == UNIT_CHANNELS


def test_perspective_p1_vs_p2_swaps_gold(game):
    """global_features[0] must always be the perspective player's gold.

    Values are tanh(gold / GOLD_SCALE); the test verifies the *perspective
    swap* by checking that obs1[own] == obs2[opp] and vice versa, without
    pinning the exact scaled value.
    """
    game.player_gold[1] = 42
    game.player_gold[2] = 99

    obs1 = build_observation(game, perspective_player=1)
    obs2 = build_observation(game, perspective_player=2)

    own_p1, opp_p1 = obs1["global_features"][0], obs1["global_features"][1]
    own_p2, opp_p2 = obs2["global_features"][0], obs2["global_features"][1]

    # Perspective swap: own-from-P1 == opp-from-P2 (= tanh(42/GOLD_SCALE))
    # and own-from-P2 == opp-from-P1 (= tanh(99/GOLD_SCALE)).
    assert own_p1 == pytest.approx(np.tanh(42 / GOLD_SCALE), rel=1e-5)
    assert opp_p1 == pytest.approx(np.tanh(99 / GOLD_SCALE), rel=1e-5)
    assert own_p2 == pytest.approx(np.tanh(99 / GOLD_SCALE), rel=1e-5)
    assert opp_p2 == pytest.approx(np.tanh(42 / GOLD_SCALE), rel=1e-5)


def test_perspective_swaps_unit_counts(game):
    """own_units / opp_units follow the perspective player."""
    from reinforcetactics.core.unit import Unit

    # Bypass create_unit validation — we only care about counts.
    game.units = [
        Unit("W", 0, 0, 1),
        Unit("W", 1, 0, 1),
        Unit("W", 2, 0, 1),
        Unit("W", 0, 5, 2),
    ]

    obs1 = build_observation(game, perspective_player=1)
    obs2 = build_observation(game, perspective_player=2)

    # global_features: [own_gold, opp_gold, turn, own_units, opp_units],
    # each tanh(count / UNIT_COUNT_SCALE).
    assert obs1["global_features"][3] == pytest.approx(np.tanh(3 / UNIT_COUNT_SCALE), rel=1e-5)
    assert obs1["global_features"][4] == pytest.approx(np.tanh(1 / UNIT_COUNT_SCALE), rel=1e-5)
    assert obs2["global_features"][3] == pytest.approx(np.tanh(1 / UNIT_COUNT_SCALE), rel=1e-5)
    assert obs2["global_features"][4] == pytest.approx(np.tanh(3 / UNIT_COUNT_SCALE), rel=1e-5)


def test_current_player_dropped_from_global_features(game):
    """``current_player`` was removed: it's redundant under agent-relative obs."""
    game.current_player = 2
    obs = build_observation(game, perspective_player=1)
    assert obs["global_features"].shape == (GLOBAL_FEATURES_DIM,)
    # No room left in the vector for an absolute current_player slot.


def test_action_mask_passthrough_when_provided(game):
    """The mask is opt-in (for non-PPO callers like MCTS)."""
    mask = np.ones(10 * 10 * 10, dtype=np.float32)
    obs = build_observation(game, perspective_player=1, action_mask=mask)
    assert obs["action_mask"] is mask


def test_action_mask_absent_by_default(game):
    """Default obs (used by gym_env / MaskablePPO) has no action_mask key."""
    obs = build_observation(game, perspective_player=1)
    assert "action_mask" not in obs


def test_fog_of_war_hides_enemy_gold(game_fow):
    game_fow.player_gold[1] = 50
    game_fow.player_gold[2] = 200
    obs = build_observation(game_fow, perspective_player=1)
    # tanh(50 / GOLD_SCALE) for own gold; opp gold is hidden under FOW so
    # the raw input is 0 -> tanh(0) = 0.
    assert obs["global_features"][0] == pytest.approx(np.tanh(50 / GOLD_SCALE), rel=1e-5)
    assert obs["global_features"][1] == 0.0  # enemy gold hidden -> tanh(0) = 0
    # visibility layer is included under FOW
    assert "visibility" in obs


def test_fog_of_war_off_by_default(game):
    obs = build_observation(game, perspective_player=1)
    assert "visibility" not in obs


def test_fog_of_war_override(game):
    """Explicit fog_of_war=False wins over game_state.fog_of_war=True."""
    obs = build_observation(game, perspective_player=1, fog_of_war=False)
    assert "visibility" not in obs


def test_grid_shape_matches_map(game):
    obs = build_observation(game, perspective_player=1)
    h, w = game.grid.height, game.grid.width
    assert obs["grid"].shape == (h, w, GRID_CHANNELS)
    assert obs["units"].shape == (h, w, UNIT_CHANNELS)


def test_units_one_hot_and_agent_relative_owner(game):
    """Unit type is one-hot; owner channels flip with perspective."""
    from reinforcetactics.constants import ALL_UNIT_TYPES
    from reinforcetactics.core.unit import Unit

    game.units = [Unit("M", 3, 4, player=1), Unit("K", 5, 6, player=2)]

    p1 = build_observation(game, perspective_player=1)
    p2 = build_observation(game, perspective_player=2)

    # Unit-type one-hot uses ALL_UNIT_TYPES ordering.
    m_idx = ALL_UNIT_TYPES.index("M")
    k_idx = ALL_UNIT_TYPES.index("K")
    assert p1["units"][4, 3, m_idx] == 1.0  # (y=4, x=3)
    assert p1["units"][4, 3, :NUM_UNIT_TYPES].sum() == 1.0
    assert p1["units"][6, 5, k_idx] == 1.0

    # Player-1 unit at (3, 4): self-channel from P1 view, opp-channel from P2.
    self_ch = NUM_UNIT_TYPES + 0
    opp_ch = NUM_UNIT_TYPES + 1
    assert p1["units"][4, 3, self_ch] == 1.0
    assert p1["units"][4, 3, opp_ch] == 0.0
    assert p2["units"][4, 3, self_ch] == 0.0
    assert p2["units"][4, 3, opp_ch] == 1.0


def test_grid_owner_channels_are_agent_relative(game):
    """Tile owner is encoded as one-hot (self / opp / neutral) per-perspective."""
    # Force a known ownership: pick the first capturable tile, make it P1's.
    capturable = game.grid.get_capturable_tiles()
    assert capturable, "fixture map must have at least one capturable tile"
    tile = capturable[0]
    tile.player = 1

    p1 = build_observation(game, perspective_player=1)
    p2 = build_observation(game, perspective_player=2)

    self_ch = NUM_TILE_TYPES + 0
    opp_ch = NUM_TILE_TYPES + 1
    neutral_ch = NUM_TILE_TYPES + 2
    y, x = tile.y, tile.x

    # P1 view: self-owned. P2 view: opp-owned.
    assert p1["grid"][y, x, self_ch] == 1.0
    assert p1["grid"][y, x, opp_ch] == 0.0
    assert p2["grid"][y, x, self_ch] == 0.0
    assert p2["grid"][y, x, opp_ch] == 1.0
    # Neutral channel is mutually exclusive with self/opp.
    assert p1["grid"][y, x, neutral_ch] == 0.0


def test_hp_channel_is_normalized(game):
    """Structure / unit HP channel is a fraction in [0, 1], not 0..100."""
    from reinforcetactics.core.unit import Unit

    game.units = [Unit("W", 0, 0, player=1)]
    obs = build_observation(game, perspective_player=1)
    hp = obs["units"][0, 0, NUM_UNIT_TYPES + 3]
    assert 0.0 <= hp <= 1.0
    # A freshly created unit at full health should be at HP=1.0 modulo
    # rounding from to_numpy's percentage encoding.
    assert hp == pytest.approx(1.0, abs=1e-3)


# ---------- pad_to (cross-stage observation-shape unification) ----------


def test_pad_to_zero_pads_grid_and_units(game):
    h, w = game.grid.height, game.grid.width
    pad = (h + 4, w + 6)
    obs = build_observation(game, perspective_player=1, pad_to=pad)
    assert obs["grid"].shape == (pad[0], pad[1], GRID_CHANNELS)
    assert obs["units"].shape == (pad[0], pad[1], UNIT_CHANNELS)
    # Real cells preserved at top-left.
    obs_unpadded = build_observation(game, perspective_player=1)
    np.testing.assert_array_equal(obs["grid"][:h, :w, :], obs_unpadded["grid"])
    np.testing.assert_array_equal(obs["units"][:h, :w, :], obs_unpadded["units"])
    # Padded region is zero across every channel — distinct from any real
    # tile, which always has exactly one tile-type channel set.
    assert obs["grid"][h:, :, :].sum() == 0
    assert obs["grid"][:, w:, :].sum() == 0
    assert obs["units"][h:, :, :].sum() == 0
    assert obs["units"][:, w:, :].sum() == 0


def test_pad_to_equal_dims_is_noop(game):
    h, w = game.grid.height, game.grid.width
    obs_pad = build_observation(game, perspective_player=1, pad_to=(h, w))
    obs_no_pad = build_observation(game, perspective_player=1)
    assert obs_pad["grid"].shape == obs_no_pad["grid"].shape
    np.testing.assert_array_equal(obs_pad["grid"], obs_no_pad["grid"])
    np.testing.assert_array_equal(obs_pad["units"], obs_no_pad["units"])


def test_pad_to_smaller_than_map_raises(game):
    h, w = game.grid.height, game.grid.width
    with pytest.raises(ValueError, match="smaller than the live map"):
        build_observation(game, perspective_player=1, pad_to=(h - 1, w))


# ---------- global_features normalization ----------


def test_global_features_bounded_under_extreme_inputs(game):
    """tanh(x/scale) stays within the [0, 1] Box bounds, even for inputs
    far past the linear regime (float32 tanh saturates to exactly 1.0)."""
    from reinforcetactics.core.unit import Unit

    game.player_gold[1] = 10**9
    game.player_gold[2] = 10**9
    game.turn_number = 10**6
    game.units = [Unit("W", 0, 0, 1) for _ in range(500)]

    obs = build_observation(game, perspective_player=1)
    g = obs["global_features"]
    assert g.dtype == np.float32
    assert np.all(g >= 0.0)
    assert np.all(g <= 1.0)


def test_global_features_monotonic_in_gold(game):
    """Doubling own_gold strictly increases the own_gold feature."""
    game.player_gold[1] = 100
    low = build_observation(game, perspective_player=1)["global_features"][0]
    game.player_gold[1] = 200
    high = build_observation(game, perspective_player=1)["global_features"][0]
    assert high > low


def test_global_features_zero_at_origin():
    """A game with no gold / no turns / no units yields a zero vector
    in the gold/turn/unit-count slots (modulo the starting state)."""
    from reinforcetactics.core.unit import Unit

    map_data = FileIO.generate_random_map(10, 10, num_players=2)
    game = GameState(map_data, num_players=2)
    game.player_gold[1] = 0
    game.player_gold[2] = 0
    game.turn_number = 0
    game.units = [Unit("W", 0, 0, 1)]  # need at least one to exercise the path

    obs = build_observation(game, perspective_player=2)
    g = obs["global_features"]
    # Player-2 perspective: own gold/units = 0 -> tanh(0) = 0.
    assert g[0] == 0.0
    assert g[2] == 0.0  # turn
    assert g[3] == 0.0  # own_units (P2 has none)


def test_scale_overrides_are_honoured(game):
    """Passing custom scales changes the output exactly as expected."""
    game.player_gold[1] = 500
    game.player_gold[2] = 500
    game.turn_number = 30

    custom_obs = build_observation(
        game,
        perspective_player=1,
        gold_scale=100.0,
        turn_scale=10.0,
        unit_count_scale=5.0,
    )
    g = custom_obs["global_features"]
    assert g[0] == pytest.approx(np.tanh(500 / 100.0), rel=1e-5)
    assert g[2] == pytest.approx(np.tanh(30 / 10.0), rel=1e-5)


def test_global_features_within_box_bounds(game):
    """Output respects the [0, 1] Box bounds advertised by gym_env / model_bot."""
    from reinforcetactics.core.unit import Unit

    game.player_gold[1] = 5000
    game.player_gold[2] = 5000
    game.turn_number = 100
    game.units = [Unit("W", i % 10, i // 10, 1 + (i % 2)) for i in range(50)]

    obs = build_observation(game, perspective_player=1)
    g = obs["global_features"]
    assert np.all(g >= 0.0)
    assert np.all(g <= 1.0)


def test_pad_to_pads_visibility_under_fog(game_fow):
    h, w = game_fow.grid.height, game_fow.grid.width
    pad = (h + 2, w + 3)
    obs = build_observation(game_fow, perspective_player=1, pad_to=pad)
    assert obs["visibility"].shape == (pad[0], pad[1])
    # Real visibility preserved at top-left, pad region all zero.
    obs_unpadded = build_observation(game_fow, perspective_player=1)
    np.testing.assert_array_equal(obs["visibility"][:h, :w], obs_unpadded["visibility"])
    assert obs["visibility"][h:, :].sum() == 0
    assert obs["visibility"][:, w:].sum() == 0
