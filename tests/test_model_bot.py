"""Tests for ModelBot's SB3 checkpoint inference path.

Regression suite for the bug where production checkpoints could not be
played outside training: ``flat_discrete`` (Discrete) checkpoints returned a
scalar index that ``_execute_action`` rejected as an invalid format (the bot
ended its turn immediately every turn), padded-curriculum checkpoints failed
SB3's obs-shape check on every predict, and MaskablePPO checkpoints were
queried without action masks.

The feudal (.pt) loading path is covered separately in
``test_model_bot_feudal.py``.
"""

import numpy as np
import pytest
from gymnasium import spaces

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.model_bot import ModelBot
from reinforcetactics.rl.gym_env import StrategyGameEnv, build_flat_actions
from reinforcetactics.rl.observation import (
    GLOBAL_FEATURES_DIM,
    GRID_CHANNELS,
    UNIT_CHANNELS,
)
from reinforcetactics.utils.file_io import FileIO

BEGINNER_MAP = "maps/1v1/beginner.csv"


# ==============================================================================
# HELPERS
# ==============================================================================


def _make_game_state(fog_of_war: bool = False) -> GameState:
    """6x6 game with a building for each player so create_unit is legal."""
    map_data = np.array([["p" for _ in range(6)] for _ in range(6)], dtype=object)
    map_data[0][0] = "h_1"
    map_data[1][1] = "b_1"
    map_data[5][5] = "h_2"
    map_data[4][4] = "b_2"
    return GameState(map_data, num_players=2, fog_of_war=fog_of_war)


def _obs_space(h: int, w: int, fog: bool = False, grid_channels: int = GRID_CHANNELS) -> spaces.Dict:
    """Observation space mirroring StrategyGameEnv's contract at (h, w)."""
    d = {
        "grid": spaces.Box(low=0.0, high=1.0, shape=(h, w, grid_channels), dtype=np.float32),
        "units": spaces.Box(low=0.0, high=1.0, shape=(h, w, UNIT_CHANNELS), dtype=np.float32),
        "global_features": spaces.Box(low=0.0, high=1.0, shape=(GLOBAL_FEATURES_DIM,), dtype=np.float32),
    }
    if fog:
        d["visibility"] = spaces.Box(low=0, high=2, shape=(h, w), dtype=np.uint8)
    return spaces.Dict(d)


class _RecordingMaskedModel:
    """Duck-typed SB3 stand-in whose ``predict`` accepts ``action_masks``.

    Plays scripted actions first, then defaults to the last legal flat index
    (end_turn -- build_flat_actions appends it last) or an end_turn 6-vector.
    """

    def __init__(self, obs_space, act_space, script=None):
        self.observation_space = obs_space
        self.action_space = act_space
        self.script = list(script or [])
        self.predict_calls = []

    def predict(self, obs, deterministic=True, action_masks=None):
        self.predict_calls.append({"obs": obs, "deterministic": deterministic, "action_masks": action_masks})
        if self.script:
            return self.script.pop(0), None
        if isinstance(self.action_space, spaces.Discrete):
            if action_masks is not None:
                legal = np.flatnonzero(np.asarray(action_masks))
                if legal.size:
                    return np.int64(legal[-1]), None
            return np.int64(0), None
        return np.array([5, 0, 0, 0, 0, 0]), None


class _RecordingUnmaskedModel:
    """Stand-in for plain PPO/A2C/DQN: ``predict`` has no ``action_masks``."""

    def __init__(self, obs_space, act_space, script=None):
        self.observation_space = obs_space
        self.action_space = act_space
        self.script = list(script or [])
        self.predict_calls = []

    def predict(self, obs, deterministic=True):
        self.predict_calls.append({"obs": obs, "deterministic": deterministic})
        if self.script:
            return self.script.pop(0), None
        if isinstance(self.action_space, spaces.Discrete):
            return np.int64(0), None
        return np.array([5, 0, 0, 0, 0, 0]), None


def _bot_with_stub(gs: GameState, stub) -> ModelBot:
    """Build a ModelBot around an injected stub model, mirroring _load_model."""
    bot = ModelBot(gs, player=2)
    bot.model = stub
    bot._configure_from_sb3_model()
    return bot


# ==============================================================================
# CONFIGURATION FROM CHECKPOINT SPACES
# ==============================================================================


class TestConfigureFromSb3Model:
    def test_flat_mode_detected_from_discrete_action_space(self):
        gs = _make_game_state()
        bot = _bot_with_stub(gs, _RecordingMaskedModel(_obs_space(6, 6), spaces.Discrete(64)))
        assert bot._action_mode == "flat"
        assert bot._max_flat_actions == 64
        assert bot._pad_to is None
        assert bot._accepts_action_masks is True

    def test_multidiscrete_mode_detected(self):
        gs = _make_game_state()
        bot = _bot_with_stub(gs, _RecordingMaskedModel(_obs_space(6, 6), spaces.MultiDiscrete([10, 8, 6, 6, 6, 6])))
        assert bot._action_mode == "multi_discrete"

    def test_padded_obs_space_sets_pad_to(self):
        gs = _make_game_state()
        bot = _bot_with_stub(gs, _RecordingMaskedModel(_obs_space(8, 9), spaces.Discrete(64)))
        assert bot._pad_to == (8, 9)

    def test_unmasked_model_detected(self):
        gs = _make_game_state()
        bot = _bot_with_stub(gs, _RecordingUnmaskedModel(_obs_space(6, 6), spaces.Discrete(64)))
        assert bot._accepts_action_masks is False

    def test_checkpoint_smaller_than_map_raises(self):
        gs = _make_game_state()
        with pytest.raises(ValueError, match="cannot see the whole board"):
            _bot_with_stub(gs, _RecordingMaskedModel(_obs_space(4, 6), spaces.Discrete(64)))

    def test_multidiscrete_grid_mismatch_raises(self):
        gs = _make_game_state()
        with pytest.raises(ValueError, match="grid"):
            _bot_with_stub(gs, _RecordingMaskedModel(_obs_space(9, 9), spaces.MultiDiscrete([10, 8, 9, 9, 9, 9])))

    def test_visibility_mismatch_raises(self):
        gs = _make_game_state(fog_of_war=False)
        with pytest.raises(ValueError, match="fog-of-war"):
            _bot_with_stub(gs, _RecordingMaskedModel(_obs_space(6, 6, fog=True), spaces.Discrete(64)))

    def test_stale_obs_schema_raises(self):
        gs = _make_game_state()
        with pytest.raises(ValueError, match="observation schema"):
            _bot_with_stub(gs, _RecordingMaskedModel(_obs_space(6, 6, grid_channels=GRID_CHANNELS + 1), spaces.Discrete(64)))


# ==============================================================================
# FLAT (Discrete) INFERENCE
# ==============================================================================


class TestFlatInference:
    def test_predict_receives_exact_flat_mask(self):
        gs = _make_game_state()
        gs.end_turn()  # player 2's turn
        gs.player_gold[2] = 1000
        n_legal = len(build_flat_actions(gs, 2, 64))

        stub = _RecordingMaskedModel(_obs_space(6, 6), spaces.Discrete(64))
        bot = _bot_with_stub(gs, stub)
        bot.take_turn()

        mask = stub.predict_calls[0]["action_masks"]
        assert mask is not None
        assert mask.dtype == np.bool_
        assert mask.shape == (64,)
        assert mask[:n_legal].all()
        assert not mask[n_legal:].any()

    def test_scripted_index_decodes_through_flat_table(self):
        # The scripted index points at a create_unit entry in the decode
        # table; pre-fix the scalar action was rejected as an invalid
        # format and the bot ended its turn having done nothing.
        gs = _make_game_state()
        gs.end_turn()
        gs.player_gold[2] = 1000
        table = build_flat_actions(gs, 2, 64)
        create_idx = next(i for i, a in enumerate(table) if int(a[0]) == 0)

        stub = _RecordingMaskedModel(_obs_space(6, 6), spaces.Discrete(64), script=[np.int64(create_idx)])
        bot = _bot_with_stub(gs, stub)
        bot.take_turn()

        assert any(u.player == 2 for u in gs.units), "scripted create_unit index should have spawned a unit"
        assert gs.current_player == 1, "turn must return to the other player"

    def test_out_of_range_index_falls_back_to_end_turn(self):
        # Plain (unmasked) models can emit any index; out-of-range must
        # decode to end_turn, mirroring StrategyGameEnv.step's fallback.
        gs = _make_game_state()
        gs.end_turn()

        stub = _RecordingUnmaskedModel(_obs_space(6, 6), spaces.Discrete(64), script=[np.int64(63)])
        bot = _bot_with_stub(gs, stub)
        bot.take_turn()

        assert gs.current_player == 1
        assert "action_masks" not in stub.predict_calls[0]

    def test_padded_checkpoint_gets_padded_observation(self):
        gs = _make_game_state()
        gs.end_turn()

        stub = _RecordingMaskedModel(_obs_space(8, 9), spaces.Discrete(64))
        bot = _bot_with_stub(gs, stub)
        bot.take_turn()

        obs = stub.predict_calls[0]["obs"]
        assert obs["grid"].shape == (8, 9, GRID_CHANNELS)
        assert obs["units"].shape == (8, 9, UNIT_CHANNELS)
        # Live 6x6 content sits in the top-left; pad rows/cols carry the
        # all-zero tile-type signature build_observation promises.
        assert not obs["grid"][6:, :, :].any()
        assert not obs["grid"][:, 6:, :].any()


# ==============================================================================
# MULTIDISCRETE INFERENCE
# ==============================================================================


class TestMultiDiscreteInference:
    def test_predict_receives_concatenated_per_dim_masks(self):
        gs = _make_game_state()
        gs.end_turn()
        gs.player_gold[2] = 1000

        stub = _RecordingMaskedModel(
            _obs_space(6, 6),
            spaces.MultiDiscrete([10, 8, 6, 6, 6, 6]),
            script=[np.array([5, 0, 0, 0, 0, 0])],
        )
        bot = _bot_with_stub(gs, stub)
        bot.take_turn()

        mask = stub.predict_calls[0]["action_masks"]
        assert mask is not None
        assert mask.dtype == np.bool_
        assert mask.shape == (10 + 8 + 6 + 6 + 6 + 6,)
        assert mask[5], "end_turn action type must always be legal"
        assert mask[0], "create_unit should be legal (own building + gold)"

    def test_unmasked_multidiscrete_predict_has_no_masks(self):
        gs = _make_game_state()
        gs.end_turn()

        stub = _RecordingUnmaskedModel(
            _obs_space(6, 6),
            spaces.MultiDiscrete([10, 8, 6, 6, 6, 6]),
            script=[np.array([5, 0, 0, 0, 0, 0])],
        )
        bot = _bot_with_stub(gs, stub)
        bot.take_turn()

        assert "action_masks" not in stub.predict_calls[0]
        assert gs.current_player == 1


# ==============================================================================
# SHARED FLAT-ACTION TABLE (env <-> free function refactor guard)
# ==============================================================================


class TestBuildFlatActionsSharedContract:
    def test_env_flat_actions_match_free_function(self):
        env = StrategyGameEnv(
            map_file=BEGINNER_MAP,
            opponent=None,
            render_mode=None,
            action_space_type="flat_discrete",
            max_flat_actions=64,
        )
        env.reset(seed=3)
        env._build_flat_actions()
        expected = build_flat_actions(env.game_state, env.agent_player, 64)
        assert len(env._current_actions) == len(expected)
        assert all(np.array_equal(a, b) for a, b in zip(env._current_actions, expected))
        env.close()


# ==============================================================================
# END-TO-END WITH A REAL MASKABLEPPO CHECKPOINT
# ==============================================================================


class TestRealCheckpointIntegration:
    def test_flat_padded_maskableppo_checkpoint_plays_a_turn(self, tmp_path, caplog):
        # The production configuration: MaskablePPO + flat_discrete + padded
        # observations (a curriculum spanning multiple map sizes always
        # pads). Untrained weights are fine -- the assertions are about the
        # encode/decode contract, not playing strength.
        sb3_contrib = pytest.importorskip("sb3_contrib")
        from reinforcetactics.rl.masking import make_maskable_env

        env = make_maskable_env(
            map_file=BEGINNER_MAP,
            opponent="noop",
            action_space_type="flat_discrete",
            max_flat_actions=64,
            pad_to_size=(8, 8),
            enabled_units=["W"],
            seed=0,
        )
        model = sb3_contrib.MaskablePPO(
            "MultiInputPolicy",
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            policy_kwargs={"net_arch": [16]},
            seed=0,
            device="cpu",
        )
        ckpt = tmp_path / "flat_padded.zip"
        model.save(str(ckpt))
        env.close()

        gs = GameState(FileIO.load_map(BEGINNER_MAP), num_players=2, enabled_units=["W"])
        bot = ModelBot(gs, player=2, model_path=str(ckpt))
        assert bot._action_mode == "flat"
        assert bot._max_flat_actions == 64
        assert bot._pad_to == (8, 8)
        assert bot._accepts_action_masks is True

        gs.end_turn()  # hand play to the bot
        with caplog.at_level("WARNING", logger="reinforcetactics.game.model_bot"):
            bot.take_turn()

        assert gs.current_player == 1 or gs.game_over
        # Pre-fix signature: every predict produced "Invalid action format".
        assert not any("Invalid action format" in rec.message for rec in caplog.records)
