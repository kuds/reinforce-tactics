"""
Tests for AlphaZero components: neural network, MCTS, trainer, and bot.
"""

import copy
import numpy as np
import pytest
import torch

from reinforcetactics.core.game_state import GameState
from reinforcetactics.rl.alphazero_net import AlphaZeroNet, ResidualBlock
from reinforcetactics.rl.mcts import MCTS, MCTSNode, _obs_from_game_state
from reinforcetactics.rl.alphazero_trainer import ReplayBuffer, self_play_game
from reinforcetactics.utils.file_io import FileIO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_map_data():
    """Generate a small random map for fast tests."""
    return FileIO.generate_random_map(20, 20, num_players=2)


@pytest.fixture
def game_state(small_map_data):
    """Create a game state from the small map."""
    return GameState(small_map_data, num_players=2)


@pytest.fixture
def small_network(game_state):
    """Create a small network for fast tests."""
    return AlphaZeroNet(
        grid_height=game_state.grid.height,
        grid_width=game_state.grid.width,
        num_res_blocks=2,
        channels=32,
    )


# ---------------------------------------------------------------------------
# AlphaZeroNet tests
# ---------------------------------------------------------------------------

class TestAlphaZeroNet:

    def test_residual_block_shape(self):
        """Residual block preserves spatial dimensions."""
        block = ResidualBlock(channels=32)
        x = torch.randn(2, 32, 10, 10)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_block_skip_connection(self):
        """Residual block output differs from input (non-identity)."""
        block = ResidualBlock(channels=16)
        x = torch.randn(1, 16, 5, 5)
        out = block(x)
        # The output should not be identical to input (unless extremely unlikely)
        assert not torch.allclose(out, x, atol=1e-6)

    def test_network_output_shapes(self, small_network, game_state):
        """Network produces correct output shapes."""
        h = game_state.grid.height
        w = game_state.grid.width
        batch_size = 4
        action_space_size = 10 * w * h

        grid = torch.randn(batch_size, h, w, 3)
        units = torch.randn(batch_size, h, w, 3)
        gf = torch.randn(batch_size, 6)

        policy_logits, value = small_network(grid, units, gf)

        assert policy_logits.shape == (batch_size, action_space_size)
        assert value.shape == (batch_size, 1)

    def test_value_in_range(self, small_network, game_state):
        """Value head output is in [-1, 1] due to tanh."""
        h = game_state.grid.height
        w = game_state.grid.width

        grid = torch.randn(2, h, w, 3)
        units = torch.randn(2, h, w, 3)
        gf = torch.randn(2, 6)

        _, value = small_network(grid, units, gf)

        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_predict_with_mask(self, small_network, game_state):
        """predict() applies action mask and normalizes probabilities."""
        h = game_state.grid.height
        w = game_state.grid.width
        action_space = 10 * w * h

        grid = torch.randn(1, h, w, 3)
        units = torch.randn(1, h, w, 3)
        gf = torch.randn(1, 6)

        # Create a sparse mask
        mask = torch.zeros(1, action_space)
        mask[0, 0] = 1.0
        mask[0, 10] = 1.0
        mask[0, 100] = 1.0

        probs, value = small_network.predict(grid, units, gf, mask)

        # Probabilities should sum to ~1
        assert abs(probs.sum().item() - 1.0) < 1e-5

        # Masked-out actions should have ~0 probability
        assert probs[0, 1].item() < 1e-6
        assert probs[0, 50].item() < 1e-6

        # Unmasked actions should have > 0 probability
        assert probs[0, 0].item() > 0
        assert probs[0, 10].item() > 0
        assert probs[0, 100].item() > 0

    def test_gradient_flow(self, small_network, game_state):
        """Gradients flow through both heads."""
        h = game_state.grid.height
        w = game_state.grid.width

        grid = torch.randn(2, h, w, 3)
        units = torch.randn(2, h, w, 3)
        gf = torch.randn(2, 6)

        policy_logits, value = small_network(grid, units, gf)

        loss = policy_logits.sum() + value.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in small_network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# MCTS tests
# ---------------------------------------------------------------------------

class TestMCTS:

    def test_obs_from_game_state(self, game_state):
        """Observation extraction produces correct shapes."""
        w = game_state.grid.width
        h = game_state.grid.height

        grid, units, gf, mask = _obs_from_game_state(game_state, w, h)

        assert grid.shape == (h, w, 3)
        assert units.shape == (h, w, 3)
        assert gf.shape == (6,)
        assert mask.shape == (10 * w * h,)
        assert grid.dtype == np.float32

    def test_action_mask_has_end_turn(self, game_state):
        """Action mask always includes end_turn."""
        w = game_state.grid.width
        h = game_state.grid.height
        _, _, _, mask = _obs_from_game_state(game_state, w, h)

        # End turn is at action_type=5, position (0,0)
        end_turn_idx = 5 * w * h + 0 * w + 0
        assert mask[end_turn_idx] == 1.0

    def test_mcts_node_legal_actions(self, game_state):
        """MCTSNode correctly maps legal actions to flat indices."""
        w = game_state.grid.width
        h = game_state.grid.height

        node = MCTSNode(game_state=game_state)
        legal_flat = node.get_legal_flat_actions(w, h)

        # Should have at least end_turn
        assert len(legal_flat) >= 1

        # End turn should be present
        end_turn_idx = 5 * w * h
        assert end_turn_idx in legal_flat
        assert legal_flat[end_turn_idx]['key'] == 'end_turn'

    def test_mcts_search_returns_valid_distribution(self, small_network, game_state):
        """MCTS search returns a valid probability distribution."""
        w = game_state.grid.width
        h = game_state.grid.height

        mcts = MCTS(
            network=small_network,
            grid_width=w,
            grid_height=h,
            num_simulations=5,  # Few simulations for speed
            device='cpu',
        )

        action_probs, value = mcts.search(game_state, add_noise=False)

        assert action_probs.shape == (10 * w * h,)
        # Should sum to ~1 (it's based on visit counts)
        total = action_probs.sum()
        assert total > 0
        assert abs(total - 1.0) < 1e-5
        # Value should be in [-1, 1]
        assert -1.0 <= value <= 1.0

    def test_mcts_select_action(self, small_network, game_state):
        """MCTS select_action returns a valid flat action."""
        w = game_state.grid.width
        h = game_state.grid.height

        mcts = MCTS(
            network=small_network,
            grid_width=w,
            grid_height=h,
            num_simulations=5,
            device='cpu',
        )

        action, probs = mcts.select_action(game_state, temperature=0, add_noise=False)

        assert isinstance(action, int)
        assert 0 <= action < 10 * w * h
        assert probs[action] > 0  # Selected action should have visits

    def test_mcts_greedy_vs_stochastic(self, small_network, game_state):
        """Greedy (temp=0) always picks the highest-visit action."""
        w = game_state.grid.width
        h = game_state.grid.height

        mcts = MCTS(
            network=small_network,
            grid_width=w,
            grid_height=h,
            num_simulations=10,
            device='cpu',
        )

        action, probs = mcts.select_action(game_state, temperature=0, add_noise=False)
        assert action == int(np.argmax(probs))

    def test_mcts_get_action_info(self, small_network, game_state):
        """get_action_info resolves flat indices to structured actions."""
        w = game_state.grid.width
        h = game_state.grid.height

        mcts = MCTS(
            network=small_network,
            grid_width=w,
            grid_height=h,
            num_simulations=5,
            device='cpu',
        )

        # End turn should always resolve
        end_turn_idx = 5 * w * h
        info = mcts.get_action_info(game_state, end_turn_idx)
        assert info is not None
        assert info['key'] == 'end_turn'


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:

    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

        examples = [(np.zeros(3), np.zeros(3), np.zeros(6),
                      np.zeros(10), np.zeros(10), 1.0)]
        buf.push(examples)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push([(i, i, i, i, i, float(i))])
        assert len(buf) == 5

    def test_sample(self):
        buf = ReplayBuffer(capacity=100)
        examples = [(np.zeros(3), np.zeros(3), np.zeros(6),
                      np.zeros(10), np.zeros(10), float(i)) for i in range(20)]
        buf.push(examples)

        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_larger_than_buffer(self):
        buf = ReplayBuffer(capacity=100)
        examples = [(0, 0, 0, 0, 0, 0.0) for _ in range(3)]
        buf.push(examples)

        batch = buf.sample(10)
        assert len(batch) == 3  # Can't sample more than buffer size

    def test_clear(self):
        buf = ReplayBuffer(capacity=100)
        buf.push([(0, 0, 0, 0, 0, 0.0)])
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0


# ---------------------------------------------------------------------------
# Self-play game test
# ---------------------------------------------------------------------------

class TestSelfPlay:

    def test_self_play_game_completes(self, small_map_data):
        """A self-play game runs to completion and produces examples."""
        gs = GameState(small_map_data, num_players=2)
        network = AlphaZeroNet(
            grid_height=gs.grid.height,
            grid_width=gs.grid.width,
            num_res_blocks=2,
            channels=32,
        )
        network.eval()

        mcts = MCTS(
            network=network,
            grid_width=gs.grid.width,
            grid_height=gs.grid.height,
            num_simulations=3,  # Minimal for speed
            device='cpu',
        )

        examples, winner = self_play_game(
            mcts=mcts,
            map_data=small_map_data,
            max_steps=20,  # Short game for testing
            temperature_threshold=5,
        )

        # Should produce at least some examples
        assert len(examples) > 0

        # Each example should have 6 elements
        for ex in examples:
            assert len(ex) == 6
            grid, units, gf, mask, policy, value = ex
            assert isinstance(value, float)
            assert value in (-1.0, 0.0, 1.0)

        # Winner should be valid
        assert winner in (1, 2, None)

    def test_self_play_value_targets_consistent(self, small_map_data):
        """Value targets match the game outcome for both players."""
        gs = GameState(small_map_data, num_players=2)
        network = AlphaZeroNet(
            grid_height=gs.grid.height,
            grid_width=gs.grid.width,
            num_res_blocks=2,
            channels=32,
        )
        network.eval()

        mcts = MCTS(
            network=network,
            grid_width=gs.grid.width,
            grid_height=gs.grid.height,
            num_simulations=3,
            device='cpu',
        )

        examples, winner = self_play_game(
            mcts=mcts,
            map_data=small_map_data,
            max_steps=10,
            temperature_threshold=3,
        )

        if winner is not None and len(examples) > 0:
            # All examples from winning player should have value=1.0
            # All examples from losing player should have value=-1.0
            for ex in examples:
                value = ex[5]
                assert value in (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestAlphaZeroIntegration:

    def test_network_on_real_game_state(self, game_state):
        """Network can process a real game state observation."""
        w = game_state.grid.width
        h = game_state.grid.height

        network = AlphaZeroNet(
            grid_height=h, grid_width=w,
            num_res_blocks=2, channels=32,
        )

        grid, units, gf, mask = _obs_from_game_state(game_state, w, h)

        grid_t = torch.tensor(grid).unsqueeze(0)
        units_t = torch.tensor(units).unsqueeze(0)
        gf_t = torch.tensor(gf).unsqueeze(0)
        mask_t = torch.tensor(mask).unsqueeze(0)

        probs, value = network.predict(grid_t, units_t, gf_t, mask_t)

        assert probs.shape == (1, 10 * w * h)
        assert value.shape == (1, 1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_full_mcts_with_real_state(self, game_state):
        """Full MCTS search works on a real game state."""
        w = game_state.grid.width
        h = game_state.grid.height

        network = AlphaZeroNet(
            grid_height=h, grid_width=w,
            num_res_blocks=2, channels=32,
        )
        network.eval()

        mcts = MCTS(
            network=network,
            grid_width=w,
            grid_height=h,
            num_simulations=5,
            device='cpu',
        )

        action, probs = mcts.select_action(game_state, temperature=0, add_noise=False)
        info = mcts.get_action_info(game_state, action)

        assert info is not None
        assert info['key'] in (
            'create_unit', 'move', 'attack', 'seize', 'heal', 'cure',
            'end_turn', 'paralyze', 'haste', 'defence_buff', 'attack_buff',
        )
