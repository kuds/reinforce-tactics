"""Tests for Feudal RL components: networks, buffer, GAE, intrinsic reward."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from reinforcetactics.rl.feudal_rl import (
    FeudalRolloutBuffer,
    ManagerNetwork,
    SpatialFeatureExtractor,
    WorkerNetwork,
    _compute_gae,
    compute_intrinsic_reward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRID_H, GRID_W = 8, 8


@pytest.fixture
def obs_space():
    """Minimal observation space matching the feature extractor expectations."""
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0, high=1, shape=(GRID_H, GRID_W, 3), dtype=np.float32),
            "units": spaces.Box(low=0, high=1, shape=(GRID_H, GRID_W, 3), dtype=np.float32),
            "global_features": spaces.Box(low=0, high=10000, shape=(6,), dtype=np.float32),
        }
    )


@pytest.fixture
def feature_extractor(obs_space):
    return SpatialFeatureExtractor(obs_space, features_dim=64)


@pytest.fixture
def manager():
    return ManagerNetwork(feature_dim=64, grid_width=GRID_W, grid_height=GRID_H, num_goal_types=4)


@pytest.fixture
def worker():
    return WorkerNetwork(feature_dim=64, goal_embedding_dim=32, action_space_dims=[10, 8, GRID_W, GRID_H, GRID_W, GRID_H])


@pytest.fixture
def sample_obs():
    """A single observation dict (unbatched numpy arrays)."""
    return {
        "grid": np.random.rand(GRID_H, GRID_W, 3).astype(np.float32),
        "units": np.random.rand(GRID_H, GRID_W, 3).astype(np.float32),
        "global_features": np.random.rand(6).astype(np.float32),
    }


@pytest.fixture
def batched_obs():
    """Batched observation tensors (batch=2)."""
    return {
        "grid": torch.rand(2, GRID_H, GRID_W, 3),
        "units": torch.rand(2, GRID_H, GRID_W, 3),
        "global_features": torch.rand(2, 6),
    }


# ---------------------------------------------------------------------------
# SpatialFeatureExtractor
# ---------------------------------------------------------------------------


class TestSpatialFeatureExtractor:
    def test_output_shape(self, feature_extractor, batched_obs):
        out = feature_extractor(batched_obs)
        assert out.shape == (2, 64)

    def test_single_sample(self, feature_extractor):
        obs = {
            "grid": torch.rand(1, GRID_H, GRID_W, 3),
            "units": torch.rand(1, GRID_H, GRID_W, 3),
            "global_features": torch.rand(1, 6),
        }
        out = feature_extractor(obs)
        assert out.shape == (1, 64)

    def test_no_global_features(self):
        """Extractor works when observation space has no global_features."""
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=1, shape=(GRID_H, GRID_W, 3), dtype=np.float32),
                "units": spaces.Box(low=0, high=1, shape=(GRID_H, GRID_W, 3), dtype=np.float32),
            }
        )
        ext = SpatialFeatureExtractor(obs_space, features_dim=64)
        assert ext.n_global == 0
        obs = {
            "grid": torch.rand(1, GRID_H, GRID_W, 3),
            "units": torch.rand(1, GRID_H, GRID_W, 3),
        }
        out = ext(obs)
        assert out.shape == (1, 64)


# ---------------------------------------------------------------------------
# ManagerNetwork
# ---------------------------------------------------------------------------


class TestManagerNetwork:
    def test_forward_shapes(self, manager):
        features = torch.rand(2, 64)
        gx, gy, gt, v = manager(features)
        assert gx.shape == (2, GRID_W)
        assert gy.shape == (2, GRID_H)
        assert gt.shape == (2, 4)
        assert v.shape == (2, 1)

    def test_sample_goal_shapes(self, manager):
        features = torch.rand(3, 64)
        goal, log_prob, value = manager.sample_goal(features)
        assert goal.shape == (3, 3)
        assert log_prob.shape == (3,)
        assert value.shape == (3, 1)

    def test_sample_goal_ranges(self, manager):
        features = torch.rand(100, 64)
        goal, _, _ = manager.sample_goal(features)
        assert (goal[:, 0] >= 0).all() and (goal[:, 0] < GRID_W).all()
        assert (goal[:, 1] >= 0).all() and (goal[:, 1] < GRID_H).all()
        assert (goal[:, 2] >= 0).all() and (goal[:, 2] < 4).all()

    def test_evaluate_goal_shapes(self, manager):
        features = torch.rand(4, 64)
        goal = torch.tensor([[1, 2, 0], [3, 4, 1], [0, 0, 2], [7, 7, 3]]).float()
        log_prob, entropy, value = manager.evaluate_goal(features, goal)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4, 1)

    def test_evaluate_matches_sample(self, manager):
        """Log probs from evaluate_goal should match sample_goal for the same goal."""
        torch.manual_seed(42)
        features = torch.rand(1, 64)
        goal, sample_lp, _ = manager.sample_goal(features)
        eval_lp, _, _ = manager.evaluate_goal(features, goal)
        assert torch.allclose(sample_lp, eval_lp, atol=1e-5)

    def test_entropy_positive(self, manager):
        features = torch.rand(2, 64)
        _, entropy, _ = manager.evaluate_goal(features, torch.tensor([[1, 2, 0], [3, 4, 1]]).float())
        assert (entropy > 0).all()


# ---------------------------------------------------------------------------
# WorkerNetwork
# ---------------------------------------------------------------------------


class TestWorkerNetwork:
    def test_forward_shapes(self, worker):
        features = torch.rand(2, 64)
        goal = torch.tensor([[1, 2, 0], [3, 4, 1]]).float()
        action_logits, value = worker(features, goal)
        assert len(action_logits) == 6
        assert action_logits[0].shape == (2, 10)
        assert action_logits[1].shape == (2, 8)
        assert value.shape == (2, 1)

    def test_sample_action_shapes(self, worker):
        features = torch.rand(3, 64)
        goal = torch.rand(3, 3)
        action, log_prob, value = worker.sample_action(features, goal)
        assert action.shape == (3, 6)
        assert log_prob.shape == (3,)
        assert value.shape == (3, 1)

    def test_evaluate_action_shapes(self, worker):
        features = torch.rand(2, 64)
        goal = torch.rand(2, 3)
        action = torch.zeros(2, 6).long()
        log_prob, entropy, value = worker.evaluate_action(features, goal, action)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert value.shape == (2, 1)

    def test_evaluate_matches_sample(self, worker):
        torch.manual_seed(0)
        features = torch.rand(1, 64)
        goal = torch.rand(1, 3)
        action, sample_lp, _ = worker.sample_action(features, goal)
        eval_lp, _, _ = worker.evaluate_action(features, goal, action)
        assert torch.allclose(sample_lp, eval_lp, atol=1e-5)


# ---------------------------------------------------------------------------
# _compute_gae
# ---------------------------------------------------------------------------


class TestComputeGAE:
    def test_single_step(self):
        rewards = np.array([1.0])
        values = np.array([0.5])
        dones = np.array([0.0])
        last_value = 0.8
        gamma, lam = 0.99, 0.95

        adv, ret = _compute_gae(rewards, values, dones, last_value, gamma, lam)
        # delta = 1.0 + 0.99*0.8 - 0.5 = 1.292
        expected_delta = 1.0 + gamma * last_value - 0.5
        assert adv.shape == (1,)
        np.testing.assert_allclose(adv[0], expected_delta, atol=1e-5)
        np.testing.assert_allclose(ret[0], adv[0] + values[0], atol=1e-5)

    def test_done_cuts_bootstrap(self):
        rewards = np.array([1.0])
        values = np.array([0.5])
        dones = np.array([1.0])
        last_value = 100.0  # Should be ignored because done=True

        adv, _ = _compute_gae(rewards, values, dones, last_value, 0.99, 0.95)
        # delta = 1.0 + 0.99*100*0 - 0.5 = 0.5
        np.testing.assert_allclose(adv[0], 0.5, atol=1e-5)

    def test_multi_step(self):
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 1.0, 1.5])
        dones = np.array([0.0, 0.0, 0.0])
        last_value = 2.0
        gamma, lam = 0.99, 0.95

        adv, ret = _compute_gae(rewards, values, dones, last_value, gamma, lam)
        assert adv.shape == (3,)
        assert ret.shape == (3,)
        # Returns = advantages + values
        np.testing.assert_allclose(ret, adv + values, atol=1e-5)

    def test_with_segment_lengths(self):
        rewards = np.array([5.0, 10.0])
        values = np.array([1.0, 2.0])
        dones = np.array([0.0, 0.0])
        last_value = 3.0
        gamma, lam = 0.99, 0.95
        seg_len = np.array([3, 5])

        adv, ret = _compute_gae(rewards, values, dones, last_value, gamma, lam, segment_lengths=seg_len)
        # Step 1 (t=1, last): delta = 10 + 0.99^5 * 3 - 2
        expected_delta_1 = 10.0 + (0.99**5) * 3.0 - 2.0
        np.testing.assert_allclose(adv[1], expected_delta_1, atol=1e-5)

    def test_all_done(self):
        rewards = np.array([1.0, 2.0])
        values = np.array([0.0, 0.0])
        dones = np.array([1.0, 1.0])
        adv, ret = _compute_gae(rewards, values, dones, 0.0, 0.99, 0.95)
        np.testing.assert_allclose(adv, rewards, atol=1e-5)


# ---------------------------------------------------------------------------
# FeudalRolloutBuffer
# ---------------------------------------------------------------------------


class TestFeudalRolloutBuffer:
    def _make_obs(self):
        return {
            "grid": np.random.rand(GRID_H, GRID_W, 3).astype(np.float32),
            "units": np.random.rand(GRID_H, GRID_W, 3).astype(np.float32),
            "global_features": np.random.rand(6).astype(np.float32),
        }

    def test_add_and_finalize_worker(self):
        buf = FeudalRolloutBuffer()
        for _ in range(5):
            buf.add_worker_step(
                self._make_obs(),
                action=np.array([0, 1, 2, 3, 4, 5]),
                log_prob=-1.0,
                value=0.5,
                goal=np.array([1.0, 2.0, 0.0]),
                extrinsic_reward=1.0,
                intrinsic_reward=0.5,
                done=False,
                worker_reward_alpha=0.5,
            )
        buf.finalize()
        assert buf.w_obs_grid.shape == (5, GRID_H, GRID_W, 3)
        assert buf.w_actions.shape == (5, 6)
        assert buf.w_rewards.shape == (5,)
        # reward = intrinsic + alpha * extrinsic = 0.5 + 0.5*1.0 = 1.0
        np.testing.assert_allclose(buf.w_rewards, 1.0)

    def test_worker_reward_formula(self):
        buf = FeudalRolloutBuffer()
        buf.add_worker_step(
            self._make_obs(),
            action=np.zeros(6),
            log_prob=0.0,
            value=0.0,
            goal=np.zeros(3),
            extrinsic_reward=2.0,
            intrinsic_reward=3.0,
            done=False,
            worker_reward_alpha=0.7,
        )
        buf.finalize()
        # 3.0 + 0.7 * 2.0 = 4.4
        np.testing.assert_allclose(buf.w_rewards[0], 4.4, atol=1e-5)

    def test_manager_segment(self):
        buf = FeudalRolloutBuffer()
        buf.add_manager_step(self._make_obs(), goal=np.array([1.0, 2.0, 0.0]), log_prob=-0.5, value=1.0)
        # Need at least one worker step for finalize to work
        buf.add_worker_step(
            self._make_obs(), np.zeros(6), 0.0, 0.0, np.zeros(3), 0.0, 0.0, False, 0.5
        )
        buf.end_manager_segment(cumulative_reward=10.0, done=False, segment_length=5)
        assert buf.has_manager_data
        buf.finalize()
        assert buf.m_rewards.shape == (1,)
        np.testing.assert_allclose(buf.m_rewards[0], 10.0)
        assert buf.m_segment_lengths[0] == 5

    def test_no_manager_data(self):
        buf = FeudalRolloutBuffer()
        buf.add_worker_step(
            self._make_obs(), np.zeros(6), 0.0, 0.0, np.zeros(3), 0.0, 0.0, False, 0.5
        )
        assert not buf.has_manager_data
        buf.finalize()
        assert len(buf.m_rewards) == 0
        assert buf.m_goals.shape == (0, 3)

    def test_compute_advantages(self):
        buf = FeudalRolloutBuffer()
        for _ in range(4):
            buf.add_worker_step(
                self._make_obs(), np.zeros(6), -1.0, 0.5, np.zeros(3), 1.0, 0.5, False, 0.5
            )
        buf.add_manager_step(self._make_obs(), np.zeros(3), -0.5, 1.0)
        buf.end_manager_segment(4.0, False, 4)
        buf.finalize()
        buf.compute_advantages(last_w_value=0.5, last_m_value=1.0, gamma=0.99, gae_lambda=0.95)
        assert buf.w_advantages.shape == (4,)
        assert buf.w_returns.shape == (4,)
        assert buf.m_advantages.shape == (1,)
        assert buf.m_returns.shape == (1,)

    def test_reset(self):
        buf = FeudalRolloutBuffer()
        buf.add_worker_step(
            self._make_obs(), np.zeros(6), 0.0, 0.0, np.zeros(3), 0.0, 0.0, False, 0.5
        )
        buf.reset()
        assert len(buf.w_obs_grid) == 0
        assert len(buf.m_obs_grid) == 0


# ---------------------------------------------------------------------------
# compute_intrinsic_reward
# ---------------------------------------------------------------------------


class TestComputeIntrinsicReward:
    def _make_state(self, player_positions=None, opponent_positions=None, grid_val=0):
        """Build a minimal state dict.

        Args:
            player_positions: list of (y, x) tuples for agent_player=1
            opponent_positions: list of (y, x) tuples for player 2
            grid_val: scalar to fill grid[:,:,0] (terrain type)
        """
        units = np.zeros((GRID_H, GRID_W, 3), dtype=np.float32)
        grid = np.zeros((GRID_H, GRID_W, 3), dtype=np.float32)
        grid[:, :, 0] = grid_val
        if player_positions:
            for y, x in player_positions:
                units[y, x, 1] = 1  # player 1
        if opponent_positions:
            for y, x in opponent_positions:
                units[y, x, 1] = 2  # player 2
        return {"grid": grid, "units": units, "global_features": np.zeros(6, dtype=np.float32)}

    def test_no_player_units(self):
        state = self._make_state()
        next_state = self._make_state()
        reward = compute_intrinsic_reward(state, np.array([0, 0, 0]), next_state, agent_player=1)
        assert reward == -10.0

    def test_unit_at_goal(self):
        next_state = self._make_state(player_positions=[(3, 4)])
        goal = np.array([4, 3, 0])  # goal_x=4, goal_y=3
        reward = compute_intrinsic_reward(self._make_state(), goal, next_state, agent_player=1)
        # distance=0 -> distance_reward=0 + 5.0 bonus
        assert reward >= 5.0

    def test_distance_penalty(self):
        next_state = self._make_state(player_positions=[(0, 0)])
        goal = np.array([7, 7, 0])  # far away
        reward = compute_intrinsic_reward(self._make_state(), goal, next_state, agent_player=1)
        # min_distance = |0-7| + |0-7| = 14, distance_reward = -1.4
        assert reward < 0

    def test_attack_goal_enemy_bonus(self):
        # Enemy near goal location
        next_state = self._make_state(player_positions=[(3, 4)], opponent_positions=[(3, 5)])
        goal = np.array([4, 3, 0])  # attack goal_type=0
        reward = compute_intrinsic_reward(self._make_state(), goal, next_state, agent_player=1)
        # Unit at goal (+5.0) + enemy within 3 tiles of goal (+1.0)
        assert reward >= 6.0

    def test_defend_goal_own_structure(self):
        next_state = self._make_state(player_positions=[(2, 2)])
        # Set grid owner at goal to player 1
        next_state["grid"][2, 2, 1] = 1
        goal = np.array([2, 2, 1])  # defend goal_type=1
        reward = compute_intrinsic_reward(self._make_state(), goal, next_state, agent_player=1)
        # Unit at goal (+5.0) + defend bonus (+3.0) + own structure (+2.0)
        assert reward >= 10.0

    def test_capture_goal_structure(self):
        next_state = self._make_state(player_positions=[(1, 1)], grid_val=2)
        goal = np.array([1, 1, 2])  # capture goal_type=2
        reward = compute_intrinsic_reward(self._make_state(), goal, next_state, agent_player=1)
        # Unit at goal (+5.0) + terrain > 0 (+4.0)
        assert reward >= 9.0

    def test_expand_goal_spread_bonus(self):
        # Multiple units near goal
        positions = [(3, 3), (3, 4), (4, 3)]
        next_state = self._make_state(player_positions=positions)
        goal = np.array([3, 3, 3])  # expand goal_type=3
        reward = compute_intrinsic_reward(self._make_state(), goal, next_state, agent_player=1)
        # All 3 units within radius 4 -> +1.5 spread bonus
        assert reward > 0

    def test_agent_player_2(self):
        """Intrinsic reward should respect agent_player parameter."""
        # Player 2 units
        units = np.zeros((GRID_H, GRID_W, 3), dtype=np.float32)
        units[0, 0, 1] = 2
        next_state = {
            "grid": np.zeros((GRID_H, GRID_W, 3), dtype=np.float32),
            "units": units,
            "global_features": np.zeros(6, dtype=np.float32),
        }
        goal = np.array([0, 0, 0])
        reward = compute_intrinsic_reward(self._make_state(), goal, next_state, agent_player=2)
        # Player 2 unit at goal -> should get bonus
        assert reward >= 5.0
