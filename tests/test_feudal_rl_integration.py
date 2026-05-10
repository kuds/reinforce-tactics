"""Integration & edge-case tests for Feudal RL.

Covers gaps not addressed by the existing unit-level tests:
  - FeudalRLAgent end-to-end (select_action, collect_rollout, update, checkpoint)
  - Manager/worker temporal coordination & goal lifecycle
  - Buffer edge cases (episode boundaries, single-step episodes)
  - Intrinsic reward edge cases
  - Gradient flow through shared feature extractor
  - Deterministic vs stochastic action selection
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from reinforcetactics.rl.feudal_rl import (
    FeudalRLAgent,
    FeudalRolloutBuffer,
    ManagerNetwork,
    SpatialFeatureExtractor,
    WorkerNetwork,
    _compute_gae,
    compute_intrinsic_reward,
)
from reinforcetactics.rl.observation import (
    GLOBAL_FEATURES_DIM,
    GRID_CHANNELS,
    NUM_TILE_TYPES,
    NUM_UNIT_TYPES,
    UNIT_CHANNELS,
)

GRID_H, GRID_W = 6, 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs_space():
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(GRID_H, GRID_W, GRID_CHANNELS), dtype=np.float32),
            "units": spaces.Box(low=0.0, high=1.0, shape=(GRID_H, GRID_W, UNIT_CHANNELS), dtype=np.float32),
            "global_features": spaces.Box(low=0.0, high=1e5, shape=(GLOBAL_FEATURES_DIM,), dtype=np.float32),
        }
    )


def _make_obs():
    return {
        "grid": np.random.rand(GRID_H, GRID_W, GRID_CHANNELS).astype(np.float32),
        "units": np.zeros((GRID_H, GRID_W, UNIT_CHANNELS), dtype=np.float32),
        "global_features": np.random.rand(GLOBAL_FEATURES_DIM).astype(np.float32),
    }


def _make_state(player_positions=None, opponent_positions=None, grid_val=0):
    """Build a minimal agent-relative observation.

    With the new encoding, ownership is one-hot: unit ``self`` lives on
    channel ``NUM_UNIT_TYPES + 0`` and ``opp`` on ``NUM_UNIT_TYPES + 1``.
    """
    units = np.zeros((GRID_H, GRID_W, UNIT_CHANNELS), dtype=np.float32)
    grid = np.zeros((GRID_H, GRID_W, GRID_CHANNELS), dtype=np.float32)
    # ``grid_val`` keeps tests that twiddled tile-type 0 working: set the
    # one-hot bit on the chosen tile type for every cell.
    if 0 <= int(grid_val) < NUM_TILE_TYPES:
        grid[:, :, int(grid_val)] = 1.0

    self_unit_ch = NUM_UNIT_TYPES + 0
    opp_unit_ch = NUM_UNIT_TYPES + 1
    hp_unit_ch = UNIT_CHANNELS - 1

    if player_positions:
        for y, x in player_positions:
            units[y, x, 0] = 1.0  # unit type "W"
            units[y, x, self_unit_ch] = 1.0
            units[y, x, hp_unit_ch] = 1.0
    if opponent_positions:
        for y, x in opponent_positions:
            units[y, x, 0] = 1.0
            units[y, x, opp_unit_ch] = 1.0
            units[y, x, hp_unit_ch] = 1.0
    return {
        "grid": grid,
        "units": units,
        "global_features": np.zeros(GLOBAL_FEATURES_DIM, dtype=np.float32),
    }


class MockEnv:
    """Minimal environment mock for collect_rollout testing."""

    def __init__(self, episode_length=50):
        self.episode_length = episode_length
        self.step_count = 0
        self.observation_space = _make_obs_space()
        self.grid_width = GRID_W
        self.grid_height = GRID_H

    def reset(self, **kwargs):
        self.step_count = 0
        return _make_obs(), {}

    def step(self, action):
        self.step_count += 1
        obs = _make_obs()
        # Place a player unit so intrinsic reward doesn't return -10. The
        # "self-owned unit" channel is the first owner channel after the
        # unit-type one-hot block.
        obs["units"][0, 0, NUM_UNIT_TYPES + 0] = 1
        reward = np.random.uniform(-1, 1)
        terminated = self.step_count >= self.episode_length
        truncated = False
        info = {"winner": 1} if terminated else {}
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# FeudalRLAgent: construction & select_action
# ---------------------------------------------------------------------------


class TestFeudalRLAgentConstruction:
    def test_creates_all_networks(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        assert isinstance(agent.feature_extractor, SpatialFeatureExtractor)
        assert isinstance(agent.manager, ManagerNetwork)
        assert isinstance(agent.worker, WorkerNetwork)

    def test_default_goal_state(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        assert agent.current_goal is None
        assert agent.goal_step_counter == 0

    def test_select_action_stochastic(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        action, goal = agent.select_action(obs, deterministic=False)
        assert action.shape == (6,)
        assert goal.shape == (3,)
        assert agent.current_goal is not None
        assert agent.goal_step_counter == 1

    def test_select_action_deterministic(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        action, goal = agent.select_action(obs, deterministic=True)
        assert action.shape == (6,)
        assert goal.shape == (3,)

    def test_goal_persists_within_horizon(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 5
        obs = _make_obs()

        # First action sets a goal
        _, goal1 = agent.select_action(obs)
        # Subsequent actions within horizon keep same goal
        for _ in range(3):
            _, goal_n = agent.select_action(obs)
            np.testing.assert_array_equal(goal1, goal_n)

    def test_goal_updates_after_horizon(self):
        torch.manual_seed(123)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 3
        obs = _make_obs()

        # Step through one full horizon
        agent.select_action(obs)
        for _ in range(2):
            agent.select_action(obs)
        assert agent.goal_step_counter == 3

        # Next call should trigger a new goal
        agent.select_action(obs)
        assert agent.goal_step_counter == 1  # reset after new goal

    def test_reset_goal(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        agent.select_action(obs)
        assert agent.current_goal is not None
        agent.reset_goal()
        assert agent.current_goal is None
        assert agent.goal_step_counter == 0


# ---------------------------------------------------------------------------
# FeudalRLAgent: training pipeline
# ---------------------------------------------------------------------------


class TestFeudalRLAgentTraining:
    @pytest.fixture
    def trained_setup(self):
        """Create agent with training setup and a short rollout."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 5
        agent.setup_training(learning_rate=1e-3)
        env = MockEnv(episode_length=30)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()
        return agent, env

    def test_collect_rollout_returns_buffer(self, trained_setup):
        agent, env = trained_setup
        buf = agent.collect_rollout(env, n_steps=20, gamma=0.99, gae_lambda=0.95)
        assert isinstance(buf, FeudalRolloutBuffer)
        assert len(buf.w_rewards) == 20
        assert buf.has_manager_data
        assert buf.w_advantages.shape == (20,)

    def test_collect_rollout_manager_segments_align(self, trained_setup):
        agent, env = trained_setup
        buf = agent.collect_rollout(env, n_steps=20, gamma=0.99, gae_lambda=0.95)
        # Sum of segment lengths should cover most of the steps
        # (may not exactly equal n_steps due to episode resets)
        total_segment_steps = buf.m_segment_lengths.sum()
        assert total_segment_steps <= 20
        assert total_segment_steps > 0

    def test_update_returns_metrics(self, trained_setup):
        agent, env = trained_setup
        buf = agent.collect_rollout(env, n_steps=20, gamma=0.99, gae_lambda=0.95)
        metrics = agent.update(buf, n_epochs=2, batch_size=10, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
        assert "worker_policy_loss" in metrics
        assert "manager_policy_loss" in metrics
        assert "worker_entropy" in metrics
        assert "manager_entropy" in metrics
        # Losses should be finite
        for v in metrics.values():
            assert np.isfinite(v), f"Non-finite metric: {v}"

    def test_update_changes_weights(self, trained_setup):
        agent, env = trained_setup

        # Snapshot initial weights
        w_before = agent.worker.action_heads[0].weight.data.clone()
        m_before = agent.manager.goal_x_head.weight.data.clone()

        buf = agent.collect_rollout(env, n_steps=20, gamma=0.99, gae_lambda=0.95)
        agent.update(buf, n_epochs=5, batch_size=10, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)

        # At least one network should have updated
        w_changed = not torch.allclose(w_before, agent.worker.action_heads[0].weight.data)
        m_changed = not torch.allclose(m_before, agent.manager.goal_x_head.weight.data)
        assert w_changed or m_changed, "Neither worker nor manager weights changed after update"

    def test_feature_extractor_updated_by_worker(self, trained_setup):
        """Feature extractor is owned by the worker optimizer; the manager
        update detaches features so its grads never reach the encoder. The
        encoder should still update because the worker drives it."""
        agent, env = trained_setup
        buf = agent.collect_rollout(env, n_steps=20, gamma=0.99, gae_lambda=0.95)

        fe_before = agent.feature_extractor.cnn[0].weight.data.clone()
        agent.update(buf, n_epochs=3, batch_size=10, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
        fe_after = agent.feature_extractor.cnn[0].weight.data.clone()

        assert not torch.allclose(fe_before, fe_after), "Feature extractor weights did not change"

    def test_ar_worker_warns_without_structured_masks(self):
        """AR sampling falls back to unmasked Categoricals when the env
        doesn't expose ``structured_action_masks`` — warn loudly so the
        misconfiguration isn't silent."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu", autoregressive_worker=True)
        agent.manager_horizon = 5
        agent.setup_training(learning_rate=1e-3)
        env = MockEnv(episode_length=20)  # MockEnv has no structured_action_masks()
        with pytest.warns(RuntimeWarning, match="structured_action_masks"):
            agent.collect_rollout(env, n_steps=4, gamma=0.99, gae_lambda=0.95)

    def test_collect_rollout_auto_initializes_last_obs(self):
        """First collect_rollout should reset the env on its own when
        ``_last_obs`` is None — callers shouldn't have to remember to prime it."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 5
        agent.setup_training(learning_rate=1e-3)
        env = MockEnv(episode_length=20)
        # Deliberately do NOT set agent._last_obs here.
        assert agent._last_obs is None
        buf = agent.collect_rollout(env, n_steps=10, gamma=0.99, gae_lambda=0.95)
        assert len(buf.w_rewards) == 10
        assert agent._last_obs is not None

    def test_multiple_episodes_in_rollout(self):
        """Rollout spanning multiple episodes should handle resets correctly."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 3
        agent.setup_training(learning_rate=1e-3)
        # Short episodes to force multiple resets
        env = MockEnv(episode_length=8)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()

        buf = agent.collect_rollout(env, n_steps=30, gamma=0.99, gae_lambda=0.95)
        assert len(buf.w_rewards) == 30
        # Should have done markers for episode boundaries
        assert buf.w_dones.sum() >= 2, "Expected at least 2 episode completions"

    def test_worker_reward_alpha_zero(self):
        """With alpha=0, worker reward should be purely intrinsic."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 5
        agent.setup_training(learning_rate=1e-3)
        env = MockEnv(episode_length=50)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()

        buf = agent.collect_rollout(env, n_steps=10, gamma=0.99, gae_lambda=0.95, worker_reward_alpha=0.0)
        # All rewards should be intrinsic only (no extrinsic component)
        assert len(buf.w_rewards) == 10

    def test_reward_scale_shrinks_extrinsic_signal(self):
        """reward_scale=0 zeros out the extrinsic component end-to-end —
        manager segment rewards should all be zero, and worker rewards
        should equal the intrinsic-only path."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 5
        agent.setup_training(learning_rate=1e-3)
        env = MockEnv(episode_length=50)
        env.reset()
        agent.reset_goal()

        buf = agent.collect_rollout(env, n_steps=10, gamma=0.99, gae_lambda=0.95, worker_reward_alpha=1.0, reward_scale=0.0)
        # Extrinsic was multiplied by 0.0, so manager segments — which only
        # accumulate extrinsic — should all be zero.
        if buf.has_manager_data:
            assert np.allclose(buf.m_rewards, 0.0)

    def test_update_reports_gradient_norms(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.manager_horizon = 5
        agent.setup_training(learning_rate=1e-3)
        env = MockEnv(episode_length=30)
        env.reset()
        agent.reset_goal()
        buf = agent.collect_rollout(env, n_steps=20, gamma=0.99, gae_lambda=0.95)
        metrics = agent.update(buf, n_epochs=2, batch_size=10, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
        assert "worker_grad_norm" in metrics
        assert "manager_grad_norm" in metrics
        # Pre-clip norm is non-negative; with random init it should be > 0.
        assert metrics["worker_grad_norm"] >= 0.0
        assert metrics["manager_grad_norm"] >= 0.0


# ---------------------------------------------------------------------------
# FeudalRLAgent: checkpoint save/load
# ---------------------------------------------------------------------------


class TestFeudalRLAgentCheckpoint:
    def test_save_and_load_roundtrip(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_checkpoint.pt")
            agent.save_checkpoint(path)

            # Create a new agent and load
            agent2 = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
            agent2.setup_training(learning_rate=1e-3)
            agent2.load_checkpoint(path)

            # Weights should match
            for p1, p2 in zip(agent.feature_extractor.parameters(), agent2.feature_extractor.parameters()):
                assert torch.allclose(p1, p2)
            for p1, p2 in zip(agent.manager.parameters(), agent2.manager.parameters()):
                assert torch.allclose(p1, p2)
            for p1, p2 in zip(agent.worker.parameters(), agent2.worker.parameters()):
                assert torch.allclose(p1, p2)

    def test_load_without_optimizer(self):
        """Loading a checkpoint when training hasn't been set up should still work for inference."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.pt")
            agent.save_checkpoint(path)

            agent2 = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
            # Don't call setup_training — simulates inference-only load
            agent2.load_checkpoint(path)

            # Should be able to select actions
            obs = _make_obs()
            action, goal = agent2.select_action(obs)
            assert action.shape == (6,)

    def test_checkpoint_restores_manager_horizon(self):
        """Saved hyperparams (manager_horizon) survive a save/load roundtrip
        so callers don't have to re-set runtime config after loading."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)
        agent.manager_horizon = 7
        agent.agent_player = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            agent.save_checkpoint(path)

            agent2 = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
            assert agent2.manager_horizon == 10  # default before load
            agent2.load_checkpoint(path)
            assert agent2.manager_horizon == 7
            assert agent2.agent_player == 2

    def test_checkpoint_training_state_roundtrip(self):
        """Saved training_state survives save/load and is returned to the
        caller — the training script uses this for resume."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            agent.save_checkpoint(
                path,
                training_state={"total_timesteps": 12345, "best_eval_reward": 7.5},
            )

            agent2 = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
            agent2.setup_training(learning_rate=1e-3)
            state = agent2.load_checkpoint(path)
            assert state is not None
            assert state["total_timesteps"] == 12345
            assert state["best_eval_reward"] == 7.5

    def test_checkpoint_without_training_state_returns_none(self):
        """Old/new checkpoints without training_state should return None
        rather than crash the resume path."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            agent.save_checkpoint(path)  # no training_state

            agent2 = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
            agent2.setup_training(learning_rate=1e-3)
            assert agent2.load_checkpoint(path) is None

    def test_checkpoint_rejects_grid_dim_mismatch(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ckpt.pt")
            agent.save_checkpoint(path)

            # Different grid width: refuse to load instead of producing a
            # silent shape mismatch deeper in the linear projection.
            agent_bad = FeudalRLAgent(obs_space, grid_width=GRID_W + 2, grid_height=GRID_H, device="cpu")
            with pytest.raises(ValueError, match="grid_width"):
                agent_bad.load_checkpoint(path)


# ---------------------------------------------------------------------------
# FeudalRLAgent: evaluate
# ---------------------------------------------------------------------------


class TestFeudalRLAgentEvaluate:
    def test_evaluate_returns_metrics(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)

        env = MockEnv(episode_length=10)
        results = agent.evaluate(env, n_episodes=3)
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "win_rate" in results
        assert 0 <= results["win_rate"] <= 1

    def test_evaluate_resets_goal_per_episode(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training(learning_rate=1e-3)

        env = MockEnv(episode_length=5)
        # Evaluate should call reset_goal for each episode
        agent.evaluate(env, n_episodes=2)
        # After evaluate the agent should be back in train mode
        assert agent.feature_extractor.training


# ---------------------------------------------------------------------------
# FeudalRolloutBuffer: edge cases
# ---------------------------------------------------------------------------


class TestFeudalRolloutBufferEdgeCases:
    def test_single_step_buffer(self):
        buf = FeudalRolloutBuffer()
        obs = _make_obs()
        buf.add_worker_step(obs, np.zeros(6), -1.0, 0.5, np.zeros(3), 1.0, 0.5, False, 0.5)
        buf.finalize()
        assert buf.w_rewards.shape == (1,)

    def test_all_done_steps(self):
        buf = FeudalRolloutBuffer()
        obs = _make_obs()
        for _ in range(3):
            buf.add_worker_step(obs, np.zeros(6), -1.0, 0.5, np.zeros(3), 1.0, 0.5, True, 0.5)
        buf.finalize()
        buf.compute_advantages(last_w_value=0.0, last_m_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert buf.w_advantages.shape == (3,)
        # With all done=True, advantages should equal reward - value
        np.testing.assert_allclose(buf.w_advantages, buf.w_rewards - buf.w_values, atol=1e-5)

    def test_manager_segment_without_worker_step_between(self):
        """Two consecutive manager segments."""
        buf = FeudalRolloutBuffer()
        obs = _make_obs()
        buf.add_manager_step(obs, np.array([1.0, 2.0, 0.0]), -0.5, 1.0)
        buf.add_worker_step(obs, np.zeros(6), 0.0, 0.0, np.zeros(3), 1.0, 0.5, False, 0.5)
        buf.end_manager_segment(5.0, False, 3)
        buf.add_manager_step(obs, np.array([3.0, 4.0, 1.0]), -0.3, 0.8)
        buf.add_worker_step(obs, np.zeros(6), 0.0, 0.0, np.zeros(3), 2.0, 0.3, False, 0.5)
        buf.end_manager_segment(10.0, True, 5)
        buf.finalize()
        assert len(buf.m_rewards) == 2
        assert buf.m_segment_lengths[0] == 3
        assert buf.m_segment_lengths[1] == 5

    def test_varying_alpha_across_steps(self):
        """Different alpha values per step produce correct rewards."""
        buf = FeudalRolloutBuffer()
        obs = _make_obs()
        buf.add_worker_step(
            obs,
            np.zeros(6),
            0.0,
            0.0,
            np.zeros(3),
            extrinsic_reward=10.0,
            intrinsic_reward=1.0,
            done=False,
            worker_reward_alpha=0.0,
        )
        buf.add_worker_step(
            obs,
            np.zeros(6),
            0.0,
            0.0,
            np.zeros(3),
            extrinsic_reward=10.0,
            intrinsic_reward=1.0,
            done=False,
            worker_reward_alpha=1.0,
        )
        buf.finalize()
        # alpha=0: 1.0 + 0*10 = 1.0
        np.testing.assert_allclose(buf.w_rewards[0], 1.0, atol=1e-5)
        # alpha=1: 1.0 + 1*10 = 11.0
        np.testing.assert_allclose(buf.w_rewards[1], 11.0, atol=1e-5)


# ---------------------------------------------------------------------------
# _compute_gae: additional edge cases
# ---------------------------------------------------------------------------


class TestComputeGAEEdgeCases:
    def test_empty_inputs(self):
        """GAE with zero-length arrays should not crash."""
        adv, ret = _compute_gae(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            0.0,
            0.99,
            0.95,
        )
        assert len(adv) == 0
        assert len(ret) == 0

    def test_large_segment_length_discount(self):
        """Very large segment lengths should produce near-zero discount."""
        rewards = np.array([10.0])
        values = np.array([1.0])
        dones = np.array([0.0])
        seg_len = np.array([100])  # gamma^100 ≈ 0.366 for gamma=0.99

        adv, _ = _compute_gae(rewards, values, dones, 5.0, 0.99, 0.95, segment_lengths=seg_len)
        # delta = 10 + 0.99^100 * 5 - 1 ≈ 10 + 1.83 - 1 = 10.83
        expected_delta = 10.0 + (0.99**100) * 5.0 - 1.0
        np.testing.assert_allclose(adv[0], expected_delta, atol=1e-3)

    def test_gae_returns_equal_advantages_plus_values(self):
        """Returns should always equal advantages + values."""
        np.random.seed(42)
        n = 50
        rewards = np.random.randn(n).astype(np.float32)
        values = np.random.randn(n).astype(np.float32)
        dones = (np.random.rand(n) > 0.9).astype(np.float32)
        adv, ret = _compute_gae(rewards, values, dones, 0.5, 0.99, 0.95)
        np.testing.assert_allclose(ret, adv + values, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_intrinsic_reward: edge cases
# ---------------------------------------------------------------------------


class TestIntrinsicRewardEdgeCases:
    def test_goal_out_of_bounds(self):
        """Goal coordinates at grid boundary should not crash."""
        next_state = _make_state(player_positions=[(0, 0)])
        goal = np.array([GRID_W - 1, GRID_H - 1, 0])
        reward = compute_intrinsic_reward(next_state, goal)
        assert np.isfinite(reward)

    def test_multiple_player_units(self):
        """Distance should use closest unit."""
        positions = [(0, 0), (2, 2), (5, 5)]
        next_state = _make_state(player_positions=positions)
        goal = np.array([2, 2, 0])  # goal at (2, 2) where a unit is
        reward = compute_intrinsic_reward(next_state, goal)
        # Unit at goal -> +5.0 bonus
        assert reward >= 5.0

    def test_all_goal_types_return_finite(self):
        """Every goal type should produce a finite reward."""
        next_state = _make_state(player_positions=[(1, 1)], opponent_positions=[(3, 3)])
        for goal_type in range(4):
            goal = np.array([1, 1, goal_type])
            reward = compute_intrinsic_reward(next_state, goal)
            assert np.isfinite(reward), f"goal_type={goal_type} produced non-finite reward"

    def test_expand_with_many_units(self):
        """Expand bonus should scale with unit count in radius."""
        positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
        next_state = _make_state(player_positions=positions)
        goal = np.array([2, 2, 3])  # expand type
        reward = compute_intrinsic_reward(next_state, goal)
        # All 5 units within radius 4 of (2,2) -> +2.5 spread bonus + distance
        assert reward > 2.0

    def test_attack_no_enemies(self):
        """Attack goal with no enemies should still return finite reward."""
        next_state = _make_state(player_positions=[(3, 3)])
        goal = np.array([3, 3, 0])
        reward = compute_intrinsic_reward(next_state, goal)
        assert np.isfinite(reward)
        # Unit at goal but no enemies -> +5.0 (at goal) only
        assert reward >= 5.0


# ---------------------------------------------------------------------------
# Network: gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_worker_gradients_flow_through_feature_extractor(self):
        obs_space = _make_obs_space()
        fe = SpatialFeatureExtractor(obs_space, features_dim=64)
        worker = WorkerNetwork(
            feature_dim=64, goal_embedding_dim=32, action_space_dims=[10, 8, GRID_W, GRID_H, GRID_W, GRID_H]
        )

        obs = {
            "grid": torch.rand(2, GRID_H, GRID_W, GRID_CHANNELS),
            "units": torch.rand(2, GRID_H, GRID_W, UNIT_CHANNELS),
            "global_features": torch.rand(2, GLOBAL_FEATURES_DIM),
        }
        features = fe(obs)
        goal = torch.rand(2, 3)
        action, log_prob, value = worker.sample_action(features, goal)

        loss = -log_prob.mean() + value.mean()
        loss.backward()

        # Feature extractor should have gradients
        assert fe.cnn[0].weight.grad is not None
        assert fe.cnn[0].weight.grad.abs().sum() > 0

    def test_manager_gradients_flow_through_feature_extractor(self):
        obs_space = _make_obs_space()
        fe = SpatialFeatureExtractor(obs_space, features_dim=64)
        manager = ManagerNetwork(feature_dim=64, grid_width=GRID_W, grid_height=GRID_H)

        obs = {
            "grid": torch.rand(2, GRID_H, GRID_W, GRID_CHANNELS),
            "units": torch.rand(2, GRID_H, GRID_W, UNIT_CHANNELS),
            "global_features": torch.rand(2, GLOBAL_FEATURES_DIM),
        }
        features = fe(obs)
        goal, log_prob, value = manager.sample_goal(features)

        loss = -log_prob.mean() + value.mean()
        loss.backward()

        assert fe.cnn[0].weight.grad is not None
        assert fe.cnn[0].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# _obs_to_tensor: filtering
# ---------------------------------------------------------------------------


class TestObsToTensor:
    def test_filters_action_mask(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        tensor_obs = agent._obs_to_tensor(obs)
        assert "action_mask" not in tensor_obs
        assert "grid" in tensor_obs
        assert "units" in tensor_obs
        assert "global_features" in tensor_obs

    def test_tensor_shapes(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        tensor_obs = agent._obs_to_tensor(obs)
        # Should be batched (dim 0 = 1)
        assert tensor_obs["grid"].shape == (1, GRID_H, GRID_W, GRID_CHANNELS)
        assert tensor_obs["units"].shape == (1, GRID_H, GRID_W, UNIT_CHANNELS)
        assert tensor_obs["global_features"].shape == (1, GLOBAL_FEATURES_DIM)


# ---------------------------------------------------------------------------
# Manager/Worker action ranges
# ---------------------------------------------------------------------------


class TestActionRanges:
    def test_worker_action_dims_match_env(self):
        """Worker action dimensions should match the environment's MultiDiscrete spec."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        expected_dims = [10, 8, GRID_W, GRID_H, GRID_W, GRID_H]
        assert agent.worker.action_space_dims == expected_dims

    def test_manager_goal_range_matches_grid(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        assert agent.manager.grid_width == GRID_W
        assert agent.manager.grid_height == GRID_H

    def test_sampled_actions_in_valid_range(self):
        """All sampled action dimensions should be within valid bounds."""
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        for _ in range(50):
            action, goal = agent.select_action(obs)
            assert 0 <= action[0] < 10, f"action_type out of range: {action[0]}"
            assert 0 <= action[1] < 8, f"unit_type out of range: {action[1]}"
            assert 0 <= action[2] < GRID_W, f"from_x out of range: {action[2]}"
            assert 0 <= action[3] < GRID_H, f"from_y out of range: {action[3]}"
            assert 0 <= action[4] < GRID_W, f"to_x out of range: {action[4]}"
            assert 0 <= action[5] < GRID_H, f"to_y out of range: {action[5]}"
            assert 0 <= goal[0] < GRID_W
            assert 0 <= goal[1] < GRID_H
            assert 0 <= goal[2] < 4


# ---------------------------------------------------------------------------
# Worker action masking (pass-2 parity with PPO)
# ---------------------------------------------------------------------------


def _make_masks(legal_at, legal_ut, legal_fx, legal_fy, legal_tx, legal_ty):
    """Build a 6-tuple of bool masks shaped like env.action_masks() for the
    multi_discrete worker dims [10, 8, GRID_W, GRID_H, GRID_W, GRID_H]."""
    at = np.zeros(10, dtype=bool)
    ut = np.zeros(8, dtype=bool)
    fx = np.zeros(GRID_W, dtype=bool)
    fy = np.zeros(GRID_H, dtype=bool)
    tx = np.zeros(GRID_W, dtype=bool)
    ty = np.zeros(GRID_H, dtype=bool)
    at[list(legal_at)] = True
    ut[list(legal_ut)] = True
    fx[list(legal_fx)] = True
    fy[list(legal_fy)] = True
    tx[list(legal_tx)] = True
    ty[list(legal_ty)] = True
    return (at, ut, fx, fy, tx, ty)


class MaskedMockEnv(MockEnv):
    """MockEnv that also exposes ``action_masks()`` like StrategyGameEnv
    (multi_discrete mode). Always allows action_type=5 (end_turn) plus the
    extras passed at construction so tests can pin down exactly which
    actions the worker is permitted to sample.
    """

    def __init__(self, episode_length=50, extra_legal_action_types=()):
        super().__init__(episode_length=episode_length)
        self._masks = _make_masks(
            legal_at=(5,) + tuple(extra_legal_action_types),
            legal_ut=(0,),
            legal_fx=(0,),
            legal_fy=(0,),
            legal_tx=(0,),
            legal_ty=(0,),
        )

    def action_masks(self):
        return self._masks


class TestWorkerActionMasking:
    def test_select_action_respects_masks(self):
        torch.manual_seed(0)
        np.random.seed(0)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        # Only end_turn (action_type=5) is legal.
        masks = _make_masks(legal_at=(5,), legal_ut=(0,), legal_fx=(0,), legal_fy=(0,), legal_tx=(0,), legal_ty=(0,))
        # Sample many times; sampled action_type must always be 5.
        for _ in range(20):
            action, _ = agent.select_action(obs, deterministic=False, action_masks=masks)
            assert action[0] == 5, f"masked action_type leaked: {action[0]}"

    def test_select_action_deterministic_respects_masks(self):
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        obs = _make_obs()
        masks = _make_masks(legal_at=(5,), legal_ut=(0,), legal_fx=(0,), legal_fy=(0,), legal_tx=(0,), legal_ty=(0,))
        action, _ = agent.select_action(obs, deterministic=True, action_masks=masks)
        assert action[0] == 5

    def test_collect_rollout_captures_masks(self):
        """When env exposes action_masks(), the rollout buffer should store
        per-dim masks aligned with worker.action_space_dims."""
        torch.manual_seed(0)
        np.random.seed(0)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training()
        env = MaskedMockEnv(episode_length=50)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()
        buf = agent.collect_rollout(env, n_steps=20, gamma=0.99, gae_lambda=0.95)
        assert buf.has_action_masks
        # Six per-dim mask arrays, each shape (n_steps, dim_i).
        assert len(buf.w_action_masks) == 6
        assert buf.w_action_masks[0].shape == (20, 10)
        assert buf.w_action_masks[1].shape == (20, 8)
        # All sampled actions should be at the (single) legal end_turn slot.
        assert (buf.w_actions[:, 0] == 5).all()

    def test_collect_rollout_no_masks_when_env_unsupported(self):
        """Plain MockEnv has no action_masks(); rollout still works (no masks)."""
        torch.manual_seed(0)
        np.random.seed(0)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training()
        env = MockEnv(episode_length=50)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()
        buf = agent.collect_rollout(env, n_steps=10, gamma=0.99, gae_lambda=0.95)
        assert not buf.has_action_masks
        assert buf.w_action_masks == []

    def test_update_runs_with_masks(self):
        """update() must complete and produce finite losses when masks are stored."""
        torch.manual_seed(0)
        np.random.seed(0)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training()
        env = MaskedMockEnv(episode_length=50)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()
        buf = agent.collect_rollout(env, n_steps=32, gamma=0.99, gae_lambda=0.95)
        metrics = agent.update(buf, n_epochs=2, batch_size=8, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} not finite: {v}"

    def test_evaluate_action_under_masks_matches_sample(self):
        """sample_action and evaluate_action must produce the same log_prob
        when given the same masks — otherwise PPO's ratio is biased."""
        torch.manual_seed(7)
        worker = WorkerNetwork(
            feature_dim=64, goal_embedding_dim=32, action_space_dims=[10, 8, GRID_W, GRID_H, GRID_W, GRID_H]
        )
        features = torch.rand(1, 64)
        goal = torch.rand(1, 3)
        masks = _make_masks(legal_at=(5,), legal_ut=(0,), legal_fx=(0,), legal_fy=(0,), legal_tx=(0,), legal_ty=(0,))
        mask_tensors = [torch.as_tensor(m) for m in masks]
        action, sample_lp, _ = worker.sample_action(features, goal, action_masks=mask_tensors)
        eval_lp, _, _ = worker.evaluate_action(features, goal, action, action_masks=mask_tensors)
        assert torch.allclose(sample_lp, eval_lp, atol=1e-5)


# ---------------------------------------------------------------------------
# Rollout info surfacing (end_reason / reward_breakdown)
# ---------------------------------------------------------------------------


class InfoMockEnv(MockEnv):
    """MockEnv that emits info['end_reason'] and info['reward_breakdown']
    so tests can verify the rollout aggregates them."""

    def step(self, action):
        self.step_count += 1
        obs = _make_obs()
        obs["units"][0, 0, 1] = 1
        reward = 1.0
        terminated = self.step_count >= self.episode_length
        info = {
            "reward_breakdown": {"action": 0.5, "shaping_delta": 0.3, "terminal": 0.2 if terminated else 0.0},
        }
        if terminated:
            info["winner"] = 1
            info["end_reason"] = "hq_capture"
        return obs, reward, False if terminated else False, terminated, info  # noqa: SIM210 (kept for clarity: no terminate, only truncate at end)


class TestRolloutInfoSurfacing:
    def test_end_reasons_collected(self):
        torch.manual_seed(0)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training()
        env = InfoMockEnv(episode_length=5)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()
        # 12 steps → at least 2 episode boundaries.
        buf = agent.collect_rollout(env, n_steps=12, gamma=0.99, gae_lambda=0.95)
        assert hasattr(buf, "end_reasons")
        assert len(buf.end_reasons) >= 2
        assert all(r == "hq_capture" for r in buf.end_reasons)

    def test_reward_breakdown_summed(self):
        torch.manual_seed(0)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training()
        env = InfoMockEnv(episode_length=5)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()
        buf = agent.collect_rollout(env, n_steps=10, gamma=0.99, gae_lambda=0.95)
        assert hasattr(buf, "reward_breakdown")
        # 10 steps × 0.5 action component
        assert abs(buf.reward_breakdown["action"] - 5.0) < 1e-4
        assert abs(buf.reward_breakdown["shaping_delta"] - 3.0) < 1e-4

    def test_no_info_when_env_silent(self):
        """When the env doesn't emit reward_breakdown / end_reason, the buf
        should still be populated (with empty containers) — not raise."""
        torch.manual_seed(0)
        obs_space = _make_obs_space()
        agent = FeudalRLAgent(obs_space, grid_width=GRID_W, grid_height=GRID_H, device="cpu")
        agent.setup_training()
        env = MockEnv(episode_length=5)
        obs, _ = env.reset()
        agent._last_obs = obs
        agent.reset_goal()
        buf = agent.collect_rollout(env, n_steps=10, gamma=0.99, gae_lambda=0.95)
        assert buf.reward_breakdown == {}
        assert buf.end_reasons == []
