"""
Tests for the AlphaStar-style autoregressive worker head.
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from reinforcetactics.rl.feudal_rl import (
    AutoregressiveActionHead,
    AutoregressiveWorkerNetwork,
    FeudalRLAgent,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def head():
    return AutoregressiveActionHead(feature_dim=64, grid_height=8, grid_width=10, num_action_types=10, num_unit_types=8)


@pytest.fixture
def features():
    return torch.randn(4, 64)  # batch of 4


@pytest.fixture
def fake_obs_space():
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0, high=255, shape=(8, 10, 3), dtype=np.float32),
            "units": spaces.Box(low=0, high=255, shape=(8, 10, 3), dtype=np.float32),
            "global_features": spaces.Box(low=0, high=10000, shape=(6,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(800,), dtype=np.float32),
        }
    )


# --------------------------------------------------------------------------- #
# AutoregressiveActionHead — shape & sampling invariants
# --------------------------------------------------------------------------- #


class TestAutoregressiveHeadShapes:
    def test_sample_returns_six_tuple(self, head, features):
        action, log_prob = head.sample(features)
        assert action.shape == (4, 6)
        assert action.dtype == torch.long
        assert log_prob.shape == (4,)

    def test_sampled_indices_in_range(self, head, features):
        action, _ = head.sample(features)
        assert (action[:, 0] >= 0).all() and (action[:, 0] < 10).all()  # atype
        assert (action[:, 1] >= 0).all() and (action[:, 1] < 8).all()  # unit_type
        assert (action[:, 2] >= 0).all() and (action[:, 2] < 10).all()  # src_x (W=10)
        assert (action[:, 3] >= 0).all() and (action[:, 3] < 8).all()  # src_y (H=8)
        assert (action[:, 4] >= 0).all() and (action[:, 4] < 10).all()  # tgt_x
        assert (action[:, 5] >= 0).all() and (action[:, 5] < 8).all()  # tgt_y

    def test_evaluate_matches_sample_logprob(self, head, features):
        action, lp_sample = head.sample(features)
        lp_eval, entropy = head.evaluate(features, action)
        assert lp_eval.shape == (4,)
        assert entropy.shape == (4,)
        # Sampling and re-evaluating must yield identical log-prob (no
        # stochasticity inside evaluate).
        torch.testing.assert_close(lp_eval, lp_sample, rtol=1e-5, atol=1e-5)

    def test_deterministic_is_repeatable(self, head, features):
        a1, _ = head.sample(features, deterministic=True)
        a2, _ = head.sample(features, deterministic=True)
        torch.testing.assert_close(a1, a2)

    def test_logprob_is_negative(self, head, features):
        _, lp = head.sample(features)
        # Joint log-prob over 4 categoricals must be < 0 (each stage's max
        # log-prob is 0 only in the degenerate single-option case).
        assert (lp <= 0).all()


# --------------------------------------------------------------------------- #
# Mask handling
# --------------------------------------------------------------------------- #


class TestAutoregressiveHeadMasks:
    def test_atype_mask_restricts_sampled_action(self, head, features):
        # Only allow atype=5 (end_turn).
        atype_mask = torch.zeros(4, 10, dtype=torch.bool)
        atype_mask[:, 5] = True
        action, _ = head.sample(features, masks={"atype": atype_mask})
        assert (action[:, 0] == 5).all()

    def test_src_mask_restricts_source(self, head, features):
        atype_mask = torch.zeros(4, 10, dtype=torch.bool)
        atype_mask[:, 5] = True
        src_mask = torch.zeros(4, 8 * 10, dtype=torch.bool)
        src_mask[:, 0] = True  # only (sx=0, sy=0) legal
        action, _ = head.sample(features, masks={"atype": atype_mask, "src": src_mask})
        assert (action[:, 2] == 0).all()  # src_x
        assert (action[:, 3] == 0).all()  # src_y

    def test_target_mask_restricts_target(self, head, features):
        atype_mask = torch.zeros(4, 10, dtype=torch.bool)
        atype_mask[:, 5] = True
        # Pin target to (tx=3, ty=2) -> idx = 2*10 + 3 = 23
        tgt_mask = torch.zeros(4, 8 * 10, dtype=torch.bool)
        tgt_mask[:, 23] = True
        action, _ = head.sample(features, masks={"atype": atype_mask, "target": tgt_mask})
        assert (action[:, 4] == 3).all()  # tgt_x
        assert (action[:, 5] == 2).all()  # tgt_y

    def test_unit_type_mask_restricts_unit(self, head, features):
        ut_mask = torch.zeros(4, 8, dtype=torch.bool)
        ut_mask[:, 3] = True
        action, _ = head.sample(features, masks={"unit_type": ut_mask})
        assert (action[:, 1] == 3).all()

    def test_evaluate_with_masks_consistent(self, head, features):
        atype_mask = torch.zeros(4, 10, dtype=torch.bool)
        atype_mask[:, [2, 5]] = True
        action, lp_sample = head.sample(features, masks={"atype": atype_mask})
        lp_eval, _ = head.evaluate(features, action, masks={"atype": atype_mask})
        torch.testing.assert_close(lp_eval, lp_sample, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# Backprop
# --------------------------------------------------------------------------- #


class TestAutoregressiveHeadBackprop:
    def test_loss_propagates_to_all_stages(self, head, features):
        features = features.requires_grad_(True)
        action, log_prob = head.sample(features)
        # Surrogate policy-gradient loss
        loss = -log_prob.mean()
        loss.backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()
        # All four stage parameter groups should have gradients.
        for name, p in head.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


# --------------------------------------------------------------------------- #
# AutoregressiveWorkerNetwork — drop-in compatibility
# --------------------------------------------------------------------------- #


class TestAutoregressiveWorkerNetwork:
    def test_sample_action_signature(self):
        worker = AutoregressiveWorkerNetwork(feature_dim=64, goal_embedding_dim=16, grid_width=10, grid_height=8)
        features = torch.randn(2, 64)
        goal = torch.tensor([[3.0, 4.0, 1.0], [0.0, 0.0, 0.0]])
        action, log_prob, value = worker.sample_action(features, goal)
        assert action.shape == (2, 6)
        assert log_prob.shape == (2,)
        assert value.shape == (2, 1)

    def test_evaluate_action_matches_sample(self):
        worker = AutoregressiveWorkerNetwork(feature_dim=64, goal_embedding_dim=16, grid_width=10, grid_height=8)
        features = torch.randn(2, 64)
        goal = torch.tensor([[3.0, 4.0, 1.0], [0.0, 0.0, 0.0]])
        action, lp_sample, _ = worker.sample_action(features, goal)
        lp_eval, entropy, _ = worker.evaluate_action(features, goal, action)
        torch.testing.assert_close(lp_eval, lp_sample, rtol=1e-5, atol=1e-5)
        assert entropy.shape == (2,)

    def test_forward_returns_none_logits(self):
        worker = AutoregressiveWorkerNetwork(feature_dim=64, goal_embedding_dim=16, grid_width=10, grid_height=8)
        features = torch.randn(2, 64)
        goal = torch.zeros(2, 3)
        logits, value = worker(features, goal)
        # AR head can't be expressed as independent per-dim logits.
        assert logits is None
        assert value.shape == (2, 1)


# --------------------------------------------------------------------------- #
# FeudalRLAgent integration
# --------------------------------------------------------------------------- #


class TestFeudalAgentAutoregressiveFlag:
    def test_agent_uses_autoregressive_worker_when_flag_set(self, fake_obs_space):
        agent = FeudalRLAgent(
            fake_obs_space,
            grid_width=10,
            grid_height=8,
            device="cpu",
            autoregressive_worker=True,
        )
        assert isinstance(agent.worker, AutoregressiveWorkerNetwork)
        assert agent.autoregressive_worker is True

    def test_agent_default_uses_independent_worker(self, fake_obs_space):
        agent = FeudalRLAgent(fake_obs_space, grid_width=10, grid_height=8, device="cpu")
        assert not isinstance(agent.worker, AutoregressiveWorkerNetwork)
        assert agent.autoregressive_worker is False

    def test_select_action_works_with_autoregressive_worker(self, fake_obs_space):
        agent = FeudalRLAgent(
            fake_obs_space,
            grid_width=10,
            grid_height=8,
            device="cpu",
            autoregressive_worker=True,
        )
        obs = {
            "grid": np.zeros((8, 10, 3), dtype=np.float32),
            "units": np.zeros((8, 10, 3), dtype=np.float32),
            "global_features": np.zeros(6, dtype=np.float32),
        }
        # Stochastic
        action, goal = agent.select_action(obs, deterministic=False)
        assert action.shape == (6,)
        assert goal.shape == (3,)
        # Deterministic — should not raise even though forward returns None logits.
        action_det, _ = agent.select_action(obs, deterministic=True)
        assert action_det.shape == (6,)
