"""
End-to-end tests for the masked autoregressive worker path: provider →
stage-by-stage AR sampling → buffer mask storage → masked PPO update.
"""

import numpy as np
import pytest
import torch

from reinforcetactics.rl.feudal_rl import (
    AutoregressiveActionHead,
    FeudalRLAgent,
    FeudalRolloutBuffer,
    StructuredMaskProvider,
)
from reinforcetactics.rl.gym_env import StrategyGameEnv


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def env():
    e = StrategyGameEnv(map_file=None, opponent="random", render_mode=None, max_steps=20)
    e.reset(seed=0)
    yield e
    e.close()


# --------------------------------------------------------------------------- #
# StructuredMaskProvider
# --------------------------------------------------------------------------- #


class TestStructuredMaskProvider:
    def test_atype_mask_matches_env(self, env):
        sm = env.structured_action_masks()
        provider = StructuredMaskProvider(sm, env.grid_height, env.grid_width)
        m = provider.atype_mask()
        assert m.shape == (1, 10)
        assert m.dtype == torch.bool
        np.testing.assert_array_equal(m.squeeze(0).numpy(), sm.atype)

    def test_src_mask_matches_source_slice(self, env):
        sm = env.structured_action_masks()
        provider = StructuredMaskProvider(sm, env.grid_height, env.grid_width)
        # End turn (atype=5) should yield a single legal source bit at (0, 0).
        m = provider.src_mask(torch.tensor([5]))
        assert m.shape == (1, env.grid_height * env.grid_width)
        assert m[0, 0].item() is True
        assert m.sum().item() == 1

    def test_target_mask_falls_back_to_all_true(self, env):
        sm = env.structured_action_masks()
        provider = StructuredMaskProvider(sm, env.grid_height, env.grid_width)
        # Pick an (atype, sx, sy) that's certainly not in the lookup —
        # e.g. atype=2 (attack) at a corner where no own unit exists.
        m = provider.target_mask(torch.tensor([2]), torch.tensor([0]), torch.tensor([env.grid_height - 1]))
        assert m.shape == (1, env.grid_height * env.grid_width)
        # Fallback: all True
        if (2, 0, env.grid_height - 1) not in sm.target:
            assert m.all().item()


# --------------------------------------------------------------------------- #
# AR head sample_with_provider — masked sampling is exact
# --------------------------------------------------------------------------- #


class TestSampleWithProvider:
    def test_returns_legal_action_for_env_masks(self, env):
        sm = env.structured_action_masks()
        provider = StructuredMaskProvider(sm, env.grid_height, env.grid_width)
        head = AutoregressiveActionHead(feature_dim=32, grid_height=env.grid_height, grid_width=env.grid_width)
        features = torch.randn(1, 32)
        # Repeat many times to catch any rare illegal sample.
        for _ in range(50):
            action, _, _ = head.sample_with_provider(features, provider)
            atype = action[0, 0].item()
            sx = action[0, 2].item()
            sy = action[0, 3].item()
            tx = action[0, 4].item()
            ty = action[0, 5].item()
            # Each sampled stage value must be allowed by the env's structured masks.
            assert sm.atype[atype], f"atype {atype} sampled but masked illegal"
            assert sm.source[atype, sy, sx], f"src ({sx},{sy}) sampled illegal for atype {atype}"
            target = sm.target.get((atype, sx, sy))
            if target is not None:
                assert target[ty, tx], f"tgt ({tx},{ty}) sampled illegal for ({atype},{sx},{sy})"

    def test_evaluate_with_returned_masks_is_exact(self, env):
        sm = env.structured_action_masks()
        provider = StructuredMaskProvider(sm, env.grid_height, env.grid_width)
        head = AutoregressiveActionHead(feature_dim=32, grid_height=env.grid_height, grid_width=env.grid_width)
        features = torch.randn(1, 32)
        action, lp_sample, masks = head.sample_with_provider(features, provider)
        lp_eval, _ = head.evaluate(features, action, masks=masks)
        torch.testing.assert_close(lp_eval, lp_sample, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# FeudalRolloutBuffer mask storage
# --------------------------------------------------------------------------- #


class TestRolloutBufferMaskStorage:
    def test_store_masks_off_by_default(self):
        buf = FeudalRolloutBuffer()
        assert buf.store_masks is False
        # Adding without masks does not raise.
        obs = {"grid": np.zeros((4, 4, 3)), "units": np.zeros((4, 4, 3)), "global_features": np.zeros(6)}
        buf.add_worker_step(obs, np.zeros(6, dtype=np.int64), 0.0, 0.0, np.zeros(3), 0.0, 0.0, False, 0.5)

    def test_store_masks_requires_masks(self):
        buf = FeudalRolloutBuffer(store_masks=True)
        obs = {"grid": np.zeros((4, 4, 3)), "units": np.zeros((4, 4, 3)), "global_features": np.zeros(6)}
        with pytest.raises(ValueError):
            buf.add_worker_step(obs, np.zeros(6, dtype=np.int64), 0.0, 0.0, np.zeros(3), 0.0, 0.0, False, 0.5)

    def test_finalize_stacks_masks(self):
        buf = FeudalRolloutBuffer(store_masks=True)
        obs = {"grid": np.zeros((4, 4, 3)), "units": np.zeros((4, 4, 3)), "global_features": np.zeros(6)}
        masks = {
            "atype": np.zeros(10, dtype=bool),
            "src": np.zeros(16, dtype=bool),
            "unit_type": np.zeros(8, dtype=bool),
            "target": np.zeros(16, dtype=bool),
        }
        for _ in range(3):
            buf.add_worker_step(obs, np.zeros(6, dtype=np.int64), 0.0, 0.0, np.zeros(3), 0.0, 0.0, False, 0.5, masks=masks)
            buf.add_manager_step(obs, np.zeros(3), 0.0, 0.0)
        buf.end_manager_segment(0.0, False, 1)
        buf.finalize()
        assert buf.w_mask_atype.shape == (3, 10)
        assert buf.w_mask_src.shape == (3, 16)
        assert buf.w_mask_unit_type.shape == (3, 8)
        assert buf.w_mask_target.shape == (3, 16)


# --------------------------------------------------------------------------- #
# FeudalRLAgent end-to-end with masked AR worker
# --------------------------------------------------------------------------- #


class TestFeudalAgentMaskedRollout:
    def test_rollout_stores_masks_when_ar_enabled(self, env):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        agent.setup_training(learning_rate=1e-3)
        agent._last_obs = env.reset(seed=1)[0]
        buf = agent.collect_rollout(env, n_steps=8, gamma=0.99, gae_lambda=0.95)
        assert buf.store_masks is True
        assert buf.w_mask_atype.shape[0] == 8
        assert buf.w_mask_atype.shape[1] == 10
        assert buf.w_mask_src.shape == (8, env.grid_height * env.grid_width)

    def test_rollout_skips_mask_storage_when_ar_disabled(self, env):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=False,
        )
        agent.setup_training(learning_rate=1e-3)
        agent._last_obs = env.reset(seed=1)[0]
        buf = agent.collect_rollout(env, n_steps=8, gamma=0.99, gae_lambda=0.95)
        assert buf.store_masks is False
        # Empty list because nothing was appended.
        assert len(buf.w_mask_atype) == 0

    def test_rollout_actions_are_legal_under_ar_masks(self, env):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        agent.setup_training(learning_rate=1e-3)
        agent._last_obs = env.reset(seed=2)[0]
        buf = agent.collect_rollout(env, n_steps=12, gamma=0.99, gae_lambda=0.95)
        # Every recorded action's atype/src/target bits must be set in the
        # mask that was actually applied at sample time.
        for i, action in enumerate(buf.w_actions):
            atype = int(action[0])
            sx, sy = int(action[2]), int(action[3])
            tx, ty = int(action[4]), int(action[5])
            assert buf.w_mask_atype[i, atype], f"step {i}: atype {atype} not in mask"
            src_idx = sy * env.grid_width + sx
            assert buf.w_mask_src[i, src_idx], f"step {i}: src ({sx},{sy}) not in mask"
            tgt_idx = ty * env.grid_width + tx
            assert buf.w_mask_target[i, tgt_idx], f"step {i}: tgt ({tx},{ty}) not in mask"

    def test_update_runs_with_masks(self, env):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        agent.setup_training(learning_rate=1e-3)
        agent._last_obs = env.reset(seed=3)[0]
        buf = agent.collect_rollout(env, n_steps=16, gamma=0.99, gae_lambda=0.95)
        metrics = agent.update(
            buf,
            n_epochs=1,
            batch_size=8,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        # Sanity: losses are finite numbers.
        assert np.isfinite(metrics["worker_policy_loss"])
        assert np.isfinite(metrics["worker_value_loss"])
        assert np.isfinite(metrics["worker_entropy"])


# --------------------------------------------------------------------------- #
# AR inference: select_action with structured_masks must produce legal actions
# --------------------------------------------------------------------------- #


class TestSelectActionWithStructuredMasks:
    def test_select_action_respects_structured_masks(self, env):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        obs = env.reset(seed=4)[0]
        sm = env.structured_action_masks()
        # Repeat to catch any stage where unmasked sampling could slip through.
        for _ in range(30):
            agent.reset_goal()
            action, _goal = agent.select_action(obs, deterministic=False, structured_masks=sm)
            atype, _ut, sx, sy, tx, ty = (int(x) for x in action)
            assert sm.atype[atype], f"select_action sampled illegal atype {atype}"
            assert sm.source[atype, sy, sx], f"select_action sampled illegal src ({sx},{sy}) for atype {atype}"
            target = sm.target.get((atype, sx, sy))
            if target is not None:
                assert target[ty, tx], f"select_action sampled illegal tgt ({tx},{ty})"

    def test_select_action_deterministic_is_legal(self, env):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        obs = env.reset(seed=5)[0]
        sm = env.structured_action_masks()
        agent.reset_goal()
        action, _goal = agent.select_action(obs, deterministic=True, structured_masks=sm)
        atype, _ut, sx, sy, _tx, _ty = (int(x) for x in action)
        assert sm.atype[atype]
        assert sm.source[atype, sy, sx]


# --------------------------------------------------------------------------- #
# Checkpoint persists autoregressive_worker flag and refuses head mismatch
# --------------------------------------------------------------------------- #


class TestCheckpointFlagPersistence:
    def test_load_into_matching_flag_succeeds(self, env, tmp_path):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        agent.setup_training(learning_rate=1e-3)
        path = tmp_path / "ar.pt"
        agent.save_checkpoint(str(path))

        agent2 = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        agent2.load_checkpoint(str(path))
        for p1, p2 in zip(agent.worker.parameters(), agent2.worker.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_load_into_mismatched_flag_raises(self, env, tmp_path):
        agent = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=True,
        )
        agent.setup_training(learning_rate=1e-3)
        path = tmp_path / "ar.pt"
        agent.save_checkpoint(str(path))

        legacy = FeudalRLAgent(
            env.observation_space,
            grid_width=env.grid_width,
            grid_height=env.grid_height,
            device="cpu",
            autoregressive_worker=False,
        )
        with pytest.raises(ValueError, match="autoregressive_worker"):
            legacy.load_checkpoint(str(path))
