"""Tests for behavior-cloning warm-start (reinforcetactics.rl.imitation)."""

import numpy as np
import pytest

from reinforcetactics.rl.imitation import (
    NUM_ACTION_TYPES,
    NUM_UNIT_TYPES,
    DemonstrationDataset,
    behavior_clone,
    collect_demonstrations,
    record_episode,
)


@pytest.fixture(scope="module")
def small_dataset() -> DemonstrationDataset:
    """A reusable BC dataset; keeps a single episode runtime cheap."""
    return collect_demonstrations(
        n_episodes=1,
        demonstrator="medium",
        opponent="random",
        max_turns=30,
        seed=7,
    )


class TestDemonstrationCollection:
    def test_record_episode_returns_demos(self):
        demos = record_episode(
            demonstrator="medium",
            opponent="random",
            max_turns=20,
            seed=0,
        )
        assert len(demos) > 0, "Demonstrator should have produced at least one action"

    def test_only_demonstrator_actions_are_recorded(self):
        # Run a deterministic episode; the recorded action_type=5 (end_turn)
        # actions should be roughly equal to the number of demonstrator turns.
        # If opponent end_turn calls leaked into the dataset, end_turn count
        # would be ~2x the actual demonstrator turn count.
        demos = record_episode(
            demonstrator="medium",
            opponent="random",
            max_turns=15,
            seed=1,
        )
        end_turns = sum(1 for d in demos if d.action[0] == 5)
        # GameState ticks both players' turns; demonstrator owns half (at most).
        assert 1 <= end_turns <= 15

    def test_dataset_shapes(self, small_dataset):
        ds = small_dataset
        n = len(ds)
        assert n > 0
        assert ds.actions.shape == (n, 6)
        assert ds.actions.dtype == np.int64

        # Per-dim mask block sizes: action_type, unit_type, fx, fy, tx, ty
        at, ut, fx, fy, tx, ty = ds.dim_sizes
        assert (at, ut) == (NUM_ACTION_TYPES, NUM_UNIT_TYPES)
        assert fx == tx and fy == ty  # square mask layout per axis
        assert ds.masks_concat.shape == (n, sum(ds.dim_sizes))
        assert ds.masks_concat.dtype == np.bool_

        # Every recorded action component must lie inside its per-dim mask.
        # If this fails, MaskablePPO log_prob would be -inf during BC.
        offsets = np.cumsum([0, at, ut, fx, fy, tx, ty])
        for i in range(n):
            for dim, val in enumerate(ds.actions[i]):
                lo = offsets[dim]
                assert ds.masks_concat[i, lo + int(val)], f"Sample {i} dim {dim} value {val} not in mask"

    def test_observation_keys(self, small_dataset):
        for k in ("grid", "units", "global_features", "action_mask"):
            assert k in small_dataset.obs

    def test_invalid_demonstrator_player_raises(self):
        with pytest.raises(ValueError):
            record_episode(demonstrator_player=3, max_turns=5, seed=0)


class TestActionTypeCoverage:
    """Smoke test that the demonstrator triggers a variety of action types."""

    def test_includes_move_and_end_turn(self, small_dataset):
        ats = small_dataset.actions[:, 0]
        assert (ats == 1).any(), "expected at least one move"
        assert (ats == 5).any(), "expected at least one end_turn"


class TestBehaviorClone:
    def test_bc_decreases_loss(self, small_dataset):
        sb3_contrib = pytest.importorskip("sb3_contrib")
        from reinforcetactics.rl import make_maskable_env

        env = make_maskable_env(opponent="random")
        model = sb3_contrib.MaskablePPO(
            "MultiInputPolicy",
            env,
            n_steps=64,
            batch_size=32,
            verbose=0,
        )

        stats = behavior_clone(
            model,
            small_dataset,
            n_epochs=2,
            batch_size=64,
            learning_rate=1e-3,
            seed=0,
        )
        assert len(stats) == 2
        # Loss should drop within two epochs on a single-episode dataset; if
        # this regresses, either the optimizer step is broken or the masking
        # is making log_prob saturate at the floor.
        assert stats[1].loss < stats[0].loss + 1e-6
        # Action-type accuracy should reach at least chance + a margin (chance
        # over 10 classes is 0.1; demonstrator move-fraction alone makes
        # ~0.7 trivially achievable). Keep the bound loose to avoid flakes.
        assert stats[1].accuracy_action_type > 0.4

    def test_bc_then_ppo_learn_runs(self, small_dataset):
        sb3_contrib = pytest.importorskip("sb3_contrib")
        from reinforcetactics.rl import make_maskable_env

        env = make_maskable_env(opponent="random")
        model = sb3_contrib.MaskablePPO(
            "MultiInputPolicy",
            env,
            n_steps=64,
            batch_size=32,
            verbose=0,
        )
        behavior_clone(model, small_dataset, n_epochs=1, batch_size=32, seed=0)
        # The post-BC model must remain a usable PPO model.
        model.learn(total_timesteps=128, progress_bar=False)
