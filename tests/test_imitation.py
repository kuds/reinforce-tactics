"""Tests for behavior-cloning warm-start (reinforcetactics.rl.imitation)."""

from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.rl.imitation import (
    NUM_ACTION_TYPES,
    NUM_UNIT_TYPES,
    DemonstrationDataset,
    DemonstrationScenario,
    behavior_clone,
    collect_demonstrations,
    collect_demonstrations_multi,
    load_scenarios_from_yaml,
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


class TestMultiScenarioCollection:
    """Verify scenario mixing produces a single concatenated dataset."""

    @staticmethod
    def _starter_map() -> str:
        # 6x6 map; cheap to simulate, exists in repo, dimensions stable.
        return str(Path("maps") / "1v1" / "starter.csv")

    def test_multi_scenario_concat(self):
        scenarios = [
            DemonstrationScenario(
                name="all",
                map_file=self._starter_map(),
                enabled_units=None,
                demonstrator="medium",
                opponent="medium",
                n_episodes=1,
                max_turns=20,
            ),
            DemonstrationScenario(
                name="warriors_only",
                map_file=self._starter_map(),
                enabled_units=["W"],
                demonstrator="medium",
                opponent="medium",
                n_episodes=1,
                max_turns=20,
            ),
        ]

        ds = collect_demonstrations_multi(scenarios, seed=42, progress=False)
        # The concat must yield a properly stacked dataset.
        assert len(ds) > 0
        assert ds.actions.shape == (len(ds), 6)
        # Per-dim sizes must be set and self-consistent.
        assert ds.dim_sizes[0] == NUM_ACTION_TYPES
        assert ds.dim_sizes[1] == NUM_UNIT_TYPES

        # All recorded actions remain mask-legal across scenarios.
        offsets = np.cumsum([0, *ds.dim_sizes])
        for i in range(len(ds)):
            for dim, val in enumerate(ds.actions[i]):
                lo = offsets[dim]
                assert ds.masks_concat[i, lo + int(val)]

    def test_unit_restriction_is_enforced(self):
        # When only Warriors are enabled, no create_unit demo should record
        # a non-Warrior unit_type (idx 0). This catches regressions where
        # enabled_units fails to propagate from scenario -> GameState.
        scenario = DemonstrationScenario(
            map_file=self._starter_map(),
            enabled_units=["W"],
            demonstrator="medium",
            opponent="medium",
            n_episodes=2,
            max_turns=20,
        )
        ds = collect_demonstrations_multi([scenario], seed=0, progress=False)

        creates = ds.actions[ds.actions[:, 0] == 0]  # action_type == create_unit
        if creates.size:
            # All create_unit demos must use unit_type idx 0 (Warrior).
            assert (creates[:, 1] == 0).all()

    def test_grid_size_mismatch_raises(self):
        # crossroads is 15x15, starter is 6x6 — concat must fail loudly
        # before any episodes run.
        scenarios = [
            DemonstrationScenario(map_file=self._starter_map(), n_episodes=1, max_turns=10),
            DemonstrationScenario(
                map_file=str(Path("maps") / "1v1" / "crossroads.csv"),
                n_episodes=1,
                max_turns=10,
            ),
        ]
        with pytest.raises(ValueError, match="grid"):
            collect_demonstrations_multi(scenarios, seed=0)

    def test_empty_scenarios_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            collect_demonstrations_multi([], seed=0)


class TestScenarioYAMLLoader:
    def test_round_trip(self, tmp_path):
        text = """\
scenarios:
  - name: a
    map_file: maps/1v1/starter.csv
    enabled_units: [W, S]
    demonstrator: medium
    opponent: medium
    n_episodes: 5
    weight: 2.0
  - name: b
    map_file: maps/1v1/starter.csv
    demonstrator: advanced
    n_episodes: 3
"""
        path = tmp_path / "scenarios.yaml"
        path.write_text(text)
        scenarios = load_scenarios_from_yaml(str(path))
        assert len(scenarios) == 2
        assert scenarios[0].name == "a"
        assert scenarios[0].enabled_units == ["W", "S"]
        assert scenarios[0].weight == 2.0
        assert scenarios[1].name == "b"
        assert scenarios[1].demonstrator == "advanced"
        # Defaults preserved on omitted fields.
        assert scenarios[1].weight == 1.0

    def test_unknown_key_raises(self, tmp_path):
        path = tmp_path / "scenarios.yaml"
        path.write_text("scenarios:\n  - name: x\n    typo_field: 1\n")
        with pytest.raises(ValueError, match="unknown keys"):
            load_scenarios_from_yaml(str(path))

    def test_empty_list_raises(self, tmp_path):
        path = tmp_path / "scenarios.yaml"
        path.write_text("scenarios: []\n")
        with pytest.raises(ValueError, match="non-empty"):
            load_scenarios_from_yaml(str(path))
