"""Rollout policy probes and eval logging must preserve training state."""

from __future__ import annotations

import csv
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat
from stable_baselines3.common.torch_layers import CombinedExtractor, FlattenExtractor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from reinforcetactics.rl.callbacks import PeriodicEvalCallback, TrainingMetricsCallback


def _consume_random_streams():
    torch.rand(7)
    np.random.random(7)
    random.random()


def _assert_random_states_unchanged(torch_rng, numpy_rng, python_rng):
    assert torch.equal(torch.get_rng_state(), torch_rng)
    after_numpy_rng = np.random.get_state()
    assert after_numpy_rng[0] == numpy_rng[0]
    np.testing.assert_array_equal(after_numpy_rng[1], numpy_rng[1])
    assert after_numpy_rng[2:] == numpy_rng[2:]
    assert random.getstate() == python_rng


def _masked_model(*, dictionary=False, forced_only=False):
    """Uniform policy over 1, 2, or 4 offered choices in a 2-env rollout."""
    obs_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    observations = np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 12
    if dictionary:
        obs_space = spaces.Dict({"board": obs_space})
        observations = {"board": observations}
    policy = MaskableActorCriticPolicy(
        obs_space,
        spaces.Discrete(4),
        lambda _: 0.001,
        features_extractor_class=CombinedExtractor if dictionary else FlattenExtractor,
    )
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
    legal_counts = [1] * 6 if forced_only else [1, 2, 4, 1, 2, 4]
    masks = (np.arange(4)[None, :] < np.asarray(legal_counts)[:, None]).reshape(3, 2, 4)
    buffer = SimpleNamespace(
        buffer_size=3,
        full=True,
        pos=3,
        n_envs=2,
        generator_ready=False,
        action_masks=masks.astype(np.float32),
        observations=observations,
    )
    return SimpleNamespace(
        action_space=spaces.Discrete(4),
        rollout_buffer=buffer,
        policy=policy,
        device=torch.device("cpu"),
        logger=Logger(folder=None, output_formats=[]),
        ent_coef=0.01,
        ep_info_buffer=[],
    )


@pytest.mark.parametrize("dictionary", [False, True])
def test_normalized_entropy_and_forced_fraction_preserve_policy_rng_and_buffer(dictionary, tmp_path):
    model = _masked_model(dictionary=dictionary)
    # Include a random draw even though distribution entropy itself is
    # deterministic, checking protection for a custom feature extractor.
    original_get_distribution = model.policy.get_distribution

    def get_distribution(*args, **kwargs):
        _consume_random_streams()
        return original_get_distribution(*args, **kwargs)

    model.policy.get_distribution = get_distribution
    model.policy.set_training_mode(True)
    original_set_training_mode = model.policy.set_training_mode

    def set_training_mode(mode):
        _consume_random_streams()
        original_set_training_mode(mode)

    model.policy.set_training_mode = set_training_mode
    before_rng = torch.get_rng_state().clone()
    before_numpy_rng = np.random.get_state()
    before_python_rng = random.getstate()
    before_masks = model.rollout_buffer.action_masks.copy()
    cb = TrainingMetricsCallback(csv_path=tmp_path / "metrics.csv")
    cb.model = model
    cb.num_timesteps = 6
    cb._on_rollout_end()
    cb._on_training_end()

    row = cb.records[0]
    assert row["rollout/normalized_policy_entropy"] == pytest.approx(1.0)
    assert row["rollout/policy_entropy"] == pytest.approx((np.log(2) + np.log(4)) / 3)
    assert row["rollout/forced_action_fraction"] == pytest.approx(1 / 3)
    assert row["rollout/legal_actions_mean"] == pytest.approx(7 / 3)
    assert row["rollout/legal_actions_p50"] == 2
    assert row["rollout/policy_diagnostic_samples"] == 6
    _assert_random_states_unchanged(before_rng, before_numpy_rng, before_python_rng)
    np.testing.assert_array_equal(model.rollout_buffer.action_masks, before_masks)
    assert model.rollout_buffer.generator_ready is False
    assert model.policy.training is True
    assert model.logger.name_to_value["rollout/normalized_policy_entropy"] == pytest.approx(1.0)
    with (tmp_path / "metrics.csv").open() as fh:
        persisted = list(csv.DictReader(fh))[0]
    assert float(persisted["rollout/normalized_policy_entropy"]) == pytest.approx(1.0)


@pytest.mark.parametrize("failure_at", ["distribution", "disable_mode", "restore_mode"])
def test_policy_probe_restores_all_random_streams_and_mode_after_failure(failure_at):
    model = _masked_model()
    original_get_distribution = model.policy.get_distribution
    original_set_training_mode = model.policy.set_training_mode
    model.policy.set_training_mode(True)

    def get_distribution(*args, **kwargs):
        _consume_random_streams()
        if failure_at == "distribution":
            raise RuntimeError("custom feature extractor failed")
        return original_get_distribution(*args, **kwargs)

    def set_training_mode(mode):
        original_set_training_mode(mode)
        _consume_random_streams()
        if failure_at == ("restore_mode" if mode else "disable_mode"):
            raise RuntimeError("custom mode hook failed")

    model.policy.get_distribution = get_distribution
    model.policy.set_training_mode = set_training_mode
    before_rng = torch.get_rng_state().clone()
    before_numpy_rng = np.random.get_state()
    before_python_rng = random.getstate()
    cb = TrainingMetricsCallback()
    cb.model = model
    cb._policy_diagnostics()
    assert cb._policy_diagnostic_warned
    assert model.policy.training is True
    _assert_random_states_unchanged(before_rng, before_numpy_rng, before_python_rng)


def test_forced_only_rollout_omits_undefined_normalized_entropy():
    cb = TrainingMetricsCallback()
    cb.model = _masked_model(forced_only=True)
    metrics = cb._policy_diagnostics()
    assert metrics["rollout/forced_action_fraction"] == 1
    assert metrics["rollout/policy_entropy"] == pytest.approx(0.0)
    assert "rollout/normalized_policy_entropy" not in metrics


def test_policy_probe_sample_cap_and_opt_out():
    cb = TrainingMetricsCallback(policy_diagnostic_samples=2)
    cb.model = _masked_model()
    assert cb._policy_diagnostics()["rollout/policy_diagnostic_samples"] == 2
    cb.policy_diagnostic_samples = 0
    assert cb._policy_diagnostics() == {}
    with pytest.raises(ValueError, match="non-negative"):
        TrainingMetricsCallback(policy_diagnostic_samples=-1)


def test_resume_upgrades_legacy_csv_atomically_preserving_rows_and_unknown_columns(tmp_path):
    path = tmp_path / "training_metrics.csv"
    path.write_text("timesteps,train/approx_kl,annotation\n8,0.01,old run\n", encoding="utf-8")
    cb = TrainingMetricsCallback(csv_path=path)
    cb._append_csv_row({"timesteps": 16, "train/approx_kl": 0.02, "rollout/normalized_policy_entropy": 0.75})
    # Repeated appends must keep the migrated header (without duplicates).
    cb._append_csv_row({"timesteps": 24, "rollout/normalized_policy_entropy": 0.5})
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames[:3] == ["timesteps", "train/approx_kl", "annotation"]
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
    assert [r["timesteps"] for r in rows] == ["8", "16", "24"]
    assert rows[0]["annotation"] == "old run"
    assert rows[0]["train/approx_kl"] == "0.01"
    assert rows[0]["rollout/normalized_policy_entropy"] == ""
    assert rows[1]["annotation"] == ""
    assert rows[1]["rollout/normalized_policy_entropy"] == "0.75"
    assert all(None not in row for row in rows)
    assert list(tmp_path.iterdir()) == [path]


def test_existing_empty_csv_gets_header_before_first_row(tmp_path):
    path = tmp_path / "training_metrics.csv"
    path.touch()
    cb = TrainingMetricsCallback(csv_path=path)
    cb._append_csv_row({"timesteps": 8, "rollout/normalized_policy_entropy": 0.75})
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        assert reader.fieldnames == list(cb._CSV_COLUMNS)
    assert rows[0]["timesteps"] == "8"
    assert rows[0]["rollout/normalized_policy_entropy"] == "0.75"


def test_failed_csv_schema_upgrade_keeps_original_file(tmp_path, monkeypatch):
    path = tmp_path / "training_metrics.csv"
    original = "timesteps,annotation\n8,old run\n"
    path.write_text(original, encoding="utf-8")
    cb = TrainingMetricsCallback(csv_path=path)
    failures = []
    cb.on_write_failure = failures.append

    def fail_replace(*args):
        raise OSError("interrupted migration")

    monkeypatch.setattr("reinforcetactics.rl.callbacks.os.replace", fail_replace)
    cb._append_csv_row({"timesteps": 16})
    assert path.read_text(encoding="utf-8") == original
    assert len(failures) == 1
    assert list(tmp_path.iterdir()) == [path]


class _MaskedChoiceEnv(Env):
    observation_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def action_masks(self):
        return np.arange(4) <= self.steps % 4

    def step(self, action):
        assert self.action_masks()[action]
        self.steps += 1
        return np.full(2, self.steps / 8, dtype=np.float32), float(action), self.steps == 8, False, {}


class _RandomizedFeaturesExtractor(FlattenExtractor):
    """Exercise custom preprocessing that samples outside torch's RNG."""

    def forward(self, observations):
        features = super().forward(observations)
        numpy_noise = torch.as_tensor(np.random.normal(size=features.shape), device=features.device, dtype=features.dtype)
        return features + 0.1 * (numpy_noise + random.random() + torch.rand_like(features))


@pytest.mark.parametrize("extractor", [FlattenExtractor, _RandomizedFeaturesExtractor])
def test_policy_probe_does_not_change_actual_maskable_ppo_updates(extractor):
    models = []
    for sample_count in (0, 8):
        model = MaskablePPO(
            "MlpPolicy",
            _MaskedChoiceEnv(),
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            seed=123,
            policy_kwargs={"net_arch": [8], "features_extractor_class": extractor},
            device="cpu",
        )
        cb = TrainingMetricsCallback(policy_diagnostic_samples=sample_count)
        model.learn(total_timesteps=16, callback=cb)
        assert len(cb.records) == 2
        if sample_count:
            assert all(row["rollout/policy_diagnostic_samples"] == 8 for row in cb.records)
        models.append(model)
    for name, actual in models[1].policy.state_dict().items():
        assert torch.equal(actual, models[0].policy.state_dict()[name]), name


def test_eval_tensorboard_keeps_each_measurement_step_without_clearing_train_metrics(tmp_path):
    output = TensorBoardOutputFormat(str(tmp_path))
    logger = Logger(folder=str(tmp_path), output_formats=[output])
    logger.record("train/approx_kl", 0.012)
    cb = PeriodicEvalCallback(object(), eval_freq=10, verbose=0)
    cb.model = SimpleNamespace(logger=logger)
    cb._record_eval_scalars({"eval/win_rate": 0.25}, 10)
    cb._record_eval_scalars({"eval/win_rate": 0.5}, 20)
    # SB3's next rollout dump must neither move nor duplicate either eval.
    assert logger.name_to_value["train/approx_kl"] == 0.012
    logger.dump(64)
    output.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert [(event.step, event.value) for event in events.Scalars("eval/win_rate")] == [(10, 0.25), (20, 0.5)]
    assert [(event.step, event.value) for event in events.Scalars("train/approx_kl")] == [(64, pytest.approx(0.012))]


def test_eval_forwards_trace_reasons_and_episode_records(monkeypatch, tmp_path):
    calls = []

    def evaluate(model, env, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr("reinforcetactics.rl.callbacks.evaluate_model", evaluate)
    cb = PeriodicEvalCallback(
        object(),
        eval_freq=10,
        trace_end_reasons=("max_steps_truncate", "max_turns_draw"),
        max_traces=3,
        include_episode_records=True,
    )
    cb.model = object()
    cb._evaluate(deterministic=True, seed=7, trace_dir=tmp_path, track_breakdown=False)
    assert calls[0]["trace_end_reasons"] == ("max_steps_truncate", "max_turns_draw")
    assert calls[0]["max_traces"] == 3
    assert calls[0]["include_episode_records"] is True
    cb = PeriodicEvalCallback(object(), eval_freq=10)
    cb.model = object()
    cb._evaluate(deterministic=True, seed=7, trace_dir=tmp_path, track_breakdown=False)
    assert "trace_end_reasons" not in calls[1]
    assert "max_traces" not in calls[1]
    assert "include_episode_records" not in calls[1]
    with pytest.raises(ValueError, match="max_traces"):
        PeriodicEvalCallback(object(), eval_freq=10, max_traces=-1)
