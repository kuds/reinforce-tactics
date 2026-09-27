"""A flat_discrete policy is only ever paired with an env of its own decode table.

SB3's space check compares ``Discrete.n`` only, so an old (version-1,
unstamped) checkpoint evaluated on, replayed on, or loaded into a default
(version-2) env ran without a word, playing different actions than it chose
wherever the legal set exceeded ``max_flat_actions``; a continued checkpoint
also kept its old stamp, so ModelBot decoded it with the wrong table
afterwards. ``check_flat_action_version`` now refuses such a pairing in
``evaluate_model``, ``record_evaluation_to_video``, ``PeriodicEvalCallback``,
``SelfPlayCallback`` and ``SelfPlayEnv.set_opponent_model``.
"""

import importlib.util
import warnings
from pathlib import Path

import pytest

from reinforcetactics.rl.gym_env import (
    FLAT_ACTION_VERSION_LATEST,
    FLAT_ACTION_VERSION_LEGACY,
    FlatActionVersionMismatch,
    check_flat_action_version,
    checkpoint_flat_action_version,
    env_flat_action_version,
    flat_action_version_of,
    stamp_flat_action_version,
)
from reinforcetactics.rl.masking import make_maskable_env

sb3_contrib = pytest.importorskip("sb3_contrib")

BEGINNER_MAP = "maps/1v1/beginner.csv"
_FLAT = {
    "map_file": BEGINNER_MAP,
    "action_space_type": "flat_discrete",
    "max_flat_actions": 16,
    "enabled_units": ["W"],
    "max_steps": 30,
}


def _flat_env(version=None, **kwargs):
    return make_maskable_env(opponent=kwargs.pop("opponent", "noop"), flat_action_version=version, **{**_FLAT, **kwargs})


@pytest.fixture(scope="module")
def old_checkpoint(tmp_path_factory):
    """A checkpoint saved before versioning existed: trained on version 1, unstamped."""
    env = _flat_env(FLAT_ACTION_VERSION_LEGACY)
    model = sb3_contrib.MaskablePPO(
        "MultiInputPolicy", env, n_steps=8, batch_size=8, n_epochs=1, policy_kwargs={"net_arch": [8]}, seed=0, device="cpu"
    )
    del model.action_space.flat_action_version
    path = tmp_path_factory.mktemp("ckpt") / "old_flat.zip"
    model.save(str(path))
    env.close()
    return path


def _load(path, **kwargs):
    return sb3_contrib.MaskablePPO.load(str(path), device="cpu", **kwargs)


class TestEnvFlatActionVersion:
    def test_reads_the_version_the_env_decodes_with(self):
        for version in (FLAT_ACTION_VERSION_LEGACY, FLAT_ACTION_VERSION_LATEST):
            env = _flat_env(version)
            assert env_flat_action_version(env) == version  # through the ActionMaskedEnv wrapper
            assert env_flat_action_version(env.unwrapped) == version
            env.close()

    def test_multi_discrete_and_foreign_envs_have_none(self):
        env = make_maskable_env(map_file=BEGINNER_MAP, opponent="noop")
        assert env_flat_action_version(env) is None
        env.close()
        assert env_flat_action_version(object()) is None

    def test_vec_envs(self):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

        vec = VecMonitor(DummyVecEnv([lambda: _flat_env(1), lambda: _flat_env(1)]))
        assert env_flat_action_version(vec) == 1
        vec.close()
        mixed = DummyVecEnv([lambda: _flat_env(1), lambda: _flat_env(2)])
        with pytest.raises(FlatActionVersionMismatch, match=r"\[1, 2\]"):
            env_flat_action_version(mixed)
        mixed.close()


class TestCheckFlatActionVersion:
    def test_old_checkpoint_on_a_default_env_is_refused(self, old_checkpoint):
        model = _load(old_checkpoint)
        env = _flat_env()
        with pytest.raises(FlatActionVersionMismatch, match="flat_action_version 1.*uses version 2"):
            check_flat_action_version(model, env)
        check_flat_action_version(model, _flat_env(checkpoint_flat_action_version(old_checkpoint)))
        env.close()

    def test_multi_discrete_and_model_less_pairs_are_not_checked(self, old_checkpoint):
        check_flat_action_version(_load(old_checkpoint), make_maskable_env(map_file=BEGINNER_MAP, opponent="noop"))
        check_flat_action_version(object(), _flat_env())  # a predict() shim with no action space


class TestEvaluationRefusesAMismatch:
    def test_evaluate_model(self, old_checkpoint):
        from reinforcetactics.rl.evaluation import evaluate_model

        model = _load(old_checkpoint)
        env = _flat_env()
        with pytest.raises(FlatActionVersionMismatch, match="evaluation env"):
            evaluate_model(model, env, n_episodes=1)
        env.close()
        matching = _flat_env(flat_action_version_of(model))
        assert evaluate_model(model, matching, n_episodes=1, seed=0)["episodes"] == 1
        matching.close()

    def test_record_evaluation_to_video(self, old_checkpoint, tmp_path):
        from reinforcetactics.utils.video import record_evaluation_to_video

        with pytest.raises(FlatActionVersionMismatch, match="replay env"):
            record_evaluation_to_video(_flat_env(), _load(old_checkpoint), output_path=str(tmp_path / "x.mp4"))

    def test_periodic_eval_callback_fails_at_training_start(self, old_checkpoint):
        from reinforcetactics.rl.callbacks import PeriodicEvalCallback

        model = _load(old_checkpoint, env=_flat_env(1))
        callback = PeriodicEvalCallback(_flat_env(), eval_freq=10_000, n_eval_episodes=1, verbose=0)
        with pytest.raises(FlatActionVersionMismatch, match="eval env"):
            model.learn(8, callback=callback)


class TestSelfPlayRefusesAMismatch:
    def _vec(self, version=None):
        from reinforcetactics.rl.self_play import make_self_play_vec_env

        kwargs = {k: v for k, v in _FLAT.items()}
        return make_self_play_vec_env(n_envs=1, use_subprocess=False, flat_action_version=version, **kwargs)

    def test_continuing_an_old_checkpoint_on_default_envs_is_refused(self, old_checkpoint):
        """The notebook hand-off: MaskablePPO.load(ckpt, env=self_play_vec_env)."""
        from reinforcetactics.rl.self_play import SelfPlayCallback

        vec = self._vec()
        model = _load(old_checkpoint, env=vec)  # SB3 accepts it: same Discrete.n
        with pytest.raises(FlatActionVersionMismatch, match="self-play training env"):
            model.learn(8, callback=SelfPlayCallback(vec, update_freq=1000, verbose=0))
        vec.close()

    def test_matching_envs_or_an_explicit_restamp_train(self, old_checkpoint):
        from reinforcetactics.rl.self_play import SelfPlayCallback

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec = self._vec(checkpoint_flat_action_version(old_checkpoint))
            model = _load(old_checkpoint, env=vec)
            model.learn(8, callback=SelfPlayCallback(vec, update_freq=1000, verbose=0))
            assert flat_action_version_of(model) == 1
            vec.close()

            vec = self._vec()
            model = _load(old_checkpoint, env=vec)
            stamp_flat_action_version(model, FLAT_ACTION_VERSION_LATEST)  # a deliberate move to version 2
            model.learn(8, callback=SelfPlayCallback(vec, update_freq=1000, verbose=0))
            assert flat_action_version_of(model) == FLAT_ACTION_VERSION_LATEST
            vec.close()

    def test_set_opponent_model(self, old_checkpoint):
        from reinforcetactics.rl.self_play import make_self_play_env

        env = make_self_play_env(**_FLAT)
        with pytest.raises(FlatActionVersionMismatch, match="self-play env"):
            env.set_opponent_model(_load(old_checkpoint))
        env.close()
        env = make_self_play_env(flat_action_version=1, **_FLAT)
        env.set_opponent_model(_load(old_checkpoint))
        env.close()


class TestSuffixlessCheckpointPath:
    """SB3's load() appends .zip to a missing path; the version read must too."""

    def test_checkpoint_flat_action_version(self, old_checkpoint):
        bare = str(old_checkpoint)[: -len(".zip")]
        assert checkpoint_flat_action_version(bare) == FLAT_ACTION_VERSION_LEGACY
        with pytest.raises(FileNotFoundError):
            checkpoint_flat_action_version(bare + "_missing")

    def test_train_self_play_resume_without_the_suffix(self, old_checkpoint):
        path = Path(__file__).resolve().parents[1] / "scripts" / "train" / "train_self_play.py"
        spec = importlib.util.spec_from_file_location("train_self_play_suffix_under_test", path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)

        bare = str(old_checkpoint)[: -len(".zip")]
        args = script.parse_args(["--action-space", "flat_discrete", "--resume-from", bare, "--map-file", BEGINNER_MAP])
        assert script.build_env_kwargs(args)["flat_action_version"] == FLAT_ACTION_VERSION_LEGACY
