"""Self-play honours the whole env config, and its latest-vs-pool opponent choice is a knob.

* Triage (self-play env-config drops): ``self_play._env_kwargs`` hard-coded
  ``engine_overrides``, the observation scales and ``opponent_kwargs`` to
  ``None`` and had no ``fog_of_war``; ``masking._build_strategy_env`` had no
  ``fog_of_war`` either; ``train_self_play.py`` mapped only part of
  ``EnvConfig``. So ``train_self_play.py --config`` with a balance overlay or
  fog of war trained the default game without a word.
* Triage (research decision): once the opponent pool has any entry, every
  reset replaces the latest snapshot with a pool sample, which contradicted
  SelfPlayCallback's docstring. The behaviour stays the default; it is now
  ``self_play.latest_opponent_prob`` (0.0), documented and validated.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from reinforcetactics.rl.masking import make_maskable_env
from reinforcetactics.rl.self_play import (
    OpponentPool,
    SelfPlayEnv,
    make_self_play_env,
    make_self_play_vec_env,
    params_checksum,
    policy_snapshot,
)

MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP = "maps/1v1/beginner.csv"
OVERRIDES = {"starting_gold": 777}


def _tiny_model(env):
    return MaskablePPO(
        "MultiInputPolicy", env, n_steps=32, batch_size=32, n_epochs=1, policy_kwargs={"net_arch": [32]}, verbose=0, seed=0
    )


@pytest.fixture(scope="module")
def train_script():
    path = REPO_ROOT / "scripts" / "train" / "train_self_play.py"
    spec = importlib.util.spec_from_file_location("train_self_play_config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Env config reaches every self-play env
# ---------------------------------------------------------------------------


class TestEnvConfigReachesSelfPlayEnvs:
    def test_masking_builder_forwards_fog_and_version(self):
        env = make_maskable_env(
            map_file=MAP, opponent="noop", fog_of_war=True, action_space_type="flat_discrete", flat_action_version=1
        )
        assert env.unwrapped.fog_of_war is True
        assert env.unwrapped.flat_action_version == 1

    def test_single_self_play_env(self):
        env = make_self_play_env(
            map_file=MAP,
            fog_of_war=True,
            engine_overrides=OVERRIDES,
            gold_scale=500.0,
            turn_scale=30.0,
            unit_count_scale=10.0,
            action_space_type="flat_discrete",
            flat_action_version=1,
            seed=0,
        )
        base = env.unwrapped
        assert base.fog_of_war is True
        assert base.game_state.starting_gold == 777
        assert (base.gold_scale, base.turn_scale, base.unit_count_scale) == (500.0, 30.0, 10.0)
        assert base.flat_action_version == 1

    def test_self_play_and_bot_workers_of_a_mixed_vec_env(self):
        vec_env = make_self_play_vec_env(
            n_envs=2,
            map_file=MAP,
            use_subprocess=False,
            bot_ratio=0.5,
            bot_opponent="random",
            bot_opponent_kwargs={"max_actions": 3},
            fog_of_war=True,
            engine_overrides=OVERRIDES,
            gold_scale=500.0,
        )
        try:
            bases = [e.unwrapped for e in vec_env.envs]
            assert [b.opponent_type for b in bases] == ["self", "random"]
            assert all(b.fog_of_war and b.gold_scale == 500.0 and b.game_state.starting_gold == 777 for b in bases)
            assert bases[0].opponent_kwargs == {} and bases[1].opponent_kwargs == {"max_actions": 3}
        finally:
            vec_env.close()

    @pytest.mark.parametrize(("bot", "kwargs"), [("self", None), ("mastr", None), ("medium", {"max_actions": 3})])
    def test_bad_bot_worker_opponent_rejected_in_the_trainer_process(self, bot, kwargs):
        with pytest.raises(ValueError):
            make_self_play_vec_env(
                n_envs=2, map_file=MAP, use_subprocess=False, bot_ratio=0.5, bot_opponent=bot, bot_opponent_kwargs=kwargs
            )

    def test_script_maps_every_env_field_from_the_config(self, train_script, tmp_path):
        config = tmp_path / "sp.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "algorithm": "self_play",
                    "env": {
                        "map_file": MAP,
                        "opponent": "random",
                        "opponent_kwargs": {"max_actions": 4},
                        "fog_of_war": True,
                        "engine_overrides": OVERRIDES,
                        "gold_scale": 500.0,
                        "turn_scale": 30.0,
                        "unit_count_scale": 10.0,
                        "action_space_type": "flat_discrete",
                        "flat_action_version": 1,
                    },
                    "ppo": {"policy_kwargs": {"net_arch": [16]}},
                    # Mixed mode with the pool on: the bot workers read
                    # env.opponent / opponent_kwargs, and the pool reads
                    # latest_opponent_prob (--strict reports them otherwise).
                    "self_play": {"latest_opponent_prob": 0.25, "use_opponent_pool": True, "mixed_training": True},
                }
            )
        )
        args = train_script.parse_args(["--config", str(config), "--strict"])
        env_kwargs = train_script.build_env_kwargs(args)
        assert env_kwargs["fog_of_war"] is True
        assert env_kwargs["engine_overrides"] == OVERRIDES
        assert (env_kwargs["gold_scale"], env_kwargs["turn_scale"], env_kwargs["unit_count_scale"]) == (500.0, 30.0, 10.0)
        assert env_kwargs["flat_action_version"] == 1
        assert (args.bot_opponent, args.bot_opponent_kwargs) == ("random", {"max_actions": 4})
        assert args.policy_kwargs == {"net_arch": [16]}
        assert args.latest_opponent_prob == 0.25

    def test_every_env_field_is_mapped(self, train_script):
        from dataclasses import fields

        from reinforcetactics.rl.config import EnvConfig

        mapped = {p.split(".", 1)[1] for p in train_script._ARG_TO_CONFIG_PATH.values() if p.startswith("env.")}
        assert mapped == {f.name for f in fields(EnvConfig)}


# ---------------------------------------------------------------------------
# latest_opponent_prob
# ---------------------------------------------------------------------------


def _env_with_pool_and_newer_latest(latest_opponent_prob: float):
    """A self-play env whose pool holds snapshot A and whose latest snapshot is a different B."""
    env = make_self_play_env(
        map_file=MAP, swap_players=False, action_space_type="flat_discrete", latest_opponent_prob=latest_opponent_prob
    )
    model = _tiny_model(env)
    pool = OpponentPool(max_size=3)
    pool_params = pool.add_model(model, timestep=1, save_to_disk=False)
    assert pool_params is not None
    env.opponent_pool = pool
    with torch.no_grad():
        for p in model.policy.parameters():
            p.add_(1.0)
    env.set_opponent_snapshot(policy_snapshot(model))
    assert env._latest_params is not None
    latest = params_checksum(env._latest_params)
    assert latest != pytest.approx(params_checksum(pool_params))
    return env, params_checksum(pool_params), latest


class TestLatestOpponentProb:
    def test_default_plays_a_pool_sample_every_episode(self):
        env, pool_sum, _ = _env_with_pool_and_newer_latest(0.0)
        assert env.latest_opponent_prob == 0.0
        for seed in range(3):
            env.reset(seed=seed)
            assert env.describe_opponent()["source"] == "pool"
            assert env.describe_opponent()["params_checksum"] == pytest.approx(pool_sum)

    def test_one_keeps_the_latest_snapshot(self):
        env, _, latest_sum = _env_with_pool_and_newer_latest(1.0)
        for seed in range(3):
            env.reset(seed=seed)
            assert env.describe_opponent()["source"] == "latest"
            assert env.describe_opponent()["params_checksum"] == pytest.approx(latest_sum)

    def test_fraction_mixes_the_two(self):
        env, _, _ = _env_with_pool_and_newer_latest(0.5)
        sources = set()
        for seed in range(12):
            env.reset(seed=seed)
            sources.add(env.describe_opponent()["source"])
        assert sources == {"pool", "latest"}

    def test_default_draws_exactly_what_the_pool_sampling_always_drew(self):
        """p=0 must leave the env's random stream as it was (no extra draw per reset)."""
        env, _, _ = _env_with_pool_and_newer_latest(0.0)
        reference, _, _ = _env_with_pool_and_newer_latest(0.0)
        env.unwrapped.np_random = np.random.default_rng(123)
        reference.unwrapped.np_random = np.random.default_rng(123)
        env._select_episode_opponent()
        reference.update_opponent_from_pool()  # the long-standing reset-time call
        assert env.unwrapped.np_random.bit_generator.state == reference.unwrapped.np_random.bit_generator.state

    def test_no_pool_plays_the_latest_whatever_the_probability(self):
        env = make_self_play_env(map_file=MAP, swap_players=False, action_space_type="flat_discrete", latest_opponent_prob=0.0)
        env.set_opponent_snapshot(policy_snapshot(_tiny_model(env)))
        env.reset(seed=0)
        assert env.describe_opponent()["source"] == "latest"

    @pytest.mark.parametrize("p", [-0.1, 1.5])
    def test_out_of_range_rejected(self, p):
        with pytest.raises(ValueError, match="latest_opponent_prob"):
            make_self_play_env(map_file=MAP, latest_opponent_prob=p)
        with pytest.raises(ValueError, match="latest_opponent_prob"):
            make_self_play_vec_env(n_envs=1, map_file=MAP, use_subprocess=False, latest_opponent_prob=p)

    def test_vec_env_forwards_it_to_every_self_play_worker(self):
        vec_env = make_self_play_vec_env(n_envs=2, map_file=MAP, use_subprocess=False, latest_opponent_prob=0.3)
        try:
            assert [e.latest_opponent_prob for e in vec_env.envs] == [0.3, 0.3]
            assert all(isinstance(e, SelfPlayEnv) for e in vec_env.envs)
        finally:
            vec_env.close()
