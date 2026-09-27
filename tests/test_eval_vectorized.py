"""Vectorized and per-seat evaluation.

* rltrain-11: eval ran one env at a time with one ``predict`` per step.
  ``evaluate_model_vec`` steps K envs of an :class:`EvalEnvPool` with one
  batched ``predict`` per step and must return exactly what the serial
  ``evaluate_model`` returns for the same seeds.
* critic-gaps-2 / critic-integration-5: seat 2 works in the env but nothing
  evaluated it. ``seats=[1, 2]`` plays every seed from both seats and
  reports per-seat win rates; ``EnvConfig.agent_seat`` reaches every env the
  curriculum builds.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest

from reinforcetactics.rl.bootstrap import _default_eval_env_factory, _default_train_env_factory, make_stage_env
from reinforcetactics.rl.config import config_from_dict
from reinforcetactics.rl.evaluation import EvalEnvPool, evaluate_model, evaluate_model_vec
from reinforcetactics.rl.masking import make_maskable_env, make_maskable_vec_env

MAP = "maps/1v1/beginner.csv"
ENV_KWARGS: dict[str, Any] = {
    "map_file": MAP,
    "action_space_type": "flat_discrete",
    "max_flat_actions": 64,
    "max_steps": 60,
    "max_turns": 6,
    "enabled_units": ["W"],
}


def _env(opponent: str = "random", agent_seat: int | str = 1):
    return make_maskable_env(opponent=opponent, agent_seat=agent_seat, **ENV_KWARGS)


class _HashPolicy:
    """Deterministic in the observation: picks a legal action from a hash of the obs.

    Handles a single observation (the serial path) and a batch (the pool).
    """

    def predict(self, obs, action_masks=None, deterministic=True):
        masks = np.asarray(action_masks)
        if masks.ndim == 1:
            return np.int64(self._pick(obs, masks)), None
        rows = [{k: v[i] for k, v in obs.items()} for i in range(masks.shape[0])]
        return np.array([self._pick(o, m) for o, m in zip(rows, masks)], dtype=np.int64), None

    @staticmethod
    def _pick(obs, mask):
        legal = np.flatnonzero(mask)
        key = int(
            abs(float(np.sum(obs["grid"])) * 31 + float(np.sum(obs["units"])) * 7 + float(np.sum(obs["global_features"])))
            * 1000
        )
        return int(legal[key % len(legal)])


def _comparable(result):
    # Every per-episode list and aggregate, in episode order.
    return {k: v for k, v in result.items() if k != "traces"}


class TestSerialVectorizedParity:
    @pytest.mark.parametrize("seats", [None, [1, 2]])
    def test_same_episodes_same_results(self, seats):
        model = _HashPolicy()
        serial = evaluate_model(model, _env(), n_episodes=5, seed=123, track_breakdown=True, seats=seats)
        pool = EvalEnvPool.from_envs([_env() for _ in range(3)])
        try:
            vec = evaluate_model_vec(model, pool, n_episodes=5, seed=123, track_breakdown=True, seats=seats)
        finally:
            pool.close()
        assert _comparable(vec) == _comparable(serial)
        assert len(vec["rewards"]) == 5 * (len(seats) if seats else 1)
        # The episodes differ from one another (the test would be vacuous otherwise).
        assert len(set(vec["lengths"])) > 1 or len(set(vec["rewards"])) > 1

    def test_pool_larger_than_the_plan_and_traces(self, tmp_path):
        model = _HashPolicy()
        kwargs = {"n_episodes": 2, "seed": 9, "trace_end_reasons": ("max_turns_draw", "max_steps_truncate", "hq_capture")}
        serial = evaluate_model(model, _env(), trace_dir=tmp_path / "serial", **kwargs)
        pool = EvalEnvPool.from_envs([_env() for _ in range(4)])
        try:
            vec = evaluate_model_vec(model, pool, trace_dir=tmp_path / "vec", **kwargs)
        finally:
            pool.close()
        assert _comparable(vec) == _comparable(serial)
        names = lambda d: sorted(p.name for p in d.iterdir())  # noqa: E731
        assert names(tmp_path / "vec") == names(tmp_path / "serial") != []

    def test_real_maskable_ppo_deterministic_parity(self):
        from sb3_contrib import MaskablePPO

        model = MaskablePPO("MultiInputPolicy", _env(), policy_kwargs={"net_arch": [16]}, seed=0, n_steps=32, batch_size=32)
        serial = evaluate_model(model, _env(), n_episodes=4, seed=77, deterministic=True)
        pool = EvalEnvPool.from_envs([_env(), _env()])
        try:
            vec = evaluate_model_vec(model, pool, n_episodes=4, seed=77, deterministic=True)
        finally:
            pool.close()
        assert _comparable(vec) == _comparable(serial)

    def test_subprocess_pool_matches(self):
        model = _HashPolicy()
        serial = evaluate_model(model, _env(), n_episodes=3, seed=5, seats=[1, 2])
        pool = EvalEnvPool([functools.partial(_env)] * 2, use_subprocess=True)
        try:
            vec = evaluate_model_vec(model, pool, n_episodes=3, seed=5, seats=[1, 2])
        finally:
            pool.close()
        assert _comparable(vec) == _comparable(serial)


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------


class TestPerSeatEvaluation:
    def test_both_seats_play_every_seed(self):
        env = _env(opponent="noop", agent_seat="random")
        result = evaluate_model(_HashPolicy(), env, n_episodes=3, seed=1, seats=[1, 2])
        assert result["episodes"] == 6
        assert result["seats"] == [1, 1, 1, 2, 2, 2]
        assert set(result["by_seat"]) == {"1", "2"}
        assert all(v["episodes"] == 3 for v in result["by_seat"].values())
        for seat, summary in result["by_seat"].items():
            assert summary["wins"] + summary["losses"] + summary["draws"] == 3
        # The env's own seat mode is restored.
        assert env.unwrapped.agent_seat == "random"

    def test_seat_two_episodes_start_on_the_agents_turn(self):
        seen = []

        class _Spy(_HashPolicy):
            def predict(self, obs, action_masks=None, deterministic=True):
                seen.append((env.unwrapped.agent_player, env.unwrapped.game_state.current_player))
                return super().predict(obs, action_masks, deterministic)

        env = _env(opponent="noop")
        evaluate_model(_Spy(), env, n_episodes=1, seed=3, seats=[2])
        assert seen and all(agent == current == 2 for agent, current in seen)
        assert env.unwrapped.agent_seat == 1

    def test_env_without_seats_rejects_a_seat_it_cannot_play(self):
        class _Fixed:
            agent_player = 1

            def reset(self, seed=None):
                return {}, {}

        with pytest.raises(ValueError, match="set_agent_seat"):
            evaluate_model(_HashPolicy(), _Fixed(), n_episodes=1, seats=[2])


class TestAgentSeatReachesEveryEnv:
    def _cfg(self, **eval_section):
        return config_from_dict(
            {
                "env": {"agent_seat": 2, "n_envs": 2, "use_subprocess": False, **ENV_KWARGS, "map_file": None},
                "eval": eval_section,
                "curriculum": {"stages": [{"name": "s", "map_file": MAP, "opponent": "noop", "max_timesteps": 100}]},
            }
        )

    def test_env_constructor_and_builders(self):
        from reinforcetactics.rl.gym_env import StrategyGameEnv

        env = StrategyGameEnv(opponent="noop", agent_seat=2, **{k: v for k, v in ENV_KWARGS.items()})
        env.reset(seed=0)
        assert env.agent_player == 2 and env.game_state.current_player == 2
        with pytest.raises(ValueError, match="agent seat"):
            StrategyGameEnv(opponent="noop", agent_seat=3, **ENV_KWARGS)
        vec = make_maskable_vec_env(n_envs=2, use_subprocess=False, opponent="noop", agent_seat=2, **ENV_KWARGS)
        try:
            assert vec.get_attr("agent_player") == [2, 2]
        finally:
            vec.close()

    def test_curriculum_train_eval_and_stage_envs(self):
        cfg = self._cfg()
        stage = cfg.curriculum.stages[0]
        vec = _default_train_env_factory(stage, cfg)
        eval_env = _default_eval_env_factory(stage, cfg)
        stage_env = make_stage_env(stage, cfg.env, seed=0)
        try:
            assert vec.get_attr("agent_player") == [2, 2]
            assert eval_env.unwrapped.agent_player == 2
            assert stage_env.unwrapped.agent_seat == 2
        finally:
            vec.close()
            eval_env.close()
            stage_env.close()

    def test_feudal_and_self_play_forward_the_seat(self):
        import importlib.util
        from pathlib import Path
        from types import SimpleNamespace

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("feudal_seat", root / "scripts" / "train" / "train_feudal_rl.py")
        assert spec is not None and spec.loader is not None
        feudal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feudal)
        cfg = self._cfg()
        args = SimpleNamespace(opponent="noop", max_steps=40, map_file=MAP, max_turns=5, gamma=0.99)
        assert feudal._env_kwargs_from_cfg(cfg.env, args)["agent_seat"] == 2
        assert "env.agent_seat" in feudal.consumed_config_fields("feudal")

        from reinforcetactics.rl.self_play import make_self_play_vec_env

        vec = make_self_play_vec_env(
            n_envs=2, use_subprocess=False, bot_ratio=0.5, bot_opponent="noop", agent_seat=2, swap_players=False, **ENV_KWARGS
        )
        try:
            # The self-play worker follows swap_players; the bot worker plays seat 2.
            assert vec.get_attr("agent_player") == [1, 2]
        finally:
            vec.close()
