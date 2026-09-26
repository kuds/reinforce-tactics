"""Regression tests for the self-play pipeline fixes (review §1.3).

Covers: the seat swap reaching the base env and producing a legal
player-2 game; ActionMaskedEnv attribute forwarding and pickling; the
opponent observing / masking / decoding for its own seat; the MaskablePPO
mask layout; game-outcome accounting for games ended by either side;
seeding from ``np_random``; opponent and pool updates reaching
SubprocVecEnv workers; mixed bot/self-play VecEnvs; and the
``train_self_play.py`` entry point honouring its env config.

Review IDs: prior-2, rltrain-1, rltrain-2, critic-gaps-1, critic-gaps-2,
rlenv-4, rlenv-12, consolidate-4, tests-3.
"""

import copy
import importlib.util
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from reinforcetactics.core.unit import Unit
from reinforcetactics.rl import self_play
from reinforcetactics.rl.gym_env import StrategyGameEnv, build_flat_actions, build_per_dim_masks
from reinforcetactics.rl.masking import ActionMaskedEnv, make_maskable_env, make_maskable_vec_env
from reinforcetactics.rl.observation import build_observation
from reinforcetactics.rl.self_play import (
    OpponentPool,
    SelfPlayCallback,
    SelfPlayEnv,
    make_self_play_env,
    make_self_play_vec_env,
)

# The snapshot helpers are reached through the module (not imported by name)
# so that this file still collects against code that predates them and each
# regression test fails on its own.
policy_snapshot: Any = getattr(self_play, "policy_snapshot", None)
params_checksum: Any = getattr(self_play, "params_checksum", None)

MaskablePPO = pytest.importorskip("sb3_contrib").MaskablePPO

MAP = "maps/1v1/beginner.csv"  # 6x6; P1 HQ at (0, 0), P2 HQ at (5, 5)
END_TURN = np.array([5, 0, 0, 0, 0, 0])
SPACES = ["multi_discrete", "flat_discrete"]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_model(env, **kwargs):
    """An untrained MaskablePPO with a small network (fast to build and run)."""
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={"net_arch": [32]},
        verbose=0,
        seed=0,
        device="cpu",
        **kwargs,
    )


def _model_checksum(model) -> float:
    return params_checksum({k: v.detach().cpu().numpy() for k, v in model.policy.state_dict().items()})


def _end_turn_action(env):
    """The end_turn action in the env's own action space (flat index or 6-vector)."""
    base = env.unwrapped
    if base.action_space_type == "flat_discrete":
        env.action_masks()  # rebuilds the agent's flat action table
        keys = [tuple(int(v) for v in a) for a in base._current_actions]
        return keys.index(tuple(int(v) for v in END_TURN))
    return END_TURN.copy()


def _flat_index(env, action) -> int:
    env.action_masks()
    keys = [tuple(int(v) for v in a) for a in env.unwrapped._current_actions]
    return keys.index(tuple(int(v) for v in action))


def _swapped_self_play_env(**kwargs) -> SelfPlayEnv:
    """A SelfPlayEnv reset into an episode where the agent plays seat 2."""
    env = make_self_play_env(map_file=MAP, swap_players=True, **kwargs)
    base: Any = env.unwrapped
    for seed in range(40):
        env.reset(seed=seed)
        if base.agent_player == 2:
            return env
    raise AssertionError("no seat-2 episode in 40 seeded resets")


def _seat_one_env(**env_kwargs) -> SelfPlayEnv:
    """SelfPlayEnv (agent = player 1) around a hand-built base env, reset with seed 0.

    Built by hand (base env with ``opponent=None``) rather than through
    ``make_self_play_env`` so the construction path is the same one older
    code supported: these tests then fail on the behaviour, not on a
    missing keyword argument.
    """
    base = StrategyGameEnv(map_file=MAP, opponent=None, **env_kwargs)
    env = SelfPlayEnv(ActionMaskedEnv(base), swap_players=False)
    env.reset(seed=0)
    return env


def _place_unit(game_state, unit_type, x, y, player):
    """Put a ready-to-act unit on the board (new units normally wait a turn)."""
    unit = Unit(unit_type, x, y, player)
    unit.can_move = True
    unit.can_attack = True
    game_state.units.append(unit)
    game_state._invalidate_cache()


# ==============================================================================
# ActionMaskedEnv attribute handling (rlenv-12)
# ==============================================================================


class TestActionMaskedEnvAttributes:
    def test_pickle_and_deepcopy_round_trip(self):
        """__getattr__ must not recurse while the instance has no ``env`` yet."""
        env = make_maskable_env(map_file=MAP, opponent="noop", seed=0)
        for clone in (pickle.loads(pickle.dumps(env)), copy.deepcopy(env)):
            assert clone.agent_player == 1
            np.testing.assert_array_equal(clone.action_masks(), env.action_masks())

    def test_attribute_write_reaches_base_env(self):
        env = make_maskable_env(map_file=MAP, opponent="noop")
        env.agent_player = 2
        assert env.unwrapped.agent_player == 2
        assert "agent_player" not in env.__dict__

    def test_unknown_attribute_write_raises(self):
        env = make_maskable_env(map_file=MAP, opponent="noop")
        with pytest.raises(AttributeError):
            env.agent_playr = 2  # typo: must not silently create a wrapper-only attribute

    def test_wrapper_owned_attribute_stays_on_wrapper(self):
        env = make_maskable_env(map_file=MAP, opponent="noop")
        env.track_stats = True
        assert env.__dict__["track_stats"] is True
        assert not hasattr(env.unwrapped, "track_stats")


# ==============================================================================
# Base-env seat support (critic-gaps-2, as far as self-play needs it)
# ==============================================================================


class TestBaseEnvSeat:
    def test_seat_two_reset_plays_player_one_opening_turn(self):
        env = StrategyGameEnv(map_file=MAP, opponent="noop", action_space_type="flat_discrete")
        env.set_agent_seat(2)
        env.reset(seed=0)
        gs = env.game_state

        assert env.agent_player == 2
        assert gs.current_player == 2, "the first observation must be on the agent's own turn"
        assert [a["type"] for a in gs.action_history] == ["end_turn"]
        assert gs.action_history[0]["player"] == 1
        # Masks, observation and potential all belong to seat 2.
        env.action_masks()
        assert [tuple(a) for a in env._current_actions] == [tuple(a) for a in build_flat_actions(gs, 2, 512)]
        for key, value in build_observation(gs, perspective_player=2).items():
            np.testing.assert_array_equal(env._get_obs()[key], value)
        assert env._prev_potential == pytest.approx(env._compute_potential())

    def test_seat_two_agent_builds_for_player_two(self):
        env = StrategyGameEnv(map_file=MAP, opponent="noop", action_space_type="flat_discrete")
        env.set_agent_seat(2)
        env.reset(seed=0)
        env.action_masks()
        create = next(i for i, a in enumerate(env._current_actions) if int(a[0]) == 0)
        env.step(create)
        assert [u.player for u in env.game_state.units] == [2]

    def test_random_seat_is_drawn_from_np_random(self):
        env = StrategyGameEnv(map_file=MAP, opponent="noop")
        env.set_agent_seat("random")
        seats = []
        for seed in range(20):
            env.reset(seed=seed)
            seats.append(env.agent_player)
            assert env.game_state.current_player == env.agent_player
        assert set(seats) == {1, 2}
        replay = []
        for seed in range(20):
            env.reset(seed=seed)
            replay.append(env.agent_player)
        assert replay == seats

    @pytest.mark.parametrize("seat", [0, 3, "p2"])
    def test_invalid_seat_rejected(self, seat):
        env = StrategyGameEnv(map_file=MAP, opponent="noop")
        with pytest.raises(ValueError):
            env.set_agent_seat(seat)


# ==============================================================================
# SelfPlayEnv seat swap (prior-2, rltrain-1, rlenv-12, tests-3)
# ==============================================================================


class TestSelfPlaySeatSwap:
    @pytest.mark.parametrize("space", SPACES)
    def test_swapped_reset_sets_base_env_seat(self, space):
        env = _swapped_self_play_env(action_space_type=space)
        base = env.unwrapped
        masked = env.env

        assert base.agent_player == 2
        assert env.agent_player == 2
        assert "agent_player" not in masked.__dict__
        # Player 1 (the opponent) has already moved: it is the agent's turn.
        assert base.game_state.current_player == 2
        assert base.game_state.action_history[-1] == {**base.game_state.action_history[-1], "player": 1}

        expected = build_per_dim_masks(base.game_state, base.grid_width, base.grid_height, player=2)[1:]
        if space == "multi_discrete":
            np.testing.assert_array_equal(env.action_masks(), np.concatenate(expected))

    def test_agent_actions_execute_for_its_own_seat(self):
        env = _swapped_self_play_env(action_space_type="flat_discrete")
        gs = env.unwrapped.game_state
        env.action_masks()
        before = {u.unit_id for u in gs.units}
        create = next(i for i, a in enumerate(env.unwrapped._current_actions) if int(a[0]) == 0)
        _, _, _, _, info = env.step(create)
        assert info["valid_action"]
        new_units = [u for u in gs.units if u.unit_id not in before]
        assert [u.player for u in new_units] == [2]

    def test_swap_players_false_keeps_seat_one(self):
        env = make_self_play_env(map_file=MAP, swap_players=False)
        for seed in range(10):
            env.reset(seed=seed)
            assert env.unwrapped.agent_player == 1
            assert env.unwrapped.game_state.current_player == 1


# ==============================================================================
# Opponent acts for its own seat (prior-2, rltrain-1, tests-3)
# ==============================================================================


class TestOpponentOwnSeat:
    @pytest.mark.parametrize("space", SPACES)
    def test_opponent_predict_gets_its_own_masks_and_obs(self, space):
        env = make_self_play_env(map_file=MAP, swap_players=False, action_space_type=space, max_steps=50)
        model = _tiny_model(env)
        env.set_opponent_snapshot(policy_snapshot(model))
        env.reset(seed=0)
        base = env.unwrapped
        gs = base.game_state

        calls = []
        policy = env._opponent_policy
        real_predict = policy.predict

        def spy(obs, **kwargs):
            player = gs.current_player
            if space == "flat_discrete":
                expected = np.zeros(base.max_flat_actions, dtype=bool)
                expected[: len(build_flat_actions(gs, player, base.max_flat_actions))] = True
            else:
                per_dim = build_per_dim_masks(gs, base.grid_width, base.grid_height, player=player)[1:]
                expected = np.concatenate(per_dim)
            calls.append((player, kwargs.get("action_masks"), expected, obs, build_observation(gs, player)))
            return real_predict(obs, **kwargs)

        policy.predict = spy
        executed = []
        real_execute = base.execute_game_action

        def execute_spy(action_dict, player):
            result, ok = real_execute(action_dict, player)
            executed.append((player, gs.current_player, ok))
            return result, ok

        base.execute_game_action = execute_spy

        for _ in range(3):
            env.step(_end_turn_action(env))

        assert calls, "the opponent policy was never consulted"
        for player, masks, expected, obs, expected_obs in calls:
            assert player == 2
            assert masks is not None, "predict() got no action_masks"
            np.testing.assert_array_equal(np.asarray(masks, dtype=bool), expected)
            for key, value in expected_obs.items():
                np.testing.assert_array_equal(obs[key], value)
        opponent_actions = [(p, cur, ok) for p, cur, ok in executed if p == 2]
        assert opponent_actions, "the opponent never executed a game action"
        assert all(cur == 2 for _, cur, _ in opponent_actions)
        if space == "flat_discrete":
            # Exact masks + its own decode table: every opponent action is legal.
            assert all(ok for _, _, ok in opponent_actions)

    def test_random_fallback_draws_legal_actions_for_the_opponent(self):
        env = make_self_play_env(map_file=MAP, swap_players=False)
        env.reset(seed=0)
        gs = env.unwrapped.game_state
        legal = {tuple(int(v) for v in a) for a in build_flat_actions(gs, 2, 512)}
        for _ in range(20):
            action = env._get_random_valid_action()
            assert tuple(int(v) for v in action) in legal


# ==============================================================================
# Seeding (rltrain-1 item 5)
# ==============================================================================


class TestSeeding:
    def _global_rng_state(self):
        return random.getstate(), np.random.get_state()[1].copy(), torch.random.get_rng_state().clone()

    def _assert_same_global_state(self, before, after):
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert torch.equal(before[2], after[2])

    def _history_after(self, env, seed, turns=4):
        env.reset(seed=seed)
        for _ in range(turns):
            _, _, terminated, truncated, _ = env.step(_end_turn_action(env))
            if terminated or truncated:
                break
        return [(a["type"], a.get("player")) for a in env.unwrapped.game_state.action_history], env.agent_player

    def test_random_opponent_and_seat_use_env_rng_only(self):
        env_a = make_self_play_env(map_file=MAP, swap_players=True, action_space_type="flat_discrete")
        env_b = make_self_play_env(map_file=MAP, swap_players=True, action_space_type="flat_discrete")
        before = self._global_rng_state()
        run_a = self._history_after(env_a, seed=5)
        after = self._global_rng_state()
        self._assert_same_global_state(before, after)
        assert run_a == self._history_after(env_b, seed=5)

    def test_policy_opponent_sampling_is_seeded_and_isolated(self):
        env_a = make_self_play_env(map_file=MAP, swap_players=False, action_space_type="flat_discrete")
        env_b = make_self_play_env(map_file=MAP, swap_players=False, action_space_type="flat_discrete")
        snapshot = policy_snapshot(_tiny_model(env_a))
        env_a.set_opponent_snapshot(snapshot)
        env_b.set_opponent_snapshot(snapshot)
        before = self._global_rng_state()
        run_a = self._history_after(env_a, seed=11)
        after = self._global_rng_state()
        self._assert_same_global_state(before, after)
        assert run_a == self._history_after(env_b, seed=11)
        assert any(player == 2 and kind != "end_turn" for kind, player in run_a[0])


# ==============================================================================
# Mask layout for MaskablePPO (critic-gaps-1)
# ==============================================================================


class TestMaskLayout:
    @pytest.mark.parametrize("space", SPACES)
    def test_action_masks_is_flat_bool_vector(self, space):
        env = _seat_one_env(action_space_type=space)
        masks = env.action_masks()
        assert masks.dtype == np.bool_ and masks.ndim == 1
        if space == "flat_discrete":
            assert masks.shape == (env.action_space.n,)
        else:
            assert masks.shape == (int(sum(env.action_space.nvec)),)
        np.testing.assert_array_equal(masks, np.concatenate(env.get_action_masks_tuple()))


# ==============================================================================
# Outcome accounting (rltrain-1 item 4, rlenv-4 terminal part)
# ==============================================================================


class TestOutcomeAccounting:
    def test_agent_win_on_its_own_move_is_recorded(self):
        reward_config = {"win": 300.0, "win_speed_bonus": 100.0}
        env = _seat_one_env(action_space_type="flat_discrete", reward_config=reward_config, max_turns=20)
        gs = env.unwrapped.game_state
        _place_unit(gs, "W", 5, 5, 1)
        _place_unit(gs, "W", 3, 4, 2)  # loser keeps a unit: an HQ capture, not an elimination
        gs.grid.get_tile(5, 5).health = 1

        _, _, terminated, _, info = env.step(_flat_index(env, [3, 0, 5, 5, 5, 5]))

        assert terminated and info["winner"] == 1 and info["end_reason"] == "hq_capture"
        remaining = 20 - gs.turn_number
        assert info["reward_breakdown"]["terminal"] == pytest.approx(300.0 + 100.0 * remaining / 20)
        assert env.stats["agent_wins"] == 1 and env.stats["total_games"] == 1
        assert env.stats["opponent_wins"] == 0 and env.stats["draws"] == 0
        assert env.get_win_rate() == 1.0

    def test_opponent_win_on_its_move_gets_base_env_terminal_accounting(self):
        env = _seat_one_env()
        base = env.unwrapped
        gs = base.game_state
        _place_unit(gs, "W", 0, 0, 2)
        _place_unit(gs, "W", 2, 1, 1)  # loser keeps a unit: an HQ capture, not an elimination
        gs.grid.get_tile(0, 0).health = 1
        # Script the opponent: seize the agent's HQ with the unit standing on it.
        env._get_opponent_action = lambda *_: np.array([3, 0, 0, 0, 0, 0])
        base._prev_potential = base._compute_potential()
        prev_potential = base._prev_potential

        _, _, terminated, _, info = env.step(_end_turn_action(env))

        assert terminated and info["winner"] == 2
        assert info["end_reason"] == "hq_capture"
        assert info["episode_stats"]["winner"] == 2
        assert info["reward_breakdown"]["terminal"] == base.reward_config["loss"]
        # Terminal step charges -Phi(s_prev) so the shaping stays policy-invariant.
        assert info["reward_breakdown"]["shaping_delta"] == pytest.approx(-prev_potential)
        assert env.stats["opponent_wins"] == 1 and env.stats["total_games"] == 1

    def test_draw_on_opponent_end_turn_is_scored_and_recorded(self):
        # The max_turns check runs when player 2 ends the round, i.e. on the
        # opponent's move when the agent is player 1.
        env = _seat_one_env(max_turns=1)
        _, reward, terminated, _, info = env.step(END_TURN)

        assert terminated and info["winner"] is None
        assert info["end_reason"] == "max_turns_draw"
        assert info["reward_breakdown"]["terminal"] == env.unwrapped.reward_config["draw"]
        assert env.stats["draws"] == 1 and env.stats["total_games"] == 1

    def test_truncation_is_recorded(self):
        env = _seat_one_env(max_steps=2)
        env.step(END_TURN)
        _, _, terminated, truncated, info = env.step(END_TURN)
        assert truncated and not terminated
        assert info["self_play_stats"]["total_games"] == 1
        assert info["self_play_stats"]["truncations"] == 1
        assert env.get_win_rate() == 0.0


# ==============================================================================
# Opponent pool (tests-3)
# ==============================================================================


class TestOpponentPoolRoundTrip:
    def test_add_model_sample_and_load_from_disk(self, tmp_path):
        env = make_self_play_env(map_file=MAP, swap_players=False, action_space_type="flat_discrete")
        model = _tiny_model(env)
        pool = OpponentPool(max_size=3, save_dir=str(tmp_path))

        params = pool.add_model(model, timestep=7, win_rate=0.6)

        assert params is not None and pool.size == 1
        assert params_checksum(params) == pytest.approx(_model_checksum(model))
        assert pool.sample_opponent(rng=np.random.default_rng(0)) is params
        assert (tmp_path / "opponent_7.zip").exists()

        reloaded = OpponentPool(max_size=3, save_dir=str(tmp_path))
        assert reloaded.load_from_disk(MaskablePPO) == 1
        assert reloaded.metadata[0]["timestep"] == 7
        assert params_checksum(reloaded.models[0]) == pytest.approx(params_checksum(params))

        # A pool member is what the opponent plays once an architecture is known.
        env.opponent_pool = reloaded
        env.set_opponent_snapshot(policy_snapshot(model))
        with torch.no_grad():
            for p in model.policy.parameters():
                p.add_(1.0)
        env.set_opponent_snapshot(policy_snapshot(model))
        env.reset(seed=0)
        described = env.describe_opponent()
        assert described["source"] == "pool"
        assert described["params_checksum"] == pytest.approx(params_checksum(params))

    def test_unloadable_parameters_are_not_added(self):
        class NoPolicy:
            pass

        pool = OpponentPool()
        assert pool.add_model(NoPolicy(), timestep=1) is None
        assert pool.size == 0


# ==============================================================================
# Vectorized envs and the callback (prior-2, rltrain-2, tests-3)
# ==============================================================================


class TestVecEnvWiring:
    def test_subproc_opponent_and_pool_updates_reach_workers(self):
        pool = OpponentPool(max_size=3)
        vec_env = make_self_play_vec_env(
            n_envs=2,
            map_file=MAP,
            use_subprocess=True,
            opponent_pool=pool,
            action_space_type="flat_discrete",
            max_steps=2,
        )
        try:
            assert type(vec_env).__name__ == "SubprocVecEnv"
            model = _tiny_model(vec_env)
            callback = SelfPlayCallback(vec_env, opponent_pool=pool, add_to_pool_freq=1, min_win_rate_for_pool=0.0, verbose=0)
            callback.init_callback(model)
            callback.on_training_start(locals(), globals())

            for described in vec_env.env_method("describe_opponent"):
                assert described["source"] == "latest"
                assert described["params_checksum"] == pytest.approx(_model_checksum(model))

            vec_env.reset()
            for _ in range(2):  # max_steps=2: every worker finishes an episode
                vec_env.step(np.array([_flat_end_turn_index(vec_env, i) for i in range(2)]))
            callback._add_to_pool()

            assert pool.size == 1
            assert [d["pool_size"] for d in vec_env.env_method("describe_opponent")] == [1, 1]
            assert sum(s["total_games"] for s in vec_env.env_method("get_self_play_stats")) == 2
        finally:
            vec_env.close()

    def test_callback_raises_without_self_play_envs(self):
        bot_vec_env = make_maskable_vec_env(n_envs=1, map_file=MAP, opponent="noop", use_subprocess=False)
        model = _tiny_model(bot_vec_env)
        callback = SelfPlayCallback(bot_vec_env, verbose=0)
        callback.init_callback(model)
        with pytest.raises(ValueError, match="SelfPlayEnv"):
            callback.on_training_start(locals(), globals())
        with pytest.raises(ValueError):
            SelfPlayCallback(envs=[])

    def test_mixed_vec_env_targets_only_self_play_workers(self):
        vec_env = make_self_play_vec_env(
            n_envs=4, map_file=MAP, use_subprocess=False, bot_ratio=0.5, action_space_type="flat_discrete"
        )
        assert vec_env.env_is_wrapped(SelfPlayEnv) == [True, True, False, False]
        assert [e.unwrapped.opponent_type for e in vec_env.envs] == ["self", "self", "bot", "bot"]
        model = _tiny_model(vec_env)
        callback = SelfPlayCallback(vec_env, verbose=0)
        callback.init_callback(model)
        callback.on_training_start(locals(), globals())
        assert [e.describe_opponent()["source"] for e in vec_env.envs[:2]] == ["latest", "latest"]
        with pytest.raises(ValueError):
            make_self_play_vec_env(n_envs=2, map_file=MAP, use_subprocess=False, bot_ratio=0.1)


def _flat_end_turn_index(vec_env, i) -> int:
    masks = vec_env.env_method("action_masks", indices=[i])[0]
    # End turn is always the last entry of the legal list.
    return int(np.flatnonzero(masks)[-1])


# ==============================================================================
# MaskablePPO smoke test (critic-gaps-1, tests-3)
# ==============================================================================


@pytest.mark.parametrize("space", SPACES)
def test_maskable_ppo_learns_on_swapped_self_play_vec_env(space):
    vec_env = make_self_play_vec_env(
        n_envs=2, map_file=MAP, use_subprocess=False, swap_players=True, action_space_type=space, max_steps=16
    )
    model = _tiny_model(vec_env)
    callback = SelfPlayCallback(vec_env, update_freq=16, add_to_pool_freq=10**9, verbose=0)
    model.learn(128, callback=callback)

    assert model.num_timesteps >= 128
    assert all(d["source"] == "latest" for d in vec_env.env_method("describe_opponent"))
    assert sum(s["total_games"] for s in vec_env.env_method("get_self_play_stats")) >= 2


# ==============================================================================
# train_self_play.py (rltrain-2, consolidate-4)
# ==============================================================================


@pytest.fixture(scope="module")
def train_script():
    path = REPO_ROOT / "scripts" / "train" / "train_self_play.py"
    spec = importlib.util.spec_from_file_location("train_self_play_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainSelfPlayScript:
    def test_config_env_section_is_honoured(self, train_script, tmp_path):
        cfg = tmp_path / "sp.yaml"
        cfg.write_text(
            "algorithm: self_play\n"
            "env:\n"
            f"  map_file: {MAP}\n"
            "  action_space_type: flat_discrete\n"
            "  max_flat_actions: 256\n"
            "  max_turns: 30\n"
            "  pad_to_size: [8, 8]\n"
            "  reward_config: {win: 500.0, draw: -50.0}\n"
            "  use_subprocess: false\n"
            "ppo:\n"
            "  gamma: 0.97\n"
            "self_play:\n"
            "  swap_players: false\n"
            "  mixed_training: true\n"
            "  bot_ratio: 0.5\n"
        )
        args = train_script.parse_args(["--config", str(cfg)])
        env_kwargs = train_script.build_env_kwargs(args)

        assert env_kwargs["map_file"] == MAP
        assert env_kwargs["action_space_type"] == "flat_discrete"
        assert env_kwargs["max_flat_actions"] == 256
        assert env_kwargs["max_turns"] == 30
        assert env_kwargs["pad_to_size"] == (8, 8)
        assert env_kwargs["reward_config"] == {"win": 500.0, "draw": -50.0}
        assert env_kwargs["gamma"] == 0.97
        assert args.subprocess is False and args.swap_players is False
        assert args.mode == "mixed" and args.bot_ratio == 0.5

        # Command-line flags still win over the file.
        args = train_script.parse_args(
            ["--config", str(cfg), "--action-space", "multi_discrete", "--swap-players", "--mode", "self-play"]
        )
        assert args.action_space_type == "multi_discrete"
        assert args.swap_players is True and args.mode == "self-play"

    def test_swap_players_can_be_disabled_from_cli(self, train_script):
        assert train_script.parse_args([]).swap_players is True
        assert train_script.parse_args(["--no-swap-players"]).swap_players is False

    def test_mixed_mode_trains_end_to_end(self, train_script, tmp_path):
        args = train_script.parse_args(
            [
                "--mode", "mixed", "--bot-ratio", "0.5", "--n-envs", "2", "--no-subprocess",
                "--map-file", MAP, "--action-space", "flat_discrete", "--max-steps", "16",
                "--total-timesteps", "64", "--n-steps", "32", "--batch-size", "32", "--n-epochs", "1",
                "--eval-freq", "64", "--n-eval-episodes", "1", "--checkpoint-freq", "1000",
                "--use-opponent-pool", "--add-to-pool-freq", "64", "--min-win-rate-for-pool", "0",
                "--no-progress-bar", "--device", "cpu", "--log-dir", str(tmp_path),
            ]
        )  # fmt: skip
        log_dir = train_script.train_self_play(args)
        assert log_dir.name.startswith("mixed_training_")
        assert (log_dir / "final_model.zip").exists()
        assert (log_dir / "final_stats.json").exists()
        # The first eval is always a new best; it is saved through the atomic
        # new-best hook to the path SB3's best_model_save_path used.
        assert (log_dir / "best_model" / "best_model.zip").exists()
        assert not list(log_dir.rglob("*.partial"))
