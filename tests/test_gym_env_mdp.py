"""MDP-correctness fixes in StrategyGameEnv (review 2026-09-26, section 2.3).

One class per finding; each class docstring says what used to happen.
"""

import random

import numpy as np
import pytest

from reinforcetactics.game.bot import MasterBot, MixedBot, RandomBot, SimpleBot
from reinforcetactics.game.bot_registry import SCRIPTED_BOTS, accepted_names
from reinforcetactics.rl.gym_env import (
    DEFAULT_REWARD_CONFIG,
    KNOWN_REWARD_KEYS,
    OPTIONAL_REWARD_KEYS,
    StrategyGameEnv,
    accepted_opponents,
    build_flat_actions,
    resolve_opponent,
    validate_opponent_kwargs,
    validate_reward_config,
)

BEGINNER_MAP = "maps/1v1/beginner.csv"
END_TURN = np.array([5, 0, 0, 0, 0, 0])

# Reward weights that isolate the term under test: no potential shaping, no
# dealt-damage or kill reward.
_ISOLATE = {"income_diff": 0.0, "unit_diff": 0.0, "structure_control": 0.0, "damage_scale": 0.0, "kill": 0.0}


def _env(**kwargs) -> StrategyGameEnv:
    kwargs.setdefault("map_file", BEGINNER_MAP)
    kwargs.setdefault("opponent", "noop")
    env = StrategyGameEnv(**kwargs)
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# rlenv-5
# ---------------------------------------------------------------------------


class TestEndReasonComesFromTheEngine:
    """rlenv-5: the end reason, and so the terminal reward, comes from the engine.

    It used to be inferred from the loser's unit count, which calls an HQ
    capture against a side with no units left an elimination.
    """

    def test_hq_capture_against_a_unitless_opponent_is_an_hq_capture(self):
        # A noop opponent never builds, so it has no units when its HQ
        # falls. The unit-count heuristic called that an elimination and
        # paid win_by_elimination.
        env = _env(reward_config={"win_by_hq_capture": 777.0, "win_by_elimination": 111.0, **_ISOLATE})
        gs = env.game_state
        gs.place_unit("W", 5, 5, env.agent_player)
        gs.grid.get_tile(5, 5).health = 1  # one seize takes the enemy HQ
        _, _, terminated, _, info = env.step(np.array([3, 0, 5, 5, 5, 5]))
        assert terminated
        assert gs.end_reason == "hq_capture"
        assert info["end_reason"] == "hq_capture"
        assert info["reward_breakdown"]["terminal"] == pytest.approx(777.0)
        env.close()

    def test_elimination_is_reported_as_elimination(self):
        env = _env(reward_config={"win_by_hq_capture": 777.0, "win_by_elimination": 111.0, **_ISOLATE})
        gs = env.game_state
        gs.place_unit("K", 2, 2, env.agent_player)
        last = gs.place_unit("W", 3, 2, 2)
        last.health = 1
        _, _, terminated, _, info = env.step(np.array([2, 0, 2, 2, 3, 2]))
        assert terminated and gs.end_reason == "elimination"
        assert info["end_reason"] == "elimination"
        assert info["reward_breakdown"]["terminal"] == pytest.approx(111.0)
        env.close()

    def test_end_reason_is_whatever_the_engine_recorded(self):
        env = _env(opponent=None)
        gs = env.game_state
        gs.place_unit("W", 1, 1, 1)
        gs.place_unit("W", 4, 4, 2)
        gs.resign(player=2)  # both sides still had units: the heuristic said hq_capture
        _, _, terminated, _, info = env.step(END_TURN)
        assert terminated and info["end_reason"] == "resign"
        assert info["reward_breakdown"]["terminal"] == pytest.approx(DEFAULT_REWARD_CONFIG["win"])
        env.close()

    def test_truncation_keeps_its_own_reason(self):
        env = _env(opponent=None, max_steps=1)
        _, _, terminated, truncated, info = env.step(np.array([1, 0, 0, 0, 1, 1]))
        assert truncated and not terminated
        assert info["end_reason"] == "max_steps_truncate"
        env.close()


# ---------------------------------------------------------------------------
# rlenv-8
# ---------------------------------------------------------------------------


class TestStructuredMasksHonourTheActionBudget:
    """rlenv-8: the per-turn action budget gates the structured (autoregressive) masks too."""

    def test_spent_budget_leaves_end_turn_alone(self):
        env = _env(opponent=None, max_actions_per_turn=1)
        gs = env.game_state
        gs.place_unit("W", 2, 2, 1)
        gs.place_unit("W", 3, 3, 1)
        _, _, _, _, info = env.step(np.array([1, 0, 2, 2, 2, 3]))  # move one Warrior
        assert info["valid_action"]
        # The other Warrior could still move: only the budget stops it.
        assert build_flat_actions(gs, 1, 512)[0][0] in (0, 1)
        masks = env.structured_action_masks()
        assert np.flatnonzero(masks.atype).tolist() == [5]
        assert np.argwhere(masks.source).tolist() == [[5, 0, 0]]
        assert list(masks.target) == [(5, 0, 0)]
        assert np.argwhere(masks.target[(5, 0, 0)]).tolist() == [[0, 0]]
        assert masks.unit_type == {}
        # Same gate as the other two builders.
        _, at_mask, *_ = env._build_masks()
        assert np.flatnonzero(at_mask).tolist() == [5]
        env.close()

    def test_end_turn_restores_the_full_masks(self):
        env = _env(opponent="noop", max_actions_per_turn=1)
        env.step(np.array([0, 0, 1, 0, 1, 0]))
        env.step(END_TURN)
        assert env.structured_action_masks().atype.sum() > 1
        env.close()


# ---------------------------------------------------------------------------
# rlenv-9
# ---------------------------------------------------------------------------


class _AttackOnce:
    """Opponent stub: one attack, then end the turn."""

    def __init__(self, game_state, attacker, target):
        self.game_state, self.attacker, self.target = game_state, attacker, target
        self.damage = None

    def take_turn(self):
        outcome = self.game_state.apply_action("attack", {"attacker": self.attacker, "target": self.target})
        assert outcome.accepted
        self.damage = outcome.result["damage"]
        self.game_state.end_turn()


class TestCombatShaping:
    """rlenv-9: combat shaping sees both halves of an exchange.

    The counter-attack an attack draws, and a unit lost to it, cost
    nothing; opponent-turn damage was netted against the agent's
    turn-start structure healing.
    """

    def test_counter_damage_is_charged_on_the_attack_step(self):
        env = _env(reward_config={**_ISOLATE, "damage_taken_scale": -1.0})
        gs = env.game_state
        attacker = gs.place_unit("W", 2, 2, 1)
        gs.place_unit("W", 3, 2, 2)
        hp_before = attacker.health
        _, _, _, _, info = env.step(np.array([2, 0, 2, 2, 3, 2]))
        counter = gs.action_history[-1]["counter_damage"]
        assert counter > 0 and attacker.health == hp_before - counter
        assert info["reward_breakdown"]["action"] == pytest.approx(-counter)
        assert env.episode_stats["damage_taken"] == pytest.approx(counter)
        assert env.episode_stats["counter_damage_taken"] == pytest.approx(counter)
        env.close()

    def test_losing_the_attacker_to_the_counter_costs_unit_lost(self):
        env = _env(reward_config={**_ISOLATE, "unit_lost": -40.0})
        gs = env.game_state
        attacker = gs.place_unit("W", 2, 2, 1)
        attacker.health = 1
        gs.place_unit("W", 3, 2, 2)
        _, _, _, _, info = env.step(np.array([2, 0, 2, 2, 3, 2]))
        assert attacker not in gs.units
        assert info["reward_breakdown"]["action"] == pytest.approx(-40.0)
        assert env.episode_stats["units_lost"] == 1
        env.close()

    def test_opponent_turn_damage_is_not_netted_against_turn_start_healing(self):
        env = _env(reward_config={**_ISOLATE, "damage_taken_scale": -1.0})
        gs = env.game_state
        mine = gs.place_unit("W", 1, 0, 1)  # on its own building: heals 2 HP at turn start
        mine.health -= 4
        enemy = gs.place_unit("W", 2, 0, 2)
        stub = _AttackOnce(gs, enemy, mine)
        env.opponent = stub
        healed_before = gs.healing_totals[1]["hp"]
        _, _, _, _, info = env.step(END_TURN)
        assert gs.healing_totals[1]["hp"] - healed_before == 2  # the heal did happen
        assert stub.damage > 2
        assert env.episode_stats["damage_taken"] == pytest.approx(stub.damage)
        assert info["reward_breakdown"]["action"] == pytest.approx(-stub.damage)
        env.close()

    def test_units_killed_on_the_opponent_turn_cost_unit_lost(self):
        env = _env(reward_config={**_ISOLATE, "unit_lost": -40.0, "damage_taken_scale": 0.0})
        gs = env.game_state
        mine = gs.place_unit("W", 3, 3, 1)
        mine.health = 1
        gs.place_unit("W", 4, 4, 1)  # a survivor, so the game goes on
        enemy = gs.place_unit("K", 3, 4, 2)
        env.opponent = _AttackOnce(gs, enemy, mine)
        _, _, terminated, _, info = env.step(END_TURN)
        assert not terminated and mine not in gs.units
        assert info["reward_breakdown"]["action"] == pytest.approx(-40.0)
        assert env.episode_stats["units_lost"] == 1
        env.close()


# ---------------------------------------------------------------------------
# rlenv-10 / rulebots-7
# ---------------------------------------------------------------------------


class TestOpponentValidation:
    """rlenv-10 / rulebots-7: opponents are validated against the bot registry.

    'master' (and any typo) used to fall through to no opponent at all.
    """

    def test_master_is_a_real_opponent(self):
        env = _env(opponent="master")
        assert isinstance(env.opponent, MasterBot)
        env.close()

    @pytest.mark.parametrize("name", sorted(accepted_names()))
    def test_every_registry_name_builds_its_bot(self, name):
        env = _env(opponent=name)
        expected = SCRIPTED_BOTS[resolve_opponent(name)]
        assert type(env.opponent) is expected
        env.close()

    @pytest.mark.parametrize("bad", ["simpel", "Master Bot", "", "selfplay", 3])
    def test_unknown_opponent_raises_listing_the_names(self, bad):
        with pytest.raises(ValueError, match="master") as excinfo:
            StrategyGameEnv(map_file=BEGINNER_MAP, opponent=bad)
        assert "self" in str(excinfo.value)

    def test_opponent_reassigned_to_a_typo_fails_at_reset(self):
        env = _env(opponent="simple")
        env.opponent_type = "simpel"
        with pytest.raises(ValueError, match="Unknown opponent"):
            env.reset(seed=1)
        env.close()

    def test_class_names_and_the_bot_alias_resolve(self):
        assert resolve_opponent("bot") == "simple"
        assert resolve_opponent("SimpleBot") == "simple"
        assert resolve_opponent(None) is None and resolve_opponent("self") == "self"
        assert isinstance(_env(opponent="bot").opponent, SimpleBot)
        assert set(accepted_opponents()) == {"self", *accepted_names()}

    def test_opponent_kwargs_reach_stochastic_bots(self):
        env = _env(opponent="random", opponent_kwargs={"max_actions": 3})
        assert isinstance(env.opponent, RandomBot) and env.opponent.max_actions == 3
        mixed = _env(opponent="mixed", opponent_kwargs={"easy": "simple", "hard": "master", "p_hard": 1.0})
        assert isinstance(mixed.opponent, MixedBot) and isinstance(mixed.opponent._inner, MasterBot)

    @pytest.mark.parametrize(
        ("opponent", "kwargs"),
        [
            ("simple", {"max_actions": 3}),  # dropped silently before
            ("random", {"max_action": 3}),  # typo
            ("mixed", {"easy": "simple", "hard": "bogus", "p_hard": 0.0}),
            ("mixed", {"p_hard": 2.0}),
            ("self", {"max_actions": 3}),
            (None, {"max_actions": 3}),
        ],
    )
    def test_bad_opponent_kwargs_raise_at_construction(self, opponent, kwargs):
        with pytest.raises(ValueError):
            StrategyGameEnv(map_file=BEGINNER_MAP, opponent=opponent, opponent_kwargs=kwargs)
        with pytest.raises(ValueError):
            validate_opponent_kwargs(opponent, kwargs)

    def test_noop_takes_no_draw_from_np_random(self):
        # NoopBot never chooses anything; building it must not consume the
        # episode rng, so seeded noop streams stay what they were.
        a = _env(opponent="noop")
        b = _env(opponent=None)
        assert a.np_random.integers(0, 2**31 - 1) == b.np_random.integers(0, 2**31 - 1)


# ---------------------------------------------------------------------------
# rlenv-16
# ---------------------------------------------------------------------------


class _KeyRecorder(dict):
    """A reward_config that records every key the env reads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()
        self.get_calls: list[str] = []

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)

    def get(self, key, default=None):
        self.get_calls.append(key)
        return super().get(key, default)


class TestRewardConfigKeys:
    """rlenv-16: reward_config keys are validated and every weight has one default."""

    def test_known_keys_cover_defaults_and_optional_keys(self):
        assert set(DEFAULT_REWARD_CONFIG) <= KNOWN_REWARD_KEYS
        assert {
            "tower_capture",
            "building_capture",
            "hq_capture",
            "win_by_hq_capture",
            "win_by_elimination",
            "truncation",
            "damage_taken_scale",
            "unit_lost",
        } <= KNOWN_REWARD_KEYS
        assert OPTIONAL_REWARD_KEYS.isdisjoint(DEFAULT_REWARD_CONFIG)

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="damage_scael"):
            StrategyGameEnv(map_file=BEGINNER_MAP, opponent=None, reward_config={"damage_scael": 0.2})

    @pytest.mark.parametrize("value", ["3e-4", None, True, [1.0]])
    def test_non_numeric_value_is_rejected(self, value):
        with pytest.raises(TypeError):
            validate_reward_config({"win": value})

    def test_non_finite_value_is_rejected(self):
        with pytest.raises(ValueError):
            validate_reward_config({"win": float("nan")})

    def test_the_env_holds_every_default_and_the_overlay(self):
        env = _env(reward_config={"damage_scale": 0.3, "hq_capture": 900.0})
        assert env.reward_config == {**DEFAULT_REWARD_CONFIG, "damage_scale": 0.3, "hq_capture": 900.0}
        env.close()

    @pytest.mark.parametrize("space", ["flat_discrete", "multi_discrete"])
    def test_the_env_reads_only_known_keys_and_never_falls_back(self, space):
        env = StrategyGameEnv(
            map_file="maps/1v1/skirmish.csv", opponent="random", action_space_type=space, max_steps=300, max_turns=15
        )
        env.reset(seed=4)
        recorder = _KeyRecorder(env.reward_config)
        env.reward_config = recorder
        rng = random.Random(4)
        for _ in range(300):
            masks = env.action_masks()
            if space == "flat_discrete":
                action = rng.choice(np.flatnonzero(masks[0]).tolist())
            else:
                action = np.array([rng.choice(np.flatnonzero(m).tolist()) for m in masks])
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                env.reset(seed=rng.randrange(100))
                env.reward_config = recorder
        assert recorder.read <= KNOWN_REWARD_KEYS
        assert recorder.get_calls == []  # no divergent .get(key, fallback) defaults
        assert {"move", "create_unit", "turn_penalty"} <= recorder.read
        if space == "multi_discrete":  # flat_discrete masks are exact: nothing is invalid
            assert "invalid_action" in recorder.read
        env.close()
