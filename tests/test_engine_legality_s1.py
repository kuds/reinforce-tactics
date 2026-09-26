"""The engine is the only enforcer of the rules (review §1.2).

``GameState``'s action methods used to trust their callers: they applied an
out-of-turn, out-of-range, already-spent or wrong-side action and recorded
it, so every caller that did not pre-filter against ``get_legal_actions``
(multi_discrete policies, LLM bots, the rule bots' knight charge, the GUI)
could break the rules (core-2, rlenv-2, aibots-1, rulebots-2). And a
defender that could not reach its attacker still countered for 1 phantom
damage (core-3, prior-11).

These tests pin the fixed behaviour: every illegal action is refused and
changes and records nothing; everything ``get_legal_actions`` offers is
accepted and nothing else is (checked exhaustively over random-play
states); the RL env scores a refused action as invalid; ModelBot and the
rule bots notice refusals; and out-of-reach fights deal no damage.
"""

import copy
import logging
import random

import numpy as np
import pytest

from reinforcetactics.constants import ALL_UNIT_TYPES
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.mechanics import GameMechanics
from reinforcetactics.core.unit import Unit
from reinforcetactics.game.bot import AdvancedBot, MediumBot, SimpleBot
from reinforcetactics.game.model_bot import ModelBot
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.replay_actions import execute_replay_action

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _game(fog_of_war: bool = False, enabled_units: list[str] | None = None) -> GameState:
    """10x10 plains with one building per player plus assorted terrain.

    (1, 0) b_1, (8, 9) b_2, (2, 5) neutral building, (6, 3) neutral tower,
    (2, 2) water, (6, 6) mountain. Both players have plenty of gold.
    """
    md = np.array([["p"] * 10 for _ in range(10)], dtype=object)
    md[0][0] = "h_1"
    md[0][1] = "b_1"
    md[9][9] = "h_2"
    md[9][8] = "b_2"
    md[2][2] = "w"
    md[5][2] = "b"
    md[3][6] = "t"
    md[6][6] = "m"
    gs = GameState(md, num_players=2, fog_of_war=fog_of_war, enabled_units=enabled_units)
    gs.player_gold[1] = 5000
    gs.player_gold[2] = 5000
    if fog_of_war:
        gs.update_visibility()
    return gs


def _snapshot(gs: GameState, structures: list | None = None) -> tuple:
    """Everything an action could touch; equal before/after == no mutation.

    Every attribute of every unit and structure tile, plus the game-level
    fields. ``structures`` may pass ``gs.grid.get_capturable_tiles()``
    precomputed (the fuzz below snapshots one state thousands of times).
    """
    if structures is None:
        structures = gs.grid.get_capturable_tiles()
    return (
        gs.current_player,
        gs.turn_number,
        gs.game_over,
        tuple(sorted(gs.player_gold.items())),
        len(gs.action_history),
        tuple(tuple(vars(u).values()) for u in gs.units),
        tuple(tuple(vars(t).values()) for t in structures),
    )


def _clone(gs: GameState) -> GameState:
    """Deep copy without the (irrelevant, ever-growing) action history."""
    return copy.deepcopy(gs, memo={id(gs.action_history): []})


# ---------------------------------------------------------------------------
# create_unit (core-2, rlenv-2)
# ---------------------------------------------------------------------------


class TestCreateUnitLegality:
    def test_create_on_own_empty_building_on_own_turn_succeeds(self):
        gs = _game()
        unit = gs.create_unit("W", 1, 0, player=1)
        assert unit is not None
        assert gs.player_gold[1] == 5000 - gs.unit_data["W"]["cost"]
        assert gs.action_history[-1]["type"] == "create_unit"

    @pytest.mark.parametrize(
        "unit_type,x,y,player",
        [
            ("W", 2, 2, 1),  # water
            ("W", 4, 4, 1),  # plains
            ("W", 8, 9, 1),  # the enemy's building
            ("W", 2, 5, 1),  # a neutral building
            ("W", 0, 0, 1),  # own HQ: HQs never spawn units
            ("W", 6, 3, 1),  # a tower
            ("W", 99, -3, 1),  # off the board
            ("W", 8, 9, 2),  # player 2's own building, but it is player 1's turn
        ],
    )
    def test_illegal_create_is_refused_and_changes_nothing(self, unit_type, x, y, player):
        gs = _game()
        before = _snapshot(gs)
        assert gs.create_unit(unit_type, x, y, player=player) is None
        assert _snapshot(gs) == before

    def test_disabled_unit_type_is_refused(self):
        gs = _game(enabled_units=["W", "M"])
        before = _snapshot(gs)
        assert gs.create_unit("K", 1, 0, player=1) is None
        assert _snapshot(gs) == before
        assert gs.create_unit("M", 1, 0, player=1) is not None

    def test_create_after_game_over_is_refused(self):
        gs = _game()
        gs.resign(2)
        assert gs.game_over
        before = _snapshot(gs)
        assert gs.create_unit("W", 1, 0, player=1) is None
        assert _snapshot(gs) == before
        assert all(u.player != 1 for u in gs.units)


# ---------------------------------------------------------------------------
# unit actions (core-2, aibots-1)
# ---------------------------------------------------------------------------


class TestUnitActionLegality:
    def test_second_attack_by_spent_unit_is_refused(self):
        gs = _game()
        attacker = gs.place_unit("K", 4, 4, 1)
        target = gs.place_unit("B", 5, 4, 2)
        assert gs.attack(attacker, target)["damage"] > 0
        before = _snapshot(gs)
        result = gs.attack(attacker, target)
        assert result["damage"] == 0 and result["counter_damage"] == 0
        assert _snapshot(gs) == before

    def test_attack_out_of_turn_is_refused(self):
        gs = _game()
        victim = gs.place_unit("W", 4, 4, 1)
        attacker = gs.place_unit("W", 5, 4, 2)  # player 2's unit during player 1's turn
        before = _snapshot(gs)
        assert gs.attack(attacker, victim)["damage"] == 0
        assert _snapshot(gs) == before

    def test_attack_out_of_range_is_refused(self):
        gs = _game()
        attacker = gs.place_unit("W", 4, 4, 1)
        target = gs.place_unit("W", 4, 9, 2)  # distance 5
        before = _snapshot(gs)
        assert gs.attack(attacker, target)["damage"] == 0
        assert _snapshot(gs) == before
        assert attacker.health == attacker.max_health and target.health == target.max_health

    def test_paralyzed_attacker_is_refused(self):
        gs = _game()
        attacker = gs.place_unit("W", 4, 4, 1)
        target = gs.place_unit("W", 5, 4, 2)
        attacker.paralyzed_turns = 2
        before = _snapshot(gs)
        assert gs.attack(attacker, target)["damage"] == 0
        assert _snapshot(gs) == before

    def test_friendly_fire_is_refused(self):
        gs = _game()
        knight = gs.place_unit("K", 4, 4, 1)
        friend = gs.place_unit("W", 5, 4, 1)
        before = _snapshot(gs)
        assert gs.attack(knight, friend)["damage"] == 0
        assert _snapshot(gs) == before

    def test_fog_hidden_target_discovered_by_moving_is_refused(self):
        """Under fog, an enemy the unit only saw after moving can't be hit."""
        gs = _game(fog_of_war=True)
        attacker = gs.place_unit("W", 4, 1, 1)
        target = gs.place_unit("W", 4, 5, 2)
        gs.update_visibility()
        assert not gs.is_position_visible(4, 5, 1)
        assert gs.move_unit(attacker, 4, 4)  # snapshot captured pre-move
        assert gs.is_position_visible(4, 5, 1)
        before = _snapshot(gs)
        assert gs.attack(attacker, target)["damage"] == 0
        assert _snapshot(gs) == before

    def test_repeated_seize_cannot_take_hq_in_one_turn(self):
        """The LLM repro: one Warrior SEIZE-spamming a 50-HP HQ (aibots-1)."""
        gs = _game()
        seizer = gs.place_unit("W", 9, 9, 1)  # on player 2's HQ
        hq = gs.grid.get_tile(9, 9)
        first = gs.seize(seizer)
        assert first["damage"] == seizer.health
        hp_after_first = hq.health
        before = _snapshot(gs)
        for _ in range(4):
            result = gs.seize(seizer)
            assert "damage" not in result and result["captured"] is False
        assert _snapshot(gs) == before
        assert hq.health == hp_after_first and hq.player == 2 and not gs.game_over

    def test_seize_of_own_structure_is_refused(self):
        gs = _game()
        unit = gs.place_unit("W", 1, 0, 1)  # own building
        before = _snapshot(gs)
        assert "damage" not in gs.seize(unit)
        assert _snapshot(gs) == before
        assert unit.can_move and unit.can_attack, "a refused seize must not spend the unit's action"

    def test_second_heal_in_a_turn_is_refused(self):
        gs = _game()
        cleric = gs.place_unit("C", 4, 4, 1)
        ally = gs.place_unit("W", 4, 5, 1)
        ally.health = 3
        assert gs.heal(cleric, ally) > 0
        before = _snapshot(gs)
        assert gs.heal(cleric, ally) == 0
        assert _snapshot(gs) == before

    def test_heal_on_enemy_is_refused(self):
        gs = _game()
        cleric = gs.place_unit("C", 4, 4, 1)
        enemy = gs.place_unit("W", 4, 5, 2)
        enemy.health = 3
        before = _snapshot(gs)
        assert gs.heal(cleric, enemy) == 0
        assert _snapshot(gs) == before

    def test_cure_by_spent_cleric_is_refused(self):
        gs = _game()
        cleric = gs.place_unit("C", 4, 4, 1)
        ally = gs.place_unit("W", 4, 5, 1)
        ally.paralyzed_turns = 2
        cleric.can_move = cleric.can_attack = False
        before = _snapshot(gs)
        assert gs.cure(cleric, ally) is False
        assert _snapshot(gs) == before

    def test_paralyze_on_already_paralyzed_enemy_is_refused(self):
        gs = _game()
        mage = gs.place_unit("M", 4, 4, 1)
        enemy = gs.place_unit("W", 4, 6, 2)
        enemy.paralyzed_turns = 1
        before = _snapshot(gs)
        assert gs.paralyze(mage, enemy) is False
        assert _snapshot(gs) == before

    def test_haste_by_spent_sorcerer_is_refused(self):
        gs = _game()
        sorcerer = gs.place_unit("S", 4, 4, 1)
        ally = gs.place_unit("K", 4, 5, 1)
        sorcerer.can_move = sorcerer.can_attack = False
        before = _snapshot(gs)
        assert gs.haste(sorcerer, ally) is False
        assert _snapshot(gs) == before

    def test_defence_buff_out_of_turn_is_refused(self):
        gs = _game()
        sorcerer = gs.place_unit("S", 4, 4, 2)
        ally = gs.place_unit("W", 4, 5, 2)
        before = _snapshot(gs)
        assert gs.defence_buff(sorcerer, ally) is False
        assert _snapshot(gs) == before

    def test_attack_buff_by_paralyzed_sorcerer_is_refused(self):
        gs = _game()
        sorcerer = gs.place_unit("S", 4, 4, 1)
        ally = gs.place_unit("W", 4, 5, 1)
        sorcerer.paralyzed_turns = 1
        before = _snapshot(gs)
        assert gs.attack_buff(sorcerer, ally) is False
        assert _snapshot(gs) == before

    def test_move_out_of_turn_is_refused(self):
        gs = _game()
        unit = gs.place_unit("W", 4, 4, 2)
        before = _snapshot(gs)
        assert gs.move_unit(unit, 4, 5) is False
        assert _snapshot(gs) == before

    def test_move_by_paralyzed_unit_is_refused(self):
        gs = _game()
        unit = gs.place_unit("W", 4, 4, 1)
        unit.paralyzed_turns = 1  # flags still say it can act
        before = _snapshot(gs)
        assert gs.move_unit(unit, 4, 5) is False
        assert _snapshot(gs) == before

    def test_actions_after_game_over_are_refused(self):
        gs = _game()
        unit = gs.place_unit("W", 4, 4, 1)
        enemy = gs.place_unit("W", 5, 4, 2)
        gs.resign(2)
        assert gs.game_over
        before = _snapshot(gs)
        assert gs.move_unit(unit, 4, 5) is False
        assert "damage" not in gs.seize(unit)
        assert gs.attack(unit, enemy)["damage"] == 0
        assert _snapshot(gs) == before


# ---------------------------------------------------------------------------
# enumeration == validation, exhaustively over random-play states
# ---------------------------------------------------------------------------

# legal-action key -> (GameState method, actor field, target field, success test)
_PAIR_ACTIONS = {
    "attack": ("attack", "attacker", "target", lambda r: r["damage"] > 0),
    "paralyze": ("paralyze", "paralyzer", "target", bool),
    "heal": ("heal", "healer", "target", lambda r: r > 0),
    "cure": ("cure", "curer", "target", bool),
    "haste": ("haste", "sorcerer", "target", bool),
    "defence_buff": ("defence_buff", "sorcerer", "target", bool),
    "attack_buff": ("attack_buff", "sorcerer", "target", bool),
}


def _execute_legal(gs: GameState, key: str, action: dict) -> None:
    """Apply one enumerated action, asserting the engine accepts it."""
    if key == "create_unit":
        assert gs.create_unit(action["unit_type"], action["x"], action["y"]) is not None
    elif key == "move":
        assert gs.move_unit(action["unit"], action["to_x"], action["to_y"])
    elif key == "seize":
        assert "damage" in gs.seize(action["unit"])
    else:
        method, actor_field, target_field, ok = _PAIR_ACTIONS[key]
        assert ok(getattr(gs, method)(action[actor_field], action[target_field]))


# Accepted candidates each need a deep copy, the dominant cost; check a
# sample of them. Every refused candidate is checked (on the live state).
_ACCEPTED_SAMPLE_RATE = 0.3


def _check_attempt(gs: GameState, expected: bool, attempt, succeeded, rng: random.Random, structures: list) -> None:
    """Run ``attempt(state)``: legal -> accepted on a clone; illegal -> refused, no mutation."""
    if expected:
        if rng.random() > _ACCEPTED_SAMPLE_RATE:
            return
        clone = _clone(gs)
        before = _snapshot(clone)
        assert succeeded(attempt(clone)), "an enumerated action was refused"
        assert _snapshot(clone) != before, "an accepted action changed nothing"
    else:
        before = _snapshot(gs, structures)
        assert not succeeded(attempt(gs)), "an action missing from get_legal_actions was accepted"
        assert _snapshot(gs, structures) == before, "a refused action mutated the state"


def _assert_engine_matches_enumeration(gs: GameState, rng: random.Random) -> None:
    player = gs.current_player
    legal = gs.get_legal_actions(player)
    n = len(gs.units)
    index = {id(u): i for i, u in enumerate(gs.units)}
    structures = gs.grid.get_capturable_tiles()

    def check(expected, attempt, succeeded):
        _check_attempt(gs, expected, attempt, succeeded, rng, structures)

    # Unit-on-unit actions: every ordered pair (both sides, self-targets too).
    for key, (method, actor_field, target_field, ok) in _PAIR_ACTIONS.items():
        legal_pairs = {(index[id(a[actor_field])], index[id(a[target_field])]) for a in legal[key]}
        for i in range(n):
            for j in range(n):
                check(
                    (i, j) in legal_pairs,
                    lambda s, i=i, j=j, m=method: getattr(s, m)(s.units[i], s.units[j]),
                    ok,
                )

    legal_seize = {index[id(a["unit"])] for a in legal["seize"]}
    for i in range(n):
        check(i in legal_seize, lambda s, i=i: s.seize(s.units[i]), lambda r: "damage" in r)

    # Moves: a few legal destinations per unit plus random tiles (moves are
    # BFS-priced, so the full unit x tile product is sampled, not swept).
    legal_moves = {(index[id(a["unit"])], a["to_x"], a["to_y"]) for a in legal["move"]}
    tiles = [(x, y) for y in range(gs.grid.height) for x in range(gs.grid.width)]
    for i in range(n):
        own = [(x, y) for (k, x, y) in legal_moves if k == i]
        for x, y in rng.sample(own, min(3, len(own))) + rng.sample(tiles, 10):
            check((i, x, y) in legal_moves, lambda s, i=i, x=x, y=y: s.move_unit(s.units[i], x, y), bool)

    # Creates: every tile and unit type for the current player, and a sample
    # for the other player (never their turn, so always refused).
    legal_creates = {(a["unit_type"], a["x"], a["y"]) for a in legal["create_unit"]}
    for x, y in tiles:
        for unit_type in ALL_UNIT_TYPES:
            check(
                (unit_type, x, y) in legal_creates,
                lambda s, t=unit_type, x=x, y=y: s.create_unit(t, x, y, player=player),
                lambda r: r is not None,
            )
    other = 3 - player
    for x, y in rng.sample(tiles, 8):
        check(False, lambda s, x=x, y=y: s.create_unit("W", x, y, player=other), lambda r: r is not None)


def _skirmish_battle(fog_of_war: bool) -> GameState:
    """skirmish.csv with a mixed army per side already in contact, so every
    ability (paralyze, heal/cure, haste, buffs, seize) comes up quickly."""
    gs = GameState(FileIO.load_map("maps/1v1/skirmish.csv"), num_players=2, fog_of_war=fog_of_war)
    gs.player_gold[1] = gs.player_gold[2] = 3000
    army_1 = [("W", 2, 3), ("M", 1, 3), ("C", 2, 4), ("A", 1, 4), ("K", 3, 2), ("R", 2, 2), ("S", 1, 2), ("B", 3, 1)]
    army_2 = [("W", 5, 4), ("M", 6, 4), ("C", 5, 3), ("A", 6, 3), ("K", 4, 5), ("R", 5, 5), ("S", 6, 5), ("B", 4, 6)]
    placed = {}
    for player, army in ((1, army_1), (2, army_2)):
        for unit_type, x, y in army:
            placed[(x, y)] = gs.place_unit(unit_type, x, y, player)
    # A paralyzed Warrior next to each side's Cleric, so cure comes up too.
    placed[(2, 3)].paralyzed_turns = 2
    placed[(5, 4)].paralyzed_turns = 3
    return gs


@pytest.mark.parametrize("fog_of_war", [False, True])
def test_engine_accepts_exactly_the_enumerated_actions(fog_of_war):
    """Random play; at many states every candidate action is accepted iff
    get_legal_actions offers it, and a refused one changes nothing."""
    rng = random.Random(1234 + fog_of_war)
    gs = _skirmish_battle(fog_of_war)
    keys = ["create_unit", "move", "seize", *_PAIR_ACTIONS]
    checked = 0
    for step in range(160):
        if gs.game_over:
            break
        if step % 10 == 0:
            _assert_engine_matches_enumeration(gs, rng)
            checked += 1
        legal = gs.get_legal_actions()
        choices = [(k, a) for k in keys for a in legal[k]]
        if not choices or rng.random() < 0.12:
            gs.end_turn()
            continue
        key, action = rng.choice(choices)
        _execute_legal(gs, key, action)
    assert checked >= 10


def test_nothing_is_accepted_after_game_over():
    gs = _skirmish_battle(fog_of_war=False)
    legal = gs.get_legal_actions()  # enumeration still answers
    gs.resign(2)
    before = _snapshot(gs)
    for key in ["create_unit", "move", "seize", *_PAIR_ACTIONS]:
        for action in legal[key]:
            if key == "create_unit":
                assert gs.create_unit(action["unit_type"], action["x"], action["y"], player=1) is None
            elif key == "move":
                assert gs.move_unit(action["unit"], action["to_x"], action["to_y"]) is False
            elif key == "seize":
                assert "damage" not in gs.seize(action["unit"])
            else:
                method, actor_field, target_field, ok = _PAIR_ACTIONS[key]
                assert not ok(getattr(gs, method)(action[actor_field], action[target_field]))
    assert _snapshot(gs) == before


# ---------------------------------------------------------------------------
# phantom counter / phantom damage (core-3, prior-11)
# ---------------------------------------------------------------------------


class _ScriptedRng:
    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.value


class TestNoPhantomDamage:
    def test_defence_reduction_of_no_hit_is_zero(self):
        assert GameMechanics.apply_defence_reduction(0, 5) == 0
        assert GameMechanics.apply_defence_reduction(-3, 0) == 0
        # The min-1 floor still applies to any real hit.
        assert GameMechanics.apply_defence_reduction(0.3, 10) == 1
        assert GameMechanics.apply_defence_reduction(10, 20) == 1

    @pytest.mark.parametrize(
        "attacker_type,defender_type,distance",
        [
            ("M", "W", 2),  # Warrior can't reach range 2
            ("S", "K", 2),
            ("A", "M", 3),  # Mage reaches 1-2 only
            ("W", "A", 1),  # Archer can't shoot point-blank
            ("K", "A", 1),
        ],
    )
    def test_defender_out_of_reach_does_not_counter(self, attacker_type, defender_type, distance):
        gs = _game()
        attacker = gs.place_unit(attacker_type, 4, 4, 1)
        defender = gs.place_unit(defender_type, 4, 4 + distance, 2)
        result = gs.attack(attacker, defender)
        assert result["damage"] > 0 and result["target_alive"]
        assert result["counter_damage"] == 0
        assert attacker.health == attacker.max_health
        assert gs.action_history[-1]["counter_damage"] == 0

    def test_defender_in_reach_still_counters(self):
        gs = _game()
        attacker = gs.place_unit("W", 4, 4, 1)
        defender = gs.place_unit("B", 4, 5, 2)
        result = gs.attack(attacker, defender)
        assert result["counter_damage"] > 0
        assert attacker.health == attacker.max_health - result["counter_damage"]

    def test_rogue_hitting_adjacent_archer_spends_no_evade_roll(self):
        """With no counter coming there is nothing to evade: no roll, no evade flag."""
        rng = _ScriptedRng(0.0)  # would force an evade if rolled
        rogue = Unit("R", 4, 4, 1)
        archer = Unit("A", 4, 5, 2)
        result = GameMechanics.attack_unit(rogue, archer, rng=rng)
        assert rng.calls == 0
        assert result["evade"] is False and result["counter_damage"] == 0
        assert rogue.health == rogue.max_health

    def test_mechanics_attack_from_out_of_range_deals_nothing(self):
        attacker = Unit("W", 0, 0, 1)
        target = Unit("W", 0, 3, 2)
        result = GameMechanics.attack_unit(attacker, target)
        assert result["damage"] == 0 and result["counter_damage"] == 0
        assert attacker.health == attacker.max_health and target.health == target.max_health

    def test_in_range_hits_keep_the_min_one_floor_under_hp_scaled(self):
        """Guard: a 1-HP Knight's charge truncates to base 0 under hp_scaled
        (int(8/18 * 1.5) == 0) but is still a hit for 1, and so is a weak
        wounded defender's counter."""
        knight = Unit("K", 4, 4, 1)
        knight.health = 1
        knight.distance_moved = 4
        target = Unit("W", 4, 5, 2)
        result = GameMechanics.attack_unit(knight, target, damage_model="hp_scaled")
        assert result["charge_bonus"] is True
        assert result["damage"] == 1

        attacker = Unit("W", 4, 4, 1)
        cleric = Unit("C", 4, 5, 2)
        cleric.health = 9  # survives the hit at 1 HP; int(2 * 0.8 * 1/10) == 0
        result = GameMechanics.attack_unit(attacker, cleric, damage_model="hp_scaled")
        assert result["target_alive"] and result["counter_damage"] == 1


# ---------------------------------------------------------------------------
# RL env: a refused action is invalid, never valid (rlenv-2)
# ---------------------------------------------------------------------------


def _env(**kwargs) -> StrategyGameEnv:
    env = StrategyGameEnv(
        map_file="maps/1v1/beginner.csv",
        opponent=None,
        action_space_type="multi_discrete",
        render_mode=None,
        **kwargs,
    )
    env.reset(seed=0)
    assert env.agent_player == 1 and env.game_state.current_player == 1
    return env


def _assert_invalid(env: StrategyGameEnv, action) -> dict:
    before = _snapshot(env.game_state)
    _obs, _reward, _term, _trunc, info = env.step(np.array(action))
    assert info["valid_action"] is False
    assert info["reward_breakdown"]["invalid_penalty"] < 0
    assert _snapshot(env.game_state) == before, "a refused action must not change the game"
    return info


class TestEnvScoresRefusalsAsInvalid:
    def test_create_off_building_is_invalid(self):
        env = _env()
        _assert_invalid(env, [0, 0, 2, 0, 2, 0])  # plains next to player 1's buildings
        assert env.game_state.units == []

    def test_create_of_disabled_type_is_invalid(self):
        env = _env(enabled_units=["W", "M"])
        env.game_state.player_gold[1] = 5000
        _assert_invalid(env, [0, 6, 1, 0, 1, 0])  # Sorcerer on b_1
        assert env.game_state.units == []

    def test_out_of_range_attack_is_invalid(self):
        env = _env()
        gs = env.game_state
        attacker = gs.place_unit("W", 0, 3, 1)
        target = gs.place_unit("W", 5, 3, 2)
        _assert_invalid(env, [2, 0, 0, 3, 5, 3])
        assert attacker.health == attacker.max_health and target.health == target.max_health

    def test_attack_by_paralyzed_unit_is_invalid(self):
        env = _env()
        gs = env.game_state
        attacker = gs.place_unit("W", 3, 3, 1)
        gs.place_unit("W", 3, 4, 2)
        attacker.paralyzed_turns = 2
        _assert_invalid(env, [2, 0, 3, 3, 3, 4])

    def test_second_seize_in_a_turn_is_invalid(self):
        env = _env()
        gs = env.game_state
        gs.place_unit("W", 2, 2, 1)  # neutral tower
        _obs, _reward, _term, _trunc, info = env.step(np.array([3, 0, 2, 2, 2, 2]))
        assert info["valid_action"] is True
        _assert_invalid(env, [3, 0, 2, 2, 2, 2])


# ---------------------------------------------------------------------------
# ModelBot notices refusals
# ---------------------------------------------------------------------------


class TestModelBotReportsRefusals:
    def test_out_of_range_attack_reports_failure(self):
        gs = _game()
        gs.place_unit("W", 4, 4, 1)
        gs.place_unit("W", 4, 9, 2)
        bot = ModelBot(gs, player=1)
        assert bot._execute_action([2, 0, 4, 4, 4, 9]) is False

    def test_second_seize_reports_failure(self):
        gs = _game()
        gs.place_unit("W", 6, 3, 1)  # neutral tower
        bot = ModelBot(gs, player=1)
        assert bot._execute_action([3, 0, 6, 3, 6, 3]) is True
        assert bot._execute_action([3, 0, 6, 3, 6, 3]) is False

    def test_create_out_of_turn_reports_failure(self):
        gs = _game()  # player 1's turn
        bot = ModelBot(gs, player=2)
        assert bot._execute_action([0, 0, 8, 9, 8, 9]) is False
        assert gs.units == []


# ---------------------------------------------------------------------------
# Rule bots: knight charge / rogue flank check the move (rulebots-2)
# ---------------------------------------------------------------------------


def _corridor_game(attacker_type: str) -> tuple[GameState, Unit, Unit]:
    """A one-tile-wide corridor at x=3. The only tile next to the enemy at
    (3, 1) is (3, 2), where an ally stands: it is in the attacker's
    reachable set (allies can be passed through) but the move onto it is
    refused. The attacker starts 4 tiles away at (3, 6)."""
    md = np.array([["w"] * 7 for _ in range(8)], dtype=object)
    for y in range(1, 8):
        md[y][3] = "p"
    md[7][0] = "h_1"
    md[7][6] = "h_2"
    gs = GameState(md, num_players=2)
    attacker = gs.place_unit(attacker_type, 3, 6, 1)
    gs.place_unit("W", 3, 2, 1)
    enemy = gs.place_unit("W", 3, 1, 2)
    enemy.health = 5
    return gs, attacker, enemy


class TestRuleBotsCheckTheMove:
    def test_knight_charge_does_not_attack_when_the_move_is_refused(self):
        gs, knight, enemy = _corridor_game("K")
        bot = AdvancedBot(gs, player=1)
        assert bot._try_knight_charge(knight) is False
        assert (knight.x, knight.y) == (3, 6)
        assert enemy.health == 5
        assert not any(a["type"] == "attack" for a in gs.action_history)
        assert "knight_charge" not in bot.get_capabilities_fired()

    def test_rogue_flank_does_not_attack_when_the_move_is_refused(self):
        gs, rogue, enemy = _corridor_game("R")
        bot = AdvancedBot(gs, player=1)
        assert bot._try_rogue_flank(rogue) is False
        assert (rogue.x, rogue.y) == (3, 6)
        assert enemy.health == 5
        assert not any(a["type"] == "attack" for a in gs.action_history)
        assert "rogue_flank" not in bot.get_capabilities_fired()

    @pytest.mark.parametrize(
        "bot_cls,method",
        [(SimpleBot, "purchase_units"), (MediumBot, "purchase_units"), (AdvancedBot, "purchase_units_enhanced")],
    )
    def test_purchase_loop_stops_when_the_engine_refuses(self, bot_cls, method):
        """Out of turn the engine refuses every create; the purchase loop,
        which re-reads an unchanged legal list, must stop instead of spin."""
        gs = _game()  # player 1's turn
        bot = bot_cls(gs, player=2)
        getattr(bot, method)()
        assert gs.units == []
        assert gs.player_gold[2] == 5000


# ---------------------------------------------------------------------------
# place_unit: the setup primitive
# ---------------------------------------------------------------------------


class TestPlaceUnit:
    def test_place_is_free_unrecorded_and_ready(self):
        gs = _game()
        unit = gs.place_unit("K", 2, 2, 2)  # water, player 2, not their turn
        assert gs.player_gold == {1: 5000, 2: 5000}
        assert gs.action_history == []
        assert unit.can_move and unit.can_attack
        assert gs.get_unit_at_position(2, 2) is unit

    def test_place_and_create_share_the_unit_id_counter(self):
        gs = _game()
        placed = gs.place_unit("W", 4, 4, 2)
        created = gs.create_unit("W", 1, 0, player=1)
        assert (placed.unit_id, created.unit_id) == (0, 1)

    def test_place_invalidates_the_legal_action_cache(self):
        gs = _game()
        assert gs.get_legal_actions(1)["move"] == []
        unit = gs.place_unit("W", 4, 4, 1)
        assert any(a["unit"] is unit for a in gs.get_legal_actions(1)["move"])

    def test_place_refreshes_fog_of_war(self):
        gs = _game(fog_of_war=True)
        assert not gs.is_position_visible(7, 5, 1)
        gs.place_unit("A", 5, 5, 1)
        assert gs.is_position_visible(7, 5, 1)

    @pytest.mark.parametrize("unit_type,x,y", [("Z", 4, 4), ("W", 10, 4), ("W", -1, 0), ("W", 4, 4)])
    def test_place_rejects_states_the_engine_cannot_represent(self, unit_type, x, y):
        gs = _game()
        if (x, y) == (4, 4) and unit_type == "W":
            gs.place_unit("M", 4, 4, 1)  # occupied
        with pytest.raises(ValueError):
            gs.place_unit(unit_type, x, y, 1)


# ---------------------------------------------------------------------------
# Replays: a refused re-execution is reported, not silently dropped
# ---------------------------------------------------------------------------


def test_replay_warns_when_the_engine_refuses_a_recorded_action(caplog):
    """A v1 replay recorded before the engine enforced the rules (e.g. an LLM
    SEIZE-spam) re-executes through the engine: the illegal repeat is
    refused and the divergence is logged."""
    gs = _game()
    gs.place_unit("W", 9, 9, 1)  # on player 2's HQ
    seize = {"type": "seize", "player": 1, "unit_type": "W", "position": [9, 9]}
    with caplog.at_level(logging.WARNING, logger="reinforcetactics.utils.replay_actions"):
        execute_replay_action(gs, dict(seize), lambda x, y: (x, y), schema_version=1)
        execute_replay_action(gs, dict(seize), lambda x, y: (x, y), schema_version=1)
    refused = [r for r in caplog.records if "refused" in r.getMessage()]
    assert len(refused) == 1
    hq = gs.grid.get_tile(9, 9)
    assert hq.health == hq.max_health - 15 and hq.player == 2
