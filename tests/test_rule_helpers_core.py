"""Each rule is written once (review core-15).

Ability ranges were hard-coded in the mask predicates, again in the
mechanics that apply the ability and again in the bots; the attack-damage
lookup restated every unit's reach next to ``Unit.get_attack_range``; five
helpers ticked the turn-start counters; and six ``GameState`` methods
repeated the validate / apply / spend / record / invalidate sequence. These
tests pin the shared versions: ``constants.ABILITY_RANGES`` feeds both the
legal-action list and execution, attack damage follows ``get_attack_range``,
``GameMechanics.tick_statuses`` runs every counter from one table, and the
ability records keep the layout replays and saves rely on.
"""

import numpy as np
import pytest

from reinforcetactics import constants
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.mechanics import GameMechanics
from reinforcetactics.core.unit import Unit


def _game():
    grid = np.full((10, 10), "p", dtype=object)
    grid[0, 0], grid[9, 9] = "h_1", "h_2"
    return GameState(grid, num_players=2)


# ---------------------------------------------------------------------------
# units_in_range
# ---------------------------------------------------------------------------


def test_units_in_range_is_an_inclusive_scan_in_list_order():
    center = Unit("W", 5, 5, 1)
    at = {d: Unit("W", 5 + d, 5, 2) for d in range(4)}  # at[d] is d tiles east
    diagonal = Unit("W", 6, 6, 2)  # Manhattan distance 2
    units = [at[3], diagonal, at[1], center, at[2]]

    assert GameMechanics.units_in_range(center, units, 1, 2) == [diagonal, at[1], at[2]]
    assert GameMechanics.units_in_range(center, units, 2, 3) == [at[3], diagonal, at[2]]
    assert GameMechanics.units_in_range(center, units, 1, 1) == [at[1]]
    assert GameMechanics.units_in_range(center, units, 4, 9) == []


def test_units_in_range_includes_the_center_only_from_distance_zero():
    center = Unit("S", 5, 5, 1)
    ally = Unit("W", 5, 6, 1)
    assert GameMechanics.units_in_range(center, [center, ally], 0, 1) == [center, ally]
    assert GameMechanics.units_in_range(center, [center, ally], 1, 1) == [ally]


def test_units_in_range_filters_with_the_predicate():
    center = Unit("W", 5, 5, 1)
    enemy, ally = Unit("W", 5, 6, 2), Unit("W", 6, 5, 1)
    assert GameMechanics.units_in_range(center, [enemy, ally], 1, 1, lambda u: u.player == 2) == [enemy]


# ---------------------------------------------------------------------------
# ABILITY_RANGES is the one source for the mask and for execution
# ---------------------------------------------------------------------------

# ability -> (caster type, target owner, target setup, mechanics function, legal-action caster key)
ABILITIES = {
    "paralyze": ("M", 2, None, "paralyze_unit", "paralyzer"),
    "heal": ("C", 1, "damaged", "heal_unit", "healer"),
    "cure": ("C", 1, "paralyzed", "cure_unit", "curer"),
    "haste": ("S", 1, None, "haste_unit", "sorcerer"),
    "defence_buff": ("S", 1, None, "defence_buff_unit", "sorcerer"),
    "attack_buff": ("S", 1, None, "attack_buff_unit", "sorcerer"),
}


def _two_tiles_apart(ability):
    caster_type, target_player, setup, _, _ = ABILITIES[ability]
    game = _game()
    caster = game.place_unit(caster_type, 4, 4, 1)
    target = game.place_unit("W", 4, 6, target_player)
    if setup == "damaged":
        target.health -= 5
    elif setup == "paralyzed":
        target.paralyzed_turns = 2
    return game, caster, target


def _listed(game, ability, target):
    return any(entry["target"] is target for entry in game.get_legal_actions(1)[ability])


@pytest.mark.parametrize("ability", sorted(ABILITIES))
def test_every_ability_reaches_two_tiles_by_default(ability):
    game, caster, target = _two_tiles_apart(ability)
    assert _listed(game, ability, target)
    assert getattr(game, ability)(caster, target)


@pytest.mark.parametrize("ability", sorted(ABILITIES))
def test_narrowing_an_ability_range_narrows_the_mask_and_execution_alike(ability, monkeypatch):
    monkeypatch.setitem(constants.ABILITY_RANGES, ability, (1, 1))
    game, caster, target = _two_tiles_apart(ability)
    _, _, _, mechanics_fn, _ = ABILITIES[ability]

    assert not _listed(game, ability, target)
    assert not getattr(game, ability)(caster, target)
    assert game.action_history == []
    # The mechanics layer reads the same table, so a direct caller is refused too.
    args = (caster, target) if ability == "haste" else (caster, target, game.teams)
    assert getattr(GameMechanics, mechanics_fn)(*args) in (False, -1)


def test_only_the_buffs_let_a_sorcerer_target_itself():
    game = _game()
    sorcerer = game.place_unit("S", 4, 4, 1)
    legal = game.get_legal_actions(1)
    assert any(e["target"] is sorcerer for e in legal["defence_buff"])
    assert any(e["target"] is sorcerer for e in legal["attack_buff"])
    assert not any(e["target"] is sorcerer for e in legal["haste"])


# ---------------------------------------------------------------------------
# Attack damage follows get_attack_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit_type", constants.ALL_UNIT_TYPES)
@pytest.mark.parametrize("on_mountain", [False, True])
def test_attack_damage_is_positive_exactly_within_get_attack_range(unit_type, on_mountain):
    unit = Unit(unit_type, 5, 5, 1)
    lo, hi = unit.get_attack_range(on_mountain)
    for dx, dy in [(d, 0) for d in range(7)] + [(1, 1), (2, 1), (2, 2)]:
        damage = unit.get_attack_damage(5 + dx, 5 + dy, on_mountain)
        assert (damage > 0) == (lo <= dx + dy <= hi), (unit_type, on_mountain, dx, dy)


@pytest.mark.parametrize("unit_type", ["M", "S"])
def test_split_attacks_hit_adjacent_and_at_range_with_their_own_values(unit_type):
    unit = Unit(unit_type, 5, 5, 1)
    attack = constants.UNIT_DATA[unit_type]["attack"]
    assert unit.get_attack_damage(5, 6) == attack["adjacent"]
    assert unit.get_attack_damage(6, 6) == attack["range"]


# ---------------------------------------------------------------------------
# tick_statuses
# ---------------------------------------------------------------------------

COUNTERS = [attr for attr, _ in GameMechanics.TURN_START_COUNTERS]


def _loaded(unit_type, player, value=2):
    unit = Unit(unit_type, 0, 0, player)
    for attr in COUNTERS:
        setattr(unit, attr, value)
    return unit


def test_tick_statuses_ticks_the_players_counters_by_unit_type():
    warrior, mage, sorcerer = _loaded("W", 1), _loaded("M", 1), _loaded("S", 1)
    enemy = _loaded("S", 2)

    assert GameMechanics.tick_statuses([warrior, mage, sorcerer, enemy], 1) == []

    statuses = ["paralyzed_turns", "defence_buff_turns", "attack_buff_turns"]
    sorcerer_cooldowns = ["haste_cooldown", "defence_buff_cooldown", "attack_buff_cooldown"]
    for unit, ticked in (
        (warrior, statuses),
        (mage, statuses + ["paralyze_cooldown"]),
        (sorcerer, statuses + sorcerer_cooldowns),
        (enemy, []),  # not its turn
    ):
        for attr in COUNTERS:
            assert getattr(unit, attr) == (1 if attr in ticked else 2), (unit.type, unit.player, attr)


def test_tick_statuses_reports_what_ran_out_and_stops_at_zero():
    mage = _loaded("M", 1, value=1)
    mage.attack_buff_turns = 0

    expired = GameMechanics.tick_statuses([mage], 1)

    assert expired == [(mage, "paralyzed_turns"), (mage, "paralyze_cooldown"), (mage, "defence_buff_turns")]
    assert mage.attack_buff_turns == 0
    assert mage.haste_cooldown == 1  # a Sorcerer's cooldown on a Mage never ticks
    assert GameMechanics.tick_statuses([mage], 1) == []


# ---------------------------------------------------------------------------
# The ability methods' shared body keeps the recorded layout
# ---------------------------------------------------------------------------

RECORD_FIELDS = {
    "paralyze": ["paralyzer_pos", "target_pos", "actor_unit_id", "target_unit_id"],
    "heal": ["healer_pos", "target_pos", "amount", "target_hp_after", "actor_unit_id", "target_unit_id"],
    "cure": ["curer_pos", "target_pos", "actor_unit_id", "target_unit_id"],
    "haste": ["sorcerer_pos", "target_pos", "target_type", "actor_unit_id", "target_unit_id"],
    "defence_buff": ["sorcerer_pos", "target_pos", "target_type", "actor_unit_id", "target_unit_id"],
    "attack_buff": ["sorcerer_pos", "target_pos", "target_type", "actor_unit_id", "target_unit_id"],
}


@pytest.mark.parametrize("ability", sorted(ABILITIES))
def test_ability_records_keep_their_field_layout(ability):
    game, caster, target = _two_tiles_apart(ability)

    result = getattr(game, ability)(caster, target)

    assert result == (5 if ability == "heal" else True)  # heal returns the HP restored
    assert type(result) is (int if ability == "heal" else bool)
    (record,) = game.action_history
    assert list(record) == ["turn", "player", "type", "timestamp", *RECORD_FIELDS[ability]]
    assert record["type"] == ability
    assert record["player"] == 1
    assert (record["actor_unit_id"], record["target_unit_id"]) == (caster.unit_id, target.unit_id)
    assert record["target_pos"] == (4, 6)
    assert not caster.can_attack  # the caster's action is spent


@pytest.mark.parametrize("ability", sorted(ABILITIES))
def test_a_refused_ability_returns_its_falsy_value_and_changes_nothing(ability):
    game, caster, target = _two_tiles_apart(ability)
    game.end_turn()  # not player 1's turn any more
    before = [u.to_dict() for u in game.units]
    history = list(game.action_history)

    result = getattr(game, ability)(caster, target)

    assert result == (0 if ability == "heal" else False)
    assert type(result) is (int if ability == "heal" else bool)
    assert [u.to_dict() for u in game.units] == before
    assert game.action_history == history
