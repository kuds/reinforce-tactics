"""Status durations count the affected unit's own turns (review core-11).

PARALYZE_DURATION used to be 3 while a paralyzed unit lost 2 of its own turns
(the counter ticks at the start of each of the victim's turns, and a
paralysis is cast on the opponent's turn), and README/docs said "3 turns".
The constant now states the gameplay -- the victim loses its next
PARALYZE_DURATION (2) turns -- and the engine stores PARALYZE_DURATION + 1 so
play is unchanged, the counter-attack window included. Buffs follow the same
rule: SORCERER_BUFF_DURATION of the buffed unit's own turns, also for a
teammate's unit buffed outside its owner's turn.
"""

import numpy as np

from reinforcetactics.constants import PARALYZE_DURATION, SORCERER_BUFF_DURATION
from reinforcetactics.core.game_state import GameState


def _game(num_players=2, teams=None):
    grid = np.full((10, 10), "p", dtype=object)
    grid[0, 0], grid[9, 9] = "h_1", "h_2"
    if num_players == 4:
        grid[0, 9], grid[9, 0] = "h_3", "h_4"
    # teams only when given, so the 1v1 checks also run on engines without teams
    return GameState(grid, num_players=num_players, **({"teams": teams} if teams else {}))


def _to_turn_of(game, player):
    game.end_turn()
    while game.current_player != player:
        game.end_turn()


def test_a_paralyzed_unit_loses_exactly_paralyze_duration_of_its_own_turns():
    game = _game()
    mage = game.place_unit("M", 4, 4, 1)
    victim = game.place_unit("W", 4, 6, 2)
    assert game.paralyze(mage, victim)

    lost = 0
    for _ in range(PARALYZE_DURATION + 3):
        _to_turn_of(game, 2)
        if victim.can_move or victim.can_attack:
            break
        lost += 1
        assert victim.is_paralyzed()
        assert not any(a["unit"] is victim for a in game.get_legal_actions(2)["move"])

    assert lost == PARALYZE_DURATION == 2


def test_no_counter_attack_or_re_paralysis_until_the_victims_first_free_turn():
    """The window the victim can't counter in is unchanged: from the cast
    until its first free turn starts, i.e. also the paralyzer's turn right
    before it -- which is also why the Mage (cooldown 2) can't chain it."""
    game = _game()
    mage = game.place_unit("M", 4, 4, 1)
    victim = game.place_unit("W", 4, 6, 2)
    hitter = game.place_unit("W", 4, 7, 1)
    game.paralyze(mage, victim)

    for _paralyzer_turn in range(PARALYZE_DURATION + 1):
        assert victim.is_paralyzed()
        assert game.attack(hitter, victim)["counter_damage"] == 0
        victim.health = victim.max_health  # keep it alive for the next round
        assert not any(a["target"] is victim for a in game.get_legal_actions(1)["paralyze"])
        _to_turn_of(game, 1)

    assert not victim.is_paralyzed()
    assert game.attack(hitter, victim)["counter_damage"] > 0


def test_an_own_units_buff_covers_buff_duration_of_its_turns():
    game = _game()
    sorcerer = game.place_unit("S", 4, 4, 1)
    ally = game.place_unit("W", 4, 5, 1)
    assert game.attack_buff(sorcerer, ally)

    active_turns = 1  # the cast turn
    for _ in range(SORCERER_BUFF_DURATION + 2):
        _to_turn_of(game, 1)
        if not ally.has_attack_buff():
            break
        active_turns += 1

    assert active_turns == SORCERER_BUFF_DURATION


def test_a_teammates_unit_buffed_off_turn_also_gets_buff_duration_of_its_turns():
    game = _game(4, teams={1: 1, 2: 2, 3: 1, 4: 2})
    sorcerer = game.place_unit("S", 4, 4, 1)
    mate = game.place_unit("W", 4, 5, 3)
    assert game.defence_buff(sorcerer, mate)

    active_turns = 0
    for _ in range(SORCERER_BUFF_DURATION + 2):
        _to_turn_of(game, 3)
        if not mate.has_defence_buff():
            break
        active_turns += 1

    assert active_turns == SORCERER_BUFF_DURATION
