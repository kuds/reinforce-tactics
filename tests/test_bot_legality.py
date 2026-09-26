"""Every scripted bot sends the engine only actions it accepts (review rulebots-14).

The engine refuses an illegal action and changes nothing (``move_unit``
returns False, ``attack`` deals 0 damage, ``seize`` returns no ``damage``, an
ability returns a falsy value, ``create_unit`` None). A bot that sends one
has lost track of the game: at review time about two thirds of SimpleBot's
moves were refused, its units recursed on actions they could not take, and
the curriculum's opponents drew most mirror games. Every one of those bugs
passed the bot unit tests, which look at single decisions.

This harness plays whole seeded games -- every scripted tier in mirror
matches on every bundled 1v1 and 1v1v1 map and the 2v2 map, a few cross-tier
pairings, and a fog-of-war subset -- and wraps the game's action methods and
the bots' per-unit act methods to assert, for every call:

* the engine accepted the action;
* the acting unit was in play (not a unit a counter-attack killed);
* a bot re-entered its per-unit decision (``_depth`` > 0) only for a unit
  the engine gave another action by haste, at most once per haste (so never
  deeper than 2 unless hasted twice);
* a bot's capture claims this turn never outnumber the units it could move
  at the start of the turn, plus hastes.
"""

import random
from collections import Counter
from pathlib import Path

import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot_registry import build_scripted
from reinforcetactics.utils.file_io import FileIO

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"

# Every scripted tier but NoopBot, which never acts.
TIERS = ("simple", "medium", "advanced", "master", "random", "balanced_random", "mixed")

# How each engine action method reports a refusal.
_REFUSED = {
    "create_unit": lambda result: result is None,
    "move_unit": lambda result: not result,
    "attack": lambda result: result["damage"] <= 0,
    "seize": lambda result: "damage" not in result,
    "paralyze": lambda result: not result,
    "heal": lambda result: not result,
    "cure": lambda result: not result,
    "haste": lambda result: not result,
    "defence_buff": lambda result: not result,
    "attack_buff": lambda result: not result,
}


def _maps(folder):
    return sorted(str(p.relative_to(MAPS_DIR)) for p in (MAPS_DIR / folder).glob("*.csv"))


MAPS_1V1 = _maps("1v1")
MAPS_1V1V1 = _maps("1v1v1")
SEATS = {"1v1": 2, "1v1v1": 3, "2v2": 4}
# Mirror-match length by map folder: long enough for the armies to meet on
# the maps that have production buildings (7 of the 1v1 maps have none, and
# nothing happens on them), short enough to keep the whole matrix near 30 s.
# The 1v1v1 maps are the largest and seat three bots.
MIRROR_TURNS = {"1v1": 14, "1v1v1": 10, "2v2": 16}


def _describe(arg):
    """A unit as ``W#3@(4, 5)``; anything else as its repr."""
    if hasattr(arg, "unit_id"):
        return f"{arg.type}#{arg.unit_id}@({arg.x}, {arg.y})"
    return repr(arg)


class Referee:
    """Wraps a game's action methods and its bots' act methods, collecting every breach of the contract."""

    def __init__(self, game, bots):
        self.game = game
        self.bots = bots
        self.breaches: list[str] = []
        self.calls: Counter = Counter()
        self.hastes: Counter = Counter()  # accepted hastes per target unit_id this turn
        self.movers_at_turn_start = 0
        for name in (*_REFUSED, "end_unit_turn"):
            setattr(game, name, self._checked(name, getattr(game, name)))
        for bot in bots.values():
            # MixedBot plays through the bot it picked.
            player = getattr(bot, "_inner", bot)
            for name in ("act_with_unit", "act_with_unit_enhanced"):
                if hasattr(player, name):
                    setattr(player, name, self._depth_checked(player, getattr(player, name)))

    def _breach(self, message):
        game = self.game
        self.breaches.append(f"turn {game.turn_number} player {game.current_player}: {message}")

    def start_turn(self):
        self.hastes.clear()
        self.movers_at_turn_start = sum(
            1 for u in self.game.units if u.player == self.game.current_player and (u.can_move or u.can_attack)
        )

    def _checked(self, name, method):
        refused = _REFUSED.get(name)

        def checked(*args, **kwargs):
            actor = args[0] if args and name != "create_unit" else None
            if actor is not None and actor not in self.game.units:
                self._breach(f"{name} by {_describe(actor)}, which is not in play")
            result = method(*args, **kwargs)
            self.calls[name] += 1
            if refused is not None and refused(result):
                self._breach(f"refused {name}({', '.join(map(_describe, args))})")
            if name == "haste" and result:
                self.hastes[args[1].unit_id] += 1
            self._check_claims()
            return result

        return checked

    def _depth_checked(self, bot, method):
        def act(unit, _depth=0):
            if _depth > self.hastes[unit.unit_id]:
                self._breach(
                    f"{type(bot).__name__} re-entered {_describe(unit)} at depth {_depth} "
                    f"after {self.hastes[unit.unit_id]} haste(s)"
                )
            return method(unit, _depth)

        return act

    def _check_claims(self):
        bot = self.bots.get(self.game.current_player)
        claims = getattr(getattr(bot, "_inner", bot), "_capture_assigned", None)
        if claims is not None and len(claims) > self.movers_at_turn_start + sum(self.hastes.values()):
            self._breach(f"{len(claims)} capture claims for {self.movers_at_turn_start} units: {sorted(claims)}")


def play(map_name, tiers, seed, max_turns, fog=False):
    """Play one seeded game with ``tiers`` by seat under the Referee; return it."""
    seats = SEATS[map_name.split("/")[0]]
    game = GameState(
        FileIO.load_map(str(MAPS_DIR / map_name)), num_players=seats, max_turns=max_turns, fog_of_war=fog, seed=seed
    )
    bots = {
        p: build_scripted(tiers[(p - 1) % len(tiers)], game, player=p, rng=random.Random(seed * 10 + p))
        for p in range(1, seats + 1)
    }
    referee = Referee(game, bots)
    for _ in range(max_turns * seats + 1):
        if game.game_over:
            break
        player = game.current_player
        referee.start_turn()
        bots[player].take_turn()
        if not game.game_over and game.current_player == player:
            referee._breach("the bot did not end its turn")
            break
    return referee


def _assert_clean(referee):
    assert not referee.breaches, "\n".join(referee.breaches[:20])
    assert referee.game.game_over, "the game outlived max_turns"


@pytest.mark.parametrize("map_name", [*MAPS_1V1, *MAPS_1V1V1, "2v2/beginner.csv"])
def test_mirror_matches_send_only_legal_actions(map_name):
    """Every tier against itself on every map (fog off)."""
    for index, tier in enumerate(TIERS):
        _assert_clean(play(map_name, (tier,), seed=index + 1, max_turns=MIRROR_TURNS[map_name.split("/")[0]]))


CROSS_TIER = [("simple", "master"), ("medium", "advanced"), ("balanced_random", "medium"), ("master", "random")]


@pytest.mark.parametrize(
    "map_name", ["1v1/beginner.csv", "1v1/crossroads.csv", "1v1/tower_rush.csv", "1v1v1/triangle_arena.csv"]
)
@pytest.mark.parametrize("tiers", CROSS_TIER, ids="-".join)
def test_cross_tier_matches_send_only_legal_actions(map_name, tiers):
    _assert_clean(play(map_name, tiers, seed=7, max_turns=16))


@pytest.mark.parametrize(
    "map_name", ["1v1/beginner.csv", "1v1/crossroads.csv", "1v1v1/triangle_arena.csv", "2v2/beginner.csv"]
)
@pytest.mark.parametrize("tier", TIERS)
def test_fog_of_war_matches_send_only_legal_actions(map_name, tier):
    """Under fog of war a bot may attack only enemies its side saw when the unit's action began."""
    _assert_clean(play(map_name, (tier,), seed=3, max_turns=12, fog=True))


def test_the_referee_catches_refused_and_dead_unit_actions():
    """The harness itself: a refused move and an action by a removed unit are both breaches."""
    referee = play("1v1/beginner.csv", ("simple",), seed=1, max_turns=2)
    game = referee.game
    unit = next(u for u in game.units if u.player == game.current_player)
    game.move_unit(unit, -1, -1)
    game.units.remove(unit)
    game.end_unit_turn(unit)
    assert any("refused move_unit" in b for b in referee.breaches)
    assert any("not in play" in b for b in referee.breaches)
