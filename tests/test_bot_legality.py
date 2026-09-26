"""Every scripted bot sends the engine only actions it accepts (review rulebots-14).

The engine refuses an illegal action and changes nothing (``move_unit``
returns False, ``attack`` deals 0 damage, ``seize`` returns no ``damage``, an
ability returns a falsy value, ``create_unit`` None). A bot that sends one
has lost track of the game: at review time about two thirds of SimpleBot's
moves were refused, its units recursed on actions they could not take, and
the curriculum's opponents drew most mirror games. Every one of those bugs
passed the bot unit tests, which look at single decisions.

This harness plays whole seeded games -- every scripted tier in mirror
matches on every bundled 1v1, 1v1v1 and 2v2 map (the 1v1 maps without
production buildings from the scenario saves they are made for, units and
all), a few cross-tier pairings, and fog-of-war games, some with every unit's
attack snapshot cut down so fog conflicts happen every turn -- and wraps the
game's action methods and the bots' per-unit act methods to assert, for
every call:

* the engine accepted the action;
* the acting unit was in play (not a unit a counter-attack killed);
* a bot ran its per-unit decision for a unit at most once per action the
  unit had: once, plus once per haste it received this turn (a re-entry,
  ``_depth`` > 0, or MasterBot's post-pass);
* a bot's capture claims this turn never outnumber the units it could move
  at the start of the turn, plus hastes.

And, for every tier but the two random ones on a map with production, that
the bots moved and attacked at all: a bot that sends nothing breaks no rule
above.

Most of the matrix is marked ``slow`` (pyproject.toml deselects it by
default; ``pytest -m slow --no-cov`` runs it, and CI does in its own step):
the default run keeps a sample of each kind of game.
"""

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot_registry import build_scripted
from reinforcetactics.rules import TileType
from reinforcetactics.utils.file_io import FileIO

ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = ROOT / "maps"
SAVES_DIR = ROOT / "saves"

# Every scripted tier but NoopBot, which never acts.
TIERS = ("simple", "medium", "advanced", "master", "random", "balanced_random", "mixed")
# RandomBot and BalancedRandomBot pick uniformly among legal actions and can
# go a short game without attacking; every other tier must move and attack.
PURPOSEFUL_TIERS = ("simple", "medium", "advanced", "master", "mixed")

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


def _has_production(csv_path):
    """Whether the map has a building (``b``, ``b_1``, ...): a tile units are bought on."""
    codes = (code.strip().split("_")[0] for line in csv_path.read_text().splitlines() for code in line.split(","))
    return TileType.BUILDING.value in codes


def _maps(folder):
    return sorted(str(p.relative_to(MAPS_DIR)) for p in (MAPS_DIR / folder).glob("*.csv"))


SCENARIOS = sorted(f"saves/{p.name}" for p in SAVES_DIR.glob("*_scenario.json"))
# A 1v1 map without production buildings is only the terrain of a scenario:
# bare, it holds no unit and nothing ever happens on it (its scenario save
# is played instead).
MAPS_1V1 = [m for m in _maps("1v1") if _has_production(MAPS_DIR / m)]
MAPS_1V1V1 = _maps("1v1v1")
MAPS_2V2 = _maps("2v2")
SEATS = {"1v1": 2, "1v1v1": 3, "2v2": 4}
# Mirror-match length by kind of game: long enough for the armies to meet,
# short enough to keep the default sample under a minute with coverage on.
# The 1v1v1 maps are the largest and seat three bots.
MIRROR_TURNS = {"1v1": 20, "1v1v1": 14, "2v2": 20, "saves": 20}

# The games the default run plays; the rest of each matrix is marked slow.
DEFAULT_MIRRORS = {"1v1/beginner.csv", "1v1/crossroads.csv", "1v1v1/triangle_arena.csv", "2v2/beginner.csv"}
DEFAULT_MIRRORS |= {
    "saves/cavalry_charge_scenario.json",
    "saves/mage_showdown_scenario.json",
    "saves/sorcerer_cabal_scenario.json",
}


def _sample(items, default):
    """``items`` for parametrize, those not in ``default`` marked slow."""
    return [item if item in default else pytest.param(item, marks=pytest.mark.slow) for item in items]


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
        # Per unit (by identity: units loaded from a scenario save have no unit_id), this turn:
        self.hastes: Counter = Counter()  # accepted hastes it received
        self.acts: Counter = Counter()  # per-unit decisions the bot ran for it
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
        self.acts.clear()
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
                self.hastes[id(args[1])] += 1
            self._check_claims()
            return result

        return checked

    def _depth_checked(self, bot, method):
        def act(unit, _depth=0):
            self.acts[id(unit)] += 1
            hastes = self.hastes[id(unit)]
            if _depth > hastes or self.acts[id(unit)] > 1 + hastes:
                self._breach(
                    f"{type(bot).__name__} ran its decision for {_describe(unit)} "
                    f"{self.acts[id(unit)]} time(s), at depth {_depth}, after {hastes} haste(s)"
                )
            return method(unit, _depth)

        return act

    def _check_claims(self):
        bot = self.bots.get(self.game.current_player)
        claims = getattr(getattr(bot, "_inner", bot), "_capture_assigned", None)
        if claims is not None and len(claims) > self.movers_at_turn_start + sum(self.hastes.values()):
            self._breach(f"{len(claims)} capture claims for {self.movers_at_turn_start} units: {sorted(claims)}")


def new_game(source, max_turns, seed, fog=False):
    """A seeded game from a map under maps/ (``1v1/beginner.csv``) or a scenario save (``saves/...json``)."""
    if source.startswith("saves/"):
        save = json.loads((ROOT / source).read_text())
        save.update(max_turns=max_turns, seed=seed)
        return GameState.from_dict(save, FileIO.load_map(str(ROOT / save["map_file"]), for_ui=True, border_size=2))
    seats = SEATS[source.split("/")[0]]
    return GameState(
        FileIO.load_map(str(MAPS_DIR / source)), num_players=seats, max_turns=max_turns, fog_of_war=fog, seed=seed
    )


def _blind(game, rng):
    """Cut every attack snapshot of the player to move to a random half of the enemies its side sees.

    As if each unit's action had begun before the rest came into sight (a
    friend's move uncovering them): the engine then refuses an attack or
    paralyze on any enemy the unit's snapshot lacks, so a bot that asks
    only whether an enemy is in range sends one almost every turn.
    """
    player = game.current_player
    seen = [(e.x, e.y) for e in game.units if game.are_enemies(e.player, player) and game.fog.is_visible(e.x, e.y, player)]
    for unit in game.units:
        if unit.player == player:
            game.fog._set_attack_snapshot(unit, {pos for pos in seen if rng.random() < 0.5})


def play(source, tiers, seed, max_turns, fog=False, blind=False):
    """Play one seeded game with ``tiers`` by seat under the Referee; return it."""
    game = new_game(source, max_turns, seed, fog=fog or blind)
    seats = game.num_players
    bots = {
        p: build_scripted(tiers[(p - 1) % len(tiers)], game, player=p, rng=random.Random(seed * 10 + p))
        for p in range(1, seats + 1)
    }
    referee = Referee(game, bots)
    rng = random.Random(seed)
    for _ in range(max_turns * seats + 1):
        if game.game_over:
            break
        player = game.current_player
        referee.start_turn()
        if blind:
            _blind(game, rng)
        bots[player].take_turn()
        if not game.game_over and game.current_player == player:
            referee._breach("the bot did not end its turn")
            break
    return referee


def _assert_clean(referee, lively=False):
    assert not referee.breaches, "\n".join(referee.breaches[:20])
    assert referee.game.game_over, "the game outlived max_turns"
    if lively:
        assert referee.calls["move_unit"] and referee.calls["attack"], f"the bots barely played: {dict(referee.calls)}"


def test_every_bundled_map_is_played():
    """A 1v1 map without production is played from its scenario save, not left out."""
    for map_name in [m for m in _maps("1v1") if m not in MAPS_1V1]:
        stem = Path(map_name).stem
        assert f"saves/{stem}_scenario.json" in SCENARIOS, f"{map_name} has no production and no scenario save"


@pytest.mark.parametrize("source", _sample([*MAPS_1V1, *MAPS_1V1V1, *MAPS_2V2, *SCENARIOS], DEFAULT_MIRRORS))
def test_mirror_matches_send_only_legal_actions(source):
    """Every tier against itself on every map and scenario (fog off)."""
    for index, tier in enumerate(TIERS):
        referee = play(source, (tier,), seed=index + 1, max_turns=MIRROR_TURNS[source.split("/")[0]])
        _assert_clean(referee, lively=tier in PURPOSEFUL_TIERS and not source.startswith("saves/"))


CROSS_TIER = [("simple", "master"), ("medium", "advanced"), ("balanced_random", "medium"), ("master", "random")]


@pytest.mark.parametrize(
    "map_name",
    _sample(
        ["1v1/beginner.csv", "1v1/crossroads.csv", "1v1/tower_rush.csv", "1v1v1/triangle_arena.csv"], {"1v1/beginner.csv"}
    ),
)
@pytest.mark.parametrize("tiers", CROSS_TIER, ids="-".join)
def test_cross_tier_matches_send_only_legal_actions(map_name, tiers):
    _assert_clean(play(map_name, tiers, seed=7, max_turns=16))


@pytest.mark.parametrize(
    "map_name",
    _sample(["1v1/beginner.csv", "1v1/crossroads.csv", "1v1v1/triangle_arena.csv", "2v2/beginner.csv"], {"2v2/beginner.csv"}),
)
@pytest.mark.parametrize("tier", TIERS)
def test_fog_of_war_matches_send_only_legal_actions(map_name, tier):
    """Under fog of war a bot may attack only enemies its side saw when the unit's action began."""
    _assert_clean(play(map_name, (tier,), seed=3, max_turns=12, fog=True))


@pytest.mark.parametrize(
    "map_name",
    _sample(["1v1/crossroads.csv", "1v1/tower_rush.csv", "2v2/beginner.csv"], {"1v1/crossroads.csv", "2v2/beginner.csv"}),
)
@pytest.mark.parametrize("tier", TIERS)
def test_fog_of_war_snapshots_cut_every_turn(map_name, tier):
    """Fog conflicts are rare in whole games; with each unit's snapshot cut to half, every turn has them."""
    referee = play(map_name, (tier,), seed=5, max_turns=16, blind=True)
    _assert_clean(referee, lively=tier in PURPOSEFUL_TIERS)


def test_the_referee_catches_every_kind_of_breach():
    """The harness itself: each check fires on the breach it is for, and only then."""
    game = new_game("1v1/beginner.csv", max_turns=10, seed=1)
    bot = build_scripted("medium", game, player=game.current_player)
    referee = Referee(game, {game.current_player: bot})
    unit = game.place_unit("W", 3, 3, game.current_player)
    referee.start_turn()

    def breached(fragment):
        found = [b for b in referee.breaches if fragment in b]
        referee.breaches.clear()
        return found

    game.move_unit(unit, -1, -1)
    assert breached("refused move_unit")

    bot.act_with_unit(unit)
    assert referee.breaches == []
    bot.act_with_unit(unit)  # a second decision for a unit without haste
    assert breached("2 time(s)")

    referee.acts.clear()
    bot.act_with_unit(unit, 1)  # a re-entry without haste
    assert breached("at depth 1")

    bot._capture_assigned = {(x, 0) for x in range(referee.movers_at_turn_start + 1)}
    game.end_unit_turn(unit)
    assert breached("capture claims")
    bot._capture_assigned = set()

    game.units.remove(unit)
    game.end_unit_turn(unit)
    assert breached("not in play")

    game.game_over = True
    with pytest.raises(AssertionError, match="barely played"):
        _assert_clean(Referee(game, {}), lively=True)
