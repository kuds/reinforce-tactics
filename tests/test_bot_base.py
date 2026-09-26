"""
Tests for the shared bot foundations in :mod:`reinforcetactics.game.bot_base`.

Covers:
  * The ``ABILITY_PROVIDERS`` table is the single source of truth: each
    ``has_X_units`` predicate returns ``is_unit_enabled(provider[X])`` and
    nothing else.
  * Every scripted bot subclasses ``BaseBot`` so the tournament/runner and
    gym env can rely on the common interface.
  * ``BotUnitMixin.has_units_with_ability`` returns False for unknown
    abilities (forward-compatible default).
  * ``BaseBot`` is abstract -- instantiation requires ``take_turn``.
  * The mixin's engine-facing helpers (reach, captures, the per-unit
    continuation, ``try_action``) lead only to accepted actions, and the
    per-turn planning cache changes no decision.
"""

import random
from contextlib import contextmanager

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import (
    AdvancedBot,
    BalancedRandomBot,
    MasterBot,
    MediumBot,
    MixedBot,
    NoopBot,
    RandomBot,
    SimpleBot,
)
from reinforcetactics.game.bot_base import (
    ABILITY_PROVIDERS,
    MELEE_UNITS,
    RANGED_UNITS,
    SUPPORT_UNITS,
    BaseBot,
    BotUnitMixin,
)
from reinforcetactics.game.bot_registry import build_scripted
from reinforcetactics.game.model_bot import ModelBot
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def game_state():
    """A fresh 2-player game state on a deterministic random map."""
    np.random.seed(123)
    map_data = FileIO.generate_random_map(10, 10, num_players=2)
    np.random.seed()
    return GameState(map_data)


SCRIPTED_BOT_CLASSES = (
    NoopBot,
    RandomBot,
    BalancedRandomBot,
    SimpleBot,
    MediumBot,
    MixedBot,
    AdvancedBot,
    MasterBot,
)


class TestBaseBotHierarchy:
    """Every bot in the project conforms to the ``BaseBot`` interface."""

    @pytest.mark.parametrize("bot_cls", SCRIPTED_BOT_CLASSES)
    def test_scripted_bots_are_base_bots(self, bot_cls):
        assert issubclass(bot_cls, BaseBot)
        assert issubclass(bot_cls, BotUnitMixin)

    def test_model_bot_is_base_bot(self):
        # ModelBot intentionally does NOT mix in BotUnitMixin -- it
        # delegates to a trained policy and doesn't need the helpers.
        assert issubclass(ModelBot, BaseBot)

    def test_base_bot_is_abstract(self):
        """Instantiating BaseBot directly should raise -- take_turn is
        abstract."""
        with pytest.raises(TypeError):
            BaseBot(game_state=None)  # type: ignore[abstract]

    def test_base_bot_sets_attributes(self, game_state):
        """A concrete subclass gets ``game_state`` and ``bot_player`` set
        via BaseBot.__init__ even if it doesn't override __init__."""

        class _ConcreteBot(BaseBot):
            def take_turn(self) -> None:  # pragma: no cover - not invoked
                pass

        bot = _ConcreteBot(game_state, player=2)
        assert bot.game_state is game_state
        assert bot.bot_player == 2


class TestAbilityTable:
    """``ABILITY_PROVIDERS`` is the only place we encode which unit type
    provides which ability."""

    def test_table_contents(self):
        # If unit roster changes, this is the row to update -- and tests
        # like this one should be updated in lockstep.
        assert ABILITY_PROVIDERS == {
            "charge": "K",
            "flank": "R",
            "buff": "S",
            "heal": "C",
            "paralyze": "M",
        }

    def test_providers_are_valid_unit_letters(self):
        # Every provider must be a recognised unit type letter -- catches
        # typos when adding a new ability.
        valid_letters = set(MELEE_UNITS) | set(RANGED_UNITS) | set(SUPPORT_UNITS)
        for ability, provider in ABILITY_PROVIDERS.items():
            assert provider in valid_letters, f"{ability!r} maps to unknown unit letter {provider!r}"

    @pytest.mark.parametrize(
        "ability,predicate_name",
        [
            ("charge", "has_charge_units"),
            ("flank", "has_flank_units"),
            ("buff", "has_buff_units"),
            ("heal", "has_heal_units"),
            ("paralyze", "has_paralyze_units"),
        ],
    )
    def test_predicate_matches_table(self, game_state, ability, predicate_name):
        bot = NoopBot(game_state)
        expected = bot.is_unit_enabled(ABILITY_PROVIDERS[ability])
        # Both the named predicate and has_units_with_ability must agree
        # with a direct is_unit_enabled() lookup on the table.
        assert getattr(bot, predicate_name)() is expected
        assert bot.has_units_with_ability(ability) is expected

    def test_unknown_ability_returns_false(self, game_state):
        bot = NoopBot(game_state)
        assert bot.has_units_with_ability("teleport") is False
        assert bot.has_units_with_ability("") is False


def _open_game(towers=()):
    """A 7x7 grass map with both HQs in the corners and neutral towers at ``towers``; player 1 to move."""
    grid = np.full((7, 7), "p", dtype=object)
    grid[0, 0], grid[6, 6] = "h_1", "h_2"
    for x, y in towers:
        grid[y, x] = "t"
    return GameState(grid, num_players=2)


class TestActingThroughTheEngine:
    """The mixin's reach, capture and continuation helpers lead only to actions the engine accepts (review rulebots-1/6)."""

    def test_get_reachable_is_the_engines_move_destinations(self):
        game = _open_game()
        walker = game.place_unit("W", 3, 3, 1)
        game.place_unit("W", 3, 2, 1)  # a friend: passable, but no move may end there
        bot = SimpleBot(game, player=1)

        reachable = bot.get_reachable(walker)

        assert reachable == game.get_move_destinations(walker)
        assert (3, 2) not in reachable and (3, 1) in reachable
        assert all(game.is_legal("move", {"unit": walker, "to_x": x, "to_y": y}) for x, y in reachable)

    def test_get_reachable_is_empty_once_the_unit_cannot_move(self):
        game = _open_game()
        moved, dead = game.place_unit("W", 3, 3, 1), game.place_unit("W", 1, 1, 1)
        bot = SimpleBot(game, player=1)
        assert game.move_unit(moved, 3, 4)
        game.units.remove(dead)  # as a counter-attack removes a unit

        assert bot.get_reachable(moved) == [] and bot.get_reachable(dead) == []
        assert bot.find_best_move_position(moved, 0, 0) is None

    def test_pick_capture_target_skips_a_structure_a_friend_stands_on(self):
        game = _open_game(towers=[(3, 2), (3, 5)])  # (3, 2) is nearer, but a friend holds it
        unit = game.place_unit("W", 3, 3, 1)
        game.place_unit("W", 3, 2, 1)
        bot = MediumBot(game, player=1)

        target = bot.pick_capture_target(unit)

        assert (target.x, target.y) == (3, 5)

    def test_continue_active_seizes_claims_the_seized_tile(self):
        game = _open_game(towers=[(3, 2)])
        tower = game.grid.get_tile(3, 2)
        tower.health = tower.max_health - 1  # capture in progress
        unit = game.place_unit("W", 3, 2, 1)
        bot = MediumBot(game, player=1)

        bot.continue_active_seizes([unit])

        assert bot._capture_assignments() == {(3, 2)}
        assert game.action_history[-1]["type"] == "seize"

    def test_a_unit_that_cannot_move_claims_nothing(self):
        """Claiming before moving let a stuck unit keep structures from its siblings."""
        game = _open_game(towers=[(3, 0)])
        unit = game.place_unit("W", 3, 3, 1)
        unit.can_move = False
        bot = MediumBot(game, player=1)

        bot.act_with_unit(unit)

        assert bot._capture_assignments() == set()
        assert (unit.x, unit.y) == (3, 3) and not unit.can_attack

    @pytest.mark.parametrize(
        "bot_cls,act_name",
        [(SimpleBot, "act_with_unit"), (MediumBot, "act_with_unit"), (AdvancedBot, "act_with_unit_enhanced")],
    )
    def test_a_move_without_an_action_ends_the_units_turn(self, bot_cls, act_name):
        """No re-entry on the ``can_attack`` a move leaves: only haste gives a unit another action.

        AdvancedBot's multi-turn capture march used to re-run the whole
        decision for the marched unit, every move branch of which the engine
        then refused.
        """
        game = _open_game(towers=[(0, 6), (1, 6), (5, 6)])  # neutral, out of reach: a march
        unit = game.place_unit("W", 3, 0, 1)
        bot = bot_cls(game, player=1)
        bot.phase = AdvancedBot.PHASE_EXPAND
        depths = []
        act = getattr(bot, act_name)

        def recording_act(u, _depth=0):
            depths.append(_depth)
            act(u, _depth)

        setattr(bot, act_name, recording_act)

        recording_act(unit)

        assert depths == [0]
        assert (unit.x, unit.y) != (3, 0)
        assert not (unit.can_move or unit.can_attack)

    def test_haste_gives_exactly_one_more_action(self):
        game = _open_game()
        warrior = game.place_unit("W", 3, 3, 1)
        game.place_unit("W", 3, 4, 2).health = 30  # survives two hits
        warrior.is_hasted = True
        bot = SimpleBot(game, player=1)

        bot.act_with_unit(warrior)

        assert [a["type"] for a in game.action_history].count("attack") == 2
        assert not (warrior.can_move or warrior.can_attack or warrior.is_hasted)

    def test_try_action_asks_the_engine_first(self, monkeypatch):
        game = _open_game()
        mage = game.place_unit("M", 3, 3, 1)
        far = game.place_unit("W", 6, 3, 2)
        bot = SimpleBot(game, player=1)
        sent = []
        monkeypatch.setattr(game, "paralyze", lambda *args: sent.append(args) or False)

        assert bot.try_action("paralyze", mage, far) is False
        assert bot.try_mage_paralyze(mage) is False
        assert sent == []  # nothing out of range reached the engine
        assert "mage_paralyze" not in bot.get_capabilities_fired()


class TestTurnContext:
    """The per-turn planning cache (review rulebots-9) must never change a decision."""

    def test_reach_is_searched_again_once_an_action_is_accepted(self):
        game = _open_game()
        walker, mover = game.place_unit("W", 3, 3, 1), game.place_unit("W", 5, 3, 1)
        bot = SimpleBot(game, player=1)

        with bot.planning_turn():
            assert (4, 3) in bot.get_reachable(walker)
            assert bot.try_move(mover, 4, 3)
            assert (4, 3) not in bot.get_reachable(walker)
            assert bot.get_reachable(walker) == game.get_move_destinations(walker)
            assert bot.capturable_structures() is bot.capturable_structures()  # one list for the turn
        assert bot._turn is None

    @pytest.mark.parametrize("tier", ["simple", "medium", "advanced", "master"])
    @pytest.mark.parametrize("fog", [False, True])
    def test_seeded_games_are_the_same_with_and_without_it(self, tier, fog, monkeypatch):
        def play():
            game = GameState(FileIO.load_map("maps/1v1/crossroads.csv"), num_players=2, max_turns=14, fog_of_war=fog, seed=5)
            bots = {p: build_scripted(tier, game, player=p, rng=random.Random(50 + p)) for p in (1, 2)}
            while not game.game_over:
                bots[game.current_player].take_turn()
            return [{k: v for k, v in a.items() if k != "timestamp"} for a in game.action_history]

        cached = play()

        @contextmanager
        def no_context(self):
            yield

        monkeypatch.setattr(BotUnitMixin, "planning_turn", no_context)
        assert play() == cached
        assert any(a["type"] == "attack" for a in cached)
