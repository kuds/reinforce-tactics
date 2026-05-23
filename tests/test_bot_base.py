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
"""

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
