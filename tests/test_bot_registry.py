"""The bot registry: name resolution, construction and player-type classification (review rulebots-14).

``reinforcetactics.game.bot_registry`` is what the gym env, the GUI factory,
the tournament and imitation learning resolve bot names through, and what
labels every replay's players. Its aliases and classifications had no direct
tests, so drift (a missing tier, a misclassified bot) went unnoticed.
"""

import random

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
from reinforcetactics.game.bot_registry import (
    SCRIPTED_BOTS,
    STOCHASTIC_BOTS,
    build_scripted,
    canonical_name,
    player_type,
    resolve_scripted,
)
from reinforcetactics.tournament.bots import BotType

EXPECTED = {
    "simple": SimpleBot,
    "medium": MediumBot,
    "advanced": AdvancedBot,
    "master": MasterBot,
    "mixed": MixedBot,
    "random": RandomBot,
    "balanced_random": BalancedRandomBot,
    "noop": NoopBot,
}


@pytest.fixture
def game():
    grid = np.full((6, 6), "p", dtype=object)
    grid[0, 0], grid[5, 5] = "h_1", "h_2"
    return GameState(grid, num_players=2)


def test_every_scripted_bot_is_registered_under_its_short_name():
    assert SCRIPTED_BOTS == EXPECTED
    assert STOCHASTIC_BOTS == {"mixed", "random", "balanced_random"}
    assert STOCHASTIC_BOTS <= SCRIPTED_BOTS.keys()


@pytest.mark.parametrize("name,cls", sorted(EXPECTED.items()))
def test_short_and_class_names_resolve(name, cls):
    for spelling in (name, name.upper(), f"  {name} ", cls.__name__, cls.__name__.lower(), cls.__name__.upper()):
        assert canonical_name(spelling) == name
        assert resolve_scripted(spelling) is cls


def test_bot_is_the_historic_alias_of_simple():
    assert canonical_name("bot") == canonical_name("BOT") == "simple"
    assert resolve_scripted("bot") is SimpleBot


@pytest.mark.parametrize(
    "member,name",
    [(BotType.SIMPLE, "simple"), (BotType.MEDIUM, "medium"), (BotType.ADVANCED, "advanced"), (BotType.MASTER, "master")],
)
def test_tournament_bot_types_resolve(member, name):
    assert canonical_name(member) == name
    assert resolve_scripted(member) is EXPECTED[name]


@pytest.mark.parametrize("unknown", ["nope", "", "llm", "model", "ModelBot", "OpenAIBot", BotType.LLM, BotType.MODEL])
def test_unknown_names_raise_a_key_error_listing_the_known_bots(unknown):
    """LLM and model bots are classified here but never built here."""
    with pytest.raises(KeyError) as excinfo:
        canonical_name(unknown)
    message = excinfo.value.args[0]
    assert message == f"Unknown scripted bot type: {unknown!r} (known: {', '.join(sorted(EXPECTED))})"
    with pytest.raises(KeyError):
        resolve_scripted(unknown)
    with pytest.raises(KeyError):
        build_scripted(unknown, None, player=2)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_build_scripted_binds_game_player_and_rng(game, name):
    rng = random.Random(3)
    bot = build_scripted(name, game, player=2, rng=rng)
    assert type(bot) is EXPECTED[name]
    assert bot.game_state is game and bot.bot_player == 2
    if name == "mixed":
        # MixedBot flips its coin with the rng, then hands it to the bot it picked.
        assert bot._rng is rng and bot._inner._rng is rng
    elif name != "noop":  # NoopBot takes no rng: it never chooses anything.
        assert bot._rng is rng


def test_build_scripted_forwards_extra_kwargs(game):
    assert build_scripted("random", game, player=1, max_actions=3).max_actions == 3
    mixed = build_scripted("mixed", game, player=2, rng=random.Random(0), easy="medium", hard="master", p_hard=1.0)
    assert isinstance(mixed._inner, MasterBot)


@pytest.mark.parametrize(
    "bot_type,expected",
    [
        # LLM bots, by class name and by the resolved type string.
        ("OpenAIBot", "llm"),
        ("ClaudeBot", "llm"),
        ("GeminiBot", "llm"),
        ("claudebot", "llm"),
        ("llm", "llm"),
        (BotType.LLM, "llm"),
        # Bots that play a trained network.
        ("ModelBot", "rl"),
        ("model", "rl"),
        ("rl", "rl"),
        (BotType.MODEL, "rl"),
        ("AlphaZeroBot", "rl"),
        ("alphazerobot", "rl"),
        # Scripted bots, by any spelling, and anything unrecognised.
        ("SimpleBot", "bot"),
        ("simple", "bot"),
        ("bot", "bot"),
        ("MasterBot", "bot"),
        ("mixed", "bot"),
        (BotType.ADVANCED, "bot"),
        ("something_else", "bot"),
        ("", "bot"),
    ],
)
def test_player_type(bot_type, expected):
    assert player_type(bot_type) == expected
