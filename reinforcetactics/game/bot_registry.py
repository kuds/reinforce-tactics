"""
Canonical bot registry.

The single source of truth for mapping a bot-type identifier — canonical
short name (``"simple"``), PascalCase class name (``"SimpleBot"``), or
tournament ``BotType`` enum member — to the scripted bot class, and for
classifying any bot type into the standardized player type
(``'bot'`` / ``'llm'`` / ``'rl'``) used in replays and player configs.

Before this module existed the name→class dispatch was hand-written in
five places (``MixedBot._build_inner``, ``app/bot_factory``,
``tournament/bots``, ``rl/imitation``, ``StrategyGameEnv.reset``) with
two naming conventions, and the player-type classifier three times
(``app/bot_factory``, ``core/game_state``, ``tournament/runner``).
Adding a scripted bot now means adding it to :data:`SCRIPTED_BOTS`
(and :data:`STOCHASTIC_BOTS` when its action choice is rng-driven).

LLM bots (``OpenAIBot`` / ``ClaudeBot`` / ``GeminiBot``) and ``ModelBot``
are deliberately not constructed here — their kwargs (API keys, model
paths, conversation logging) are context-specific, so the GUI factory
and tournament factory keep those branches and use this module only for
classification.
"""

from __future__ import annotations

import random
from typing import Any

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

# Canonical short name → class for every scripted (non-LLM, non-RL) bot.
SCRIPTED_BOTS: dict[str, type] = {
    "simple": SimpleBot,
    "medium": MediumBot,
    "advanced": AdvancedBot,
    "master": MasterBot,
    "mixed": MixedBot,
    "random": RandomBot,
    "balanced_random": BalancedRandomBot,
    "noop": NoopBot,
}

# Bots whose *action selection* consumes the rng stream. The deterministic
# ladder (simple/medium/advanced/master) accepts an rng too, but uses it
# only for shuffle-before-sort tiebreaks — callers that keep separate rng
# streams for the two groups (see rl/imitation) key off this set.
STOCHASTIC_BOTS = frozenset({"mixed", "random", "balanced_random"})

# Accepted aliases → canonical short name. Class names resolve
# case-insensitively ("SimpleBot" → "simple"); "bot" is the historic
# alias for the default scripted opponent.
_ALIASES: dict[str, str] = {"bot": "simple"}
_ALIASES.update({cls.__name__.lower(): name for name, cls in SCRIPTED_BOTS.items()})

_LLM_TYPES = frozenset({"openaibot", "claudebot", "geminibot", "llm"})
# AlphaZeroBot plays a trained network too (through MCTS), so like ModelBot
# it is an 'rl' player; it used to fall through to 'bot' (review rulebots-23).
_RL_TYPES = frozenset({"modelbot", "alphazerobot", "model", "rl"})


def _key_of(bot_type: Any) -> str:
    """Normalize an identifier (str, class name, or Enum member) to a key."""
    raw = getattr(bot_type, "value", bot_type)
    return str(raw).strip().lower()


def canonical_name(bot_type: Any) -> str:
    """Resolve any accepted identifier to the canonical short name.

    Raises:
        KeyError: If the identifier names no scripted bot.
    """
    key = _key_of(bot_type)
    key = _ALIASES.get(key, key)
    if key not in SCRIPTED_BOTS:
        raise KeyError(f"Unknown scripted bot type: {bot_type!r} (known: {', '.join(sorted(SCRIPTED_BOTS))})")
    return key


def resolve_scripted(bot_type: Any) -> type:
    """The scripted bot class for any accepted identifier (KeyError if unknown)."""
    return SCRIPTED_BOTS[canonical_name(bot_type)]


def build_scripted(
    bot_type: Any,
    game_state: Any,
    player: int,
    rng: random.Random | None = None,
    **kwargs: Any,
) -> Any:
    """Construct a scripted bot bound to ``(game_state, player)``.

    ``rng`` is forwarded to every bot except ``NoopBot`` (which takes no
    rng — it never chooses anything). Extra kwargs are forwarded verbatim.

    Raises:
        KeyError: If ``bot_type`` names no scripted bot.
    """
    cls = resolve_scripted(bot_type)
    if cls is NoopBot:
        return cls(game_state, player=player, **kwargs)
    return cls(game_state, player=player, rng=rng, **kwargs)


def player_type(bot_type: Any) -> str:
    """Standardized player type for replays/configs: ``'llm'``, ``'rl'``, or ``'bot'``.

    Accepts class names ("OpenAIBot"), short names, tournament ``BotType``
    members, and already-resolved type strings ("llm"/"rl"). Anything
    unrecognized classifies as ``'bot'`` — matching the historic fallback
    in every one of the three classifiers this replaces.
    """
    key = _key_of(bot_type)
    if key in _LLM_TYPES:
        return "llm"
    if key in _RL_TYPES:
        return "rl"
    return "bot"
