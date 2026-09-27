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

import inspect
import random
from collections.abc import Mapping
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

# Short aliases → canonical short name. "bot" is the historic alias for the
# default scripted opponent. These, with the canonical names, are what
# configs and CLIs list (see :func:`accepted_names`).
_SHORT_ALIASES: dict[str, str] = {"bot": "simple"}

# Every accepted alias → canonical short name. Class names also resolve,
# case-insensitively ("SimpleBot" → "simple").
_ALIASES: dict[str, str] = dict(_SHORT_ALIASES)
_ALIASES.update({cls.__name__.lower(): name for name, cls in SCRIPTED_BOTS.items()})

# Constructor arguments every scripted bot takes from its caller rather than
# from ``kwargs`` (see :func:`build_scripted`); never valid as extra kwargs.
_SUPPLIED_ARGS = frozenset({"game_state", "player", "rng"})

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


def accepted_names() -> tuple[str, ...]:
    """The scripted-opponent names configs, CLIs and the gym env accept, sorted.

    Every canonical short name in :data:`SCRIPTED_BOTS` (``noop`` included)
    plus the short aliases (``"bot"``). The single list that CLI ``choices``
    and config validation derive from, so a new registry entry (``master``
    was the one that drifted) is accepted everywhere at once.
    :func:`canonical_name` also resolves class names (``"SimpleBot"``);
    those are left out here to keep listings short.
    """
    return tuple(sorted(set(SCRIPTED_BOTS) | set(_SHORT_ALIASES)))


def is_scripted_name(bot_type: Any) -> bool:
    """Whether :func:`canonical_name` resolves ``bot_type`` (no exception)."""
    key = _key_of(bot_type)
    return _ALIASES.get(key, key) in SCRIPTED_BOTS


def constructor_kwargs(bot_type: Any) -> frozenset[str] | None:
    """The extra keyword arguments the scripted bot's constructor takes.

    Everything but ``game_state`` / ``player`` / ``rng``, which callers
    supply themselves. ``None`` means the constructor takes ``**kwargs``
    (anything goes). Raises ``KeyError`` for an unknown bot.
    """
    cls = resolve_scripted(bot_type)
    params = inspect.signature(cls).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return frozenset(
        p.name
        for p in params
        if p.name not in _SUPPLIED_ARGS and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def validate_scripted_kwargs(bot_type: Any, kwargs: Mapping[str, Any] | None) -> None:
    """Check extra constructor kwargs for a scripted bot before it is built.

    Keys must be parameters of the bot's constructor (see
    :func:`constructor_kwargs`): today only ``random`` (``max_actions``)
    and ``mixed`` take any, so kwargs given for the deterministic ladder,
    which the gym env used to drop silently, are rejected. The values are
    checked too, by each bot's ``validate_config``: RandomBot's
    ``max_actions`` must be an integer >= 1, and ``MixedBot``'s are checked
    in depth (inner bot names, ``p_hard`` in ``[0, 1]``, and
    ``easy_kwargs`` / ``hard_kwargs`` against the inner bots' own
    constructors and values) so a bad bridge stage fails when it is
    configured, not at the random reset whose coin flip first picks the bad
    side.

    Raises:
        KeyError: ``bot_type`` names no scripted bot.
        TypeError: ``kwargs`` (or a nested ``*_kwargs``) is not a mapping.
        ValueError: A key the bot does not take, or a bad MixedBot value.
    """
    name = canonical_name(bot_type)
    if kwargs is None:
        return
    if not isinstance(kwargs, Mapping):
        raise TypeError(f"kwargs for scripted bot {name!r} must be a mapping, got {type(kwargs).__name__}")
    if not kwargs:
        return
    allowed = constructor_kwargs(name)
    if allowed is not None:
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise ValueError(
                f"unknown kwargs {unknown} for scripted bot {name!r}; it takes: {', '.join(sorted(allowed)) or '(none)'}"
            )
    if name == "mixed":
        MixedBot.validate_config(**kwargs)
    elif name == "random":
        RandomBot.validate_config(**kwargs)


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
