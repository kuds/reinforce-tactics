"""
LLM-powered bots for playing Reinforce Tactics using OpenAI, Claude, and Gemini.
"""

import email.utils
import json
import logging
import math
import os
import random
import re
import time
from abc import abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from reinforcetactics import __version__
from reinforcetactics.constants import UNIT_DATA
from reinforcetactics.game.bot_base import BaseBot
from reinforcetactics.game.llm_prompts import (
    DEFAULT_PROMPT,
    PROMPT_TWO_PHASE_EXECUTE,
    PROMPT_TWO_PHASE_PLAN,
    get_prompt,
)

# Configure logging
logger = logging.getLogger(__name__)


# Supported models for each provider
OPENAI_MODELS = [
    # GPT-5.2 (flagship reasoning model)
    "gpt-5.2",
    # GPT-5 family
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano-2025-08-07",
]

ANTHROPIC_MODELS = [
    # Claude Fable 5.1 / Fable 5
    "claude-fable-5-1",
    "claude-fable-5",
    # Claude Opus 5.5 (launching) / Opus 5
    "claude-opus-5-5",
    "claude-opus-5",
    # Claude Sonnet 5
    "claude-sonnet-5",
    # Claude Opus 4.8 / 4.7 / 4.6
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    # Claude Sonnet 4.6
    "claude-sonnet-4-6",
    # Claude Haiku 4.5 (ClaudeBot's default: the cheapest current model)
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
    # Older models that are still served
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
]

# Claude IDs that configs may still name but that should not be picked.
# ClaudeBot logs the note at construction; a retired model then fails its
# first request with a 404, which surfaces as LLMBotError.
ANTHROPIC_UNAVAILABLE_MODELS: dict[str, str] = {
    "claude-opus-4-1-20250805": "retired on 2026-08-05, so every request will fail",
    "claude-sonnet-4-20250514": "deprecated and scheduled for retirement",
    "claude-opus-4-20250514": "deprecated and scheduled for retirement",
}

GEMINI_MODELS = [
    # Gemini 3.0 (latest generation, preview)
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    # Gemini 2.5 (current production)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]


# Legacy alias - the actual prompt used by bots is DEFAULT_PROMPT from llm_prompts.py
# which includes all 8 unit types (W, M, C, A, K, R, S, B) with correct stats.
# See reinforcetactics/game/llm_prompts.py for the full prompt definitions.
SYSTEM_PROMPT = DEFAULT_PROMPT


# Seconds before one LLM request is abandoned (and retried as a timeout).
# The SDK defaults (600 s for OpenAI/Anthropic, none for google-genai) let a
# single stuck request freeze a GUI game or a tournament worker for ten
# minutes or more. Five minutes is far above a normal turn; with the default
# request_timeout="auto" it is raised for large max_tokens (see
# default_request_timeout).
DEFAULT_REQUEST_TIMEOUT_S = 300.0

# Seconds allowed per requested output token when sizing the "auto" timeout.
# This is the Anthropic SDK's own estimate for a non-streamed request (128K
# tokens an hour, ~36 tokens/s), which is slow for current models: a reply
# that uses all of max_tokens should finish well inside the timeout rather
# than be cut off, retried and billed again.
_TIMEOUT_S_PER_OUTPUT_TOKEN = 3600 / 128_000


def default_request_timeout(max_tokens: int | None) -> float:
    """The "auto" request timeout: DEFAULT_REQUEST_TIMEOUT_S, or longer when
    ``max_tokens`` is large enough that a full-length reply could need more
    (16K tokens -> 450 s; 32K -> 900 s)."""
    return max(DEFAULT_REQUEST_TIMEOUT_S, (max_tokens or 0) * _TIMEOUT_S_PER_OUTPUT_TOKEN)


# Turns in a row the LLM API may fail to answer at all (after retries) before
# the bot raises LLMBotError. One failed turn is treated as a blip and passed;
# a streak means an outage or misconfiguration that would otherwise look like
# a passive opponent and be scored as a loss for the model. A reply that
# arrives but is useless (prose, truncated JSON, a refusal) is the model's
# own play, not an outage, and doesn't count (see take_turn).
DEFAULT_MAX_CONSECUTIVE_FAILED_TURNS = 3

# Seconds one LLM call may spend (requests plus backoff) retrying failures
# that are known to pass: rate limits, 5xx/overloaded, timeouts and
# connection errors. These are retried past max_retries while the call is
# inside this budget, so a minute-long overload is ridden out instead of
# passing turns (three of which raise LLMBotError). The SDKs' own retries are
# off (max_retries=0), so this is the whole retry budget for such failures.
DEFAULT_RETRY_BUDGET_S = 60.0

# Backoff between attempts of one request: exponential from the base,
# capped, then jittered (see LLMBot._retry_delay).
_RETRY_BASE_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0
# Upper bound on a server-requested Retry-After, so a bad header can't stall
# a game indefinitely.
_RETRY_AFTER_CAP_S = 60.0

# HTTP statuses worth retrying: request timeout, conflict/lock contention,
# rate limiting. Every 5xx (including Anthropic's 529 "overloaded") is
# retried too. Every other 4xx is a request the server will keep rejecting.
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 429})
_HTTP_STATUS_REASONS = {
    400: "bad request",
    401: "authentication failed (check the API key)",
    403: "permission denied",
    404: "model or endpoint not found",
    408: "request timeout",
    409: "conflict",
    413: "request too large",
    422: "unprocessable request",
    429: "rate limited",
}

# Claude models from this (major, minor) version on reject temperature,
# top_p and top_k with a 400: Opus 4.7, Opus 4.8 and every 5.x model
# (Opus 5 / 5.5, Sonnet 5, Fable 5 / 5.1). Opus 4.6, Sonnet 4.6, Haiku 4.5
# and older still accept them.
_CLAUDE_NO_SAMPLING_PARAMS_FROM = (4, 7)
# Matches "claude-opus-4-7", "claude-fable-5-1", "claude-haiku-4-5-20251001"
# and the older "claude-3-5-sonnet-20241022" form. The minor version is one
# or two digits so an 8-digit date suffix ("claude-opus-4-20250514") is not
# read as one.
_CLAUDE_VERSION_RE = re.compile(r"^claude-(?:[a-z]+-)?(\d+)(?:-(\d{1,2})(?!\d))?")


def claude_model_accepts_sampling_params(model: str) -> bool:
    """Whether ``model`` accepts temperature/top_p/top_k without a 400.

    An ID that can't be parsed is treated as not accepting them: dropping a
    temperature only changes sampling, while sending one to a model that
    rejects it fails every request.
    """
    match = _CLAUDE_VERSION_RE.match(model)
    if match is None:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version < _CLAUDE_NO_SAMPLING_PARAMS_FROM


# OpenAI reasoning models that reject a non-default temperature with a 400
# ("Unsupported value: 'temperature' does not support 0.5 with this model"):
# the o-series, the original GPT-5 family (gpt-5 / -mini / -nano, which can't
# turn reasoning off) and the -pro models (reasoning only). GPT-5.1 and later
# accept temperature at their default reasoning effort of "none", which
# OpenAIBot never changes, so e.g. gpt-5.2 keeps it.
_OPENAI_NO_TEMPERATURE_RE = re.compile(
    r"^(?:o\d"  # o1, o3, o4-mini, o3-pro, ...
    r"|gpt-5(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?$"  # gpt-5, gpt-5-mini-2025-08-07, ...
    r"|gpt-5(?:\.\d+)?-pro)"  # gpt-5-pro, gpt-5.2-pro, ...
)


def openai_model_accepts_temperature(model: str) -> bool:
    """Whether OpenAI ``model`` accepts a non-default temperature without a 400."""
    return _OPENAI_NO_TEMPERATURE_RE.match(model) is None


class LLMBotError(RuntimeError):
    """An LLM bot can't reach its model, and passing more turns would hide it.

    Raised from ``take_turn()`` in two cases:

    * at once, for a failure no retry can fix: the provider SDK is missing,
      the API key is rejected, the key lacks permission, the model doesn't
      exist, the request is rejected as malformed, or the SDK call itself
      fails locally (e.g. a TypeError from an SDK version that doesn't take
      an argument) (``retryable=False``);
    * after ``max_consecutive_failed_turns`` turns in a row where the API
      gave no reply at all: every retry of a rate limit, outage, timeout or
      unreadable response was used up (``retryable=True``).

    Both are infrastructure failures, not the model's play. A reply that
    arrives but holds no usable actions (prose, JSON cut off at max_tokens,
    a refusal or a safety block) never raises: the turn passes and the game
    is scored on the board, like any other bad move.

    The turn in progress is *not* ended, so the caller decides what happens
    to the game. The tournament runner ends the game with an error result,
    which TournamentResults leaves out of wins/losses/draws and Elo (it is
    counted under ``errors``) instead of scoring it as a loss or a draw. The
    GUI hands the seat to SimpleBot and tells the player.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class _UnreadableResponseError(Exception):
    """The provider answered, but the reply isn't the shape its SDK promises.

    Raised by ``_reading_response`` for an error while reading a reply (e.g.
    an SDK that hands back a proxy's HTML page as a ``str``), so it is
    classified apart from local errors raised while building the request.
    """


@contextmanager
def _reading_response(response: Any) -> Iterator[None]:
    """Re-raise errors from reading ``response`` as _UnreadableResponseError.

    A TypeError/AttributeError/ValueError from the SDK call itself means the
    request couldn't be built, which no retry fixes (see _classify_llm_error).
    The same exceptions raised while reading what came back mean a garbled
    reply, which may be a one-off; without this wrapper one such reply ended
    the game.
    """
    try:
        yield
    except (AttributeError, TypeError, ValueError, KeyError, IndexError) as exc:
        raise _UnreadableResponseError(f"couldn't read the {type(response).__name__} reply: {exc}") from exc


@dataclass(frozen=True)
class _LLMErrorInfo:
    """How ``_call_llm_with_retry`` should treat one failed request.

    ``transient`` marks failures known to pass with time (rate limits,
    5xx/overloaded, timeouts, connection errors). They are retried within
    the retry budget even past ``max_retries``; other retryable failures get
    ``max_retries`` attempts only.
    """

    retryable: bool
    reason: str
    retry_after: float | None = None
    transient: bool = False


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status carried by a provider SDK error, if any.

    OpenAI and Anthropic errors expose ``status_code``; google-genai's
    ``APIError`` exposes ``code``. Falls back to ``exc.response.status_code``.
    """
    candidates = (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _header(headers: Any, name: str) -> str | None:
    """Read one header from an httpx/requests header map or a plain dict."""
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get(name)
    if value is None:
        value = headers.get(name.title())
    return str(value) if value is not None else None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The server's requested wait before retrying, in seconds, if it sent one.

    Reads ``retry-after-ms`` / ``retry-after`` (seconds or an HTTP date) from
    the error's HTTP response (OpenAI, Anthropic), or the ``RetryInfo``
    detail in a google-genai error body (``"retryDelay": "30s"``).
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    candidates: list[float] = []
    retry_after_ms = _header(headers, "retry-after-ms")
    if retry_after_ms is not None:
        try:
            candidates.append(float(retry_after_ms) / 1000.0)
        except ValueError:
            pass
    retry_after = _header(headers, "retry-after")
    if retry_after is not None and not candidates:
        try:
            candidates.append(float(retry_after))
        except ValueError:
            try:
                when = email.utils.parsedate_to_datetime(retry_after)
                candidates.append((when - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError):
                pass
    body = getattr(exc, "details", None)
    if not candidates and isinstance(body, dict):
        error = body.get("error")
        details = error.get("details") if isinstance(error, dict) else None
        for item in details if isinstance(details, list) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
                delay = str(item.get("retryDelay", ""))
                try:
                    candidates.append(float(delay.removesuffix("s")))
                except ValueError:
                    pass
    for seconds in candidates:
        if math.isfinite(seconds):
            return max(0.0, seconds)
    return None


# Transport-error class names with no "Timeout"/"Connect" in them that still
# mean the connection failed: httpx's NetworkError family (ReadError,
# WriteError, CloseError) and RemoteProtocolError (the server hung up
# mid-reply). google-genai raises these raw, unlike the OpenAI and Anthropic
# SDKs, which wrap them in APIConnectionError. Other httpx TransportErrors
# (UnsupportedProtocol, LocalProtocolError) are local and stay "unexpected".
_CONNECTION_ERROR_CLASS_NAMES = frozenset({"NetworkError", "RemoteProtocolError"})


def _quota_exhausted(exc: BaseException) -> bool:
    """Whether a 429 means the account is out of quota, not rate-limited.

    Waiting doesn't clear these, so retrying them only stalls the game: an
    OpenAI ``insufficient_quota`` 429 took 21 requests and ~131 s of backoff
    over three turns before the failed-turn limit named the wrong cause.
    OpenAI marks them with code/type ``insufficient_quota``; Gemini's daily
    limits carry a QuotaFailure whose ``quotaId`` says ``PerDay`` (per-minute
    quotas stay retryable).
    """
    if "insufficient_quota" in (getattr(exc, "code", None), getattr(exc, "type", None)):
        return True
    if "insufficient_quota" in str(exc):
        return True

    def quota_ids(node: Any) -> list[str]:
        if isinstance(node, dict):
            found = [str(node["quotaId"])] if "quotaId" in node else []
            return found + [q for value in node.values() for q in quota_ids(value)]
        if isinstance(node, list):
            return [q for value in node for q in quota_ids(value)]
        return []

    return any("PerDay" in quota_id for quota_id in quota_ids(getattr(exc, "details", None)))


def _classify_llm_error(exc: BaseException) -> _LLMErrorInfo:
    """Sort a failed LLM request into retryable or not.

    Not retryable: a missing SDK, 4xx responses other than 408/409/429 (bad
    key, missing permission, unknown model, malformed request), and local
    TypeError/AttributeError/ValueError from the SDK call (see below). The
    same request would fail the same way, so retrying only burns time and
    passes turns. Retryable and transient (retried within the retry budget):
    408/409/429, 5xx, timeouts and connection errors. Retryable for
    ``max_retries`` attempts: an unreadable reply, and anything unrecognised
    (the pre-classification behaviour; a streak of those still ends in
    LLMBotError via the failed-turn limit).
    """
    if isinstance(exc, ImportError):
        return _LLMErrorInfo(False, "LLM SDK not installed")
    if isinstance(exc, _UnreadableResponseError):
        return _LLMErrorInfo(True, "unreadable reply")
    status = _http_status(exc)
    if status is not None:
        reason = f"HTTP {status}"
        if status in _HTTP_STATUS_REASONS:
            reason += f" {_HTTP_STATUS_REASONS[status]}"
        if status == 429 and _quota_exhausted(exc):
            return _LLMErrorInfo(False, f"{reason} (quota exhausted; retrying won't help)")
        if status in _RETRYABLE_HTTP_STATUSES or status >= 500:
            return _LLMErrorInfo(True, reason, _retry_after_seconds(exc), transient=True)
        if 400 <= status < 500:
            return _LLMErrorInfo(False, reason)
    # SDK transport errors carry no status. Match on class names so this
    # works without importing each SDK (APIConnectionError, APITimeoutError,
    # httpx.ConnectError / ReadTimeout / ReadError, ...).
    class_names = [cls.__name__ for cls in type(exc).__mro__]
    if isinstance(exc, TimeoutError) or any("Timeout" in name for name in class_names):
        return _LLMErrorInfo(True, "timeout", transient=True)
    if (
        isinstance(exc, ConnectionError)
        or any("Connect" in name for name in class_names)
        or not _CONNECTION_ERROR_CLASS_NAMES.isdisjoint(class_names)
    ):
        return _LLMErrorInfo(True, "connection error", transient=True)
    # With no HTTP status, these come from building or sending the request
    # in-process: a keyword the installed SDK doesn't take (anthropic 1.x
    # raises TypeError for temperature=), a client attribute an old SDK
    # lacks, a config value the SDK's validation rejects. Nothing reached the
    # server, and every retry fails identically. The providers read replies
    # inside _reading_response, so the same exceptions from a garbled reply
    # arrive as _UnreadableResponseError instead. A reply the SDK itself
    # couldn't decode (JSONDecodeError, google-genai's UnknownApiResponseError,
    # both ValueErrors) can be a garbled one-off too, so it stays retryable.
    # So does a pydantic ValidationError titled after a response model:
    # google-genai validates every reply into a GenerateContentResponse inside
    # the SDK call, while one titled after a request model (e.g.
    # GenerateContentConfig) is a local config error.
    decode_error = (
        isinstance(exc, json.JSONDecodeError | UnicodeError)
        or any("Response" in name for name in class_names)
        or "Response" in str(getattr(exc, "title", ""))
    )
    if decode_error:
        return _LLMErrorInfo(True, f"unreadable reply ({type(exc).__name__})")
    if isinstance(exc, TypeError | AttributeError | ValueError):
        return _LLMErrorInfo(False, f"local {type(exc).__name__} (SDK/config mismatch?)")
    return _LLMErrorInfo(True, f"unexpected {type(exc).__name__}")


# LLM action type -> (key in get_legal_actions(), key naming the acting unit
# in each of that key's entries).
_UNIT_ACTION_LEGAL_KEYS: dict[str, tuple[str, str]] = {
    "MOVE": ("move", "unit"),
    "ATTACK": ("attack", "attacker"),
    "PARALYZE": ("paralyze", "paralyzer"),
    "HEAL": ("heal", "healer"),
    "CURE": ("cure", "curer"),
    "SEIZE": ("seize", "unit"),
}

# Why get_legal_actions may not list an action, per type: logged when the
# bot skips one so the conversation log explains the rejection.
_ILLEGAL_ACTION_HINTS: dict[str, str] = {
    "MOVE": "the unit has already moved or acted, is paralyzed, or can't reach that tile",
    "ATTACK": "the unit has already acted this turn, or the target isn't a visible enemy in its range",
    "PARALYZE": "only a Mage that hasn't acted, with paralyze off cooldown, can paralyze an unparalyzed enemy in range",
    "HEAL": "only a Cleric that hasn't acted can heal a damaged ally in range",
    "CURE": "only a Cleric that hasn't acted can cure a paralyzed ally in range",
    "SEIZE": "the unit has already acted this turn, or isn't standing on a structure it doesn't own",
    "CREATE_UNIT": "not an empty owned building, the unit type is disabled or unaffordable, or the unit cap is reached",
}


class LLMBot(BaseBot):  # pylint: disable=too-few-public-methods
    """
    Abstract base class for LLM-powered bots.

    This class provides the foundation for bots that use Large Language Models
    to play Reinforce Tactics. It handles game state serialization, API communication,
    and action execution.

    Subclasses must implement provider-specific methods for API key handling
    and model invocation.
    """

    def __init__(
        self,
        game_state,
        player: int = 2,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        log_conversations: bool = False,
        conversation_log_dir: str | None = None,
        game_session_id: str | None = None,
        pretty_print_logs: bool = True,
        stateful: bool = False,
        should_reason: bool = False,
        max_tokens: int | None = 8_000,
        temperature: float | None = None,
        system_prompt: str | None = None,
        two_phase_planning: bool = False,
        request_timeout: float | Literal["auto"] | None = "auto",
        max_consecutive_failed_turns: int | None = DEFAULT_MAX_CONSECUTIVE_FAILED_TURNS,
        retry_budget_s: float | None = DEFAULT_RETRY_BUDGET_S,
    ):
        """
        Initialize the LLM bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot (default 2)
            api_key: API key for the LLM provider (optional, uses env var if not provided)
            model: Model name to use (optional, uses default if not provided)
            max_retries: Attempts per API call for a retryable failure
                (default 3). Rate limits, 5xx/overloaded, timeouts and
                connection errors are retried past this while the call is
                inside ``retry_budget_s``. Failures no retry can fix aren't
                retried at all; see LLMBotError.
            log_conversations: Enable conversation logging to JSON files (default False)
            conversation_log_dir: Directory for conversation logs (default: logs/llm_conversations/)
            game_session_id: Unique game session identifier (default: auto-generated)
            pretty_print_logs: Format JSON logs with indentation for readability (default True)
            stateful: Maintain conversation history across turns (default False)
            should_reason: Include reasoning field in response format (default False).
                When True, includes "reasoning" field prompting for strategy explanation.
                When False, the reasoning field is omitted entirely from the prompt.
            max_tokens: Maximum number of tokens for LLM response (default 8000).
                Set to 0 or None to not pass max_tokens to the LLM provider.
                If not specified, uses DEFAULT_MAX_TOKENS (8000).
            temperature: Temperature for LLM response (default None).
                Set to None to use the LLM provider's default temperature.
                Set to a value (e.g., 0, 0.5, 1.0) to override the default.
            system_prompt: Custom system prompt to use (default None uses DEFAULT_PROMPT).
                Can be a prompt string or a prompt name from llm_prompts (e.g., "strategic").
                See reinforcetactics.game.llm_prompts for available prompts.
            two_phase_planning: Enable two-phase planning mode (default False).
                When True, the bot first generates a strategic plan, then executes it.
                This encourages deeper strategic thinking about action sequences.
                Note: This doubles the number of API calls per turn.
            request_timeout: Seconds before a single API request is abandoned.
                "auto" (default) is default_request_timeout(max_tokens):
                DEFAULT_REQUEST_TIMEOUT_S, raised for large max_tokens. None
                uses the SDK default.
            max_consecutive_failed_turns: Turns in a row where the API gave
                no reply (after retries) before take_turn() raises
                LLMBotError instead of passing another turn (default 3).
                None never raises.
            retry_budget_s: Seconds one API call may spend, counting requests
                and backoff, retrying rate limits, 5xx/overloaded, timeouts
                and connection errors beyond ``max_retries`` attempts
                (default DEFAULT_RETRY_BUDGET_S, 60 s). None retries those
                only ``max_retries`` times.

        Raises:
            ValueError: If no API key is available.
            ImportError: If the provider's SDK isn't installed.
        """
        self.game_state = game_state
        self.bot_player = player
        self.api_key = api_key or self._get_api_key_from_env()
        self.model = model or self._get_default_model()
        self.max_retries = max_retries
        self.log_conversations = log_conversations
        self.conversation_log_dir = conversation_log_dir or "logs/llm_conversations/"
        self.pretty_print_logs = pretty_print_logs
        self.stateful = stateful
        self.should_reason = should_reason
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.two_phase_planning = two_phase_planning
        if isinstance(request_timeout, str) and request_timeout != "auto":
            raise ValueError(f'request_timeout must be seconds, None or "auto", not {request_timeout!r}')
        self.request_timeout: float | None = (
            default_request_timeout(max_tokens) if isinstance(request_timeout, str) else request_timeout
        )
        self.max_consecutive_failed_turns = max_consecutive_failed_turns
        self.consecutive_failed_turns = 0
        # Why the most recent unanswered call failed, for the streak LLMBotError.
        self._last_failure_reason = ""
        self.retry_budget_s = retry_budget_s

        # Resolve system prompt - can be a name or a full prompt string
        if system_prompt is None:
            self.system_prompt = DEFAULT_PROMPT
        elif len(system_prompt) < 100 and "\n" not in system_prompt:
            # Looks like a prompt name, try to resolve it
            try:
                self.system_prompt = get_prompt(system_prompt)
            except ValueError:
                # Not a known name, treat as custom prompt
                self.system_prompt = system_prompt
        else:
            # Full prompt string
            self.system_prompt = system_prompt

        # Initialize conversation history for stateful mode
        self.conversation_history: list[dict[str, str]] = []

        # Initialize token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Per-call token tracking (set by subclasses that support it)
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        # Per-call stop reason tracking (set by subclasses)
        self._last_stop_reason = ""

        # Generate or use provided game session ID
        if game_session_id:
            self.game_session_id = game_session_id
        else:
            # Generate unique session ID: timestamp + random component
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_component = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
            self.game_session_id = f"{timestamp}_{random_component}"

        if not self.api_key:
            raise ValueError(
                f"API key not provided. Set {self._get_env_var_name()} environment variable or pass api_key parameter."
            )

        # Validate model (warning only, to support newly released models)
        self._validate_model()

        # Build the provider client once, here rather than per request: a
        # missing SDK then raises ImportError at bot creation (bot_factory
        # falls back to SimpleBot on it) instead of silently passing every
        # turn, and one HTTP connection pool is reused for the whole game.
        self._client: Any = self._create_client()

    # --- Provider configuration (set by subclasses) ---
    # Subclasses must define these class attributes:
    _env_var_name: str = ""  # e.g., "OPENAI_API_KEY"
    _default_model_name: str = ""  # e.g., "gpt-5-mini-2025-08-07"
    _supported_model_list: list[str] = []  # e.g., OPENAI_MODELS
    # Known model IDs that shouldn't be used, with the reason (retired or
    # deprecated); _validate_model warns about them specifically.
    _unavailable_model_notes: dict[str, str] = {}

    def _get_api_key_from_env(self) -> str | None:
        """Get API key from environment variable."""
        return os.getenv(self._env_var_name)

    def _get_env_var_name(self) -> str:
        """Get the name of the environment variable for the API key."""
        return self._env_var_name

    def _get_default_model(self) -> str:
        """Get the default model name."""
        return self._default_model_name

    def _get_supported_models(self) -> list[str]:
        """Get the list of supported models for this provider."""
        return self._supported_model_list

    def _create_client(self) -> Any:
        """Import the provider SDK and build its client (called once, from __init__).

        Providers override this; the base returns None for bots that don't
        talk to an SDK (e.g. test doubles overriding ``_call_llm``).

        Raises:
            ImportError: If the provider's SDK isn't installed.
        """
        return None

    @abstractmethod
    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM API and return the response text."""

    @abstractmethod
    def _get_llm_sdk_version(self) -> str:
        """Get the version of the LLM SDK being used."""

    def _validate_model(self) -> None:
        """
        Validate that the requested model is in the supported list.

        Logs a warning if the model is not in the known supported list,
        but does not raise an error to allow for newly released models.
        """
        supported_models = self._get_supported_models()
        note = self._unavailable_model_notes.get(self.model)
        if note:
            logger.warning("Model '%s' is %s. Current models: %s", self.model, note, ", ".join(supported_models[:5]) + "...")
        elif self.model not in supported_models:
            logger.warning(
                "Model '%s' not in known supported models. It may still work if newly released. Supported models: %s",
                self.model,
                ", ".join(supported_models[:5]) + "...",
            )

    def get_token_usage(self) -> dict[str, int]:
        """
        Get unified token usage statistics across all providers.

        Returns:
            Dictionary with total_input_tokens, total_output_tokens, and total_tokens.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    @property
    def illegal_action_count(self) -> int:
        """Actions from the LLM this game that were skipped as illegal or malformed.

        Per-type counts (``llm_illegal_seize``, ...) are in
        ``get_capabilities_fired()``, which the tournament runner stores in
        each replay's ``game_info``.
        """
        return self.get_capabilities_fired().get("llm_illegal_action", 0)

    def _get_effective_system_prompt(self) -> str:
        """
        Get the effective system prompt considering enabled units.

        If some units are disabled, appends a note to the system prompt
        informing the LLM about the restricted unit types.

        Returns:
            The system prompt with any disabled unit information appended.
        """
        # Get enabled units from game state
        enabled_units = getattr(self.game_state, "enabled_units", None)

        # If all units are enabled or enabled_units is not set, use original prompt
        all_units = ["W", "M", "C", "A", "K", "R", "S", "B"]
        if enabled_units is None or set(enabled_units) == set(all_units):
            return self.system_prompt

        # Find disabled units
        disabled_units = [u for u in all_units if u not in enabled_units]

        if not disabled_units:
            return self.system_prompt

        # Get unit names for disabled units
        disabled_names: list[str] = [str(UNIT_DATA[u]["name"]) for u in disabled_units]

        # Append disabled units note to the system prompt
        disabled_note = (
            f"\n\nIMPORTANT - DISABLED UNITS:\n"
            f"The following unit types are DISABLED for this game and cannot be created: "
            f"{', '.join(disabled_names)} ({', '.join(disabled_units)}).\n"
            f"Do NOT attempt to create these units. Only the following units are available: "
            f"{', '.join([str(UNIT_DATA[u]['name']) for u in enabled_units])} ({', '.join(enabled_units)})."
        )

        return self.system_prompt + disabled_note

    def _log_conversation_to_json(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        stop_reason: str = "",
    ) -> None:
        """
        Log the conversation to a JSON file (single file per game).

        Only logs if log_conversations is True.
        Creates a single log file per game with all turns appended.

        Args:
            system_prompt: The system prompt sent to the LLM
            user_prompt: The user prompt (formatted game state)
            assistant_response: The LLM's response
            input_tokens: Number of input tokens used for this turn
            output_tokens: Number of output tokens used for this turn
            stop_reason: The stop/finish reason from the LLM API response
        """
        # Only log if enabled
        if not self.log_conversations:
            return

        try:
            # Create log directory if it doesn't exist
            log_dir = Path(self.conversation_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Get provider name from class name
            provider = self.__class__.__name__.replace("Bot", "")

            # Generate filename: game_{session_id}_player{player}_model{model}.json
            safe_model = self.model.replace("/", "_").replace(":", "_")
            filename = f"game_{self.game_session_id}_player{self.bot_player}_model{safe_model}.json"
            filepath = log_dir / filename

            # Current timestamp and turn
            timestamp = datetime.now()
            turn = self.game_state.turn_number

            # Build turn data with token usage and stop reason
            turn_data = {
                "turn_number": turn,
                "timestamp": timestamp.isoformat(),
                "user_prompt": user_prompt,
                "assistant_response": assistant_response,
            }

            # Include stop reason if available
            if stop_reason:
                turn_data["stop_reason"] = stop_reason

            # Include token usage if tracked
            if input_tokens > 0 or output_tokens > 0:
                turn_data["token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

            # Check if file exists
            if filepath.exists():
                # Load existing data and append new turn
                with open(filepath, encoding="utf-8") as f:
                    log_data = json.load(f)

                # Append new turn
                log_data["turns"].append(turn_data)

                # Update cumulative token usage
                if input_tokens > 0 or output_tokens > 0:
                    log_data["total_token_usage"] = self.get_token_usage()
            else:
                # Create new log file with metadata
                log_data = {
                    "game_session_id": self.game_session_id,
                    "version": {"reinforce_tactics": __version__, "llm_sdk": self._get_llm_sdk_version()},
                    "model": self.model,
                    "max_tokens": self.max_tokens if self.max_tokens is not None else "None",
                    "temperature": self.temperature,
                    "provider": provider,
                    "player": self.bot_player,
                    "start_time": timestamp.isoformat(),
                    "map_file": self.game_state.map_file_used,
                    "map_dimensions": {
                        "width": self.game_state.original_map_width,
                        "height": self.game_state.original_map_height,
                    },
                    "system_prompt": system_prompt,
                    "turns": [turn_data],
                }

                # Add cumulative token usage
                if input_tokens > 0 or output_tokens > 0:
                    log_data["total_token_usage"] = self.get_token_usage()

            # Write to file with configurable formatting
            indent = 2 if self.pretty_print_logs else None
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=indent, ensure_ascii=False)

            logger.debug("Logged conversation to %s (turn %s)", filepath, turn)

        except Exception as e:
            # Don't let logging errors break the bot
            logger.error("Failed to log conversation: %s", e)

    def take_turn(self):
        """
        Execute the bot's turn using LLM guidance.

        This method orchestrates the entire turn-taking process:
        1. Serializes the current game state into JSON
        2. Optionally runs a planning phase (if two_phase_planning is enabled)
        3. Calls the LLM API with retry logic (including conversation history if stateful)
        4. Parses the LLM response
        5. Validates and executes the suggested actions

        Actions that aren't in ``get_legal_actions`` at the moment they would
        run are skipped and counted (see ``illegal_action_count``). A reply
        with no usable actions (empty, prose, JSON cut off at max_tokens, a
        refusal or safety block) ends the turn without actions and is counted
        (``llm_empty_reply`` / ``llm_unparseable_reply``): that is the model's
        play, scored on the board. If the API gives no reply at all after
        retries, the turn also ends without actions, up to
        ``max_consecutive_failed_turns`` turns in a row.

        Raises:
            LLMBotError: On a non-retryable API failure (missing SDK, bad key,
                no permission, unknown model, rejected request), or when the
                streak of turns without a reply reaches
                ``max_consecutive_failed_turns``. The turn is left un-ended
                for the caller to handle.
        """
        logger.info("LLM Bot (Player %s) is thinking...", self.bot_player)

        # Serialize game state
        game_state_json = self._serialize_game_state()

        # Two-phase planning: first get a strategic plan, then execute
        strategic_plan = None
        if self.two_phase_planning:
            strategic_plan = self._run_planning_phase(game_state_json)
            if strategic_plan:
                logger.info("Strategic plan generated: %s", strategic_plan.get("primary_objective", "No objective"))

        # Format the user prompt (include plan if two-phase mode)
        user_prompt = self._format_prompt(game_state_json, strategic_plan=strategic_plan)

        # Determine which system prompt to use for execution
        if self.two_phase_planning and strategic_plan:
            # Use the execution prompt for phase 2
            execution_system_prompt = PROMPT_TWO_PHASE_EXECUTE.format(plan=json.dumps(strategic_plan, indent=2))
        else:
            # Use the effective prompt that includes disabled unit information
            execution_system_prompt = self._get_effective_system_prompt()

        # Get LLM response with retries
        response_text = self._call_llm_with_retry(execution_system_prompt, user_prompt)

        if response_text is None:
            # No reply at all: the API is down, overloaded, unreachable or
            # timing out. May raise LLMBotError once that has lasted
            # max_consecutive_failed_turns turns.
            self._record_failed_turn()
            self.game_state.end_turn()
            return

        # The provider answered, so any outage streak is over, whatever the
        # reply holds.
        self.consecutive_failed_turns = 0

        # Store conversation in history if stateful mode is enabled. An empty
        # reply is left out: providers reject an empty assistant message.
        if self.stateful and response_text:
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response_text})

        # Log the conversation if enabled (include token usage and stop reason)
        self._log_conversation_to_json(
            execution_system_prompt,
            user_prompt,
            response_text,
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens,
            stop_reason=self._last_stop_reason,
        )

        # An empty or unparseable reply (a refusal, a safety block, prose, JSON
        # cut off at max_tokens) passes the turn but never raises LLMBotError:
        # it is the model failing at the task, not the infrastructure failing
        # the model. Raising would cancel the game in a tournament, which
        # would favour models that break the output format over ones that
        # play a losing position out. Both are counted in the bot's stats.
        if not response_text:
            logger.warning(
                "LLM returned an empty reply (stop reason: %s); ending turn without actions",
                self._last_stop_reason or "n/a",
            )
            self._record("llm_empty_reply")
        else:
            self._execute_actions(response_text)

        # End turn (advance game state to next player, collect income, etc.)
        # Skip if game is already over (e.g., due to resignation)
        if not self.game_state.game_over:
            self.game_state.end_turn()

    def _record_failed_turn(self) -> None:
        """Count a turn the API didn't answer; raise once the streak hits the limit."""
        self.consecutive_failed_turns += 1
        self._record("llm_failed_turn")
        limit = self.max_consecutive_failed_turns
        if limit is not None and self.consecutive_failed_turns >= limit:
            last = self._last_failure_reason
            message = (
                f"{self.__class__.__name__} ({self.model}) got no usable reply from the API for "
                f"{self.consecutive_failed_turns} turns in a row"
                + (f" (last error: {last})" if last else "")
                + "; stopping instead of passing more turns"
            )
            logger.error(message)
            raise LLMBotError(message, retryable=True)
        logger.warning(
            "No response from the LLM API. Ending turn without actions (%d consecutive failed turn(s)).",
            self.consecutive_failed_turns,
        )

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        """Seconds to wait before retry number ``attempt + 1``.

        Exponential backoff with "equal jitter" (half fixed, half random):
        concurrent tournament games that hit the same 429 don't retry in
        lockstep, yet no retry is instant. A server-sent Retry-After, when
        longer, wins (capped at _RETRY_AFTER_CAP_S).
        """
        ceiling = min(_RETRY_MAX_DELAY_S, _RETRY_BASE_DELAY_S * 2**attempt)
        delay = ceiling / 2 + random.uniform(0, ceiling / 2)
        if retry_after is not None:
            delay = max(delay, min(retry_after, _RETRY_AFTER_CAP_S))
        return delay

    def _should_retry(self, attempts_made: int, info: _LLMErrorInfo, spent_after_delay: float) -> bool:
        """Whether to try again after ``attempts_made`` retryable failures.

        Every retryable failure gets ``max_retries`` attempts. A transient
        one (rate limit, 5xx/overloaded, timeout, connection error) is then
        retried for as long as the call, counting the next backoff, stays
        inside ``retry_budget_s``: with the SDKs' own retries off, three
        quick attempts (about 3 s of backoff) gave up on a short overload,
        and three such turns raise LLMBotError.
        """
        over_budget = self.retry_budget_s is not None and spent_after_delay > self.retry_budget_s
        if info.retry_after is not None and over_budget:
            # The server asked for a wait that alone takes this call past its
            # budget. Sleeping through it (up to _RETRY_AFTER_CAP_S, even
            # within max_retries) froze a GUI game for minutes per turn.
            return False
        if attempts_made < max(1, self.max_retries):
            return True
        return info.transient and self.retry_budget_s is not None and not over_budget

    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str | None:
        """
        Call the LLM, retrying transient failures with jittered exponential backoff.

        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt with game state

        Returns:
            The LLM response text (possibly empty), or None if no attempt got
            a reply (see _should_retry for how long it keeps trying)

        Raises:
            LLMBotError: On the first non-retryable failure (see _classify_llm_error).
        """
        if self.stateful and self.conversation_history:
            # In stateful mode, include full conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_prompt})
        else:
            # In stateless mode, only send current turn
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        attempts_made = 0
        # Seconds this call has used: time in requests plus backoff. Summed
        # from the parts rather than read off a clock around the whole loop
        # so the budget holds however sleeping is done.
        spent = 0.0
        while True:
            attempts_made += 1
            # Reset per-call telemetry so a failed attempt can't leave the
            # previous call's tokens or stop reason behind.
            self._last_input_tokens = 0
            self._last_output_tokens = 0
            self._last_stop_reason = ""
            started = time.monotonic()
            try:
                response_text = self._call_llm(messages)
            except LLMBotError:
                raise
            except Exception as exc:
                spent += time.monotonic() - started
                info = _classify_llm_error(exc)
                if not info.retryable:
                    message = f"{self.__class__.__name__} ({self.model}): {info.reason}: {exc}"
                    logger.error("LLM request failed and won't be retried: %s", message)
                    raise LLMBotError(message, retryable=False) from exc
                delay = self._retry_delay(attempts_made - 1, info.retry_after)
                if not self._should_retry(attempts_made, info, spent + delay):
                    self._last_failure_reason = f"{info.reason}: {exc}"[:300]
                    logger.warning(
                        "LLM request failed (%s), attempt %d after %.0fs; giving up for this turn: %s",
                        info.reason,
                        attempts_made,
                        spent,
                        exc,
                    )
                    return None
                logger.warning(
                    "LLM request failed (%s), attempt %d; retrying in %.1fs: %s",
                    info.reason,
                    attempts_made,
                    delay,
                    exc,
                )
                time.sleep(delay)
                spent += delay
                continue
            # Accumulate token usage (only if tracked by subclass)
            self.total_input_tokens += self._last_input_tokens
            self.total_output_tokens += self._last_output_tokens
            return response_text

    def _run_planning_phase(self, game_state_json: dict[str, Any]) -> dict | None:
        """
        Run the planning phase for two-phase planning mode.

        This phase asks the LLM to analyze the situation and create a strategic
        plan before deciding on specific actions. This encourages deeper thinking
        about multi-step tactical sequences.

        Args:
            game_state_json: The serialized game state

        Returns:
            The strategic plan as a dictionary, or None if planning failed
        """
        logger.info("Running planning phase...")

        # Format the planning prompt
        planning_prompt = f"""Analyze this game state and create a strategic plan:

{json.dumps(game_state_json, indent=2)}

Consider:
1. What buildings can be captured this turn?
2. Which enemies need to be killed to enable captures?
3. What order should units act?
4. Are there any threats to address?

Respond with your strategic plan in JSON format."""

        # Call LLM for planning (use planning prompt)
        response_text = self._call_llm_with_retry(PROMPT_TWO_PHASE_PLAN, planning_prompt)

        if not response_text:
            logger.warning("Planning phase failed, falling back to single-phase")
            return None

        # Log the planning conversation if enabled
        self._log_conversation_to_json(
            PROMPT_TWO_PHASE_PLAN,
            planning_prompt,
            response_text,
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens,
            stop_reason=self._last_stop_reason,
        )

        # Parse the plan
        try:
            plan = self._extract_json(response_text)
            if plan:
                return plan
        except Exception as e:
            logger.warning("Failed to parse strategic plan: %s", e)

        return None

    def _serialize_game_state(self) -> dict[str, Any]:
        """
        Serialize the current game state to a dictionary.

        Returns:
            Dictionary containing game state information
        """
        # Get legal actions first
        legal_actions = self.game_state.get_legal_actions(self.bot_player)

        # Serialize player's units with IDs (convert to original coordinates)
        player_units = []
        unit_id = 0
        unit_id_map = {}  # Map unit objects to IDs for later reference

        for unit in self.game_state.units:
            if unit.player == self.bot_player:
                orig_x, orig_y = self.game_state.padded_to_original_coords(unit.x, unit.y)
                unit_data = {
                    "id": unit_id,
                    "type": unit.type,
                    "position": [orig_x, orig_y],
                    "hp": unit.health,
                    "max_hp": UNIT_DATA[unit.type]["health"],
                    "can_move": unit.can_move,
                    "can_attack": unit.can_attack,
                    "is_paralyzed": unit.is_paralyzed(),
                }
                player_units.append(unit_data)
                unit_id_map[unit] = unit_id
                unit_id += 1

        # Serialize enemy units (less detail, convert to original coordinates)
        # With fog of war, only include visible enemy units
        enemy_units = []
        for unit in self.game_state.units:
            if unit.player != self.bot_player:
                # FOW: Skip enemies that are not visible
                if self.game_state.fog_of_war:
                    if not self.game_state.is_position_visible(unit.x, unit.y, self.bot_player):
                        continue

                orig_x, orig_y = self.game_state.padded_to_original_coords(unit.x, unit.y)
                enemy_data = {
                    "type": unit.type,
                    "position": [orig_x, orig_y],
                    "hp": unit.health,
                    "max_hp": UNIT_DATA[unit.type]["health"],
                }
                enemy_units.append(enemy_data)

        # Serialize buildings (convert to original coordinates)
        # With fog of war, only include structures the bot knows of, with the
        # owner it knows (GameState.known_structure, the same view the RL
        # observation and the renderer use): live while in sight, else as
        # last seen, and every HQ from the start. Reading the live owner of
        # an out-of-sight structure leaked enemy captures to LLM bots
        # (review critic-integration-3).
        player_buildings = []
        enemy_buildings = []
        neutral_buildings = []

        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type in ["b", "h", "t"]:
                    owner = tile.player
                    known = None
                    if self.game_state.fog_of_war:
                        known = self.game_state.known_structure(self.bot_player, tile.x, tile.y)
                        if known is None:
                            continue  # never seen
                        owner = known.owner

                    orig_x, orig_y = self.game_state.padded_to_original_coords(tile.x, tile.y)
                    building_info = {
                        "type": tile.type,
                        "position": [orig_x, orig_y],
                        "income": 100 if tile.type == "h" else (100 if tile.type == "b" else 50),
                    }

                    # FOW: an out-of-sight structure is reported as last seen
                    if known is not None and not self.game_state.is_position_visible(tile.x, tile.y, self.bot_player):
                        building_info["last_seen"] = True
                        building_info["turn_seen"] = known.turn_seen
                        building_info["hp"] = known.health

                    if owner == self.bot_player:
                        player_buildings.append(building_info)
                    elif owner is not None:
                        enemy_buildings.append(building_info)
                    else:
                        neutral_buildings.append(building_info)

        # Format legal actions for the LLM
        formatted_legal_actions = self._format_legal_actions(legal_actions, unit_id_map)

        # Extract map name from file path
        map_name = "unknown"
        if self.game_state.map_file_used:
            map_name = Path(self.game_state.map_file_used).stem

        # Get enabled units (for informing LLM which units can be created)
        enabled_units = getattr(self.game_state, "enabled_units", ["W", "M", "C", "A", "K", "R", "S", "B"])

        # Build the state dictionary
        state = {
            "map_name": map_name,
            "map_width": self.game_state.original_map_width,
            "map_height": self.game_state.original_map_height,
            "turn_number": self.game_state.turn_number,
            "player_gold": self.game_state.player_gold[self.bot_player],
            "enabled_units": enabled_units,
            "player_units": player_units,
            "enemy_units": enemy_units,
            "player_buildings": player_buildings,
            "enemy_buildings": enemy_buildings,
            "neutral_buildings": neutral_buildings,
            "legal_actions": formatted_legal_actions,
        }

        # FOW: Include fog of war status and hide enemy gold
        if self.game_state.fog_of_war:
            state["fog_of_war"] = True
            state["opponent_gold"] = "hidden"  # Hide enemy gold in FOW mode
        else:
            state["fog_of_war"] = False
            state["opponent_gold"] = self.game_state.player_gold[1 if self.bot_player == 2 else 2]

        return state

    def _compute_move_then_actions(
        self, unit, unit_id: int, reachable_positions: list[tuple]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Compute actions that become available after moving to reachable positions.

        Args:
            unit: The unit to check
            unit_id: The unit's ID for the LLM
            reachable_positions: List of (x, y) positions the unit can move to

        Returns:
            Dict with move_then_attack, move_then_seize, etc. combinations
        """
        result: dict[str, list[dict[str, Any]]] = {
            "move_then_attack": [],
            "move_then_seize": [],
            "move_then_heal": [],
            "move_then_cure": [],
            "move_then_paralyze": [],
        }

        # Get enemy units for attack calculations.
        # Under FOW, only consider enemies currently visible: a move-then-attack
        # combo against an enemy the bot can't see right now would either be a
        # FOW info leak (we'd reveal hidden enemy positions) or a "move to
        # discover, then attack" exploit (forbidden by is_enemy_attackable_by_unit).
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player]
        if self.game_state.fog_of_war:
            enemy_units = [u for u in enemy_units if self.game_state.is_position_visible(u.x, u.y, self.bot_player)]

        # Get ally units for heal/cure calculations (Cleric only)
        ally_units = [u for u in self.game_state.units if u.player == self.bot_player and u != unit]

        for to_x, to_y in reachable_positions:
            # Convert to original coords for output
            orig_to_x, orig_to_y = self.game_state.padded_to_original_coords(to_x, to_y)

            # Check if moving here allows attacking enemies
            # Temporarily calculate what would be in range from this position
            tile = self.game_state.grid.get_tile(to_x, to_y)
            on_mountain = tile.type == "m" if tile else False

            for enemy in enemy_units:
                # Calculate distance from potential new position
                distance = abs(to_x - enemy.x) + abs(to_y - enemy.y)

                # Check if enemy would be in attack range from new position
                min_range, max_range = unit.get_attack_range(on_mountain)
                if min_range <= distance <= max_range:
                    orig_enemy_x, orig_enemy_y = self.game_state.padded_to_original_coords(enemy.x, enemy.y)
                    result["move_then_attack"].append(
                        {"unit_id": unit_id, "move_to": [orig_to_x, orig_to_y], "then_attack": [orig_enemy_x, orig_enemy_y]}
                    )

                    # Mage can also paralyze when the target is within its valid range
                    if unit.type == "M" and min_range <= distance <= max_range:
                        result["move_then_paralyze"].append(
                            {
                                "unit_id": unit_id,
                                "move_to": [orig_to_x, orig_to_y],
                                "then_paralyze": [orig_enemy_x, orig_enemy_y],
                            }
                        )

            # Check if moving here allows seizing a structure. Under fog of war
            # the move set includes tiles the bot has never seen, so judge by
            # the structure it knows of (GameState.known_structure, the same
            # view as its building lists): offering then_seize on an unseen
            # tile would reveal that a structure stands there.
            if tile and tile.is_capturable():
                known_to_bot, owner = True, tile.player
                if self.game_state.fog_of_war:
                    known = self.game_state.known_structure(self.bot_player, to_x, to_y)
                    known_to_bot = known is not None
                    owner = known.owner if known is not None else None
                if known_to_bot and owner != self.bot_player:
                    result["move_then_seize"].append(
                        {"unit_id": unit_id, "move_to": [orig_to_x, orig_to_y], "then_seize": True}
                    )

            # Cleric-specific: check for heal/cure opportunities
            if unit.type == "C":
                adjacent_positions = [(to_x, to_y - 1), (to_x, to_y + 1), (to_x - 1, to_y), (to_x + 1, to_y)]

                for ally in ally_units:
                    if (ally.x, ally.y) in adjacent_positions:
                        orig_ally_x, orig_ally_y = self.game_state.padded_to_original_coords(ally.x, ally.y)
                        # Heal if damaged
                        if ally.health < ally.max_health:
                            result["move_then_heal"].append(
                                {
                                    "unit_id": unit_id,
                                    "move_to": [orig_to_x, orig_to_y],
                                    "then_heal": [orig_ally_x, orig_ally_y],
                                }
                            )
                        # Cure if paralyzed
                        if ally.is_paralyzed():
                            result["move_then_cure"].append(
                                {
                                    "unit_id": unit_id,
                                    "move_to": [orig_to_x, orig_to_y],
                                    "then_cure": [orig_ally_x, orig_ally_y],
                                }
                            )

        return result

    def _format_legal_actions(self, legal_actions: dict[str, list[Any]], unit_id_map: dict) -> dict[str, list[dict[str, Any]]]:
        """Format legal actions for LLM consumption with original map coordinates."""
        formatted: dict[str, list[dict[str, Any]]] = {
            "create_unit": [],
            "move": [],
            "attack": [],
            "paralyze": [],
            "heal": [],
            "cure": [],
            "seize": [],
            # Move-then-action combinations
            "move_then_attack": [],
            "move_then_seize": [],
            "move_then_heal": [],
            "move_then_cure": [],
            "move_then_paralyze": [],
        }

        # Create unit actions (convert coordinates)
        for action in legal_actions["create_unit"]:
            orig_x, orig_y = self.game_state.padded_to_original_coords(action["x"], action["y"])
            formatted["create_unit"].append(
                {
                    "unit_type": action["unit_type"],
                    "position": [orig_x, orig_y],
                    "cost": UNIT_DATA[action["unit_type"]]["cost"],
                }
            )

        # Move actions (convert coordinates)
        for action in legal_actions["move"]:
            if action["unit"] in unit_id_map:
                from_x, from_y = self.game_state.padded_to_original_coords(action["from_x"], action["from_y"])
                to_x, to_y = self.game_state.padded_to_original_coords(action["to_x"], action["to_y"])
                formatted["move"].append(
                    {"unit_id": unit_id_map[action["unit"]], "from": [from_x, from_y], "to": [to_x, to_y]}
                )

        # Attack actions (convert coordinates)
        for action in legal_actions["attack"]:
            if action["attacker"] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(action["target"].x, action["target"].y)
                formatted["attack"].append(
                    {"unit_id": unit_id_map[action["attacker"]], "target_position": [target_x, target_y]}
                )

        # Paralyze actions (convert coordinates)
        for action in legal_actions["paralyze"]:
            if action["paralyzer"] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(action["target"].x, action["target"].y)
                formatted["paralyze"].append(
                    {"unit_id": unit_id_map[action["paralyzer"]], "target_position": [target_x, target_y]}
                )

        # Heal actions (convert coordinates)
        for action in legal_actions["heal"]:
            if action["healer"] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(action["target"].x, action["target"].y)
                formatted["heal"].append({"unit_id": unit_id_map[action["healer"]], "target_position": [target_x, target_y]})

        # Cure actions (convert coordinates)
        for action in legal_actions["cure"]:
            if action["curer"] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(action["target"].x, action["target"].y)
                formatted["cure"].append({"unit_id": unit_id_map[action["curer"]], "target_position": [target_x, target_y]})

        # Seize actions (convert coordinates)
        for action in legal_actions["seize"]:
            if action["unit"] in unit_id_map:
                tile_x, tile_y = self.game_state.padded_to_original_coords(action["tile"].x, action["tile"].y)
                formatted["seize"].append({"unit_id": unit_id_map[action["unit"]], "position": [tile_x, tile_y]})

        # Compute move-then-action combinations for units that can move
        # Group move actions by unit to get all reachable positions per unit
        unit_reachable_positions: dict[Any, list[tuple]] = {}
        for action in legal_actions["move"]:
            unit = action["unit"]
            if unit not in unit_reachable_positions:
                unit_reachable_positions[unit] = []
            unit_reachable_positions[unit].append((action["to_x"], action["to_y"]))

        # For each movable unit, compute what actions become available after moving
        for unit, positions in unit_reachable_positions.items():
            if unit in unit_id_map:
                unit_id = unit_id_map[unit]
                move_then_actions = self._compute_move_then_actions(unit, unit_id, positions)

                # Merge results into formatted output
                for key in ["move_then_attack", "move_then_seize", "move_then_heal", "move_then_cure", "move_then_paralyze"]:
                    formatted[key].extend(move_then_actions[key])

        return formatted

    def _format_prompt(self, game_state_json: dict[str, Any], strategic_plan: dict | None = None) -> str:
        """
        Format the game state into a prompt for the LLM.

        Args:
            game_state_json: The serialized game state
            strategic_plan: Optional strategic plan from two-phase planning mode

        Returns:
            The formatted user prompt string
        """
        reasoning_line = (
            '    "reasoning": "Brief explanation of your strategy (1-2 sentences)",\n'
            if self.should_reason or strategic_plan
            else ""
        )

        # Include strategic plan context if provided
        plan_context = ""
        if strategic_plan:
            plan_context = f"""
STRATEGIC PLAN TO EXECUTE:
{json.dumps(strategic_plan, indent=2)}

Execute the actions according to this plan.

"""

        return f"""Current Game State:
{json.dumps(game_state_json, indent=2)}
{plan_context}
Respond with a JSON object in the following format:
{{
{reasoning_line}    "actions": [
        {{"type": "CREATE_UNIT", "unit_type": "W|M|C|A", "position": [x, y]}},
        {{"type": "MOVE", "unit_id": 0, "from": [x, y], "to": [x, y]}},
        {{"type": "ATTACK", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "PARALYZE", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "HEAL", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "CURE", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "SEIZE", "unit_id": 0}},
        {{"type": "END_TURN"}},
        {{"type": "RESIGN"}}
    ]
}}

Only include actions that are legal based on the legal_actions provided.
You can take multiple actions in one turn.
Use RESIGN only as a last resort when victory is impossible."""

    def _execute_actions(self, response_text: str) -> bool:
        """Parse the LLM response and execute its actions in order.

        Each action is checked against ``get_legal_actions`` as the state is
        at that moment (see ``_is_legal_unit_action``) and skipped if it isn't
        listed, so an LLM that repeats or invents actions can't do what the
        rules forbid: seize twice in a turn, attack its own units, or act
        again with a spent unit.

        Returns:
            True if the reply held an ``actions`` list (even an empty one, or
            one whose actions were all skipped); False if no such list could
            be parsed from it (counted as ``llm_unparseable_reply``).
        """
        try:
            response_json = self._extract_json(response_text)
        except Exception as e:  # e.g. RecursionError on absurdly nested JSON
            logger.error("Error parsing LLM response: %s", e)
            response_json = None
        if not isinstance(response_json, dict) or not isinstance(response_json.get("actions"), list):
            logger.warning(
                "Invalid response format: no 'actions' list found (stop reason: %s); ending turn without actions",
                self._last_stop_reason or "n/a",
            )
            self._record("llm_unparseable_reply")
            return False
        actions = response_json["actions"]

        try:
            # Log reasoning if provided
            if "reasoning" in response_json:
                logger.info("Bot reasoning: %s", response_json["reasoning"])

            # Build unit ID to unit object mapping
            unit_map = self._get_unit_by_id()

            # Execute each action
            for index, action in enumerate(actions):
                # A seize or kill earlier in the list can end the game; nothing
                # after that point is a legal move.
                if self.game_state.game_over:
                    logger.info("Game over; ignoring the LLM's remaining %d action(s)", len(actions) - index)
                    break

                if not isinstance(action, dict) or "type" not in action:
                    self._reject_action(action, "not an object with a 'type' field")
                    continue

                action_type = action["type"]

                try:
                    executed = False
                    if action_type == "CREATE_UNIT":
                        executed = self._execute_create_unit(action)
                    elif action_type == "MOVE":
                        executed = self._execute_move(action, unit_map)
                    elif action_type == "ATTACK":
                        executed = self._execute_attack(action, unit_map)
                    elif action_type == "PARALYZE":
                        executed = self._execute_paralyze(action, unit_map)
                    elif action_type == "HEAL":
                        executed = self._execute_heal(action, unit_map)
                    elif action_type == "CURE":
                        executed = self._execute_cure(action, unit_map)
                    elif action_type == "SEIZE":
                        executed = self._execute_seize(action, unit_map)
                    elif action_type == "END_TURN":
                        logger.info("Bot chose to end turn")
                        break
                    elif action_type == "RESIGN":
                        self._execute_resign()
                        return True  # Exit immediately after resignation
                    else:
                        self._reject_action(action, f"unknown action type {action_type!r}")
                    if executed:
                        self._record("llm_action_executed")
                except Exception as e:
                    logger.error("Error executing action %s: %s", action, e)
                    continue

        except Exception as e:
            logger.error("Error executing LLM response: %s", e)
        return True

    def _extract_json(self, text: str) -> dict | None:
        """Extract JSON from response text, handling markdown code blocks."""
        # Try to parse the whole response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object anywhere in the text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _get_unit_by_id(self) -> dict[int, Any]:
        """Create a mapping from unit IDs to unit objects."""
        unit_map = {}
        unit_id = 0
        for unit in self.game_state.units:
            if unit.player == self.bot_player:
                unit_map[unit_id] = unit
                unit_id += 1
        return unit_map

    @staticmethod
    def _parse_xy(value: Any) -> tuple[int, int] | None:
        """``[x, y]`` from an LLM action as ints, or None if it isn't one."""
        if not isinstance(value, list | tuple) or len(value) != 2:
            return None
        coords = []
        for v in value:
            if isinstance(v, bool):
                return None
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            if not isinstance(v, int):
                return None
            coords.append(v)
        return coords[0], coords[1]

    @staticmethod
    def _lookup_unit(action: dict[str, Any], unit_map: dict[int, Any]) -> Any | None:
        """The bot's unit named by the action's ``unit_id``, or None."""
        unit_id = action.get("unit_id")
        if isinstance(unit_id, bool) or not isinstance(unit_id, int):
            return None
        return unit_map.get(unit_id)

    def _reject_action(self, action: Any, reason: str) -> None:
        """Skip an LLM action, logging why and counting it in the bot's stats."""
        action_type = action.get("type") if isinstance(action, dict) else None
        logger.warning("Skipping illegal LLM action %s: %s", action, reason)
        self._record("llm_illegal_action")
        known = action_type in _ILLEGAL_ACTION_HINTS
        self._record(f"llm_illegal_{str(action_type).lower()}" if known else "llm_illegal_other")

    def _is_legal_unit_action(self, action_type: str, unit: Any, target_xy: tuple[int, int] | None = None) -> bool:
        """Whether ``unit`` may do ``action_type`` (at padded ``target_xy``) right now.

        Re-queries get_legal_actions on every call. Each executed action
        invalidates the engine's cache, so this sees the state after the
        LLM's earlier actions this turn: a unit that has attacked or seized is
        no longer listed. Matching on the acting unit object and the target
        square is what rejects a repeated SEIZE, friendly fire (``attack``
        lists only enemies), attacks on units hidden by fog of war, and any
        second action by a spent unit.
        """
        legal_key, actor_key = _UNIT_ACTION_LEGAL_KEYS[action_type]
        for entry in self.game_state.get_legal_actions(self.bot_player).get(legal_key, []):
            if entry[actor_key] is not unit:
                continue
            if target_xy is None:
                return True
            if legal_key == "move":
                entry_xy = (entry["to_x"], entry["to_y"])
            else:
                entry_xy = (entry["target"].x, entry["target"].y)
            if entry_xy == target_xy:
                return True
        return False

    def _execute_create_unit(self, action: dict[str, Any]) -> bool:
        """Execute a CREATE_UNIT action (converts from original to padded coordinates)."""
        unit_type = action.get("unit_type")
        position = self._parse_xy(action.get("position"))

        if not unit_type or position is None:
            self._reject_action(action, "needs unit_type and position [x, y]")
            return False

        # Convert from original to padded coordinates
        orig_x, orig_y = position
        x, y = self.game_state.original_to_padded_coords(orig_x, orig_y)

        # Validate this is a legal action (using padded coordinates)
        legal_actions = self.game_state.get_legal_actions(self.bot_player)
        is_legal = any(
            a["unit_type"] == unit_type and a["x"] == x and a["y"] == y for a in legal_actions.get("create_unit", [])
        )

        if not is_legal:
            self._reject_action(
                action, f"not in legal_actions.create_unit (padded [{x}, {y}]): {_ILLEGAL_ACTION_HINTS['CREATE_UNIT']}"
            )
            return False

        if self.game_state.create_unit(unit_type, x, y, self.bot_player) is None:
            logger.warning("Engine refused CREATE_UNIT %s", action)
            return False
        logger.info("Created %s at original coords (%s, %s) / padded coords (%s, %s)", unit_type, orig_x, orig_y, x, y)
        return True

    def _execute_move(self, action: dict[str, Any], unit_map: dict[int, Any]) -> bool:
        """Execute a MOVE action (converts from original to padded coordinates)."""
        unit = self._lookup_unit(action, unit_map)
        to_pos = self._parse_xy(action.get("to"))

        if unit is None or to_pos is None:
            self._reject_action(action, "needs a unit_id from player_units and to [x, y]")
            return False

        # Convert from original to padded coordinates
        orig_to_x, orig_to_y = to_pos
        to_x, to_y = self.game_state.original_to_padded_coords(orig_to_x, orig_to_y)

        # "from" is optional, but when given it must be where the unit is: a
        # mismatch means the LLM has confused its unit IDs, and moving
        # whichever unit the ID happens to name would not be what it meant.
        if "from" in action:
            from_pos = self._parse_xy(action.get("from"))
            if from_pos is None or self.game_state.original_to_padded_coords(*from_pos) != (unit.x, unit.y):
                orig_pos = list(self.game_state.padded_to_original_coords(unit.x, unit.y))
                self._reject_action(action, f"'from' doesn't match unit {action.get('unit_id')}'s position {orig_pos}")
                return False

        if not self._is_legal_unit_action("MOVE", unit, (to_x, to_y)):
            self._reject_action(action, f"not in legal_actions.move: {_ILLEGAL_ACTION_HINTS['MOVE']}")
            return False

        if not self.game_state.move_unit(unit, to_x, to_y):
            logger.warning("Engine refused MOVE %s", action)
            return False
        # Log where the unit really is: a fog-of-war ambush stops it short.
        if unit.ambushed:
            logger.info(
                "Unit %s was ambushed on its way to original coords (%s, %s)", action.get("unit_id"), orig_to_x, orig_to_y
            )
        orig_x, orig_y = self.game_state.padded_to_original_coords(unit.x, unit.y)
        logger.info(
            "Moved unit %s to original coords (%s, %s) / padded coords (%s, %s)",
            action.get("unit_id"),
            orig_x,
            orig_y,
            unit.x,
            unit.y,
        )
        return True

    def _resolve_targeted_action(
        self, action: dict[str, Any], unit_map: dict[int, Any]
    ) -> tuple[Any, Any, tuple[int, int]] | None:
        """Validate an ATTACK/PARALYZE/HEAL/CURE action against the legal actions.

        Returns ``(unit, target, original_target_xy)`` if legal; otherwise
        rejects the action (logged and counted) and returns None.
        """
        action_type = action["type"]
        unit = self._lookup_unit(action, unit_map)
        target_pos = self._parse_xy(action.get("target_position"))

        if unit is None or target_pos is None:
            self._reject_action(action, "needs a unit_id from player_units and target_position [x, y]")
            return None

        # Convert from original to padded coordinates
        target_x, target_y = self.game_state.original_to_padded_coords(*target_pos)
        if not self._is_legal_unit_action(action_type, unit, (target_x, target_y)):
            legal_key = _UNIT_ACTION_LEGAL_KEYS[action_type][0]
            self._reject_action(action, f"not in legal_actions.{legal_key}: {_ILLEGAL_ACTION_HINTS[action_type]}")
            return None

        target = self.game_state.get_unit_at_position(target_x, target_y)
        return unit, target, target_pos

    def _execute_attack(self, action: dict[str, Any], unit_map: dict[int, Any]) -> bool:
        """Execute an ATTACK action (converts from original to padded coordinates)."""
        resolved = self._resolve_targeted_action(action, unit_map)
        if resolved is None:
            return False
        unit, target, (orig_target_x, orig_target_y) = resolved

        self.game_state.attack(unit, target)
        logger.info(
            "Unit %s attacked enemy at original coords (%s, %s) / padded coords (%s, %s)",
            action.get("unit_id"),
            orig_target_x,
            orig_target_y,
            target.x,
            target.y,
        )
        return True

    def _execute_paralyze(self, action: dict[str, Any], unit_map: dict[int, Any]) -> bool:
        """Execute a PARALYZE action (converts from original to padded coordinates)."""
        resolved = self._resolve_targeted_action(action, unit_map)
        if resolved is None:
            return False
        unit, target, (orig_target_x, orig_target_y) = resolved

        if not self.game_state.paralyze(unit, target):
            logger.warning("Engine refused PARALYZE %s", action)
            return False
        logger.info(
            "Unit %s paralyzed enemy at original coords (%s, %s) / padded coords (%s, %s)",
            action.get("unit_id"),
            orig_target_x,
            orig_target_y,
            target.x,
            target.y,
        )
        return True

    def _execute_heal(self, action: dict[str, Any], unit_map: dict[int, Any]) -> bool:
        """Execute a HEAL action (converts from original to padded coordinates)."""
        resolved = self._resolve_targeted_action(action, unit_map)
        if resolved is None:
            return False
        unit, target, (orig_target_x, orig_target_y) = resolved

        if not self.game_state.heal(unit, target):
            logger.warning("Engine refused HEAL %s", action)
            return False
        logger.info(
            "Unit %s healed ally at original coords (%s, %s) / padded coords (%s, %s)",
            action.get("unit_id"),
            orig_target_x,
            orig_target_y,
            target.x,
            target.y,
        )
        return True

    def _execute_cure(self, action: dict[str, Any], unit_map: dict[int, Any]) -> bool:
        """Execute a CURE action (converts from original to padded coordinates)."""
        resolved = self._resolve_targeted_action(action, unit_map)
        if resolved is None:
            return False
        unit, target, (orig_target_x, orig_target_y) = resolved

        if not self.game_state.cure(unit, target):
            logger.warning("Engine refused CURE %s", action)
            return False
        logger.info(
            "Unit %s cured ally at original coords (%s, %s) / padded coords (%s, %s)",
            action.get("unit_id"),
            orig_target_x,
            orig_target_y,
            target.x,
            target.y,
        )
        return True

    def _execute_seize(self, action: dict[str, Any], unit_map: dict[int, Any]) -> bool:
        """Execute a SEIZE action on the structure under the unit."""
        unit = self._lookup_unit(action, unit_map)

        if unit is None:
            self._reject_action(action, "needs a unit_id from player_units")
            return False

        # Without this check a repeated SEIZE hits the structure once per
        # repetition: one Warrior could take a 50-HP HQ, and the game, in a
        # single turn.
        if not self._is_legal_unit_action("SEIZE", unit):
            self._reject_action(action, f"not in legal_actions.seize: {_ILLEGAL_ACTION_HINTS['SEIZE']}")
            return False

        self.game_state.seize(unit)
        logger.info("Unit %s is seizing structure at (%s, %s)", action.get("unit_id"), unit.x, unit.y)
        return True

    def _execute_resign(self):
        """Execute a RESIGN action - the bot concedes the game."""
        logger.info("LLM Bot (Player %s) has decided to resign.", self.bot_player)
        self.game_state.resign(self.bot_player)


class OpenAIBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using OpenAI's GPT models.

    Supports OpenAI GPT-5+ models:
    - GPT-5.2: Flagship reasoning model, most capable
    - GPT-5: gpt-5, gpt-5-mini, gpt-5-nano

    Default model: gpt-5-mini-2025-08-07 (good balance of cost and performance)

    Cost tiers:
    - Budget: gpt-5-nano, gpt-5-mini (~$0.15-0.50/1M input tokens)
    - Premium: gpt-5.2 (~$10-15/1M input tokens)

    The original GPT-5 models (gpt-5 / -mini / -nano), the -pro models and
    the o-series reject a non-default temperature with a 400, so
    ``temperature`` is ignored (with a warning) for them.
    """

    _env_var_name = "OPENAI_API_KEY"
    _default_model_name = "gpt-5-mini-2025-08-07"
    _supported_model_list = OPENAI_MODELS

    def __init__(self, *args, **kwargs):
        """Initialize OpenAIBot, dropping a temperature the model would reject."""
        super().__init__(*args, **kwargs)
        # A 400 is not retried, so sending it would stop the bot on its first
        # turn (LLMBotError); dropping it only changes sampling.
        if self.temperature is not None and not openai_model_accepts_temperature(self.model):
            logger.warning(
                "Model '%s' only supports the default temperature; ignoring temperature=%s.",
                self.model,
                self.temperature,
            )
            # Cleared rather than just not sent, so conversation logs, the
            # GUI's player config and tournament replays record the value used.
            self.temperature = None

    def _get_llm_sdk_version(self) -> str:
        """Get the OpenAI SDK version."""
        try:
            import openai

            return f"openai=={openai.__version__}"
        except (ImportError, AttributeError):
            return "openai==unknown"

    def _create_client(self) -> Any:
        """Build the OpenAI client once per bot."""
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package not installed. Install with: pip install openai>=1.0.0") from exc

        # max_retries=0: retrying is _call_llm_with_retry's job, so every
        # attempt is classified, logged and backed off by one policy. SDK
        # retries underneath it would multiply the attempts per turn.
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout
        return openai.OpenAI(**client_kwargs)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call OpenAI API."""
        # Build request kwargs, conditionally including max_completion_tokens and temperature
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self.max_tokens is not None:
            request_kwargs["max_completion_tokens"] = self.max_tokens
        if self.temperature is not None and openai_model_accepts_temperature(self.model):
            request_kwargs["temperature"] = self.temperature

        response = self._client.chat.completions.create(**request_kwargs)

        with _reading_response(response):
            # Capture token usage from OpenAI API response
            if response.usage:
                self._last_input_tokens = response.usage.prompt_tokens
                self._last_output_tokens = response.usage.completion_tokens

            # Capture finish reason from OpenAI API response
            if response.choices and response.choices[0].finish_reason:
                self._last_stop_reason = response.choices[0].finish_reason

            if not response.choices:
                return ""
            # content is None on a refusal; "" makes take_turn pass the turn.
            return response.choices[0].message.content or ""


class ClaudeBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using Anthropic's Claude models.

    Supports the models in ANTHROPIC_MODELS:
    - Claude Fable 5.1 / 5 (claude-fable-5-1, claude-fable-5)
    - Claude Opus 5.5 / 5 (claude-opus-5-5, claude-opus-5)
    - Claude Sonnet 5 (claude-sonnet-5)
    - Claude Opus 4.8 / 4.7 / 4.6, Sonnet 4.6
    - Claude Haiku 4.5 (claude-haiku-4-5-20251001)
    - Older, still served: Opus 4.5, Sonnet 4.5

    Default model: claude-haiku-4-5-20251001 (fast and economical)

    Request constraints on current models:
    - No assistant prefill: a conversation ending on an assistant turn is
      rejected with a 400 by Opus 4.6+, Sonnet 4.6+ and every 5.x model.
      JSON output comes from the system-prompt instruction instead.
    - Opus 4.7, Opus 4.8 and every 5.x model reject temperature/top_p/top_k,
      so ``temperature`` is ignored (with a warning) for them. Older models
      get it through ``extra_body``: the anthropic 1.x SDK has no
      ``temperature=`` keyword.
    """

    _env_var_name = "ANTHROPIC_API_KEY"
    _default_model_name = "claude-haiku-4-5-20251001"
    _supported_model_list = ANTHROPIC_MODELS
    _unavailable_model_notes = ANTHROPIC_UNAVAILABLE_MODELS

    def __init__(self, *args, **kwargs):
        """Initialize ClaudeBot, dropping a temperature the model would reject."""
        super().__init__(*args, **kwargs)
        if self.temperature is not None and not claude_model_accepts_sampling_params(self.model):
            logger.warning(
                "Model '%s' rejects sampling parameters (temperature/top_p/top_k); "
                "ignoring temperature=%s and using the model's default.",
                self.model,
                self.temperature,
            )
            # Cleared rather than just not sent, so conversation logs, the
            # GUI's player config and tournament replays record the value used.
            self.temperature = None

    def _get_llm_sdk_version(self) -> str:
        """Get the Anthropic SDK version."""
        try:
            import anthropic

            return f"anthropic=={anthropic.__version__}"
        except (ImportError, AttributeError):
            return "anthropic==unknown"

    def _create_client(self) -> Any:
        """Build the Anthropic client once per bot."""
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic>=0.18.0") from exc

        # max_retries=0: retrying is _call_llm_with_retry's job (see OpenAIBot).
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout
        return anthropic.Anthropic(**client_kwargs)

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call Anthropic API."""
        # Extract system message and conversation messages
        system_message = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                chat_messages.append(msg)

        # No assistant prefill (e.g. a trailing {"role": "assistant",
        # "content": "{"}): Opus 4.6+, Sonnet 4.6+ and every 5.x model reject
        # a conversation ending on an assistant turn with a 400. The system
        # prompt asks for JSON only, and _extract_json tolerates prose or
        # code fences around the object.
        # Anthropic API requires max_tokens so default to 4096 (0 would 400 too).
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": self.max_tokens or 4096,
        }
        if system_message:
            request_kwargs["system"] = system_message
        if self.temperature is not None and claude_model_accepts_sampling_params(self.model):
            # Through extra_body, not temperature=: anthropic 1.x dropped the
            # sampling keywords from messages.create (passing one is a
            # TypeError before any request is sent), while the API still
            # honours them on these models. Both 0.x and 1.x SDKs merge
            # extra_body into the request JSON as-is.
            request_kwargs["extra_body"] = {"temperature": self.temperature}

        response = self._client.messages.create(**request_kwargs)

        with _reading_response(response):
            # Capture token usage from Claude API response
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._last_input_tokens = usage.input_tokens
                self._last_output_tokens = usage.output_tokens

            # Capture stop reason from Claude API response
            if response.stop_reason:
                self._last_stop_reason = response.stop_reason

            # The reply is a list of content blocks, not always one text block
            # (it can be empty, or hold thinking/tool blocks), so join every
            # text block instead of indexing content[0].
            return "".join(block.text for block in response.content or [] if getattr(block, "type", None) == "text")


class GeminiBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using Google's Gemini models via the google-genai SDK.

    Supports Gemini 2.5+ models:
    - Gemini 3.0: Latest generation (gemini-3-pro-preview, gemini-3-flash-preview)
    - Gemini 2.5: Production models with thinking (gemini-2.5-pro, gemini-2.5-flash)

    Default model: gemini-2.5-flash (production Flash with thinking capabilities)

    Cost tiers:
    - Budget: gemini-2.5-flash-lite (~$0.075/1M input tokens)
    - Standard: gemini-2.5-flash, gemini-3-flash-preview (~$0.15/1M input tokens)
    - Premium: gemini-2.5-pro, gemini-3-pro-preview (~$1.25/1M input tokens)

    Token limits:
    - Gemini 3.0/2.5: Up to 1M token context window

    Best use cases:
    - gemini-3-flash-preview: Latest generation, fast, frontier-class
    - gemini-2.5-flash: Best production balance of speed and quality
    - gemini-2.5-pro: Complex reasoning tasks
    """

    _env_var_name = "GOOGLE_API_KEY"
    _default_model_name = "gemini-2.5-flash"
    _supported_model_list = GEMINI_MODELS

    def __init__(self, *args, **kwargs):
        """Initialize GeminiBot with optional chat session for stateful mode."""
        super().__init__(*args, **kwargs)
        self._chat_session = None

    def _get_llm_sdk_version(self) -> str:
        """Get the Google GenAI SDK version."""
        try:
            from google import genai

            return f"google-genai=={genai.__version__}"
        except (ImportError, AttributeError):
            return "google-genai==unknown"

    def _create_client(self) -> Any:
        """Build the Gemini client once per bot."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai") from exc

        http_options = None
        if self.request_timeout is not None:
            # google-genai takes the timeout in milliseconds.
            http_options = types.HttpOptions(timeout=int(self.request_timeout * 1000))
        return genai.Client(api_key=self.api_key, http_options=http_options)

    def _get_client(self):
        """Get (or, if it was cleared, rebuild) the Gemini client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call Google Gemini API using the new google-genai SDK."""
        try:
            from google.genai import types
        except ImportError as exc:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai") from exc

        client = self._get_client()

        # Extract system instruction and build conversation contents
        system_instruction = None
        contents = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
            elif msg["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])]))

        # Build generation config, conditionally including max_output_tokens and temperature
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
        }
        if self.max_tokens is not None:
            config_kwargs["max_output_tokens"] = self.max_tokens
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature

        config = types.GenerateContentConfig(**config_kwargs)

        # API errors propagate to _call_llm_with_retry, which classifies,
        # logs and retries them.
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        with _reading_response(response):
            # Track token usage from response metadata
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                self._last_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                self._last_output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            # Capture finish reason from Gemini API response
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    # Convert enum to string if necessary
                    finish_reason = candidate.finish_reason
                    self._last_stop_reason = str(finish_reason.name) if hasattr(finish_reason, "name") else str(finish_reason)

            # Blocked or empty responses return "" rather than a synthetic
            # END_TURN reply: take_turn still passes the turn, but logs and
            # counts it (llm_empty_reply) as the model's own failure, where
            # a synthetic END_TURN looked like a deliberate pass.
            if not response.text:
                block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
                if block_reason:
                    logger.warning("Gemini response blocked: %s", block_reason)
                    self._last_stop_reason = self._last_stop_reason or f"blocked: {block_reason}"
                else:
                    logger.warning("Empty response from Gemini API")
                return ""

            return response.text
