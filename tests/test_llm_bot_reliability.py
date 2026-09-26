"""Reliability tests for the LLM bots (review §1.9: aibots-1, -3, -4, -14).

Covers the Claude request shape (no assistant prefill, no sampling params
for models that reject them, temperature via extra_body for the rest, text
read from every content block), OpenAI temperature handling, client timeouts,
error classification and retry policy for all three providers, the
failed-turn limit that raises LLMBotError (unparseable replies included),
how a tournament records such a game, and the LLM-side legality check that
stops repeated SEIZEs, friendly fire and repeated attacks.

No network: each provider SDK is replaced by a fake module in sys.modules
(the fake anthropic client takes exactly the anthropic 1.x keywords).
New names are reached through the ``llm_bot`` module object rather than
imported at the top, so each test fails on its own against older code.
"""

import inspect
import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.unit import Unit
from reinforcetactics.game import llm_bot
from reinforcetactics.game.llm_bot import ClaudeBot, GeminiBot, LLMBot, OpenAIBot

END_TURN_REPLY = json.dumps({"actions": [{"type": "END_TURN"}]})

# Player 2 owns the building at (8, 9); creating a Warrior there is legal
# on the first player-2 turn, so a reply containing it shows the reply was
# parsed and executed.
CREATE_REPLY = json.dumps({"actions": [{"type": "CREATE_UNIT", "unit_type": "W", "position": [8, 9]}]})


# ---------------------------------------------------------------------------
# Game fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def game():
    """10x10 plains; HQs at (0, 0) for P1 and (9, 9) for P2; P2 building at (8, 9)."""
    map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[0][0] = "h_1"
    map_data[9][9] = "h_2"
    map_data[0][1] = "b_1"
    map_data[9][8] = "b_2"
    state = GameState(map_data, num_players=2)
    state.player_gold[1] = 5000
    state.player_gold[2] = 5000
    return state


def _place(game: GameState, unit_type: str, x: int, y: int, player: int) -> Unit:
    """Put a unit on the board without create_unit's building/gold rules."""
    unit = Unit(unit_type, x, y, player, stats=game.unit_data[unit_type])
    unit.unit_id = game._next_unit_id
    game._next_unit_id += 1
    game.units.append(unit)
    game._invalidate_cache()
    return unit


def _start_player2_turn(game: GameState) -> None:
    """End player 1's turn; player 2's units become ready to act."""
    game.end_turn()
    assert game.current_player == 2


def _count_actions(game: GameState, action_type: str) -> int:
    return sum(1 for a in game.action_history if a.get("type") == action_type)


class ScriptedBot(LLMBot):
    """LLM bot whose "API" returns scripted replies or raises scripted errors."""

    _env_var_name = "TEST_API_KEY"
    _default_model_name = "test-model"
    _supported_model_list = ["test-model"]

    def __init__(self, *args, script: list[Any] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.script = list(script or [])
        self.calls = 0

    def _call_llm(self, messages):
        self.calls += 1
        item = self.script.pop(0) if self.script else END_TURN_REPLY
        if isinstance(item, BaseException):
            raise item
        return item

    def _get_llm_sdk_version(self):
        return "test-sdk"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff sleeps instead of waiting."""
    recorded: list[float] = []
    monkeypatch.setattr(llm_bot.time, "sleep", recorded.append)
    return recorded


# ---------------------------------------------------------------------------
# Fake provider SDKs
# ---------------------------------------------------------------------------


class FakeServer:
    """Shared script and call log behind one fake SDK module."""

    def __init__(self, default_response: Any):
        self.default_response = default_response
        self.script: list[Any] = []
        self.calls: list[dict[str, Any]] = []
        self.clients: list[dict[str, Any]] = []

    def respond(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else self.default_response
        if isinstance(item, BaseException):
            raise item
        return item


class FakeHTTPError(Exception):
    """Mimics openai/anthropic APIStatusError: ``status_code`` plus ``response.headers``."""

    def __init__(self, status_code: int, headers: dict[str, str] | None = None):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class APIConnectionError(Exception):
    """Same class name as the SDKs' transport error, which carries no status."""


class UnknownApiResponseError(ValueError):
    """Same name and base as google-genai's error for a reply that isn't JSON."""


class FakeGenaiAPIError(Exception):
    """Mimics google.genai.errors.APIError: ``code`` plus the JSON body in ``details``."""

    def __init__(self, code: int, details: dict[str, Any] | None = None):
        super().__init__(f"{code} error")
        self.code = code
        self.details = details or {}
        self.response = None


def claude_response(*blocks: Any, stop_reason: str = "end_turn") -> SimpleNamespace:
    content = [SimpleNamespace(type="text", text=b) if isinstance(b, str) else b for b in blocks]
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        stop_reason=stop_reason,
    )


def openai_response(content: str | None, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


# Keyword arguments of messages.create in the anthropic SDK 1.x (1.8.0),
# which `pip install anthropic` resolves to today. 1.x removed temperature,
# top_p and top_k, so passing one is a TypeError raised before any request;
# a fake that accepted **kw hid exactly that bug.
ANTHROPIC_1X_CREATE_PARAMS = frozenset(
    {
        "max_tokens",
        "messages",
        "model",
        "cache_control",
        "container",
        "inference_geo",
        "metadata",
        "output_config",
        "service_tier",
        "stop_sequences",
        "stream",
        "system",
        "thinking",
        "tool_choice",
        "tools",
        "user_profile_id",
        "workspace_id",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
    }
)


@pytest.fixture
def fake_anthropic(monkeypatch):
    server = FakeServer(claude_response(END_TURN_REPLY))

    def create(**kw):
        """messages.create with the anthropic 1.x signature."""
        unexpected = sorted(set(kw) - ANTHROPIC_1X_CREATE_PARAMS)
        if unexpected:
            raise TypeError(f"Messages.create() got an unexpected keyword argument '{unexpected[0]}'")
        missing = sorted({"max_tokens", "messages", "model"} - set(kw))
        if missing:
            raise TypeError(f"Messages.create() missing required keyword argument '{missing[0]}'")
        return server.respond(kw)

    class Anthropic:
        def __init__(self, **kwargs):
            server.clients.append(kwargs)
            self.messages = SimpleNamespace(create=create)

    module = types.ModuleType("anthropic")
    module.__version__ = "0.0.0-test"  # type: ignore[attr-defined]
    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return server


@pytest.fixture
def fake_openai(monkeypatch):
    server = FakeServer(openai_response(END_TURN_REPLY))

    class OpenAI:
        def __init__(self, **kwargs):
            server.clients.append(kwargs)
            completions = SimpleNamespace(create=lambda **kw: server.respond(kw))
            self.chat = SimpleNamespace(completions=completions)

    module = types.ModuleType("openai")
    module.__version__ = "0.0.0-test"  # type: ignore[attr-defined]
    module.OpenAI = OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return server


@pytest.fixture
def fake_genai(monkeypatch):
    server = FakeServer(SimpleNamespace(text=END_TURN_REPLY, usage_metadata=None, candidates=[], prompt_feedback=None))

    class Client:
        def __init__(self, **kwargs):
            server.clients.append(kwargs)
            self.models = SimpleNamespace(generate_content=lambda **kw: server.respond(kw))

    types_module = types.ModuleType("google.genai.types")
    types_module.Content = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    types_module.Part = SimpleNamespace(from_text=lambda text: SimpleNamespace(text=text))  # type: ignore[attr-defined]
    types_module.GenerateContentConfig = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    types_module.HttpOptions = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
    genai_module = types.ModuleType("google.genai")
    genai_module.__version__ = "0.0.0-test"  # type: ignore[attr-defined]
    genai_module.Client = Client  # type: ignore[attr-defined]
    genai_module.types = types_module  # type: ignore[attr-defined]
    google_module = types.ModuleType("google")
    google_module.genai = genai_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    return server


# ---------------------------------------------------------------------------
# aibots-3: the Claude request
# ---------------------------------------------------------------------------


class TestClaudeRequest:
    def test_no_assistant_prefill_and_full_reply_parsed(self, game, fake_anthropic):
        """The request ends on the user turn, and the reply isn't glued to a "{"."""
        _start_player2_turn(game)
        fake_anthropic.script.append(claude_response(CREATE_REPLY))
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test")

        bot.take_turn()

        request = fake_anthropic.calls[-1]
        assert [m["role"] for m in request["messages"]] == ["user"]
        assert all(m["content"] != "{" for m in request["messages"])
        assert "Reinforce Tactics" in request["system"]
        # The CREATE_UNIT in the reply ran, so it was parsed as sent.
        assert any(u.player == 2 and (u.x, u.y) == (8, 9) for u in game.units)

    def test_stateful_history_still_ends_on_user_turn(self, game, fake_anthropic):
        _start_player2_turn(game)
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test", stateful=True)
        bot.take_turn()
        game.end_turn()
        bot.take_turn()

        roles = [m["role"] for m in fake_anthropic.calls[-1]["messages"]]
        assert roles == ["user", "assistant", "user"]

    @pytest.mark.parametrize(
        "model",
        [
            "claude-fable-5-1",
            "claude-fable-5",
            "claude-opus-5-5",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
            "claude-opus-4-7",
        ],
    )
    def test_no_sampling_params_for_models_that_reject_them(self, game, fake_anthropic, model):
        _start_player2_turn(game)
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test", model=model, temperature=0.5)

        bot.take_turn()

        request = fake_anthropic.calls[-1]
        assert request["model"] == model
        assert not {"temperature", "top_p", "top_k"} & set(request)
        assert not {"temperature", "top_p", "top_k"} & set(request.get("extra_body") or {})
        # Cleared so logs and replays record what was actually used.
        assert bot.temperature is None

    @pytest.mark.parametrize(
        "model", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929"]
    )
    def test_temperature_reaches_models_that_accept_it_via_extra_body(self, game, fake_anthropic, sleeps, model):
        """anthropic 1.x has no temperature= keyword; extra_body works on 0.x and 1.x.

        Sent as a keyword, every request raised TypeError inside the SDK and
        each turn was passed after three attempts.
        """
        _start_player2_turn(game)
        fake_anthropic.script.append(claude_response(CREATE_REPLY))
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test", model=model, temperature=0.5)

        bot.take_turn()

        request = fake_anthropic.calls[-1]
        assert "temperature" not in request
        assert request["extra_body"] == {"temperature": 0.5}
        assert len(fake_anthropic.calls) == 1
        assert sleeps == []
        assert any(u.player == 2 and (u.x, u.y) == (8, 9) for u in game.units)

    def test_no_extra_body_without_temperature(self, game, fake_anthropic):
        _start_player2_turn(game)
        ClaudeBot(game, player=2, api_key="sk-ant-test").take_turn()
        assert "extra_body" not in fake_anthropic.calls[-1]

    def test_request_fits_installed_anthropic_sdk_signature(self, game):
        """With a real anthropic SDK installed, every keyword ClaudeBot sends is one it takes."""
        anthropic = pytest.importorskip("anthropic")
        signature = inspect.signature(anthropic.Anthropic(api_key="sk-ant-test").messages.create)
        requests = []  # keyword arguments of each create() call

        def create(**kw):
            signature.bind(**kw)  # TypeError on a keyword the installed SDK doesn't take
            requests.append(kw)
            return claude_response(END_TURN_REPLY)

        _start_player2_turn(game)
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test", model="claude-haiku-4-5-20251001", temperature=0.5)
        bot._client = SimpleNamespace(messages=SimpleNamespace(create=create))

        bot.take_turn()

        assert len(requests) == 1
        assert requests[0]["extra_body"] == {"temperature": 0.5}

    def test_text_read_from_every_text_block(self, game, fake_anthropic, sleeps):
        """A reply that starts with a non-text block and splits its text still parses."""
        _start_player2_turn(game)
        head, tail = CREATE_REPLY[:20], CREATE_REPLY[20:]
        thinking = SimpleNamespace(type="thinking", thinking="Build a warrior.")
        fake_anthropic.script.append(claude_response(thinking, head, tail))
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test")

        bot.take_turn()

        assert len(fake_anthropic.calls) == 1
        assert any(u.player == 2 and (u.x, u.y) == (8, 9) for u in game.units)

    def test_empty_content_is_a_failed_turn_not_a_crash(self, game, fake_anthropic, sleeps):
        _start_player2_turn(game)
        fake_anthropic.script.append(claude_response(stop_reason="max_tokens"))
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test")

        bot.take_turn()

        assert len(fake_anthropic.calls) == 1  # an empty reply isn't an error to retry
        assert bot.consecutive_failed_turns == 1
        assert game.current_player == 1

    def test_replies_truncated_at_max_tokens_hit_the_failed_turn_limit(self, game, fake_anthropic, sleeps):
        """Without the prefill, a reply cut off mid-JSON has no actions list; it's a failed turn."""
        _start_player2_turn(game)
        fake_anthropic.default_response = claude_response(CREATE_REPLY[:30], stop_reason="max_tokens")
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test")

        for _ in range(2):
            bot.take_turn()
            game.end_turn()
        with pytest.raises(llm_bot.LLMBotError, match="3 turns in a row"):
            bot.take_turn()

        assert len(fake_anthropic.calls) == 3
        assert not any(u.player == 2 for u in game.units)

    def test_retired_model_is_called_out(self, game, fake_anthropic, caplog):
        assert "claude-opus-4-1-20250805" not in llm_bot.ANTHROPIC_MODELS
        with caplog.at_level("WARNING", logger="reinforcetactics.game.llm_bot"):
            ClaudeBot(game, player=2, api_key="sk-ant-test", model="claude-opus-4-1-20250805")
        assert "retired" in caplog.text

    def test_default_model_unchanged(self):
        assert ClaudeBot._default_model_name == "claude-haiku-4-5-20251001"
        assert "claude-haiku-4-5-20251001" in llm_bot.ANTHROPIC_MODELS


# ---------------------------------------------------------------------------
# aibots-4: sampling parameters OpenAI reasoning models reject (a 400 is now
# fatal instead of silently passing turns)
# ---------------------------------------------------------------------------


class TestOpenAIRequest:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07", "gpt-5-2025-08-07", "gpt-5", "gpt-5-pro", "o3-mini", "o1"],
    )
    def test_no_temperature_for_models_that_reject_it(self, game, fake_openai, sleeps, model):
        """These 400 on a non-default temperature, which is now fatal (not retried) on turn 1."""
        _start_player2_turn(game)
        bot = OpenAIBot(game, player=2, api_key="sk-test", model=model, temperature=0.5)

        bot.take_turn()

        assert "temperature" not in fake_openai.calls[-1]
        assert bot.temperature is None

    @pytest.mark.parametrize("model", ["gpt-5.2", "gpt-5.1", "gpt-4.1", "gpt-5-chat-latest"])
    def test_temperature_sent_to_models_that_accept_it(self, game, fake_openai, model):
        _start_player2_turn(game)
        bot = OpenAIBot(game, player=2, api_key="sk-test", model=model, temperature=0.5)

        bot.take_turn()

        assert fake_openai.calls[-1]["temperature"] == 0.5
        assert bot.temperature == 0.5


# ---------------------------------------------------------------------------
# aibots-14 / aibots-4: SDK import and client construction
# ---------------------------------------------------------------------------


class TestClientConstruction:
    def test_claude_client_built_once_with_timeout(self, game, fake_anthropic):
        _start_player2_turn(game)
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test")
        bot.take_turn()
        game.end_turn()
        bot.take_turn()

        assert len(fake_anthropic.calls) == 2
        assert len(fake_anthropic.clients) == 1
        client_kwargs = fake_anthropic.clients[0]
        assert client_kwargs["timeout"] == llm_bot.DEFAULT_REQUEST_TIMEOUT_S
        # Retries belong to LLMBot's classified loop, not the SDK underneath it.
        assert client_kwargs["max_retries"] == 0

    def test_openai_client_built_once_with_timeout(self, game, fake_openai):
        _start_player2_turn(game)
        bot = OpenAIBot(game, player=2, api_key="sk-test", request_timeout=42.0)
        bot.take_turn()
        game.end_turn()
        bot.take_turn()

        assert len(fake_openai.calls) == 2
        assert fake_openai.clients == [{"api_key": "sk-test", "max_retries": 0, "timeout": 42.0}]

    def test_gemini_client_gets_timeout_in_milliseconds(self, game, fake_genai):
        GeminiBot(game, player=2, api_key="g-test", request_timeout=30.0)
        assert len(fake_genai.clients) == 1
        assert fake_genai.clients[0]["http_options"].timeout == 30_000

    @pytest.mark.parametrize(
        "max_tokens, expected_timeout",
        [(None, 300.0), (8_000, 300.0), (16_000, 450.0), (32_000, 900.0)],
    )
    def test_default_timeout_grows_with_max_tokens(self, game, fake_anthropic, max_tokens, expected_timeout):
        """A fixed 300 s cut off full-length 16K-token replies, then retried (and re-billed) them."""
        ClaudeBot(game, player=2, api_key="sk-ant-test", max_tokens=max_tokens)
        assert fake_anthropic.clients[0]["timeout"] == pytest.approx(expected_timeout)

    def test_explicit_timeout_is_used_as_given(self, game, fake_anthropic):
        ClaudeBot(game, player=2, api_key="sk-ant-test", max_tokens=32_000, request_timeout=42.0)
        ClaudeBot(game, player=2, api_key="sk-ant-test", request_timeout=None)
        assert fake_anthropic.clients[0]["timeout"] == 42.0
        assert "timeout" not in fake_anthropic.clients[1]  # None: the SDK's default
        with pytest.raises(ValueError, match="request_timeout"):
            ClaudeBot(game, player=2, api_key="sk-ant-test", request_timeout="soon")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bot_class, modules",
        [
            (OpenAIBot, ["openai"]),
            (ClaudeBot, ["anthropic"]),
            (GeminiBot, ["google", "google.genai"]),
        ],
    )
    def test_missing_sdk_raises_at_construction(self, game, monkeypatch, bot_class, modules):
        for name in modules:
            monkeypatch.setitem(sys.modules, name, None)  # makes `import name` raise ImportError
        with pytest.raises(ImportError, match="not installed"):
            bot_class(game, player=2, api_key="key")

    def test_bot_factory_falls_back_when_sdk_missing(self, game, monkeypatch):
        from reinforcetactics.app.bot_factory import create_bots_from_config
        from reinforcetactics.game.bot import SimpleBot

        monkeypatch.setitem(sys.modules, "anthropic", None)
        settings = SimpleNamespace(get_api_key=lambda provider: "sk-ant-test")
        configs = [{"type": "human"}, {"type": "computer", "bot_type": "ClaudeBot"}]

        bots = create_bots_from_config(game, configs, settings)

        assert isinstance(bots[2], SimpleBot)
        assert configs[1]["player_name"] == "SimpleBot"


# ---------------------------------------------------------------------------
# aibots-4: error classification, retries and the failed-turn limit
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
    def test_claude_non_retryable_errors_fail_fast(self, game, fake_anthropic, sleeps, status):
        _start_player2_turn(game)
        fake_anthropic.script.append(FakeHTTPError(status))
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test")

        with pytest.raises(llm_bot.LLMBotError) as excinfo:
            bot.take_turn()

        assert excinfo.value.retryable is False
        assert str(status) in str(excinfo.value)
        assert len(fake_anthropic.calls) == 1
        assert sleeps == []
        # The turn is left for the caller to decide about, not silently passed.
        assert game.current_player == 2

    def test_openai_auth_error_fails_fast(self, game, fake_openai, sleeps):
        _start_player2_turn(game)
        fake_openai.script.append(FakeHTTPError(401))
        bot = OpenAIBot(game, player=2, api_key="sk-bad")

        with pytest.raises(llm_bot.LLMBotError, match="authentication"):
            bot.take_turn()
        assert len(fake_openai.calls) == 1
        assert sleeps == []

    def test_gemini_not_found_fails_fast(self, game, fake_genai, sleeps):
        _start_player2_turn(game)
        fake_genai.script.append(FakeGenaiAPIError(404))
        bot = GeminiBot(game, player=2, api_key="g-test", model="gemini-0-retired")

        with pytest.raises(llm_bot.LLMBotError, match="not found"):
            bot.take_turn()
        assert len(fake_genai.calls) == 1
        assert sleeps == []

    def test_retryable_errors_are_retried_then_succeed(self, game, fake_anthropic, sleeps):
        _start_player2_turn(game)
        fake_anthropic.script.extend(
            [
                FakeHTTPError(429, headers={"retry-after": "7"}),
                FakeHTTPError(529),  # Anthropic "overloaded"
                APIConnectionError("connection reset"),
                claude_response(CREATE_REPLY),
            ]
        )
        bot = ClaudeBot(game, player=2, api_key="sk-ant-test", max_retries=4)

        bot.take_turn()

        assert len(fake_anthropic.calls) == 4
        assert any(u.player == 2 and (u.x, u.y) == (8, 9) for u in game.units)
        assert bot.consecutive_failed_turns == 0
        assert game.current_player == 1
        # Retry-After honoured on the 429; jittered exponential backoff after.
        assert len(sleeps) == 3
        assert sleeps[0] >= 7
        assert 1.0 <= sleeps[1] <= 2.0
        assert 2.0 <= sleeps[2] <= 4.0

    def test_exhausted_retries_pass_the_turn_once(self, game, sleeps):
        _start_player2_turn(game)
        bot = ScriptedBot(game, player=2, api_key="k", max_retries=2, script=[TimeoutError(), TimeoutError()])

        bot.take_turn()

        assert bot.calls == 2
        assert len(sleeps) == 1
        assert game.current_player == 1
        assert bot.consecutive_failed_turns == 1
        assert bot.get_capabilities_fired()["llm_failed_turn"] == 1

    def test_llm_bot_error_after_consecutive_failed_turns(self, game, sleeps):
        _start_player2_turn(game)
        outage = [FakeHTTPError(503)] * 10
        bot = ScriptedBot(game, player=2, api_key="k", max_retries=1, script=outage)

        for _ in range(2):
            bot.take_turn()  # a blip: the turn passes
            assert game.current_player == 1
            game.end_turn()

        with pytest.raises(llm_bot.LLMBotError) as excinfo:
            bot.take_turn()

        assert excinfo.value.retryable is True
        assert "3 turns in a row" in str(excinfo.value)
        assert game.current_player == 2

    def test_a_good_turn_resets_the_failure_streak(self, game, sleeps):
        _start_player2_turn(game)
        script = [TimeoutError(), TimeoutError(), END_TURN_REPLY, TimeoutError(), TimeoutError()]
        bot = ScriptedBot(game, player=2, api_key="k", max_retries=1, script=script)

        for _ in range(5):
            bot.take_turn()
            game.end_turn()

        assert bot.consecutive_failed_turns == 2

    def test_failed_turn_limit_can_be_disabled(self, game, sleeps):
        _start_player2_turn(game)
        bot = ScriptedBot(
            game, player=2, api_key="k", max_retries=1, max_consecutive_failed_turns=None, script=[TimeoutError()] * 5
        )
        for _ in range(5):
            bot.take_turn()
            game.end_turn()
        assert bot.consecutive_failed_turns == 5

    def test_gemini_blocked_responses_count_as_failed_turns(self, game, fake_genai, sleeps):
        """Blocked replies used to become a synthetic END_TURN, passing turns forever."""
        _start_player2_turn(game)
        blocked = SimpleNamespace(
            text="", usage_metadata=None, candidates=[], prompt_feedback=SimpleNamespace(block_reason="SAFETY")
        )
        fake_genai.default_response = blocked
        bot = GeminiBot(game, player=2, api_key="g-test")

        for _ in range(2):
            bot.take_turn()
            game.end_turn()
        with pytest.raises(llm_bot.LLMBotError):
            bot.take_turn()

    @pytest.mark.parametrize(
        "error, retryable",
        [
            (ImportError("no sdk"), False),
            (FakeHTTPError(400), False),
            (FakeHTTPError(401), False),
            (FakeHTTPError(403), False),
            (FakeHTTPError(404), False),
            (FakeHTTPError(408), True),
            (FakeHTTPError(409), True),
            (FakeHTTPError(429), True),
            (FakeHTTPError(500), True),
            (FakeHTTPError(503), True),
            (FakeHTTPError(529), True),
            (FakeGenaiAPIError(429), True),
            (FakeGenaiAPIError(400), False),
            (APIConnectionError("reset"), True),
            (TimeoutError(), True),
            (ConnectionResetError(), True),
            (RuntimeError("something unexpected"), True),
            # Raised in-process before any request: same result every time.
            (TypeError("Messages.create() got an unexpected keyword argument 'temperature'"), False),
            (AttributeError("module 'openai' has no attribute 'OpenAI'"), False),
            (ValueError("temperature: Input should be less than or equal to 2"), False),
            # A reply the SDK couldn't decode may be a garbled one-off.
            (json.JSONDecodeError("Expecting value", "<html>", 0), True),
            (UnknownApiResponseError("response is not JSON"), True),
        ],
    )
    def test_classification(self, error, retryable):
        assert llm_bot._classify_llm_error(error).retryable is retryable

    def test_local_sdk_error_fails_fast(self, game, sleeps):
        """A TypeError from the SDK call used to be retried and then passed, turn after turn."""
        _start_player2_turn(game)
        error = TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
        bot = ScriptedBot(game, player=2, api_key="k", script=[error] * 3)

        with pytest.raises(llm_bot.LLMBotError, match="unexpected keyword argument") as excinfo:
            bot.take_turn()

        assert excinfo.value.retryable is False
        assert bot.calls == 1
        assert sleeps == []
        assert game.current_player == 2

    def test_unparseable_replies_count_toward_the_limit(self, game, sleeps):
        """Prose, truncated or non-object JSON used to reset the streak, so the bot passed forever."""
        _start_player2_turn(game)
        replies = [
            "I will build a warrior.",  # no JSON at all
            '{"actions": [{"type": "MOVE", "unit_id": 0, "to": [1',  # cut off at max_tokens
            '["END_TURN"]',  # JSON, but not an object with an actions list
        ]
        bot = ScriptedBot(game, player=2, api_key="k", script=replies)

        for _ in range(2):
            bot.take_turn()
            assert game.current_player == 1  # a single bad reply still passes the turn
            game.end_turn()
        with pytest.raises(llm_bot.LLMBotError, match="3 turns in a row"):
            bot.take_turn()

        assert bot.calls == 3  # the requests succeeded, so nothing was retried
        assert sleeps == []
        assert bot.get_capabilities_fired()["llm_unparseable_reply"] == 3

    def test_parsed_reply_resets_the_streak_even_if_every_action_is_illegal(self, game):
        _start_player2_turn(game)
        illegal_only = json.dumps({"actions": [{"type": "SEIZE", "unit_id": 0}]})  # P2 has no units
        script = ["no json", "no json", illegal_only, "no json", "no json"]
        bot = ScriptedBot(game, player=2, api_key="k", script=script)

        for _ in range(5):
            bot.take_turn()
            game.end_turn()

        assert bot.consecutive_failed_turns == 2
        assert bot.illegal_action_count == 1

    def test_empty_actions_list_is_a_usable_reply(self, game):
        _start_player2_turn(game)
        bot = ScriptedBot(game, player=2, api_key="k", script=['{"actions": []}'] * 5)
        for _ in range(5):
            bot.take_turn()
            game.end_turn()
        assert bot.consecutive_failed_turns == 0

    def test_retry_after_sources(self):
        assert llm_bot._retry_after_seconds(FakeHTTPError(429, {"retry-after": "12"})) == 12.0
        assert llm_bot._retry_after_seconds(FakeHTTPError(429, {"retry-after-ms": "2500"})) == 2.5
        assert llm_bot._retry_after_seconds(FakeHTTPError(429, {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})) == 0.0
        assert llm_bot._retry_after_seconds(FakeHTTPError(429)) is None
        genai_body = {
            "error": {
                "code": 429,
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31s"}],
            }
        }
        assert llm_bot._retry_after_seconds(FakeGenaiAPIError(429, genai_body)) == 31.0

    def test_backoff_is_jittered_exponential_and_capped(self, game):
        bot = ScriptedBot(game, player=2, api_key="k")
        for attempt in range(8):
            ceiling = min(llm_bot._RETRY_MAX_DELAY_S, llm_bot._RETRY_BASE_DELAY_S * 2**attempt)
            delays = {bot._retry_delay(attempt, None) for _ in range(20)}
            assert all(ceiling / 2 <= d <= ceiling for d in delays)
            assert len(delays) > 1  # jittered, not fixed
        # A long Retry-After wins over the backoff, but is capped.
        assert bot._retry_delay(0, 10.0) >= 10.0
        assert bot._retry_delay(0, 10_000.0) == llm_bot._RETRY_AFTER_CAP_S


# ---------------------------------------------------------------------------
# aibots-1: LLM-side legality
# ---------------------------------------------------------------------------


class TestActionLegality:
    def test_repeated_seize_cannot_capture_hq_in_one_turn(self, game):
        hq = game.grid.get_tile(0, 0)
        assert hq.type == "h" and hq.player == 1
        _place(game, "W", 0, 0, player=2)  # a 15-HP Warrior on the 50-HP enemy HQ
        _start_player2_turn(game)
        reply = json.dumps({"actions": [{"type": "SEIZE", "unit_id": 0}] * 8})
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        assert not game.game_over
        assert hq.player == 1
        assert _count_actions(game, "seize") == 1
        assert bot.illegal_action_count == 7
        assert bot.get_capabilities_fired()["llm_illegal_seize"] == 7

    def test_friendly_fire_and_repeated_attacks_rejected(self, game):
        knight = _place(game, "K", 5, 5, player=2)
        friend = _place(game, "W", 5, 6, player=2)
        enemy = _place(game, "W", 6, 5, player=1)
        _start_player2_turn(game)
        reply = json.dumps(
            {
                "actions": [
                    {"type": "ATTACK", "unit_id": 0, "target_position": [5, 6]},  # own Warrior
                    {"type": "ATTACK", "unit_id": 0, "target_position": [6, 5]},
                    {"type": "ATTACK", "unit_id": 0, "target_position": [6, 5]},  # already attacked
                ]
            }
        )
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        assert friend.health == friend.max_health
        assert _count_actions(game, "attack") == 1
        assert [a["attacker_unit_id"] for a in game.action_history if a.get("type") == "attack"] == [knight.unit_id]
        assert enemy.health < enemy.max_health
        assert bot.illegal_action_count == 2
        assert bot.get_capabilities_fired()["llm_illegal_attack"] == 2

    def test_out_of_range_attack_rejected(self, game):
        _place(game, "W", 2, 2, player=2)
        enemy = _place(game, "W", 7, 7, player=1)
        _start_player2_turn(game)
        reply = json.dumps({"actions": [{"type": "ATTACK", "unit_id": 0, "target_position": [7, 7]}]})
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        assert enemy.health == enemy.max_health
        assert _count_actions(game, "attack") == 0
        assert bot.illegal_action_count == 1

    def test_repeated_heal_applies_once(self, game):
        _place(game, "C", 3, 3, player=2)
        ally = _place(game, "W", 3, 4, player=2)
        ally.health = 3
        _start_player2_turn(game)
        reply = json.dumps({"actions": [{"type": "HEAL", "unit_id": 0, "target_position": [3, 4]}] * 3})
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        assert _count_actions(game, "heal") == 1
        assert bot.illegal_action_count == 2

    def test_move_then_attack_in_one_turn_still_works(self, game):
        """The legality check re-reads the state, so a legal sequence isn't rejected."""
        warrior = _place(game, "W", 2, 5, player=2)
        enemy = _place(game, "W", 4, 5, player=1)
        _start_player2_turn(game)
        reply = json.dumps(
            {
                "actions": [
                    {"type": "MOVE", "unit_id": 0, "from": [2, 5], "to": [3, 5]},
                    {"type": "ATTACK", "unit_id": 0, "target_position": [4, 5]},
                ]
            }
        )
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        assert (warrior.x, warrior.y) == (3, 5)
        assert enemy.health < enemy.max_health
        assert bot.illegal_action_count == 0
        assert bot.get_capabilities_fired()["llm_action_executed"] == 2

    def test_move_with_wrong_from_rejected(self, game):
        warrior = _place(game, "W", 2, 5, player=2)
        _start_player2_turn(game)
        reply = json.dumps({"actions": [{"type": "MOVE", "unit_id": 0, "from": [7, 7], "to": [3, 5]}]})
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        assert (warrior.x, warrior.y) == (2, 5)
        assert bot.illegal_action_count == 1

    def test_actions_after_game_over_ignored(self, game):
        hq = game.grid.get_tile(0, 0)
        hq.health = 1
        _place(game, "W", 0, 0, player=2)
        _start_player2_turn(game)
        reply = json.dumps(
            {
                "actions": [
                    {"type": "SEIZE", "unit_id": 0},
                    {"type": "CREATE_UNIT", "unit_type": "W", "position": [8, 9]},
                ]
            }
        )
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        assert game.game_over and game.winner == 2
        # Checked on the board: record_action drops post-game-over records,
        # so the action history wouldn't show a unit created after the win.
        assert not any(u.player == 2 and (u.x, u.y) == (8, 9) for u in game.units)

    def test_unknown_and_malformed_actions_counted(self, game):
        _start_player2_turn(game)
        reply = json.dumps({"actions": [{"type": "HASTE", "unit_id": 0}, "SEIZE", {"type": "SEIZE"}]})
        bot = ScriptedBot(game, player=2, api_key="k", script=[reply])

        bot.take_turn()

        stats = bot.get_capabilities_fired()
        assert bot.illegal_action_count == 3
        assert stats["llm_illegal_other"] == 2
        assert stats["llm_illegal_seize"] == 1


# ---------------------------------------------------------------------------
# aibots-4: a tournament surfaces a broken LLM bot instead of scoring it
# ---------------------------------------------------------------------------


def _run_tournament(tmp_path, llm_descriptor, *, max_turns: int = 4, save_replays: bool = False):
    """SimpleBot vs ``llm_descriptor``, one game per side on the starter map."""
    from reinforcetactics.tournament import BotDescriptor, MapConfig, TournamentConfig, TournamentRunner

    config = TournamentConfig(
        name="llm_errors",
        maps=[MapConfig(path="maps/1v1/starter.csv", max_turns=max_turns)],
        games_per_side=1,
        max_turns=max_turns,
        save_replays=save_replays,
        replay_dir=str(tmp_path / "replays"),
        output_dir=str(tmp_path / "out"),
        llm_api_delay=0,
    )
    return TournamentRunner(config).run([BotDescriptor.simple_bot("SimpleBot"), llm_descriptor])


def _claude_descriptor(model: str = "claude-haiku-4-5-20251001", **kwargs):
    from reinforcetactics.tournament import BotDescriptor

    return BotDescriptor.llm_bot("Claude", "anthropic", model, api_key="sk-ant-test", **kwargs)


class TestTournamentErrors:
    @staticmethod
    def _game(game_id, bot1, bot2, winner, error=None):
        from reinforcetactics.tournament import GameResult

        winner_name = "Error" if error else {0: "Draw", 1: bot1, 2: bot2}[winner]
        return GameResult(game_id, bot1, bot2, winner, winner_name, turns=10, map_name="m.csv", error=error)

    def test_errored_game_is_not_scored_as_a_draw(self):
        """The runner reports errors as winner 0, which used to count as a draw with an Elo update."""
        from reinforcetactics.tournament import TournamentResults

        finished = self._game(2, "SimpleBot", "Claude", winner=1)
        results = TournamentResults()
        results.add_game_result(self._game(1, "Claude", "SimpleBot", winner=0, error="LLMBotError: HTTP 401"))
        results.add_game_result(finished)
        reference = TournamentResults()
        reference.add_game_result(finished)

        standings = {s.bot_name: s for s in results.get_standings()}
        assert (standings["Claude"].wins, standings["Claude"].losses, standings["Claude"].draws) == (0, 1, 0)
        assert (standings["SimpleBot"].wins, standings["SimpleBot"].losses, standings["SimpleBot"].draws) == (1, 0, 0)
        assert standings["Claude"].errors == standings["SimpleBot"].errors == 1
        assert standings["Claude"].win_rate == 0.0 and standings["Claude"].total_games == 1
        assert [m.draws for m in results.get_matchups()] == [0]
        for bot in ("Claude", "SimpleBot"):
            assert results.elo_system.get_rating(bot) == reference.elo_system.get_rating(bot)
        data = results.to_dict()
        assert data["total_games"] == 2 and data["errored_games"] == 1
        assert {s["bot"]: s["errors"] for s in data["standings"]} == {"Claude": 1, "SimpleBot": 1}

    def test_bot_with_only_errored_games_is_still_listed(self):
        from reinforcetactics.tournament import TournamentResults

        results = TournamentResults()
        results.add_game_result(self._game(1, "Claude", "SimpleBot", winner=0, error="boom"))

        standings = {s.bot_name: s for s in results.get_standings()}
        assert set(standings) == {"Claude", "SimpleBot"}
        assert standings["Claude"].errors == 1 and standings["Claude"].total_games == 0

    def test_llm_bot_error_mid_game_is_recorded_as_an_error(self, tmp_path, fake_anthropic, sleeps):
        """A rejected API key fails fast; the games show the error rather than two draws."""
        fake_anthropic.default_response = FakeHTTPError(401)

        results = _run_tournament(tmp_path, _claude_descriptor())

        assert len(results.game_results) == 2
        assert all(g.error and "authentication" in g.error for g in results.game_results)
        assert len(fake_anthropic.calls) == 2  # one request per game, not retried
        assert sleeps == []
        standings = {s.bot_name: s for s in results.get_standings()}
        assert standings["Claude"].errors == 2 and standings["Claude"].total_games == 0
        assert results.elo_system.get_rating("Claude") == results.elo_system.get_rating("SimpleBot")

    def test_missing_sdk_does_not_abort_a_sequential_tournament(self, tmp_path, monkeypatch):
        """Bot construction ran outside the runner's try, so this ImportError ended the tournament."""
        monkeypatch.setitem(sys.modules, "anthropic", None)

        results = _run_tournament(tmp_path, _claude_descriptor())

        assert len(results.game_results) == 2
        assert all(g.error and "not installed" in g.error for g in results.game_results)
        assert {s.bot_name: s.errors for s in results.get_standings()} == {"SimpleBot": 2, "Claude": 2}

    def test_replay_records_the_temperature_actually_used(self, tmp_path, fake_anthropic):
        """Opus 4.7 rejects temperature, so ClaudeBot drops it; the replay said 0.5 anyway."""
        _run_tournament(tmp_path, _claude_descriptor("claude-opus-4-7", temperature=0.5), save_replays=True)

        replays = sorted((tmp_path / "replays").rglob("*.json"))
        assert len(replays) == 2
        for path in replays:
            configs = json.loads(path.read_text(encoding="utf-8"))["game_info"]["player_configs"]
            assert [c["temperature"] for c in configs if c["type"] == "llm"] == [None]
        assert not {"temperature"} & set(fake_anthropic.calls[-1].get("extra_body") or {})
