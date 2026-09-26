"""Tests for LLM bot module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from reinforcetactics.constants import MIN_MAP_SIZE
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.llm_bot import LLMBot
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def simple_game():
    """Create a simple game state for testing."""
    # Create a 10x10 map with basic tiles
    map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
    # Add HQ for player 1 and 2
    map_data[0][0] = "h_1"
    map_data[9][9] = "h_2"
    # Add some buildings
    map_data[0][1] = "b_1"
    map_data[9][8] = "b_2"
    return GameState(map_data, num_players=2)


class TestLLMBotBase:
    """Test the base LLMBot class."""

    def test_api_key_required(self, simple_game):
        """Test that API key is required."""
        with pytest.raises(ValueError, match="API key not provided"):
            # Mock the subclass methods since LLMBot is abstract
            class TestBot(LLMBot):
                def _get_api_key_from_env(self):
                    return None

                def _get_env_var_name(self):
                    return "TEST_API_KEY"

                def _get_default_model(self):
                    return "test-model"

                def _get_supported_models(self):
                    return ["test-model"]

                def _call_llm(self, messages):
                    return '{"reasoning": "test", "actions": []}'

                def _get_llm_sdk_version(self):
                    return "test-sdk-1.0.0"

            TestBot(simple_game, player=2)

    def test_game_state_serialization(self, simple_game):
        """Test that game state can be serialized."""

        # Create a mock bot with API key
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": []}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        bot = TestBot(simple_game, player=2, api_key="test-key")
        game_state_json = bot._serialize_game_state()

        # Check that serialization includes expected keys
        assert "turn_number" in game_state_json
        assert "player_gold" in game_state_json
        assert "opponent_gold" in game_state_json
        assert "player_units" in game_state_json
        assert "enemy_units" in game_state_json
        assert "player_buildings" in game_state_json
        assert "enemy_buildings" in game_state_json
        assert "legal_actions" in game_state_json

    def test_json_extraction_plain(self, simple_game):
        """Test JSON extraction from plain JSON response."""

        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": []}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        bot = TestBot(simple_game, player=2, api_key="test-key")
        json_text = '{"reasoning": "test strategy", "actions": [{"type": "END_TURN"}]}'
        extracted = bot._extract_json(json_text)

        assert extracted is not None
        assert extracted["reasoning"] == "test strategy"
        assert len(extracted["actions"]) == 1

    def test_json_extraction_markdown(self, simple_game):
        """Test JSON extraction from markdown code blocks."""

        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": []}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        bot = TestBot(simple_game, player=2, api_key="test-key")
        json_text = """Here is the response:
```json
{"reasoning": "test strategy", "actions": [{"type": "END_TURN"}]}
```
"""
        extracted = bot._extract_json(json_text)

        assert extracted is not None
        assert extracted["reasoning"] == "test strategy"

    def test_take_turn_ends_turn(self, simple_game):
        """Test that take_turn() properly ends the turn and advances game state."""

        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        # Game starts at player 1's turn
        assert simple_game.current_player == 1
        initial_turn = simple_game.turn_number

        # End player 1's turn manually
        simple_game.end_turn()
        assert simple_game.current_player == 2

        # Bot plays as player 2
        bot = TestBot(simple_game, player=2, api_key="test-key")

        # Bot takes turn - should call end_turn() and advance to player 1
        bot.take_turn()

        # After bot's turn, current player should be back to 1
        assert simple_game.current_player == 1
        # Turn number should have advanced by 1 (since we went through full cycle)
        assert simple_game.turn_number == initial_turn + 1

    def test_take_turn_ends_turn_on_llm_failure(self, simple_game):
        """Test that take_turn() ends the turn even when LLM fails."""
        call_count = 0

        class FailingBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                nonlocal call_count
                call_count += 1
                raise Exception("API Error")

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        # End player 1's turn
        simple_game.end_turn()
        assert simple_game.current_player == 2

        # Bot plays as player 2 with max_retries=1 for faster test
        bot = FailingBot(simple_game, player=2, api_key="test-key", max_retries=1)

        # Bot takes turn - should call end_turn() even on failure
        bot.take_turn()

        # After bot's turn, current player should be back to 1
        assert simple_game.current_player == 1
        # Verify that the LLM was called (to confirm the failure path was taken)
        assert call_count == 1


class TestOpenAIBot:
    """Test OpenAIBot implementation."""

    def test_env_var_name(self):
        """Test that correct environment variable name is configured."""
        from reinforcetactics.game.llm_bot import OpenAIBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._env_var_name == "OPENAI_API_KEY"  # pylint: disable=protected-access

    def test_default_model(self):
        """Test default model selection."""
        from reinforcetactics.game.llm_bot import OpenAIBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._default_model_name == "gpt-5-mini-2025-08-07"  # pylint: disable=protected-access

    def test_supported_models(self):
        """Test that supported models list is configured."""
        from reinforcetactics.game.llm_bot import OPENAI_MODELS
        from reinforcetactics.game.llm_bot import OpenAIBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._supported_model_list is OPENAI_MODELS  # pylint: disable=protected-access
        # Verify some expected models are present
        assert "gpt-5.2" in OPENAI_MODELS
        assert "gpt-5-mini-2025-08-07" in OPENAI_MODELS
        assert "gpt-5-nano-2025-08-07" in OPENAI_MODELS


class TestClaudeBot:
    """Test ClaudeBot implementation."""

    def test_env_var_name(self):
        """Test that correct environment variable name is configured."""
        from reinforcetactics.game.llm_bot import ClaudeBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._env_var_name == "ANTHROPIC_API_KEY"  # pylint: disable=protected-access

    def test_default_model(self):
        """Test default model selection."""
        from reinforcetactics.game.llm_bot import ClaudeBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._default_model_name == "claude-haiku-4-5-20251001"  # pylint: disable=protected-access

    def test_supported_models(self):
        """Test that supported models list is configured."""
        from reinforcetactics.game.llm_bot import ANTHROPIC_MODELS, ANTHROPIC_UNAVAILABLE_MODELS
        from reinforcetactics.game.llm_bot import ClaudeBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._supported_model_list is ANTHROPIC_MODELS  # pylint: disable=protected-access
        # Verify some expected models are present
        assert "claude-opus-4-6" in ANTHROPIC_MODELS
        assert "claude-sonnet-4-5-20250929" in ANTHROPIC_MODELS
        assert "claude-haiku-4-5-20251001" in ANTHROPIC_MODELS
        # The deprecated Sonnet 4 used to be listed here; retired and
        # deprecated IDs are now only in the "don't use" notes.
        assert "claude-sonnet-4-20250514" not in ANTHROPIC_MODELS
        assert "claude-sonnet-4-20250514" in ANTHROPIC_UNAVAILABLE_MODELS


class TestGeminiBot:
    """Test GeminiBot implementation."""

    def test_env_var_name(self):
        """Test that correct environment variable name is configured."""
        from reinforcetactics.game.llm_bot import GeminiBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._env_var_name == "GOOGLE_API_KEY"  # pylint: disable=protected-access

    def test_default_model(self):
        """Test default model selection."""
        from reinforcetactics.game.llm_bot import GeminiBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._default_model_name == "gemini-2.5-flash"  # pylint: disable=protected-access

    def test_supported_models(self):
        """Test that supported models list is configured."""
        from reinforcetactics.game.llm_bot import GEMINI_MODELS
        from reinforcetactics.game.llm_bot import GeminiBot as TestBot  # pylint: disable=import-outside-toplevel

        assert TestBot._supported_model_list is GEMINI_MODELS  # pylint: disable=protected-access
        # Verify some expected models are present
        assert "gemini-2.5-flash" in GEMINI_MODELS
        assert "gemini-2.5-pro" in GEMINI_MODELS
        assert "gemini-3-pro-preview" in GEMINI_MODELS
        assert "gemini-2.5-flash-lite" in GEMINI_MODELS


class TestConversationLogging:
    """Test conversation logging functionality."""

    @pytest.fixture
    def test_bot_class(self):
        """Create a test bot class for testing."""

        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        return TestBot

    def test_log_conversations_parameter(self, simple_game, test_bot_class):
        """Test that log_conversations parameter is properly set."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key", log_conversations=True)
        assert bot.log_conversations is True

        bot2 = test_bot_class(simple_game, player=2, api_key="test-key", log_conversations=False)
        assert bot2.log_conversations is False

        bot3 = test_bot_class(simple_game, player=2, api_key="test-key")
        assert bot3.log_conversations is False  # Default should be False

    def test_conversation_log_dir_parameter(self, simple_game, test_bot_class):
        """Test that conversation_log_dir parameter is properly set."""
        custom_dir = "/tmp/custom_logs/"
        bot = test_bot_class(simple_game, player=2, api_key="test-key", conversation_log_dir=custom_dir)
        assert bot.conversation_log_dir == custom_dir

        bot2 = test_bot_class(simple_game, player=2, api_key="test-key")
        assert bot2.conversation_log_dir == "logs/llm_conversations/"  # Default

    def test_no_logging_when_disabled(self, simple_game, test_bot_class):
        """Test that no logging occurs when log_conversations is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game, player=2, api_key="test-key", log_conversations=False, conversation_log_dir=tmpdir
            )

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # No files should be created
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 0

    def test_logging_when_enabled(self, simple_game, test_bot_class):
        """Test that logging occurs when log_conversations is True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game, player=2, api_key="test-key", log_conversations=True, conversation_log_dir=tmpdir
            )

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # One file should be created
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

    def test_json_file_structure(self, simple_game, test_bot_class):
        """Test that JSON log file has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game, player=2, api_key="test-key", log_conversations=True, conversation_log_dir=tmpdir
            )

            response = '{"reasoning": "test strategy", "actions": [{"type": "END_TURN"}]}'
            # Mock _call_llm to avoid actual API calls
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # Read the log file
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], encoding="utf-8") as f:
                log_data = json.load(f)

            # Verify structure
            assert "game_session_id" in log_data
            assert "model" in log_data
            assert log_data["model"] == "test-model"
            assert "temperature" in log_data
            assert log_data["temperature"] is None
            assert "provider" in log_data
            assert log_data["provider"] == "Test"
            assert "player" in log_data
            assert log_data["player"] == 2
            assert "start_time" in log_data
            assert "system_prompt" in log_data
            assert "turns" in log_data

            # Verify system prompt contains game rules
            assert "Reinforce Tactics" in log_data["system_prompt"]
            assert "GAME OBJECTIVE" in log_data["system_prompt"]

            # Verify turns structure
            assert len(log_data["turns"]) == 1
            turn = log_data["turns"][0]
            assert "turn_number" in turn
            assert "timestamp" in turn
            assert "user_prompt" in turn
            assert "assistant_response" in turn
            assert turn["assistant_response"] == response

    def test_custom_log_directory(self, simple_game, test_bot_class):
        """Test that custom log directory is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "my_custom_logs"
            bot = test_bot_class(
                simple_game, player=2, api_key="test-key", log_conversations=True, conversation_log_dir=str(custom_dir)
            )

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # Verify directory was created
            assert custom_dir.exists()
            assert custom_dir.is_dir()

            # Verify file was created in custom directory
            log_files = list(custom_dir.glob("*.json"))
            assert len(log_files) == 1

    def test_filename_format(self, simple_game, test_bot_class):
        """Test that log filename has correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game, player=2, api_key="test-key", log_conversations=True, conversation_log_dir=tmpdir
            )

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # Verify filename format
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            filename = log_files[0].name
            # Should be like: game_{session_id}_player2_model{model}.json
            assert filename.startswith("game_")
            assert "_player2_" in filename
            assert "_modeltest-model.json" in filename or "model" in filename
            assert filename.endswith(".json")

    def test_game_session_id_parameter(self, simple_game, test_bot_class):
        """Test that custom game_session_id is used when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_session_id = "test_session_12345"
            bot = test_bot_class(
                simple_game,
                player=2,
                api_key="test-key",
                log_conversations=True,
                conversation_log_dir=tmpdir,
                game_session_id=custom_session_id,
            )

            assert bot.game_session_id == custom_session_id

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # Verify filename includes custom session ID
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1
            assert custom_session_id in log_files[0].name

            # Verify session ID is in the log data
            with open(log_files[0], encoding="utf-8") as f:
                log_data = json.load(f)
            assert log_data["game_session_id"] == custom_session_id

    def test_auto_generated_session_id(self, simple_game, test_bot_class):
        """Test that session ID is auto-generated if not provided."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key")

        # Session ID should be auto-generated
        assert bot.game_session_id is not None
        assert len(bot.game_session_id) > 0

        # Should have format: YYYYMMDD_HHMMSS_{random}
        parts = bot.game_session_id.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 6  # random component

    def test_multiple_games_create_separate_files(self, simple_game, test_bot_class):
        """Test that multiple games create separate log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two bots with different session IDs (simulating different games)
            bot1 = test_bot_class(
                simple_game,
                player=2,
                api_key="test-key",
                log_conversations=True,
                conversation_log_dir=tmpdir,
                game_session_id="game1",
            )
            bot2 = test_bot_class(
                simple_game,
                player=2,
                api_key="test-key",
                log_conversations=True,
                conversation_log_dir=tmpdir,
                game_session_id="game2",
            )

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot1, "_call_llm", return_value=response):
                with patch.object(bot2, "_call_llm", return_value=response):
                    bot1.take_turn()
                    bot2.take_turn()

            # Two files should be created (one per game)
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 2

            # Verify they have different session IDs
            session_ids = set()
            for log_file in log_files:
                with open(log_file, encoding="utf-8") as f:
                    log_data = json.load(f)
                    session_ids.add(log_data["game_session_id"])

            assert len(session_ids) == 2
            assert "game1" in session_ids
            assert "game2" in session_ids

    def test_pretty_print_logs_enabled(self, simple_game, test_bot_class):
        """Test that pretty_print_logs=True creates indented JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game,
                player=2,
                api_key="test-key",
                log_conversations=True,
                conversation_log_dir=tmpdir,
                pretty_print_logs=True,
            )

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # Read the file as text
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], encoding="utf-8") as f:
                content = f.read()

            # Pretty-printed JSON should have newlines and indentation
            assert "\n" in content
            assert "  " in content  # Indentation

    def test_pretty_print_logs_disabled(self, simple_game, test_bot_class):
        """Test that pretty_print_logs=False creates compact JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game,
                player=2,
                api_key="test-key",
                log_conversations=True,
                conversation_log_dir=tmpdir,
                pretty_print_logs=False,
            )

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            # Read the file as text
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], encoding="utf-8") as f:
                content = f.read()

            # Compact JSON should be mostly on one line (no indentation)
            # It may have some newlines but shouldn't have the 2-space indentation pattern
            lines = content.split("\n")
            # For compact JSON, most content is on fewer lines
            assert len(lines) < 10  # Pretty version would have many more lines

    def test_logging_error_handling(self, simple_game, test_bot_class):
        """Test that logging errors don't break the bot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game, player=2, api_key="test-key", log_conversations=True, conversation_log_dir=tmpdir
            )

            # Mock Path.mkdir to raise an exception
            with patch("reinforcetactics.game.llm_bot.Path.mkdir", side_effect=OSError("Permission denied")):
                # Mock _call_llm to avoid actual API calls
                # This should not raise an exception even if logging fails
                response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
                with patch.object(bot, "_call_llm", return_value=response):
                    bot.take_turn()  # Should complete without exception

    def test_multiple_turns_create_single_file(self, simple_game, test_bot_class):
        """Test that multiple turns create a single log file with all turns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                simple_game, player=2, api_key="test-key", log_conversations=True, conversation_log_dir=tmpdir
            )

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()
                # Advance turn
                bot.game_state.turn_number += 1
                bot.take_turn()

            # Only one file should be created
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            # Verify it contains two turns
            with open(log_files[0], encoding="utf-8") as f:
                log_data = json.load(f)

            assert "turns" in log_data
            assert len(log_data["turns"]) == 2

            # Verify they have different turn numbers
            turn_numbers = [turn["turn_number"] for turn in log_data["turns"]]
            assert len(set(turn_numbers)) == 2  # Should be different
            assert turn_numbers[0] < turn_numbers[1]  # Should be in order


class TestStatefulConversation:
    """Test stateful conversation functionality."""

    @pytest.fixture
    def test_bot_class(self):
        """Create a test bot class for testing."""

        class TestBot(LLMBot):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0
                self.messages_received = []

            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                self.call_count += 1
                self.messages_received.append(messages)
                return f'{{"reasoning": "Turn {self.call_count}", "actions": [{{"type": "END_TURN"}}]}}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        return TestBot

    def test_stateful_parameter_default(self, simple_game, test_bot_class):
        """Test that stateful parameter defaults to False."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key")
        assert bot.stateful is False
        assert bot.conversation_history == []

    def test_stateful_parameter_enabled(self, simple_game, test_bot_class):
        """Test that stateful parameter can be enabled."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key", stateful=True)
        assert bot.stateful is True
        assert bot.conversation_history == []

    def test_stateless_mode_no_history(self, simple_game, test_bot_class):
        """Test that stateless mode doesn't accumulate conversation history."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key", stateful=False)

        # Take 3 turns
        for _ in range(3):
            bot.take_turn()
            simple_game.turn_number += 1

        # In stateless mode, history should remain empty
        assert len(bot.conversation_history) == 0

        # Each call should only have system + user message (no history)
        assert bot.call_count == 3
        for messages in bot.messages_received:
            assert len(messages) == 2  # system + user only
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    def test_stateful_mode_accumulates_history(self, simple_game, test_bot_class):
        """Test that stateful mode accumulates conversation history."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key", stateful=True)

        # Take 3 turns
        for i in range(3):
            bot.take_turn()
            simple_game.turn_number += 1

        # In stateful mode, history should accumulate (2 messages per turn: user + assistant)
        assert len(bot.conversation_history) == 6  # 3 turns * 2 messages

        # Verify the pattern: user, assistant, user, assistant, ...
        for i in range(0, 6, 2):
            assert bot.conversation_history[i]["role"] == "user"
            assert bot.conversation_history[i + 1]["role"] == "assistant"

    def test_stateful_mode_sends_history_to_llm(self, simple_game, test_bot_class):
        """Test that stateful mode sends conversation history to LLM."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key", stateful=True)

        # First turn
        bot.take_turn()
        assert len(bot.messages_received[0]) == 2  # system + user

        # Second turn should include history
        simple_game.turn_number += 1
        bot.take_turn()
        assert len(bot.messages_received[1]) == 4  # system + prev_user + prev_assistant + current_user
        assert bot.messages_received[1][0]["role"] == "system"
        assert bot.messages_received[1][1]["role"] == "user"  # previous turn
        assert bot.messages_received[1][2]["role"] == "assistant"  # previous response
        assert bot.messages_received[1][3]["role"] == "user"  # current turn

        # Third turn should include even more history
        simple_game.turn_number += 1
        bot.take_turn()
        assert len(bot.messages_received[2]) == 6  # system + 2 prev exchanges + current_user

    def test_stateful_mode_history_content(self, simple_game, test_bot_class):
        """Test that conversation history contains actual content."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key", stateful=True)

        # Take 2 turns
        bot.take_turn()
        simple_game.turn_number += 1
        bot.take_turn()

        # Check that history has content
        assert len(bot.conversation_history) == 4

        # First user message should have game state
        assert "Current Game State" in bot.conversation_history[0]["content"]

        # First assistant message should have reasoning
        assert "Turn 1" in bot.conversation_history[1]["content"]

        # Second user message should have game state
        assert "Current Game State" in bot.conversation_history[2]["content"]

        # Second assistant message should have reasoning
        assert "Turn 2" in bot.conversation_history[3]["content"]


class TestLLMCoordinatesAreTheGrids:
    """The LLM sees and answers in the game's own grid coordinates.

    A GUI game is played on the UI-padded map (FileIO.load_map(for_ui=True)),
    so its prompt coordinates include that padding; map_width/map_height
    describe the same padded grid, and whatever the LLM sends back is read on
    it. There is no second coordinate frame to convert to or from.
    """

    @pytest.fixture
    def padded_game(self):
        """A 6x6 map padded the way the GUI pads it (to 20x20, then a 2-tile border)."""
        small_map = np.array([["p" for _ in range(6)] for _ in range(6)], dtype=object)
        small_map[0][0] = "h_1"
        small_map[5][5] = "h_2"
        small_map[0][1] = "b_1"
        small_map[5][4] = "b_2"
        padded_map, offset_x, offset_y = FileIO.pad_for_display(small_map, MIN_MAP_SIZE, border_size=2)
        assert (offset_x, offset_y) == (9, 9)

        game = GameState(padded_map, num_players=2)
        game.map_file_used = "maps/1v1/beginner.csv"
        return game

    @pytest.fixture
    def test_bot_class(self):
        """Create a test bot class for testing."""

        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        return TestBot

    def test_serialized_state_describes_the_grid(self, padded_game, test_bot_class):
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        state = bot._serialize_game_state()

        assert state["map_name"] == "beginner"
        assert (state["map_width"], state["map_height"]) == (24, 24)

    def test_serialized_positions_are_grid_positions(self, padded_game, test_bot_class):
        padded_game.place_unit("W", 9, 10, player=2)
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        state = bot._serialize_game_state()

        assert [u["position"] for u in state["player_units"]] == [[9, 10]]
        enemy_hq = next(b for b in state["enemy_buildings"] if b["type"] == "h")
        assert enemy_hq["position"] == [9, 9]
        assert padded_game.grid.get_tile(9, 9).type == "h"

    def test_legal_moves_are_the_engines(self, padded_game, test_bot_class):
        unit = padded_game.place_unit("W", 9, 10, player=2)
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        state = bot._serialize_game_state()

        engine_moves = {(m["to_x"], m["to_y"]) for m in padded_game.get_legal_actions(2)["move"] if m["unit"] is unit}
        assert engine_moves
        assert {tuple(m["to"]) for m in state["legal_actions"]["move"]} == engine_moves
        assert all(m["from"] == [9, 10] for m in state["legal_actions"]["move"])

    def test_offered_create_unit_is_carried_out_where_offered(self, padded_game, test_bot_class):
        padded_game.current_player = 2  # the engine only creates on the player's own turn
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        offered = bot._serialize_game_state()["legal_actions"]["create_unit"][0]

        assert bot._execute_create_unit({"type": "CREATE_UNIT", **offered})

        new_unit = padded_game.units[-1]
        assert [new_unit.x, new_unit.y] == offered["position"] == [13, 14]

    def test_offered_move_is_carried_out_where_offered(self, padded_game, test_bot_class):
        unit = padded_game.place_unit("W", 9, 10, player=2)
        padded_game.current_player = 2  # the engine only moves the current player's units
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        state = bot._serialize_game_state()
        move = next(m for m in state["legal_actions"]["move"] if m["to"] == [9, 11])

        assert bot._execute_move({"type": "MOVE", **move}, bot._get_unit_by_id())

        assert (unit.x, unit.y) == (9, 11)

    def test_conversation_log_includes_map_metadata(self, padded_game, test_bot_class):
        """Test that conversation logs include map metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(
                padded_game, player=2, api_key="test-key", log_conversations=True, conversation_log_dir=tmpdir
            )

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, "_call_llm", return_value=response):
                bot.take_turn()

            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1
            with open(log_files[0], encoding="utf-8") as f:
                log_data = json.load(f)

            assert log_data["map_file"] == "maps/1v1/beginner.csv"
            assert log_data["map_dimensions"] == {"width": 24, "height": 24}

    def test_unpadded_map(self, simple_game, test_bot_class):
        """A map used as is (tournaments, the RL env) is described as is."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key")
        simple_game.place_unit("W", 5, 5, player=2)

        state = bot._serialize_game_state()

        assert [u["position"] for u in state["player_units"]] == [[5, 5]]
        assert (state["map_width"], state["map_height"]) == (10, 10)

    def test_action_history_uses_grid_coordinates(self, padded_game):
        """Recorded actions are on the grid, the one a replay stores as its initial_map."""
        # Player 1's building is at grid (10, 9) -- create_unit only spawns on
        # an owned building.
        unit = padded_game.create_unit("W", 10, 9, player=1)
        create_action = padded_game.action_history[-1]
        assert (create_action["type"], create_action["x"], create_action["y"]) == ("create_unit", 10, 9)

        unit.can_move = True
        padded_game.move_unit(unit, 10, 10)
        move_action = padded_game.action_history[-1]
        assert move_action["type"] == "move"
        assert (move_action["from_x"], move_action["from_y"], move_action["to_x"], move_action["to_y"]) == (10, 9, 10, 10)

        enemy = padded_game.place_unit("W", 10, 11, player=2)
        unit.can_attack = True
        padded_game.attack(unit, enemy)
        attack_action = padded_game.action_history[-1]
        assert attack_action["type"] == "attack"
        assert (attack_action["attacker_pos"], attack_action["target_pos"]) == ((10, 10), (10, 11))

    def test_action_history_no_padding(self, simple_game):
        """Test that action history works correctly when there's no padding."""
        simple_game.create_unit("W", 1, 0, player=1)

        create_action = simple_game.action_history[-1]
        assert create_action["type"] == "create_unit"
        assert create_action["x"] == 1
        assert create_action["y"] == 0


class TestLLMBotFogOfWar:
    """Verify the LLM adapter respects fog of war when serializing state.

    The LLM bot must not leak information about hidden enemies, including
    via the ``move_then_attack``/``move_then_paralyze`` combo planner.
    """

    @pytest.fixture
    def fow_game(self):
        map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
        map_data[0][0] = "h_1"
        map_data[9][9] = "h_2"
        map_data[0][1] = "b_1"
        map_data[9][8] = "b_2"
        game = GameState(map_data, num_players=2, fog_of_war=True)
        game.player_gold[1] = 10000
        game.player_gold[2] = 10000
        game.update_visibility()
        return game

    @pytest.fixture
    def bot_class(self):
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return "TEST_API_KEY"

            def _get_default_model(self):
                return "test-model"

            def _get_supported_models(self):
                return ["test-model"]

            def _call_llm(self, messages):
                return '{"actions": []}'

            def _get_llm_sdk_version(self):
                return "test-sdk-1.0.0"

        return TestBot

    def test_serialized_state_hides_enemy_units(self, fow_game, bot_class):
        """Hidden enemies must not appear in the serialized enemy_units list."""
        # Bot plays player 1; enemy at (8, 8) is far outside HQ vision (range 4)
        fow_game.place_unit("W", 8, 8, player=2)
        fow_game.update_visibility()
        assert not fow_game.is_position_visible(8, 8, player=1)

        bot = bot_class(fow_game, player=1, api_key="test-key")
        state = bot._serialize_game_state()

        assert state["fog_of_war"] is True
        assert state["opponent_gold"] == "hidden"
        # The hidden enemy unit must be omitted from enemy_units
        assert all(tuple(u["position"]) != (8, 8) for u in state["enemy_units"])

    def test_move_then_attack_excludes_hidden_enemies(self, fow_game, bot_class):
        """Move-then-attack combos must not be generated for hidden enemies.

        Without this filter, telling the LLM "if you move here you can attack X"
        leaks the hidden position of X and effectively enables the
        "move-to-discover, then attack" exploit.
        """
        # Player 1 unit positioned where a 1-tile move would put it adjacent
        # to a hidden enemy. Without the FOW filter, _compute_move_then_actions
        # would emit a move_then_attack against the hidden enemy.
        attacker = fow_game.place_unit("W", 4, 4, player=1)
        attacker.can_move = True
        attacker.can_attack = True

        # Enemy positioned outside attacker's pre-move vision (Warrior range 3)
        # but within 1-tile move + attack reach.
        hidden_enemy = fow_game.place_unit("W", 8, 4, player=2)
        fow_game.update_visibility(player=1)
        assert not fow_game.is_position_visible(8, 4, player=1)

        bot = bot_class(fow_game, player=1, api_key="test-key")
        state = bot._serialize_game_state()

        # No move_then_attack combo should target the hidden enemy
        for combo in state["legal_actions"].get("move_then_attack", []):
            assert tuple(combo["then_attack"]) != (hidden_enemy.x, hidden_enemy.y)

    def test_move_then_attack_includes_visible_enemies(self, fow_game, bot_class):
        """Visible enemies must still appear in move-then-attack combos."""
        attacker = fow_game.place_unit("W", 3, 3, player=1)
        attacker.can_move = True
        attacker.can_attack = True

        # Enemy within attacker's pre-move vision range (Warrior vision 3).
        # We only need the unit to exist on the board; we look it up by
        # position in the serialized output below.
        fow_game.place_unit("W", 5, 3, player=2)
        fow_game.update_visibility(player=1)
        assert fow_game.is_position_visible(5, 3, player=1)

        bot = bot_class(fow_game, player=1, api_key="test-key")
        state = bot._serialize_game_state()

        # At least one move_then_attack combo should target the visible enemy
        targets = [tuple(c["then_attack"]) for c in state["legal_actions"].get("move_then_attack", [])]
        assert (5, 3) in targets

    def test_move_then_actions_unaffected_without_fow(self, simple_game, bot_class):
        """Without FOW, the FOW filter must not accidentally drop enemies."""
        attacker = simple_game.place_unit("W", 3, 3, player=1)
        attacker.can_move = True
        attacker.can_attack = True
        target = simple_game.place_unit("W", 8, 3, player=2)

        bot = bot_class(simple_game, player=1, api_key="test-key")
        state = bot._serialize_game_state()

        # Without FOW, the enemy must appear in the serialized state — the new
        # filter in _compute_move_then_actions only kicks in under FOW.
        assert state["fog_of_war"] is False
        assert any(u["position"] == [target.x, target.y] for u in state["enemy_units"])
