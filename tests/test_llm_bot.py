"""Tests for LLM bot module."""
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.llm_bot import LLMBot


@pytest.fixture
def simple_game():
    """Create a simple game state for testing."""
    # Create a 10x10 map with basic tiles
    map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
    # Add HQ for player 1 and 2
    map_data[0][0] = 'h_1'
    map_data[9][9] = 'h_2'
    # Add some buildings
    map_data[0][1] = 'b_1'
    map_data[9][8] = 'b_2'
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
                    return 'TEST_API_KEY'

                def _get_default_model(self):
                    return 'test-model'

                def _get_supported_models(self):
                    return ['test-model']

                def _call_llm(self, messages):
                    return '{"reasoning": "test", "actions": []}'

            TestBot(simple_game, player=2)

    def test_game_state_serialization(self, simple_game):
        """Test that game state can be serialized."""
        # Create a mock bot with API key
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": []}'

        bot = TestBot(simple_game, player=2, api_key="test-key")
        game_state_json = bot._serialize_game_state()

        # Check that serialization includes expected keys
        assert 'turn_number' in game_state_json
        assert 'player_gold' in game_state_json
        assert 'opponent_gold' in game_state_json
        assert 'player_units' in game_state_json
        assert 'enemy_units' in game_state_json
        assert 'player_buildings' in game_state_json
        assert 'enemy_buildings' in game_state_json
        assert 'legal_actions' in game_state_json

    def test_json_extraction_plain(self, simple_game):
        """Test JSON extraction from plain JSON response."""
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": []}'

        bot = TestBot(simple_game, player=2, api_key="test-key")
        json_text = '{"reasoning": "test strategy", "actions": [{"type": "END_TURN"}]}'
        extracted = bot._extract_json(json_text)

        assert extracted is not None
        assert extracted['reasoning'] == "test strategy"
        assert len(extracted['actions']) == 1

    def test_json_extraction_markdown(self, simple_game):
        """Test JSON extraction from markdown code blocks."""
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": []}'

        bot = TestBot(simple_game, player=2, api_key="test-key")
        json_text = '''Here is the response:
```json
{"reasoning": "test strategy", "actions": [{"type": "END_TURN"}]}
```
'''
        extracted = bot._extract_json(json_text)

        assert extracted is not None
        assert extracted['reasoning'] == "test strategy"

    def test_take_turn_ends_turn(self, simple_game):
        """Test that take_turn() properly ends the turn and advances game state."""
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'

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
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                nonlocal call_count
                call_count += 1
                raise Exception("API Error")

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
        """Test that correct environment variable name is returned."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import OpenAIBot as TestBot  # pylint: disable=import-outside-toplevel
            assert TestBot._get_env_var_name(Mock()) == 'OPENAI_API_KEY'  # pylint: disable=protected-access

    def test_default_model(self):
        """Test default model selection."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import OpenAIBot as TestBot  # pylint: disable=import-outside-toplevel
            assert TestBot._get_default_model(Mock()) == 'gpt-4o-mini'  # pylint: disable=protected-access

    def test_supported_models(self):
        """Test that supported models list is returned."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import OpenAIBot as TestBot, OPENAI_MODELS  # pylint: disable=import-outside-toplevel
            assert TestBot._get_supported_models(Mock()) == OPENAI_MODELS  # pylint: disable=protected-access
            # Verify some expected models are present
            assert 'gpt-4o' in OPENAI_MODELS
            assert 'gpt-4o-mini' in OPENAI_MODELS
            assert 'gpt-4-turbo' in OPENAI_MODELS
            assert 'o1' in OPENAI_MODELS


class TestClaudeBot:
    """Test ClaudeBot implementation."""

    def test_env_var_name(self):
        """Test that correct environment variable name is returned."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import ClaudeBot as TestBot  # pylint: disable=import-outside-toplevel
            assert TestBot._get_env_var_name(Mock()) == 'ANTHROPIC_API_KEY'  # pylint: disable=protected-access

    def test_default_model(self):
        """Test default model selection."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import ClaudeBot as TestBot  # pylint: disable=import-outside-toplevel
            assert TestBot._get_default_model(Mock()) == 'claude-3-5-haiku-20241022'  # pylint: disable=protected-access

    def test_supported_models(self):
        """Test that supported models list is returned."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import ClaudeBot as TestBot, ANTHROPIC_MODELS  # pylint: disable=import-outside-toplevel
            assert TestBot._get_supported_models(Mock()) == ANTHROPIC_MODELS  # pylint: disable=protected-access
            # Verify some expected models are present
            assert 'claude-sonnet-4-20250514' in ANTHROPIC_MODELS
            assert 'claude-3-5-sonnet-20241022' in ANTHROPIC_MODELS
            assert 'claude-3-5-haiku-20241022' in ANTHROPIC_MODELS
            assert 'claude-3-opus-20240229' in ANTHROPIC_MODELS


class TestGeminiBot:
    """Test GeminiBot implementation."""

    def test_env_var_name(self):
        """Test that correct environment variable name is returned."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import GeminiBot as TestBot  # pylint: disable=import-outside-toplevel
            assert TestBot._get_env_var_name(Mock()) == 'GOOGLE_API_KEY'  # pylint: disable=protected-access

    def test_default_model(self):
        """Test default model selection."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import GeminiBot as TestBot  # pylint: disable=import-outside-toplevel
            assert TestBot._get_default_model(Mock()) == 'gemini-2.0-flash'  # pylint: disable=protected-access

    def test_supported_models(self):
        """Test that supported models list is returned."""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
            from reinforcetactics.game.llm_bot import GeminiBot as TestBot, GEMINI_MODELS  # pylint: disable=import-outside-toplevel
            assert TestBot._get_supported_models(Mock()) == GEMINI_MODELS  # pylint: disable=protected-access
            # Verify some expected models are present
            assert 'gemini-2.0-flash' in GEMINI_MODELS
            assert 'gemini-1.5-pro' in GEMINI_MODELS
            assert 'gemini-1.5-flash' in GEMINI_MODELS
            assert 'gemini-2.0-flash-thinking-exp' in GEMINI_MODELS


class TestConversationLogging:
    """Test conversation logging functionality."""

    @pytest.fixture
    def test_bot_class(self):
        """Create a test bot class for testing."""
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'

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
        bot = test_bot_class(simple_game, player=2, api_key="test-key",
                            conversation_log_dir=custom_dir)
        assert bot.conversation_log_dir == custom_dir

        bot2 = test_bot_class(simple_game, player=2, api_key="test-key")
        assert bot2.conversation_log_dir == 'logs/llm_conversations/'  # Default

    def test_no_logging_when_disabled(self, simple_game, test_bot_class):
        """Test that no logging occurs when log_conversations is False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=False,
                                conversation_log_dir=tmpdir)

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()

            # No files should be created
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 0

    def test_logging_when_enabled(self, simple_game, test_bot_class):
        """Test that logging occurs when log_conversations is True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()

            # One file should be created
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

    def test_json_file_structure(self, simple_game, test_bot_class):
        """Test that JSON log file has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            response = '{"reasoning": "test strategy", "actions": [{"type": "END_TURN"}]}'
            # Mock _call_llm to avoid actual API calls
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()

            # Read the log file
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], 'r', encoding='utf-8') as f:
                log_data = json.load(f)

            # Verify structure
            assert 'game_session_id' in log_data
            assert 'model' in log_data
            assert log_data['model'] == 'test-model'
            assert 'provider' in log_data
            assert log_data['provider'] == 'Test'
            assert 'player' in log_data
            assert log_data['player'] == 2
            assert 'start_time' in log_data
            assert 'system_prompt' in log_data
            assert 'turns' in log_data

            # Verify system prompt contains game rules
            assert 'Reinforce Tactics' in log_data['system_prompt']
            assert 'GAME OBJECTIVE' in log_data['system_prompt']

            # Verify turns structure
            assert len(log_data['turns']) == 1
            turn = log_data['turns'][0]
            assert 'turn_number' in turn
            assert 'timestamp' in turn
            assert 'user_prompt' in turn
            assert 'assistant_response' in turn
            assert turn['assistant_response'] == response

    def test_custom_log_directory(self, simple_game, test_bot_class):
        """Test that custom log directory is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "my_custom_logs"
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=str(custom_dir))

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
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
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
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
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir,
                                game_session_id=custom_session_id)

            assert bot.game_session_id == custom_session_id

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()

            # Verify filename includes custom session ID
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1
            assert custom_session_id in log_files[0].name

            # Verify session ID is in the log data
            with open(log_files[0], 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            assert log_data['game_session_id'] == custom_session_id

    def test_auto_generated_session_id(self, simple_game, test_bot_class):
        """Test that session ID is auto-generated if not provided."""
        bot = test_bot_class(simple_game, player=2, api_key="test-key")

        # Session ID should be auto-generated
        assert bot.game_session_id is not None
        assert len(bot.game_session_id) > 0

        # Should have format: YYYYMMDD_HHMMSS_{random}
        parts = bot.game_session_id.split('_')
        assert len(parts) == 3
        assert len(parts[0]) == 8  # YYYYMMDD
        assert len(parts[1]) == 6  # HHMMSS
        assert len(parts[2]) == 6  # random component

    def test_multiple_games_create_separate_files(self, simple_game, test_bot_class):
        """Test that multiple games create separate log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two bots with different session IDs (simulating different games)
            bot1 = test_bot_class(simple_game, player=2, api_key="test-key",
                                 log_conversations=True,
                                 conversation_log_dir=tmpdir,
                                 game_session_id="game1")
            bot2 = test_bot_class(simple_game, player=2, api_key="test-key",
                                 log_conversations=True,
                                 conversation_log_dir=tmpdir,
                                 game_session_id="game2")

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot1, '_call_llm', return_value=response):
                with patch.object(bot2, '_call_llm', return_value=response):
                    bot1.take_turn()
                    bot2.take_turn()

            # Two files should be created (one per game)
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 2

            # Verify they have different session IDs
            session_ids = set()
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    session_ids.add(log_data['game_session_id'])

            assert len(session_ids) == 2
            assert "game1" in session_ids
            assert "game2" in session_ids

    def test_pretty_print_logs_enabled(self, simple_game, test_bot_class):
        """Test that pretty_print_logs=True creates indented JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir,
                                pretty_print_logs=True)

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()

            # Read the file as text
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], 'r', encoding='utf-8') as f:
                content = f.read()

            # Pretty-printed JSON should have newlines and indentation
            assert '\n' in content
            assert '  ' in content  # Indentation

    def test_pretty_print_logs_disabled(self, simple_game, test_bot_class):
        """Test that pretty_print_logs=False creates compact JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir,
                                pretty_print_logs=False)

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()

            # Read the file as text
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], 'r', encoding='utf-8') as f:
                content = f.read()

            # Compact JSON should be mostly on one line (no indentation)
            # It may have some newlines but shouldn't have the 2-space indentation pattern
            lines = content.split('\n')
            # For compact JSON, most content is on fewer lines
            assert len(lines) < 10  # Pretty version would have many more lines

    def test_logging_error_handling(self, simple_game, test_bot_class):
        """Test that logging errors don't break the bot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Mock Path.mkdir to raise an exception
            with patch('reinforcetactics.game.llm_bot.Path.mkdir', side_effect=OSError("Permission denied")):
                # Mock _call_llm to avoid actual API calls
                # This should not raise an exception even if logging fails
                response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
                with patch.object(bot, '_call_llm', return_value=response):
                    bot.take_turn()  # Should complete without exception

    def test_multiple_turns_create_single_file(self, simple_game, test_bot_class):
        """Test that multiple turns create a single log file with all turns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Mock _call_llm to avoid actual API calls
            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()
                # Advance turn
                bot.game_state.turn_number += 1
                bot.take_turn()

            # Only one file should be created
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            # Verify it contains two turns
            with open(log_files[0], 'r', encoding='utf-8') as f:
                log_data = json.load(f)

            assert 'turns' in log_data
            assert len(log_data['turns']) == 2

            # Verify they have different turn numbers
            turn_numbers = [turn['turn_number'] for turn in log_data['turns']]
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
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                self.call_count += 1
                self.messages_received.append(messages)
                return f'{{"reasoning": "Turn {self.call_count}", "actions": [{{"type": "END_TURN"}}]}}'

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
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'

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
            assert bot.conversation_history[i]['role'] == 'user'
            assert bot.conversation_history[i + 1]['role'] == 'assistant'

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
        assert bot.messages_received[1][0]['role'] == 'system'
        assert bot.messages_received[1][1]['role'] == 'user'  # previous turn
        assert bot.messages_received[1][2]['role'] == 'assistant'  # previous response
        assert bot.messages_received[1][3]['role'] == 'user'  # current turn

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
        assert 'Current Game State' in bot.conversation_history[0]['content']

        # First assistant message should have reasoning
        assert 'Turn 1' in bot.conversation_history[1]['content']

        # Second user message should have game state
        assert 'Current Game State' in bot.conversation_history[2]['content']

        # Second assistant message should have reasoning
        assert 'Turn 2' in bot.conversation_history[3]['content']


class TestMapCoordinateConversion:
    """Test map coordinate conversion for padded maps."""

    @pytest.fixture
    def padded_game(self):
        """Create a game state with a small map that will be padded."""
        # Create a 6x6 map (smaller than MIN_MAP_SIZE of 20)
        small_map = np.array([['p' for _ in range(6)] for _ in range(6)], dtype=object)
        # Add HQ for player 1 and 2 at opposite corners
        small_map[0][0] = 'h_1'
        small_map[5][5] = 'h_2'
        # Add some buildings
        small_map[0][1] = 'b_1'
        small_map[5][4] = 'b_2'
        
        # Manually pad the map to 20x20 to simulate what load_map does
        # For a 6x6 map padded to 20x20, padding is 7 on each side
        padded_map = np.full((20, 20), 'o', dtype=object)
        padded_map[7:13, 7:13] = small_map
        
        game = GameState(padded_map, num_players=2)
        
        # Set the metadata to indicate this was padded from a 6x6 map
        game.set_map_metadata(
            original_width=6,
            original_height=6,
            padding_offset_x=7,
            padding_offset_y=7,
            map_file="maps/1v1/beginner.csv",
            original_map_data=small_map.tolist()
        )
        
        return game

    @pytest.fixture
    def test_bot_class(self):
        """Create a test bot class for testing."""
        class TestBot(LLMBot):
            def _get_api_key_from_env(self):
                return "test-key"

            def _get_env_var_name(self):
                return 'TEST_API_KEY'

            def _get_default_model(self):
                return 'test-model'

            def _get_supported_models(self):
                return ['test-model']

            def _call_llm(self, messages):
                return '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'

        return TestBot

    def test_coordinate_conversion_methods(self, padded_game):
        """Test basic coordinate conversion methods."""
        # Padded position [7, 7] should convert to original [0, 0]
        orig_x, orig_y = padded_game.padded_to_original_coords(7, 7)
        assert orig_x == 0
        assert orig_y == 0
        
        # Padded position [12, 12] should convert to original [5, 5]
        orig_x, orig_y = padded_game.padded_to_original_coords(12, 12)
        assert orig_x == 5
        assert orig_y == 5
        
        # Test reverse conversion
        pad_x, pad_y = padded_game.original_to_padded_coords(0, 0)
        assert pad_x == 7
        assert pad_y == 7
        
        pad_x, pad_y = padded_game.original_to_padded_coords(5, 5)
        assert pad_x == 12
        assert pad_y == 12

    def test_serialized_state_has_map_metadata(self, padded_game, test_bot_class):
        """Test that serialized game state includes map metadata."""
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        game_state_json = bot._serialize_game_state()
        
        # Check map metadata fields
        assert 'map_name' in game_state_json
        assert game_state_json['map_name'] == 'beginner'
        
        assert 'map_width' in game_state_json
        assert game_state_json['map_width'] == 6
        
        assert 'map_height' in game_state_json
        assert game_state_json['map_height'] == 6
        
        # map_padding_applied field has been removed from serialization

    def test_serialized_coordinates_are_original(self, padded_game, test_bot_class):
        """Test that serialized coordinates are in original map system."""
        # Add a unit at padded position [7, 7] (which is original [0, 0])
        padded_game.create_unit('W', 7, 7, player=2)
        
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        game_state_json = bot._serialize_game_state()
        
        # Check that unit position is in original coordinates
        assert len(game_state_json['player_units']) == 1
        unit = game_state_json['player_units'][0]
        assert unit['position'] == [0, 0]  # Original coordinates, not [7, 7]

    def test_serialized_building_coordinates_are_original(self, padded_game, test_bot_class):
        """Test that building coordinates are in original map system."""
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        game_state_json = bot._serialize_game_state()
        
        # The HQ at padded [7, 7] should be at original [0, 0]
        # But it belongs to player 1, so check enemy_buildings
        assert len(game_state_json['enemy_buildings']) >= 1
        
        # Find the HQ
        enemy_hq = None
        for building in game_state_json['enemy_buildings']:
            if building['type'] == 'h':
                enemy_hq = building
                break
        
        assert enemy_hq is not None
        # The padded position would be [7, 7], original should be [0, 0]
        assert enemy_hq['position'] == [0, 0]

    def test_legal_actions_use_original_coordinates(self, padded_game, test_bot_class):
        """Test that legal actions use original coordinates."""
        # Add a unit at padded [7, 8]
        padded_game.create_unit('W', 7, 8, player=2)
        
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        game_state_json = bot._serialize_game_state()
        
        # Check move actions - they should use original coordinates
        legal_moves = game_state_json['legal_actions']['move']
        
        if len(legal_moves) > 0:
            # At least one move should exist
            # Positions should be in original coordinate system (0-5, not 7-12)
            for move in legal_moves:
                assert 0 <= move['from'][0] < 6
                assert 0 <= move['from'][1] < 6
                assert 0 <= move['to'][0] < 6
                assert 0 <= move['to'][1] < 6

    def test_create_unit_action_converts_coordinates(self, padded_game, test_bot_class):
        """Test that CREATE_UNIT action converts from original to padded coordinates."""
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        
        # Action in original coordinates [4, 5] (building at padded [11, 12])
        action = {
            'type': 'CREATE_UNIT',
            'unit_type': 'W',
            'position': [4, 5]  # Original coordinates
        }
        
        initial_unit_count = len(padded_game.units)
        
        # Mock the legal actions check
        with patch.object(padded_game, 'get_legal_actions') as mock_legal:
            mock_legal.return_value = {
                'create_unit': [{'unit_type': 'W', 'x': 11, 'y': 12}],
                'move': [], 'attack': [], 'paralyze': [],
                'heal': [], 'cure': [], 'seize': [], 'end_turn': True
            }
            
            bot._execute_create_unit(action)
            
            # Unit should be created at padded position [11, 12]
            assert len(padded_game.units) == initial_unit_count + 1
            new_unit = padded_game.units[-1]
            assert new_unit.x == 11  # Padded coordinate
            assert new_unit.y == 12  # Padded coordinate

    def test_move_action_converts_coordinates(self, padded_game, test_bot_class):
        """Test that MOVE action converts from original to padded coordinates."""
        # Create a unit at padded [7, 7]
        unit = padded_game.create_unit('W', 7, 7, player=2)
        unit.can_move = True  # Enable movement for this test
        
        bot = test_bot_class(padded_game, player=2, api_key="test-key")
        
        # Get unit ID (it should be 0 since it's the first unit for player 2)
        unit_map = bot._get_unit_by_id()
        
        # Action in original coordinates: move from [0, 0] to [0, 1]
        # This corresponds to padded [7, 7] to [7, 8]
        action = {
            'type': 'MOVE',
            'unit_id': 0,
            'to': [0, 1]  # Original coordinates
        }
        
        bot._execute_move(action, unit_map)
        
        # Unit should now be at padded position [7, 8]
        assert unit.x == 7
        assert unit.y == 8

    def test_conversation_log_includes_map_metadata(self, padded_game, test_bot_class):
        """Test that conversation logs include map metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(padded_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
            with patch.object(bot, '_call_llm', return_value=response):
                bot.take_turn()

            # Read the log file
            log_files = list(Path(tmpdir).glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], 'r', encoding='utf-8') as f:
                log_data = json.load(f)

            # Verify map metadata is present
            assert 'map_file' in log_data
            assert log_data['map_file'] == "maps/1v1/beginner.csv"

            assert 'map_dimensions' in log_data
            assert log_data['map_dimensions']['width'] == 6
            assert log_data['map_dimensions']['height'] == 6

    def test_non_padded_map_works_correctly(self, simple_game, test_bot_class):
        """Test that non-padded maps (20x20+) work correctly without coordinate conversion."""
        # simple_game is 10x10, but if it's not padded, offsets should be 0
        assert simple_game.map_padding_offset_x == 0
        assert simple_game.map_padding_offset_y == 0
        assert simple_game.original_map_width == 10
        assert simple_game.original_map_height == 10
        
        bot = test_bot_class(simple_game, player=2, api_key="test-key")
        
        # Create a unit at position [5, 5]
        simple_game.create_unit('W', 5, 5, player=2)
        
        game_state_json = bot._serialize_game_state()
        
        # With no padding, coordinates should be identical
        assert len(game_state_json['player_units']) == 1
        unit = game_state_json['player_units'][0]
        assert unit['position'] == [5, 5]  # Same as padded position
        
        # Map dimensions should match grid dimensions
        assert game_state_json['map_width'] == 10
        assert game_state_json['map_height'] == 10
        # map_padding_applied field has been removed from serialization

    def test_action_history_uses_original_coordinates(self, padded_game):
        """Test that action history records use original/unpadded coordinates."""
        # Create a unit at padded position [7, 7] (original [0, 0])
        unit = padded_game.create_unit('W', 7, 7, player=1)
        
        # Check the create_unit action was recorded with original coordinates
        assert len(padded_game.action_history) >= 1
        create_action = padded_game.action_history[-1]
        assert create_action['type'] == 'create_unit'
        assert create_action['x'] == 0  # Original coordinate, not 7
        assert create_action['y'] == 0  # Original coordinate, not 7
        
        # Move the unit from padded [7, 7] to [7, 8] (original [0, 0] to [0, 1])
        unit.can_move = True
        padded_game.move_unit(unit, 7, 8)
        
        # Check the move action was recorded with original coordinates
        move_action = padded_game.action_history[-1]
        assert move_action['type'] == 'move'
        assert move_action['from_x'] == 0  # Original coordinate, not 7
        assert move_action['from_y'] == 0  # Original coordinate, not 7
        assert move_action['to_x'] == 0    # Original coordinate, not 7
        assert move_action['to_y'] == 1    # Original coordinate, not 8
        
        # Create an enemy unit at padded [7, 9] (original [0, 2])
        enemy = padded_game.create_unit('W', 7, 9, player=2)
        
        # Attack the enemy
        unit.can_attack = True
        padded_game.attack(unit, enemy)
        
        # Check the attack action was recorded with original coordinates
        attack_action = padded_game.action_history[-1]
        assert attack_action['type'] == 'attack'
        assert attack_action['attacker_pos'] == (0, 1)  # Original coords
        assert attack_action['target_pos'] == (0, 2)    # Original coords

    def test_action_history_no_padding(self, simple_game):
        """Test that action history works correctly when there's no padding."""
        # Create a unit at position [5, 5] (no padding, so original = padded)
        unit = simple_game.create_unit('W', 5, 5, player=1)
        
        # Check the create_unit action
        create_action = simple_game.action_history[-1]
        assert create_action['type'] == 'create_unit'
        assert create_action['x'] == 5  # Same as input since no padding
        assert create_action['y'] == 5
