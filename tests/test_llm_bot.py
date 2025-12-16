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

    def test_no_logging_when_not_debug_level(self, simple_game, test_bot_class):
        """Test that no logging occurs when logging level is not DEBUG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Set logging level to INFO (not DEBUG)
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.INFO)

            try:
                # Mock _call_llm to avoid actual API calls
                response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
                with patch.object(bot, '_call_llm', return_value=response):
                    bot.take_turn()

                # No files should be created
                log_files = list(Path(tmpdir).glob("*.json"))
                assert len(log_files) == 0
            finally:
                logger.setLevel(original_level)

    def test_logging_at_debug_level(self, simple_game, test_bot_class):
        """Test that logging occurs when enabled and at DEBUG level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
                # Mock _call_llm to avoid actual API calls
                response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
                with patch.object(bot, '_call_llm', return_value=response):
                    bot.take_turn()

                # One file should be created
                log_files = list(Path(tmpdir).glob("*.json"))
                assert len(log_files) == 1
            finally:
                logger.setLevel(original_level)

    def test_json_file_structure(self, simple_game, test_bot_class):
        """Test that JSON log file has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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

            finally:
                logger.setLevel(original_level)

    def test_custom_log_directory(self, simple_game, test_bot_class):
        """Test that custom log directory is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "my_custom_logs"
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=str(custom_dir))

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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
            finally:
                logger.setLevel(original_level)

    def test_filename_format(self, simple_game, test_bot_class):
        """Test that log filename has correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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
            finally:
                logger.setLevel(original_level)

    def test_game_session_id_parameter(self, simple_game, test_bot_class):
        """Test that custom game_session_id is used when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_session_id = "test_session_12345"
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir,
                                game_session_id=custom_session_id)

            assert bot.game_session_id == custom_session_id

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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
            finally:
                logger.setLevel(original_level)

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

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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
            finally:
                logger.setLevel(original_level)

    def test_pretty_print_logs_enabled(self, simple_game, test_bot_class):
        """Test that pretty_print_logs=True creates indented JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir,
                                pretty_print_logs=True)

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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
            finally:
                logger.setLevel(original_level)

    def test_pretty_print_logs_disabled(self, simple_game, test_bot_class):
        """Test that pretty_print_logs=False creates compact JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir,
                                pretty_print_logs=False)

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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
            finally:
                logger.setLevel(original_level)

    def test_logging_error_handling(self, simple_game, test_bot_class):
        """Test that logging errors don't break the bot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
                # Mock Path.mkdir to raise an exception
                with patch('reinforcetactics.game.llm_bot.Path.mkdir', side_effect=OSError("Permission denied")):
                    # Mock _call_llm to avoid actual API calls
                    # This should not raise an exception even if logging fails
                    response = '{"reasoning": "test", "actions": [{"type": "END_TURN"}]}'
                    with patch.object(bot, '_call_llm', return_value=response):
                        bot.take_turn()  # Should complete without exception

            finally:
                logger.setLevel(original_level)

    def test_multiple_turns_create_single_file(self, simple_game, test_bot_class):
        """Test that multiple turns create a single log file with all turns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot = test_bot_class(simple_game, player=2, api_key="test-key",
                                log_conversations=True,
                                conversation_log_dir=tmpdir)

            # Set logging level to DEBUG
            logger = logging.getLogger('reinforcetactics.game.llm_bot')
            original_level = logger.level
            logger.setLevel(logging.DEBUG)

            try:
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
            finally:
                logger.setLevel(original_level)
