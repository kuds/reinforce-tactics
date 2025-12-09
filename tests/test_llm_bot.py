"""Tests for LLM bot module."""
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
            assert 'gpt' in TestBot._get_default_model(Mock()).lower()  # pylint: disable=protected-access


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
            assert 'claude' in TestBot._get_default_model(Mock()).lower()  # pylint: disable=protected-access


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
            assert 'gemini' in TestBot._get_default_model(Mock()).lower()  # pylint: disable=protected-access
