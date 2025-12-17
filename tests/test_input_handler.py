"""Tests for InputHandler class, specifically bot turn processing."""
from unittest.mock import Mock
import pygame
import pytest
from game.input_handler import InputHandler
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot


@pytest.fixture
def mock_game():
    """Create a mock game state for testing."""
    game = Mock(spec=GameState)
    game.current_player = 1
    game.game_over = False
    game.end_turn = Mock()
    game.grid = Mock()
    game.grid.width = 10
    game.grid.height = 10
    return game


@pytest.fixture
def mock_renderer():
    """Create a mock renderer for testing."""
    renderer = Mock()
    renderer.screen = Mock()
    renderer.end_turn_button = pygame.Rect(0, 0, 100, 50)
    renderer.resign_button = pygame.Rect(0, 60, 100, 50)
    return renderer


@pytest.fixture
def simple_bot():
    """Create a simple bot for testing."""
    bot = Mock(spec=SimpleBot)
    bot.take_turn = Mock()
    return bot


class TestInputHandlerBotTurns:
    """Test InputHandler bot turn processing."""

    def test_process_bot_turns_does_not_double_end_turn(self, mock_game, mock_renderer, simple_bot):
        """
        Test that _process_bot_turns() does not call end_turn() since bots already do it.

        This test verifies that when a bot takes its turn, end_turn() is not called
        by the InputHandler, as all bot implementations already call end_turn() internally.
        """
        # Setup: Bot is player 2
        bots = {2: simple_bot}
        handler = InputHandler(mock_game, mock_renderer, bots, num_players=2)

        # Set current player to bot player
        mock_game.current_player = 2

        # Track end_turn calls
        end_turn_call_count = 0
        original_end_turn = mock_game.end_turn

        def count_end_turn_calls(*args, **kwargs):
            nonlocal end_turn_call_count
            end_turn_call_count += 1
            # Simulate end_turn changing current_player to next player
            mock_game.current_player = 1  # Switch to human player
            return original_end_turn(*args, **kwargs)

        mock_game.end_turn = Mock(side_effect=count_end_turn_calls)

        # Make bot.take_turn() simulate what real bots do: call end_turn() internally
        def bot_take_turn_with_end_turn():
            # Simulate bot calling end_turn() like real bots do
            mock_game.end_turn()

        simple_bot.take_turn = Mock(side_effect=bot_take_turn_with_end_turn)

        # Execute bot turn processing
        handler._process_bot_turns()

        # Verify bot's take_turn was called once
        assert simple_bot.take_turn.call_count == 1, \
            f"Bot's take_turn should be called once, but was called {simple_bot.take_turn.call_count} times"

        # Verify end_turn is only called once (by the bot internally)
        # Before the fix: end_turn was called twice (once by bot, once by InputHandler)
        # After the fix: end_turn is called once (only by bot)
        assert end_turn_call_count == 1, \
            f"end_turn should be called once (by bot only), but was called {end_turn_call_count} times. " \
            f"This indicates InputHandler is also calling end_turn(), which is a bug."

    def test_process_bot_turns_handles_multiple_consecutive_bots(self, mock_game, mock_renderer):
        """Test that multiple consecutive bot turns are processed correctly."""
        # Setup: Two bots
        bot1 = Mock(spec=SimpleBot)
        bot2 = Mock(spec=SimpleBot)

        bots = {2: bot1, 3: bot2}
        handler = InputHandler(mock_game, mock_renderer, bots, num_players=3)

        # Track end_turn calls
        end_turn_call_count = [0]

        # Simulate: Current player is bot 2, then bot 3, then human player 1
        def simulate_end_turn():
            end_turn_call_count[0] += 1
            # First call switches from player 2 to player 3
            if end_turn_call_count[0] == 1:
                mock_game.current_player = 3
            # Second call switches from player 3 to player 1 (human)
            elif end_turn_call_count[0] == 2:
                mock_game.current_player = 1

        mock_game.current_player = 2  # Start with bot 2
        mock_game.end_turn = Mock(side_effect=simulate_end_turn)

        # Make bots call end_turn() like real bots do
        def bot1_take_turn():
            mock_game.end_turn()

        def bot2_take_turn():
            mock_game.end_turn()

        bot1.take_turn = Mock(side_effect=bot1_take_turn)
        bot2.take_turn = Mock(side_effect=bot2_take_turn)

        # Execute
        handler._process_bot_turns()

        # Verify both bots took their turns
        assert bot1.take_turn.call_count == 1
        assert bot2.take_turn.call_count == 1

        # Verify end_turn is called the correct number of times
        # After fix: end_turn is called 2 times (once by each bot internally)
        # Before fix: was called 4 times (2 by InputHandler + 2 by bots)
        assert mock_game.end_turn.call_count == 2, \
            f"end_turn should be called 2 times (once by each bot), but was called {mock_game.end_turn.call_count} times"

    def test_process_bot_turns_stops_on_game_over(self, mock_game, mock_renderer, simple_bot):
        """Test that bot turn processing stops when game is over."""
        bots = {2: simple_bot}
        handler = InputHandler(mock_game, mock_renderer, bots, num_players=2)

        # Set current player to bot and game over
        mock_game.current_player = 2
        mock_game.game_over = True

        # Execute
        handler._process_bot_turns()

        # Verify bot didn't take turn since game is over
        assert simple_bot.take_turn.call_count == 0
        assert mock_game.end_turn.call_count == 0

    def test_process_bot_turns_safety_limit(self, mock_game, mock_renderer, simple_bot):
        """Test that bot turn processing has a safety limit to prevent infinite loops."""
        bots = {2: simple_bot}
        handler = InputHandler(mock_game, mock_renderer, bots, num_players=2)

        # Setup: Bot player stays as current player (simulating a bug)
        mock_game.current_player = 2

        # Execute
        handler._process_bot_turns()

        # Verify there's a safety limit (num_players * 2 = 4 in this case)
        # Should stop after max_bot_turns iterations
        assert simple_bot.take_turn.call_count <= 4
