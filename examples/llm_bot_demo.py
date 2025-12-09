#!/usr/bin/env python3
"""
Demo script showing how to use LLM bots in Reinforce Tactics.

This script demonstrates:
1. Setting up a game with an LLM bot
2. Playing a few turns
3. Handling bot actions

Requirements:
- Set API key as environment variable (e.g., OPENAI_API_KEY)
- Install the appropriate LLM package (e.g., pip install openai)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from reinforcetactics.core.game_state import GameState
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.game.llm_bot import OpenAIBot, ClaudeBot, GeminiBot


def main():
    """Run the LLM bot demo."""
    print("=" * 60)
    print("LLM Bot Demo for Reinforce Tactics")
    print("=" * 60)

    # Select bot type
    print("\nAvailable LLM bots:")
    print("1. OpenAI GPT (requires OPENAI_API_KEY)")
    print("2. Anthropic Claude (requires ANTHROPIC_API_KEY)")
    print("3. Google Gemini (requires GOOGLE_API_KEY)")

    choice = input("\nSelect bot type (1-3): ").strip()

    bot_class = None
    if choice == "1":
        bot_class = OpenAIBot
        print("Using OpenAI GPT bot")
    elif choice == "2":
        bot_class = ClaudeBot
        print("Using Anthropic Claude bot")
    elif choice == "3":
        bot_class = GeminiBot
        print("Using Google Gemini bot")
    else:
        print("Invalid choice. Exiting.")
        return

    # Load a map
    print("\nLoading map...")
    try:
        map_data = FileIO.load_map('maps/1v1/simple.csv')
        if map_data is None:
            print("Map not found, generating random map...")
            map_data = FileIO.generate_random_map(10, 10, num_players=2)
    except Exception:
        print("Error loading map, generating random map...")
        map_data = FileIO.generate_random_map(10, 10, num_players=2)

    # Create game state
    game = GameState(map_data, num_players=2)
    print(f"Game initialized with {game.grid.width}x{game.grid.height} map")

    # Create LLM bot
    print("\nCreating LLM bot...")
    try:
        bot = bot_class(game, player=2)
        print(f"✅ Bot created successfully: {bot.__class__.__name__}")
    except ValueError as e:
        print(f"❌ Error creating bot: {e}")
        print("\nMake sure to set the appropriate API key environment variable:")
        print("  export OPENAI_API_KEY='your-key'  # for OpenAI")
        print("  export ANTHROPIC_API_KEY='your-key'  # for Anthropic")
        print("  export GOOGLE_API_KEY='your-key'  # for Google")
        return
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return

    # Run a few turns
    print("\n" + "=" * 60)
    print("Running demo turns...")
    print("=" * 60)

    max_turns = 3
    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")
        print(f"Current player: {game.current_player}")
        print(f"Player 1 gold: {game.player_gold[1]}")
        print(f"Player 2 gold: {game.player_gold[2]}")

        if game.current_player == 2:
            print("\n🤖 LLM Bot is thinking...")
            try:
                bot.take_turn()
                print("✅ Bot completed turn")
            except Exception as e:
                print(f"❌ Bot error: {e}")

        # End turn
        game.end_turn()

        # Check game over
        if game.game_over:
            print(f"\n🎉 Game Over! Player {game.winner} wins!")
            break

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo play a full game with LLM bots:")
    print("1. Set your API key as an environment variable")
    print("2. Run main.py and select an LLM bot in the player configuration")
    print("3. Or use the bot programmatically as shown in this script")


if __name__ == "__main__":
    main()
