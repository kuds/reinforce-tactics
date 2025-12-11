"""
Bot Factory for Reinforce Tactics.

This module provides factory functions for creating bot instances,
eliminating duplication between start_new_game() and load_saved_game().
"""


def create_bot(game, player_num, bot_type, settings, model_path=None):
    """
    Create a single bot instance.

    Args:
        game: The GameState instance
        player_num: The player number for this bot
        bot_type: String identifier for bot type ('SimpleBot', 'OpenAIBot', etc.)
        settings: Settings instance for API keys
        model_path: Path to model file (required for ModelBot)

    Returns:
        Bot instance

    Raises:
        ValueError: If bot creation fails due to configuration issues
        ImportError: If required dependencies for bot type are missing
    """
    from reinforcetactics.game.bot import SimpleBot
    from reinforcetactics.game.llm_bot import OpenAIBot, ClaudeBot, GeminiBot

    if bot_type == 'SimpleBot':
        return SimpleBot(game, player=player_num)
    if bot_type == 'OpenAIBot':
        api_key = settings.get_api_key('openai') or None
        return OpenAIBot(game, player=player_num, api_key=api_key)
    if bot_type == 'ClaudeBot':
        api_key = settings.get_api_key('anthropic') or None
        return ClaudeBot(game, player=player_num, api_key=api_key)
    if bot_type == 'GeminiBot':
        api_key = settings.get_api_key('google') or None
        return GeminiBot(game, player=player_num, api_key=api_key)
    if bot_type == 'ModelBot':
        from reinforcetactics.game.model_bot import ModelBot
        if not model_path:
            raise ValueError("model_path is required for ModelBot")
        return ModelBot(game, player=player_num, model_path=model_path)
    print(f"⚠️  Unknown bot type '{bot_type}', using SimpleBot")
    return SimpleBot(game, player=player_num)


def create_bots_from_config(game, player_configs, settings):
    """
    Create bots based on player configurations.

    Args:
        game: The GameState instance
        player_configs: List of player configuration dictionaries
        settings: Settings instance for API keys

    Returns:
        Dictionary mapping player numbers to bot instances
    """
    bots = {}

    if not player_configs:
        return bots

    for i, config in enumerate(player_configs):
        player_num = i + 1
        if config['type'] == 'computer':
            bot_type = config.get('bot_type', 'SimpleBot')
            model_path = config.get('model_path', None)
            try:
                bots[player_num] = create_bot(game, player_num, bot_type, settings, model_path)
                print(f"Bot created for Player {player_num} ({bot_type})")
            except ValueError as e:
                print(f"❌ Error creating {bot_type} for Player {player_num}: {e}")
                print("   Falling back to SimpleBot")
                bots[player_num] = create_bot(game, player_num, 'SimpleBot', settings)
            except ImportError as e:
                print(f"❌ Missing dependency for {bot_type}: {e}")
                print("   Falling back to SimpleBot")
                bots[player_num] = create_bot(game, player_num, 'SimpleBot', settings)

    return bots
