"""
Bot Factory for Reinforce Tactics.

This module provides factory functions for creating bot instances,
eliminating duplication between start_new_game() and load_saved_game().
"""

from pathlib import Path


def get_player_name(bot, bot_type, model_path=None):
    """
    Get the player name for a bot.

    Args:
        bot: The bot instance
        bot_type: String identifier for bot type ('SimpleBot', 'OpenAIBot', etc.)
        model_path: Path to model file (for ModelBot)

    Returns:
        String name for the player
    """
    # For basic bots (SimpleBot, MediumBot, AdvancedBot, MasterBot), use the class name
    if bot_type in ('SimpleBot', 'MediumBot', 'AdvancedBot', 'MasterBot'):
        return bot_type

    # For LLM bots (OpenAIBot, ClaudeBot, GeminiBot), use the model name
    if bot_type in ('OpenAIBot', 'ClaudeBot', 'GeminiBot'):
        return getattr(bot, 'model', bot_type)

    # For ModelBot, use the base filename from model_path
    if bot_type == 'ModelBot' and model_path:
        return Path(model_path).stem

    # Fallback to bot_type
    return bot_type


def get_player_type(bot_type):
    """
    Get the standardized player type for a bot.

    Args:
        bot_type: String identifier for bot type ('SimpleBot', 'OpenAIBot', etc.)

    Returns:
        Player type string: 'bot', 'llm', or 'rl'
    """
    # LLM bots
    if bot_type in ('OpenAIBot', 'ClaudeBot', 'GeminiBot'):
        return 'llm'

    # RL model bots
    if bot_type == 'ModelBot':
        return 'rl'

    # Standard bots (SimpleBot, MediumBot, AdvancedBot)
    return 'bot'


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
    from reinforcetactics.game.bot import SimpleBot, MediumBot, AdvancedBot, MasterBot
    from reinforcetactics.game.llm_bot import OpenAIBot, ClaudeBot, GeminiBot

    if bot_type == 'SimpleBot':
        return SimpleBot(game, player=player_num)
    if bot_type == 'MediumBot':
        return MediumBot(game, player=player_num)
    if bot_type == 'AdvancedBot':
        return AdvancedBot(game, player=player_num)
    if bot_type == 'MasterBot':
        return MasterBot(game, player=player_num)
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

    Updates player_configs with:
    - 'player_name': Display name for the player
    - 'player_type': Standardized type ('human', 'bot', 'llm', 'rl')
    - For LLM bots: 'temperature' and 'max_tokens' from bot instance

    Player name sources:
    - Human players: "Human"
    - SimpleBot/MediumBot/AdvancedBot: Class name (e.g., "SimpleBot")
    - LLM bots: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
    - ModelBot: Base filename from model_path (e.g., "agent_v1")

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
                bot = create_bot(game, player_num, bot_type, settings, model_path)
                bots[player_num] = bot
                config['player_name'] = get_player_name(bot, bot_type, model_path)
                config['player_type'] = get_player_type(bot_type)

                # Add LLM-specific fields
                if config['player_type'] == 'llm':
                    config['temperature'] = getattr(bot, 'temperature', None)
                    config['max_tokens'] = getattr(bot, 'max_tokens', None)

                print(f"Bot created for Player {player_num} ({bot_type})")
            except ValueError as e:
                print(f"❌ Error creating {bot_type} for Player {player_num}: {e}")
                print("   Falling back to SimpleBot")
                bot = create_bot(game, player_num, 'SimpleBot', settings)
                bots[player_num] = bot
                config['player_name'] = 'SimpleBot'
                config['player_type'] = 'bot'
            except ImportError as e:
                print(f"❌ Missing dependency for {bot_type}: {e}")
                print("   Falling back to SimpleBot")
                bot = create_bot(game, player_num, 'SimpleBot', settings)
                bots[player_num] = bot
                config['player_name'] = 'SimpleBot'
                config['player_type'] = 'bot'
        else:
            # Human player
            config['player_name'] = 'Human'
            config['player_type'] = 'human'

    return bots
