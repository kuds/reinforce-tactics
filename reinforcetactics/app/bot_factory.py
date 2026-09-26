"""
Bot Factory for Reinforce Tactics.

This module provides factory functions for creating bot instances,
eliminating duplication between start_new_game() and load_saved_game().
"""

from pathlib import Path

# Retry limits for LLM bots created for GUI games (see create_bot).
GUI_LLM_RETRY_BUDGET_S = 15.0
GUI_LLM_MAX_FAILED_TURNS = 2
# One request may take this long before it counts as a (transient) timeout.
# The library default ("auto", up to 300 s) is sized for unattended runs of
# slow reasoning models; in the GUI a hung endpoint blocked the window for
# ~15 minutes per turn. 60 s still leaves room for a slow legitimate reply.
GUI_LLM_REQUEST_TIMEOUT_S = 60.0
# Attempts guaranteed whatever the budget; fast transient failures (429/5xx)
# are still retried within GUI_LLM_RETRY_BUDGET_S after this.
GUI_LLM_MAX_RETRIES = 1


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
    if bot_type in ("SimpleBot", "MediumBot", "AdvancedBot", "MasterBot"):
        return bot_type

    # For LLM bots (OpenAIBot, ClaudeBot, GeminiBot), use the model name
    if bot_type in ("OpenAIBot", "ClaudeBot", "GeminiBot"):
        return getattr(bot, "model", bot_type)

    # For ModelBot, use the base filename from model_path
    if bot_type == "ModelBot" and model_path:
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
    from reinforcetactics.game.bot_registry import player_type

    return player_type(bot_type)


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
    from reinforcetactics.game.bot_registry import build_scripted, canonical_name
    from reinforcetactics.game.llm_bot import ClaudeBot, GeminiBot, OpenAIBot

    # LLM turns run on the GUI thread, so an outage must give up quickly: the
    # library defaults (60 s of transient retries per call, 3 failed turns)
    # suit unattended tournaments but froze the window for minutes. After
    # GUI_LLM_MAX_FAILED_TURNS unanswered turns the bot raises LLMBotError
    # and InputHandler hands the seat to SimpleBot.
    llm_kwargs = {
        "retry_budget_s": GUI_LLM_RETRY_BUDGET_S,
        "max_consecutive_failed_turns": GUI_LLM_MAX_FAILED_TURNS,
        "request_timeout": GUI_LLM_REQUEST_TIMEOUT_S,
        "max_retries": GUI_LLM_MAX_RETRIES,
    }
    if bot_type == "OpenAIBot":
        api_key = settings.get_api_key("openai") or None
        return OpenAIBot(game, player=player_num, api_key=api_key, **llm_kwargs)
    if bot_type == "ClaudeBot":
        api_key = settings.get_api_key("anthropic") or None
        return ClaudeBot(game, player=player_num, api_key=api_key, **llm_kwargs)
    if bot_type == "GeminiBot":
        api_key = settings.get_api_key("google") or None
        return GeminiBot(game, player=player_num, api_key=api_key, **llm_kwargs)
    if bot_type == "ModelBot":
        from reinforcetactics.game.model_bot import ModelBot

        if not model_path:
            raise ValueError("model_path is required for ModelBot")
        return ModelBot(game, player=player_num, model_path=model_path)
    # Scripted bots ('SimpleBot' .. 'MasterBot' and their short-name
    # aliases) resolve through the registry. GUI bots take no rng — they
    # stay deterministic, matching historic behavior.
    try:
        return build_scripted(canonical_name(bot_type), game, player=player_num)
    except KeyError:
        pass
    print(f"⚠️  Unknown bot type '{bot_type}', using SimpleBot")
    return SimpleBot(game, player=player_num)


def create_bots_from_config(game, player_configs, settings, notices=None):
    """
    Create bots based on player configurations.

    A bot that cannot be built is replaced by SimpleBot rather than aborting
    the game, whatever the reason: a missing API key or unknown type
    (ValueError), a missing optional dependency (ImportError), or a ModelBot
    whose model file has moved since the game was saved (FileNotFoundError,
    which used to escape and abort loading the save).

    Updates player_configs with:
    - 'player_name': Display name for the player
    - 'player_type': Standardized type ('human', 'bot', 'llm', 'rl')
    - For LLM bots: 'temperature' and 'max_tokens' from bot instance

    Player name sources:
    - Human players: "Human"
    - SimpleBot/MediumBot/AdvancedBot: Class name (e.g., "SimpleBot")
    - LLM bots: Model name (e.g., "gpt-5-mini-2025-08-07", "claude-sonnet-4-6")
    - ModelBot: Base filename from model_path (e.g., "agent_v1")

    Args:
        game: The GameState instance
        player_configs: List of player configuration dictionaries
        settings: Settings instance for API keys
        notices: Optional list that receives one player-facing message per
            fallback, so the GUI can show it on screen (stdout is invisible
            to a GUI player).

    Returns:
        Dictionary mapping player numbers to bot instances
    """
    bots = {}

    if not player_configs:
        return bots

    for i, config in enumerate(player_configs):
        player_num = i + 1
        if config["type"] == "computer":
            bot_type = config.get("bot_type", "SimpleBot")
            model_path = config.get("model_path", None)
            try:
                bot = create_bot(game, player_num, bot_type, settings, model_path)
                bots[player_num] = bot
                config["player_name"] = get_player_name(bot, bot_type, model_path)
                config["player_type"] = get_player_type(bot_type)

                # Add LLM-specific fields
                if config["player_type"] == "llm":
                    config["temperature"] = getattr(bot, "temperature", None)
                    config["max_tokens"] = getattr(bot, "max_tokens", None)

                print(f"Bot created for Player {player_num} ({bot_type})")
            except Exception as e:
                reason = f"missing dependency: {e}" if isinstance(e, ImportError) else str(e)
                print(f"❌ Error creating {bot_type} for Player {player_num}: {reason}")
                print("   Falling back to SimpleBot")
                if notices is not None:
                    notices.append(f"Player {player_num}: {bot_type} unavailable ({reason}); using SimpleBot")
                bot = create_bot(game, player_num, "SimpleBot", settings)
                bots[player_num] = bot
                config["player_name"] = "SimpleBot"
                config["player_type"] = "bot"
        else:
            # Human player
            config["player_name"] = "Human"
            config["player_type"] = "human"

    return bots
