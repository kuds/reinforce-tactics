"""
Bot descriptors and factory for tournament participants.

This module provides a unified way to describe and create bot instances
for tournament play.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from reinforcetactics.core.game_state import GameState

logger = logging.getLogger(__name__)


class BotType(Enum):
    """Types of bots that can participate in tournaments."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    ADVANCED = "advanced"
    LLM = "llm"
    MODEL = "model"


@dataclass
class BotDescriptor:
    """
    Describes a bot that can participate in a tournament.

    This is a lightweight descriptor that holds configuration for creating
    bot instances. The actual bot is created lazily when needed.

    Attributes:
        name: Display name for the bot
        bot_type: Type of bot (simple, medium, advanced, llm, model)
        model: For LLM bots, the model name (e.g., "gpt-4")
        provider: For LLM bots, the provider (openai, anthropic, google)
        model_path: For model bots, path to the trained model
        temperature: For LLM bots, sampling temperature
        max_tokens: For LLM bots, max tokens in response
        api_key: Optional API key (if not using environment variables)
        extra_kwargs: Additional keyword arguments for bot creation
    """

    name: str
    bot_type: BotType
    model: Optional[str] = None
    provider: Optional[str] = None
    model_path: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: int = 8000
    api_key: Optional[str] = None
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Class-level cache for bot classes
    _bot_classes: Dict[str, Type] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Validate the descriptor after initialization."""
        if isinstance(self.bot_type, str):
            self.bot_type = BotType(self.bot_type)

    @classmethod
    def simple_bot(cls, name: str = "SimpleBot") -> "BotDescriptor":
        """Create a SimpleBot descriptor."""
        return cls(name=name, bot_type=BotType.SIMPLE)

    @classmethod
    def medium_bot(cls, name: str = "MediumBot") -> "BotDescriptor":
        """Create a MediumBot descriptor."""
        return cls(name=name, bot_type=BotType.MEDIUM)

    @classmethod
    def advanced_bot(cls, name: str = "AdvancedBot") -> "BotDescriptor":
        """Create an AdvancedBot descriptor."""
        return cls(name=name, bot_type=BotType.ADVANCED)

    @classmethod
    def llm_bot(
        cls,
        name: str,
        provider: str,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: int = 8000,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> "BotDescriptor":
        """
        Create an LLM bot descriptor.

        Args:
            name: Display name
            provider: LLM provider (openai, anthropic, google)
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            temperature: Sampling temperature
            max_tokens: Max response tokens
            api_key: Optional API key
            **kwargs: Additional arguments passed to bot
        """
        return cls(
            name=name,
            bot_type=BotType.LLM,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            extra_kwargs=kwargs,
        )

    @classmethod
    def model_bot(cls, name: str, model_path: str) -> "BotDescriptor":
        """
        Create a trained model bot descriptor.

        Args:
            name: Display name
            model_path: Path to the trained model file
        """
        return cls(name=name, bot_type=BotType.MODEL, model_path=model_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotDescriptor":
        """
        Create BotDescriptor from dictionary.

        Args:
            data: Dictionary with bot configuration

        Returns:
            BotDescriptor instance
        """
        bot_type = data.get("type", data.get("bot_type", "simple"))
        return cls(
            name=data["name"],
            bot_type=BotType(bot_type),
            model=data.get("model"),
            provider=data.get("provider"),
            model_path=data.get("model_path"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens", 8000),
            api_key=data.get("api_key"),
            extra_kwargs=data.get("extra_kwargs", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "type": self.bot_type.value,
        }
        if self.model:
            result["model"] = self.model
        if self.provider:
            result["provider"] = self.provider
        if self.model_path:
            result["model_path"] = self.model_path
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.max_tokens != 8000:
            result["max_tokens"] = self.max_tokens
        if self.extra_kwargs:
            result["extra_kwargs"] = self.extra_kwargs
        return result

    def get_display_info(self) -> str:
        """Get a human-readable description of the bot."""
        if self.bot_type == BotType.LLM:
            model_str = f" ({self.model})" if self.model else ""
            temp_str = f" [temp={self.temperature}]" if self.temperature else ""
            return f"{self.name}: {self.provider}{model_str}{temp_str}"
        elif self.bot_type == BotType.MODEL:
            return f"{self.name}: Model ({self.model_path})"
        else:
            return f"{self.name}: {self.bot_type.value.title()}Bot"

    def __repr__(self) -> str:
        return f"BotDescriptor(name={self.name!r}, type={self.bot_type.value!r})"


def create_bot_instance(
    descriptor: BotDescriptor,
    game_state: "GameState",
    player: int,
    log_conversations: bool = False,
    conversation_log_dir: Optional[str] = None,
    game_session_id: Optional[str] = None,
    should_reason: bool = False,
) -> Any:
    """
    Create a bot instance from a descriptor.

    Args:
        descriptor: Bot descriptor
        game_state: Game state instance
        player: Player number (1 or 2)
        log_conversations: Enable conversation logging for LLM bots
        conversation_log_dir: Directory for conversation logs
        game_session_id: Unique session ID for logging
        should_reason: Enable reasoning output for LLM bots

    Returns:
        Bot instance ready to play

    Raises:
        ValueError: If bot type is unknown
        ImportError: If required package is not installed
    """
    bot_type = descriptor.bot_type

    if bot_type == BotType.SIMPLE:
        from reinforcetactics.game.bot import SimpleBot

        return SimpleBot(game_state, player)

    elif bot_type == BotType.MEDIUM:
        from reinforcetactics.game.bot import MediumBot

        return MediumBot(game_state, player)

    elif bot_type == BotType.ADVANCED:
        from reinforcetactics.game.bot import AdvancedBot

        return AdvancedBot(game_state, player)

    elif bot_type == BotType.LLM:
        return _create_llm_bot(
            descriptor,
            game_state,
            player,
            log_conversations,
            conversation_log_dir,
            game_session_id,
            should_reason,
        )

    elif bot_type == BotType.MODEL:
        from reinforcetactics.game.model_bot import ModelBot

        if not descriptor.model_path:
            raise ValueError(f"Model bot {descriptor.name} requires model_path")
        return ModelBot(game_state, player, model_path=descriptor.model_path)

    else:
        raise ValueError(f"Unknown bot type: {bot_type}")


def _create_llm_bot(
    descriptor: BotDescriptor,
    game_state: "GameState",
    player: int,
    log_conversations: bool,
    conversation_log_dir: Optional[str],
    game_session_id: Optional[str],
    should_reason: bool,
) -> Any:
    """Create an LLM bot instance."""
    provider = descriptor.provider

    if not provider:
        raise ValueError(f"LLM bot {descriptor.name} requires provider")

    # Build kwargs for bot creation
    kwargs = {
        "game_state": game_state,
        "player": player,
        "model": descriptor.model,
        "max_tokens": descriptor.max_tokens,
        "should_reason": should_reason,
        "log_conversations": log_conversations,
        "conversation_log_dir": conversation_log_dir,
        "game_session_id": game_session_id,
    }

    if descriptor.temperature is not None:
        kwargs["temperature"] = descriptor.temperature

    if descriptor.api_key:
        kwargs["api_key"] = descriptor.api_key

    # Add any extra kwargs
    kwargs.update(descriptor.extra_kwargs)

    if provider == "openai":
        from reinforcetactics.game.llm_bot import OpenAIBot

        return OpenAIBot(**kwargs)

    elif provider == "anthropic":
        from reinforcetactics.game.llm_bot import ClaudeBot

        return ClaudeBot(**kwargs)

    elif provider == "google":
        from reinforcetactics.game.llm_bot import GeminiBot

        return GeminiBot(**kwargs)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def discover_builtin_bots() -> List[BotDescriptor]:
    """
    Discover built-in rule-based bots.

    Returns:
        List of BotDescriptors for SimpleBot, MediumBot, AdvancedBot
    """
    return [
        BotDescriptor.simple_bot("SimpleBot"),
        BotDescriptor.medium_bot("MediumBot"),
        BotDescriptor.advanced_bot("AdvancedBot"),
    ]


def discover_llm_bots(
    test_keys: bool = True,
    log_conversations: bool = False,
    conversation_log_dir: Optional[str] = None,
) -> List[BotDescriptor]:
    """
    Discover available LLM bots based on configured API keys.

    Args:
        test_keys: If True, test API keys before adding bots
        log_conversations: Enable conversation logging
        conversation_log_dir: Directory for logs

    Returns:
        List of BotDescriptors for available LLM bots
    """
    from reinforcetactics.utils.settings import get_settings

    bots = []
    settings = get_settings()

    # Check OpenAI
    openai_key = settings.get_api_key("openai")
    if openai_key:
        if not test_keys or _test_openai_key(openai_key):
            bots.append(
                BotDescriptor.llm_bot(
                    name="OpenAIBot",
                    provider="openai",
                    model="gpt-4",
                    api_key=openai_key,
                )
            )
            logger.info("Discovered OpenAIBot")

    # Check Anthropic
    anthropic_key = settings.get_api_key("anthropic")
    if anthropic_key:
        if not test_keys or _test_anthropic_key(anthropic_key):
            bots.append(
                BotDescriptor.llm_bot(
                    name="ClaudeBot",
                    provider="anthropic",
                    model="claude-3-sonnet-20240229",
                    api_key=anthropic_key,
                )
            )
            logger.info("Discovered ClaudeBot")

    # Check Google
    google_key = settings.get_api_key("google")
    if google_key:
        if not test_keys or _test_google_key(google_key):
            bots.append(
                BotDescriptor.llm_bot(
                    name="GeminiBot",
                    provider="google",
                    model="gemini-pro",
                    api_key=google_key,
                )
            )
            logger.info("Discovered GeminiBot")

    return bots


def discover_model_bots(
    models_dir: str, test_models: bool = True
) -> List[BotDescriptor]:
    """
    Discover trained model bots from a directory.

    Args:
        models_dir: Directory containing .zip model files
        test_models: If True, test that models can be loaded

    Returns:
        List of BotDescriptors for valid model bots
    """
    bots = []
    models_path = Path(models_dir)

    if not models_path.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return bots

    # Find all .zip files
    model_files = list(models_path.glob("*.zip"))

    for model_file in model_files:
        if not test_models or _test_model_file(model_file):
            bot_name = f"Model_{model_file.stem}"
            bots.append(BotDescriptor.model_bot(bot_name, str(model_file)))
            logger.info(f"Discovered model bot: {bot_name}")

    return bots


def discover_all_bots(
    models_dir: Optional[str] = None,
    test_keys: bool = True,
    test_models: bool = True,
    include_llm: bool = True,
    include_models: bool = True,
) -> List[BotDescriptor]:
    """
    Discover all available bots.

    Args:
        models_dir: Directory for trained models
        test_keys: Test API keys before adding LLM bots
        test_models: Test model files before adding model bots
        include_llm: Include LLM bots in discovery
        include_models: Include model bots in discovery

    Returns:
        List of all discovered BotDescriptors
    """
    bots = discover_builtin_bots()

    if include_llm:
        bots.extend(discover_llm_bots(test_keys=test_keys))

    if include_models and models_dir:
        bots.extend(discover_model_bots(models_dir, test_models=test_models))

    logger.info(f"Discovered {len(bots)} bots total")
    return bots


def _test_openai_key(api_key: str) -> bool:
    """Test if OpenAI API key is valid."""
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        logger.warning(f"OpenAI API key test failed: {e}")
        return False


def _test_anthropic_key(api_key: str) -> bool:
    """Test if Anthropic API key is valid."""
    try:
        import anthropic

        anthropic.Anthropic(api_key=api_key)
        return True
    except Exception as e:
        logger.warning(f"Anthropic API key test failed: {e}")
        return False


def _test_google_key(api_key: str) -> bool:
    """Test if Google API key is valid."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        list(genai.list_models())
        return True
    except Exception as e:
        logger.warning(f"Google API key test failed: {e}")
        return False


def _test_model_file(model_path: Path) -> bool:
    """Test if a model file can be loaded."""
    try:
        from reinforcetactics.game.model_bot import ModelBot
        from reinforcetactics.core.game_state import GameState
        from reinforcetactics.utils.file_io import FileIO

        # Create a dummy game state for testing
        map_data = FileIO.load_map("maps/1v1/6x6_beginner.csv")
        dummy_state = GameState(map_data, num_players=2)

        # Try to create the bot
        bot = ModelBot(dummy_state, player=2, model_path=str(model_path))
        return bot.model is not None
    except Exception as e:
        logger.warning(f"Failed to load model {model_path.name}: {e}")
        return False
