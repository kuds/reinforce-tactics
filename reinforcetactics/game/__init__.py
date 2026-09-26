"""
Game mechanics module.
"""

# GameMechanics moved to reinforcetactics.core.mechanics (the engine layer);
# re-exported here so existing `from reinforcetactics.game import GameMechanics`
# imports keep working.
from reinforcetactics.core.mechanics import GameMechanics
from reinforcetactics.game.bot import NoopBot, RandomBot, SimpleBot
from reinforcetactics.game.bot_base import ABILITY_PROVIDERS, BaseBot, BotUnitMixin
from reinforcetactics.game.llm_bot import ClaudeBot, GeminiBot, LLMBot, LLMBotError, OpenAIBot
from reinforcetactics.game.llm_prompts import (
    PROMPT_BASIC,
    PROMPT_STRATEGIC,
    PROMPT_TWO_PHASE_EXECUTE,
    PROMPT_TWO_PHASE_PLAN,
    get_prompt,
    list_prompts,
    register_prompt,
)
from reinforcetactics.game.model_bot import ModelBot

__all__ = [
    "GameMechanics",
    "BaseBot",
    "BotUnitMixin",
    "ABILITY_PROVIDERS",
    "NoopBot",
    "RandomBot",
    "SimpleBot",
    "LLMBot",
    "LLMBotError",
    "OpenAIBot",
    "ClaudeBot",
    "GeminiBot",
    "ModelBot",
    # Prompts
    "PROMPT_BASIC",
    "PROMPT_STRATEGIC",
    "PROMPT_TWO_PHASE_PLAN",
    "PROMPT_TWO_PHASE_EXECUTE",
    "get_prompt",
    "list_prompts",
    "register_prompt",
]
