"""
Game mechanics module.
"""
from reinforcetactics.game.mechanics import GameMechanics
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.game.llm_bot import LLMBot, OpenAIBot, ClaudeBot, GeminiBot
from reinforcetactics.game.model_bot import ModelBot
from reinforcetactics.game.llm_prompts import (
    PROMPT_BASIC,
    PROMPT_STRATEGIC,
    PROMPT_TWO_PHASE_PLAN,
    PROMPT_TWO_PHASE_EXECUTE,
    get_prompt,
    list_prompts,
    register_prompt,
)

__all__ = [
    'GameMechanics',
    'SimpleBot',
    'LLMBot',
    'OpenAIBot',
    'ClaudeBot',
    'GeminiBot',
    'ModelBot',
    # Prompts
    'PROMPT_BASIC',
    'PROMPT_STRATEGIC',
    'PROMPT_TWO_PHASE_PLAN',
    'PROMPT_TWO_PHASE_EXECUTE',
    'get_prompt',
    'list_prompts',
    'register_prompt',
]
