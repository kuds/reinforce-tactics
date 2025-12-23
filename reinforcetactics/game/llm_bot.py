"""
LLM-powered bots for playing Reinforce Tactics using OpenAI, Claude, and Gemini.
"""
import os
import json
import re
import time
import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from reinforcetactics.constants import UNIT_DATA
from reinforcetactics import __version__

# Configure logging
logger = logging.getLogger(__name__)


# Supported models for each provider
OPENAI_MODELS = [
    # GPT-4o family (latest, most capable)
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4o-2024-11-20',
    'gpt-4o-2024-08-06',
    'gpt-4o-2024-05-13',
    'gpt-4o-mini-2024-07-18',
    # GPT-4 Turbo (high performance)
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-turbo-preview',
    'gpt-4-0125-preview',
    'gpt-4-1106-preview',
    # GPT-4 (stable, proven)
    'gpt-4',
    'gpt-4-0613',
    # GPT-3.5 Turbo (fast and economical)
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-1106',
    # O1 Reasoning models (advanced reasoning)
    'o1',
    'o1-2024-12-17',
    'o1-mini',
    'o1-mini-2024-09-12',
    'o1-preview',
    'o1-preview-2024-09-12',
    # O3 models (if available)
    'o3-mini',
    'o3-mini-2025-01-31',
    'gpt-5.2-2025-12-11',
    'gpt-5-mini-2025-08-07',
]

ANTHROPIC_MODELS = [
    # Claude 4 (latest generation)
    'claude-sonnet-4-5-20250929',
    'claude-sonnet-4-20250514',
    # Claude 3.5 (high performance)
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-3-5-haiku-20241022',
    # Claude 3 (proven models)
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-haiku-4-5-20251001',
    'claude-opus-4-5-20251101',
]

GEMINI_MODELS = [
    # Gemini 2.0 (latest generation)
    'gemini-2.0-flash',
    'gemini-2.0-flash-exp',
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash-thinking-exp',
    # Gemini 1.5 (stable and capable)
    'gemini-1.5-pro',
    'gemini-1.5-pro-latest',
    'gemini-1.5-flash',
    'gemini-1.5-flash-latest',
    'gemini-1.5-flash-8b',
    # Gemini 1.0 (legacy)
    'gemini-1.0-pro',
    'gemini-pro',
    'gemini-3-pro-preview',
    'gemini-3-flash-preview',
]


# System prompt explaining the game rules (compact version)
SYSTEM_PROMPT = """Reinforce Tactics: Turn-based strategy. Win by capturing enemy HQ or eliminating all enemies.

UNITS (Cost/HP/Atk/Def/Mv):
W=Warrior(200/15/10/6/3): Melee only
M=Mage(250/10/8-12/4/2): Range 1-2, can PARALYZE
C=Cleric(200/8/2/4/2): HEAL allies, CURE paralysis
A=Archer(250/15/5/1/3): Range 1-3(mountain), no adjacent attack, melee can't counter

BUILDINGS: h=HQ(150g,lose=defeat), b=Building(100g,recruit), t=Tower(50g)

ACTIONS: CREATE_UNIT, MOVE, ATTACK, PARALYZE(M), HEAL(C), CURE(C), SEIZE, END_TURN

RULES:
- Orthogonal attacks only (not diagonal)
- Counter-attacks occur except: melee vs Archer
- Move+attack or attack+move allowed
- One unit per tile
- Actions execute sequentially
- Defend HQ if enemies within 2-3 tiles

STATE FORMAT: units have {id,t(type),pos,hp,max,mv,atk,para?}, buildings are [type,x,y], actions.create=[type,x,y,cost], move/attack/etc=[unit_id,x,y]

Reply with JSON only."""


class LLMBot(ABC):  # pylint: disable=too-few-public-methods
    """
    Abstract base class for LLM-powered bots.

    This class provides the foundation for bots that use Large Language Models
    to play Reinforce Tactics. It handles game state serialization, API communication,
    and action execution.

    Subclasses must implement provider-specific methods for API key handling
    and model invocation.
    """

    def __init__(self, game_state, player: int = 2, api_key: Optional[str] = None,
                 model: Optional[str] = None, max_retries: int = 3,
                 log_conversations: bool = False,
                 conversation_log_dir: Optional[str] = None,
                 game_session_id: Optional[str] = None,
                 pretty_print_logs: bool = True,
                 stateful: bool = False,
                 should_reason: bool = False):
        """
        Initialize the LLM bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot (default 2)
            api_key: API key for the LLM provider (optional, uses env var if not provided)
            model: Model name to use (optional, uses default if not provided)
            max_retries: Maximum number of retries for API calls
            log_conversations: Enable conversation logging to JSON files (default False)
            conversation_log_dir: Directory for conversation logs (default: logs/llm_conversations/)
            game_session_id: Unique game session identifier (default: auto-generated)
            pretty_print_logs: Format JSON logs with indentation for readability (default True)
            stateful: Maintain conversation history across turns (default False)
            should_reason: Include reasoning field in response format (default False).
                When True, includes "reasoning" field prompting for strategy explanation.
                When False, the reasoning field is omitted entirely from the prompt.
        """
        self.game_state = game_state
        self.bot_player = player
        self.api_key = api_key or self._get_api_key_from_env()
        self.model = model or self._get_default_model()
        self.max_retries = max_retries
        self.log_conversations = log_conversations
        self.conversation_log_dir = conversation_log_dir or 'logs/llm_conversations/'
        self.pretty_print_logs = pretty_print_logs
        self.stateful = stateful
        self.should_reason = should_reason

        # Initialize conversation history for stateful mode
        self.conversation_history = []

        # Initialize token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Per-call token tracking (set by subclasses that support it)
        self._last_input_tokens = 0
        self._last_output_tokens = 0

        # Generate or use provided game session ID
        if game_session_id:
            self.game_session_id = game_session_id
        else:
            # Generate unique session ID: timestamp + random component
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_component = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
            self.game_session_id = f"{timestamp}_{random_component}"

        if not self.api_key:
            raise ValueError(
                f"API key not provided. Set {self._get_env_var_name()} environment variable "
                f"or pass api_key parameter."
            )

        # Validate model (warning only, to support newly released models)
        self._validate_model()

    @abstractmethod
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""

    @abstractmethod
    def _get_env_var_name(self) -> str:
        """Get the name of the environment variable for the API key."""

    @abstractmethod
    def _get_default_model(self) -> str:
        """Get the default model name."""

    @abstractmethod
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API and return the response text."""

    @abstractmethod
    def _get_supported_models(self) -> List[str]:
        """Get the list of supported models for this provider."""

    def _validate_model(self) -> None:
        """
        Validate that the requested model is in the supported list.

        Logs a warning if the model is not in the known supported list,
        but does not raise an error to allow for newly released models.
        """
        supported_models = self._get_supported_models()
        if self.model not in supported_models:
            logger.warning(
                "Model '%s' not in known supported models. "
                "It may still work if newly released. "
                "Supported models: %s",
                self.model,
                ', '.join(supported_models[:5]) + '...'
            )

    def _log_conversation_to_json(self, system_prompt: str, user_prompt: str,
                                   assistant_response: str,
                                   input_tokens: int = 0,
                                   output_tokens: int = 0) -> None:
        """
        Log the conversation to a JSON file (single file per game).

        Only logs if log_conversations is True.
        Creates a single log file per game with all turns appended.

        Args:
            system_prompt: The system prompt sent to the LLM
            user_prompt: The user prompt (formatted game state)
            assistant_response: The LLM's response
            input_tokens: Number of input tokens used for this turn (Claude only)
            output_tokens: Number of output tokens used for this turn (Claude only)
        """
        # Only log if enabled
        if not self.log_conversations:
            return

        try:
            # Create log directory if it doesn't exist
            log_dir = Path(self.conversation_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Get provider name from class name
            provider = self.__class__.__name__.replace('Bot', '')

            # Generate filename: game_{session_id}_player{player}_model{model}.json
            safe_model = self.model.replace('/', '_').replace(':', '_')
            filename = f"game_{self.game_session_id}_player{self.bot_player}_model{safe_model}.json"
            filepath = log_dir / filename

            # Current timestamp and turn
            timestamp = datetime.now()
            turn = self.game_state.turn_number

            # Build turn data with token usage
            turn_data = {
                "turn_number": turn,
                "timestamp": timestamp.isoformat(),
                "user_prompt": user_prompt,
                "assistant_response": assistant_response
            }

            # Include token usage if tracked (Claude models)
            if input_tokens > 0 or output_tokens > 0:
                turn_data["token_usage"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }

            # Check if file exists
            if filepath.exists():
                # Load existing data and append new turn
                with open(filepath, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                # Append new turn
                log_data['turns'].append(turn_data)

                # Update cumulative token usage for Claude models
                if input_tokens > 0 or output_tokens > 0:
                    log_data['total_token_usage'] = {
                        "total_input_tokens": self.total_input_tokens,
                        "total_output_tokens": self.total_output_tokens,
                        "total_tokens": self.total_input_tokens + self.total_output_tokens
                    }
            else:
                # Create new log file with metadata
                # Extract map name from file path
                map_name = "unknown"
                if self.game_state.map_file_used:
                    map_name = Path(self.game_state.map_file_used).stem

                log_data = {
                    "game_session_id": self.game_session_id,
                    "version": __version__,
                    "model": self.model,
                    "provider": provider,
                    "player": self.bot_player,
                    "start_time": timestamp.isoformat(),
                    "map_file": self.game_state.map_file_used,
                    "map_dimensions": {
                        "width": self.game_state.original_map_width,
                        "height": self.game_state.original_map_height
                    },
                    "system_prompt": system_prompt,
                    "turns": [turn_data]
                }

                # Add cumulative token usage for Claude models
                if input_tokens > 0 or output_tokens > 0:
                    log_data['total_token_usage'] = {
                        "total_input_tokens": self.total_input_tokens,
                        "total_output_tokens": self.total_output_tokens,
                        "total_tokens": self.total_input_tokens + self.total_output_tokens
                    }

            # Write to file with configurable formatting
            indent = 2 if self.pretty_print_logs else None
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=indent, ensure_ascii=False)

            logger.debug("Logged conversation to %s (turn %s)", filepath, turn)

        except Exception as e:
            # Don't let logging errors break the bot
            logger.error("Failed to log conversation: %s", e)

    def take_turn(self):
        """
        Execute the bot's turn using LLM guidance.

        This method orchestrates the entire turn-taking process:
        1. Serializes the current game state into JSON
        2. Calls the LLM API with retry logic (including conversation history if stateful)
        3. Parses the LLM response
        4. Validates and executes the suggested actions

        The method handles errors gracefully and will fall back to ending
        the turn if the LLM fails to respond or provides invalid actions.
        """
        logger.info("LLM Bot (Player %s) is thinking...", self.bot_player)

        # Serialize game state
        game_state_json = self._serialize_game_state()

        # Format the user prompt
        user_prompt = self._format_prompt(game_state_json)

        # Get LLM response with retries
        response_text = None
        for attempt in range(self.max_retries):
            try:
                # Build messages list
                if self.stateful and self.conversation_history:
                    # In stateful mode, include full conversation history
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    messages.extend(self.conversation_history)
                    messages.append({"role": "user", "content": user_prompt})
                else:
                    # In stateless mode, only send current turn
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]

                response_text = self._call_llm(messages)
                # Accumulate token usage (only if tracked by subclass)
                self.total_input_tokens += self._last_input_tokens
                self.total_output_tokens += self._last_output_tokens
                break
            except Exception as e:
                logger.error("Attempt %s/%s failed: %s", attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info("Retrying in %s seconds...", wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Ending turn.")
                    self.game_state.end_turn()
                    return

        if not response_text:
            logger.error("No response from LLM. Ending turn.")
            self.game_state.end_turn()
            return

        # Store conversation in history if stateful mode is enabled
        if self.stateful:
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response_text})

        # Log the conversation if enabled (include token usage for Claude)
        self._log_conversation_to_json(
            SYSTEM_PROMPT, user_prompt, response_text,
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens
        )

        # Parse and execute actions
        self._execute_actions(response_text)

        # End turn (advance game state to next player, collect income, etc.)
        self.game_state.end_turn()

    def _serialize_game_state(self) -> Dict[str, Any]:
        """
        Serialize the current game state to a dictionary.

        Returns:
            Dictionary containing game state information
        """
        # Get legal actions first
        legal_actions = self.game_state.get_legal_actions(self.bot_player)

        # Serialize player's units with IDs (convert to original coordinates)
        player_units = []
        unit_id = 0
        unit_id_map = {}  # Map unit objects to IDs for later reference

        for unit in self.game_state.units:
            if unit.player == self.bot_player:
                orig_x, orig_y = self.game_state.padded_to_original_coords(unit.x, unit.y)
                unit_data = {
                    'id': unit_id,
                    't': unit.type,
                    'pos': [orig_x, orig_y],
                    'hp': unit.health,
                    'max': UNIT_DATA[unit.type]['health'],
                    'mv': unit.can_move,
                    'atk': unit.can_attack,
                }
                # Only include paralyzed flag if true (saves tokens)
                if unit.is_paralyzed():
                    unit_data['para'] = True
                player_units.append(unit_data)
                unit_id_map[unit] = unit_id
                unit_id += 1

        # Serialize enemy units (less detail, convert to original coordinates)
        enemy_units = []
        for unit in self.game_state.units:
            if unit.player != self.bot_player:
                orig_x, orig_y = self.game_state.padded_to_original_coords(unit.x, unit.y)
                enemy_data = {
                    't': unit.type,
                    'pos': [orig_x, orig_y],
                    'hp': unit.health,
                    'max': UNIT_DATA[unit.type]['health']
                }
                enemy_units.append(enemy_data)

        # Serialize buildings (convert to original coordinates)
        player_buildings = []
        enemy_buildings = []
        neutral_buildings = []

        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type in ['b', 'h', 't']:
                    orig_x, orig_y = self.game_state.padded_to_original_coords(tile.x, tile.y)
                    # Compact building format: [type, x, y] - income is derivable from type
                    building_info = [tile.type, orig_x, orig_y]

                    if tile.player == self.bot_player:
                        player_buildings.append(building_info)
                    elif tile.player is not None:
                        enemy_buildings.append(building_info)
                    else:
                        neutral_buildings.append(building_info)

        # Format legal actions for the LLM (filter out empty action types)
        formatted_legal_actions = self._format_legal_actions(legal_actions, unit_id_map)
        filtered_legal_actions = {k: v for k, v in formatted_legal_actions.items() if v}

        # Extract map name from file path
        map_name = "unknown"
        if self.game_state.map_file_used:
            map_name = Path(self.game_state.map_file_used).stem

        return {
            'map': map_name,
            'w': self.game_state.original_map_width,
            'h': self.game_state.original_map_height,
            'turn': self.game_state.turn_number,
            'gold': self.game_state.player_gold[self.bot_player],
            'enemy_gold': self.game_state.player_gold[
                1 if self.bot_player == 2 else 2
            ],
            'units': player_units,
            'enemies': enemy_units,
            'buildings': player_buildings,
            'enemy_buildings': enemy_buildings,
            'neutral_buildings': neutral_buildings,
            'actions': filtered_legal_actions
        }

    def _format_legal_actions(self, legal_actions: Dict[str, List[Any]],
                              unit_id_map: Dict) -> Dict[str, List[Dict[str, Any]]]:
        """Format legal actions for LLM consumption with original map coordinates."""
        formatted = {
            'create': [],
            'move': [],
            'attack': [],
            'paralyze': [],
            'heal': [],
            'cure': [],
            'seize': []
        }

        # Create unit actions (convert coordinates) - compact: [type, x, y, cost]
        for action in legal_actions['create_unit']:
            orig_x, orig_y = self.game_state.padded_to_original_coords(
                action['x'], action['y']
            )
            formatted['create'].append([
                action['unit_type'], orig_x, orig_y,
                UNIT_DATA[action['unit_type']]['cost']
            ])

        # Move actions - compact: [unit_id, to_x, to_y]
        for action in legal_actions['move']:
            if action['unit'] in unit_id_map:
                to_x, to_y = self.game_state.padded_to_original_coords(
                    action['to_x'], action['to_y']
                )
                formatted['move'].append([
                    unit_id_map[action['unit']], to_x, to_y
                ])

        # Attack actions - compact: [unit_id, target_x, target_y]
        for action in legal_actions['attack']:
            if action['attacker'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['attack'].append([
                    unit_id_map[action['attacker']], target_x, target_y
                ])

        # Paralyze actions - compact: [unit_id, target_x, target_y]
        for action in legal_actions['paralyze']:
            if action['paralyzer'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['paralyze'].append([
                    unit_id_map[action['paralyzer']], target_x, target_y
                ])

        # Heal actions - compact: [unit_id, target_x, target_y]
        for action in legal_actions['heal']:
            if action['healer'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['heal'].append([
                    unit_id_map[action['healer']], target_x, target_y
                ])

        # Cure actions - compact: [unit_id, target_x, target_y]
        for action in legal_actions['cure']:
            if action['curer'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['cure'].append([
                    unit_id_map[action['curer']], target_x, target_y
                ])

        # Seize actions - compact: [unit_id]
        for action in legal_actions['seize']:
            if action['unit'] in unit_id_map:
                formatted['seize'].append([unit_id_map[action['unit']]])

        return formatted

    def _format_prompt(self, game_state_json: Dict[str, Any]) -> str:
        """Format the game state into a prompt for the LLM."""
        reasoning_line = (
            '"reasoning":"<strategy>",\n'
            if self.should_reason
            else ""
        )
        # Compact response format - actions format matches legal actions arrays
        # State uses: units[id,t,pos,hp,max,mv,atk], buildings[t,x,y], actions.create[t,x,y,cost], move/attack/etc[uid,x,y]
        return f"""State:{json.dumps(game_state_json, separators=(',', ':'))}

Reply JSON:{{{reasoning_line}"actions":[{{"type":"CREATE_UNIT","unit_type":"W|M|C|A","position":[x,y]}},{{"type":"MOVE","unit_id":0,"to":[x,y]}},{{"type":"ATTACK|PARALYZE|HEAL|CURE","unit_id":0,"target_position":[x,y]}},{{"type":"SEIZE","unit_id":0}},{{"type":"END_TURN"}}]}}
Use only legal actions from state."""

    def _execute_actions(self, response_text: str):
        """Parse LLM response and execute the actions."""
        try:
            # Try to parse JSON from response
            response_json = self._extract_json(response_text)

            if not response_json or 'actions' not in response_json:
                logger.error("Invalid response format. No actions found.")
                return

            # Log reasoning if provided
            if 'reasoning' in response_json:
                logger.info("Bot reasoning: %s", response_json['reasoning'])

            # Build unit ID to unit object mapping
            unit_map = self._get_unit_by_id()

            actions = response_json['actions']
            if not isinstance(actions, list):
                logger.error("Actions must be a list")
                return

            # Execute each action
            for action in actions:
                if not isinstance(action, dict) or 'type' not in action:
                    logger.warning("Skipping invalid action: %s", action)
                    continue

                action_type = action['type']

                try:
                    if action_type == 'CREATE_UNIT':
                        self._execute_create_unit(action)
                    elif action_type == 'MOVE':
                        self._execute_move(action, unit_map)
                    elif action_type == 'ATTACK':
                        self._execute_attack(action, unit_map)
                    elif action_type == 'PARALYZE':
                        self._execute_paralyze(action, unit_map)
                    elif action_type == 'HEAL':
                        self._execute_heal(action, unit_map)
                    elif action_type == 'CURE':
                        self._execute_cure(action, unit_map)
                    elif action_type == 'SEIZE':
                        self._execute_seize(action, unit_map)
                    elif action_type == 'END_TURN':
                        logger.info("Bot chose to end turn")
                        break
                    else:
                        logger.warning("Unknown action type: %s", action_type)
                except Exception as e:
                    logger.error("Error executing action %s: %s", action, e)
                    continue

        except Exception as e:
            logger.error("Error parsing/executing LLM response: %s", e)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from response text, handling markdown code blocks."""
        # Try to parse the whole response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object anywhere in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _get_unit_by_id(self) -> Dict[int, Any]:
        """Create a mapping from unit IDs to unit objects."""
        unit_map = {}
        unit_id = 0
        for unit in self.game_state.units:
            if unit.player == self.bot_player:
                unit_map[unit_id] = unit
                unit_id += 1
        return unit_map

    def _execute_create_unit(self, action: Dict[str, Any]):
        """Execute a CREATE_UNIT action (converts from original to padded coordinates)."""
        unit_type = action.get('unit_type')
        position = action.get('position')

        if not unit_type or not position or len(position) != 2:
            logger.warning("Invalid CREATE_UNIT action: %s", action)
            return

        # Convert from original to padded coordinates
        orig_x, orig_y = position
        x, y = self.game_state.original_to_padded_coords(orig_x, orig_y)

        # Validate this is a legal action (using padded coordinates)
        legal_actions = self.game_state.get_legal_actions(self.bot_player)
        is_legal = any(
            a['unit_type'] == unit_type and a['x'] == x and a['y'] == y
            for a in legal_actions['create_unit']
        )

        if not is_legal:
            logger.warning("Illegal CREATE_UNIT action: %s (converted to padded: [%s, %s])", 
                         action, x, y)
            return

        self.game_state.create_unit(unit_type, x, y, self.bot_player)
        logger.info("Created %s at original coords (%s, %s) / padded coords (%s, %s)", 
                   unit_type, orig_x, orig_y, x, y)

    def _execute_move(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a MOVE action (converts from original to padded coordinates)."""
        unit_id = action.get('unit_id')
        to_pos = action.get('to')

        if unit_id not in unit_map or not to_pos or len(to_pos) != 2:
            logger.warning("Invalid MOVE action: %s", action)
            return

        unit = unit_map[unit_id]
        # Convert from original to padded coordinates
        orig_to_x, orig_to_y = to_pos
        to_x, to_y = self.game_state.original_to_padded_coords(orig_to_x, orig_to_y)

        # Validate this is a legal move
        if not unit.can_move:
            logger.warning("Unit %s cannot move", unit_id)
            return

        self.game_state.move_unit(unit, to_x, to_y)
        logger.info("Moved unit %s to original coords (%s, %s) / padded coords (%s, %s)", 
                   unit_id, orig_to_x, orig_to_y, to_x, to_y)

    def _execute_attack(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute an ATTACK action (converts from original to padded coordinates)."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')

        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning("Invalid ATTACK action: %s", action)
            return

        unit = unit_map[unit_id]
        # Convert from original to padded coordinates
        orig_target_x, orig_target_y = target_pos
        target_x, target_y = self.game_state.original_to_padded_coords(
            orig_target_x, orig_target_y
        )

        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning("No unit at target position (%s, %s)", target_x, target_y)
            return

        self.game_state.attack(unit, target)
        logger.info("Unit %s attacked enemy at original coords (%s, %s) / padded coords (%s, %s)", 
                   unit_id, orig_target_x, orig_target_y, target_x, target_y)

    def _execute_paralyze(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a PARALYZE action (converts from original to padded coordinates)."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')

        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning("Invalid PARALYZE action: %s", action)
            return

        unit = unit_map[unit_id]
        # Convert from original to padded coordinates
        orig_target_x, orig_target_y = target_pos
        target_x, target_y = self.game_state.original_to_padded_coords(
            orig_target_x, orig_target_y
        )

        if unit.type != 'M':
            logger.warning("Unit %s is not a Mage, cannot paralyze", unit_id)
            return

        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning("No unit at target position (%s, %s)", target_x, target_y)
            return

        self.game_state.paralyze(unit, target)
        logger.info("Unit %s paralyzed enemy at original coords (%s, %s) / padded coords (%s, %s)", 
                   unit_id, orig_target_x, orig_target_y, target_x, target_y)

    def _execute_heal(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a HEAL action (converts from original to padded coordinates)."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')

        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning("Invalid HEAL action: %s", action)
            return

        unit = unit_map[unit_id]
        # Convert from original to padded coordinates
        orig_target_x, orig_target_y = target_pos
        target_x, target_y = self.game_state.original_to_padded_coords(
            orig_target_x, orig_target_y
        )

        if unit.type != 'C':
            logger.warning("Unit %s is not a Cleric, cannot heal", unit_id)
            return

        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning("No unit at target position (%s, %s)", target_x, target_y)
            return

        self.game_state.heal(unit, target)
        logger.info("Unit %s healed ally at original coords (%s, %s) / padded coords (%s, %s)", 
                   unit_id, orig_target_x, orig_target_y, target_x, target_y)

    def _execute_cure(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a CURE action (converts from original to padded coordinates)."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')

        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning("Invalid CURE action: %s", action)
            return

        unit = unit_map[unit_id]
        # Convert from original to padded coordinates
        orig_target_x, orig_target_y = target_pos
        target_x, target_y = self.game_state.original_to_padded_coords(
            orig_target_x, orig_target_y
        )

        if unit.type != 'C':
            logger.warning("Unit %s is not a Cleric, cannot cure", unit_id)
            return

        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning("No unit at target position (%s, %s)", target_x, target_y)
            return

        self.game_state.cure(unit, target)
        logger.info("Unit %s cured ally at original coords (%s, %s) / padded coords (%s, %s)", 
                   unit_id, orig_target_x, orig_target_y, target_x, target_y)

    def _execute_seize(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a SEIZE action."""
        unit_id = action.get('unit_id')

        if unit_id not in unit_map:
            logger.warning("Invalid SEIZE action: %s", action)
            return

        unit = unit_map[unit_id]
        self.game_state.seize(unit)
        logger.info("Unit %s is seizing structure at (%s, %s)", unit_id, unit.x, unit.y)


class OpenAIBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using OpenAI's GPT models.

    Supports the full range of OpenAI models including:
    - GPT-4o family: Latest generation with best performance (gpt-4o, gpt-4o-mini)
    - GPT-4 Turbo: High performance models (gpt-4-turbo)
    - GPT-4: Stable and proven models (gpt-4)
    - GPT-3.5 Turbo: Fast and economical (gpt-3.5-turbo)
    - O1 models: Advanced reasoning capabilities (o1, o1-mini)

    Default model: gpt-4o-mini (excellent balance of cost and performance)

    Cost tiers:
    - Budget: gpt-4o-mini, gpt-3.5-turbo (~$0.15-0.50/1M input tokens)
    - Standard: gpt-4o, gpt-4-turbo (~$2.50-10/1M input tokens)
    - Premium: o1 series (~$15/1M input tokens)
    """

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('OPENAI_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'OPENAI_API_KEY'

    def _get_default_model(self) -> str:
        return 'gpt-4o-mini'

    def _get_supported_models(self) -> List[str]:
        return OPENAI_MODELS

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Install with: pip install openai>=1.0.0"
            ) from exc

        client = openai.OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content


class ClaudeBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using Anthropic's Claude models.

    Supports Claude models across multiple generations:
    - Claude 4: Latest generation (claude-sonnet-4-20250514)
    - Claude 3.5: High performance (claude-3-5-sonnet, claude-3-5-haiku)
    - Claude 3: Proven models (claude-3-opus, claude-3-sonnet, claude-3-haiku)

    Default model: claude-3-5-haiku-20241022 (latest Haiku, fast and economical)

    Cost tiers:
    - Budget: claude-3-haiku, claude-3-5-haiku (~$0.25/1M input tokens)
    - Standard: claude-3-sonnet, claude-3-5-sonnet (~$3/1M input tokens)
    - Premium: claude-3-opus, claude-sonnet-4 (~$15/1M input tokens)

    Token limits:
    - Claude 3.5/4: 200K context window
    - Claude 3: 200K context window (Opus), 200K (Sonnet/Haiku)
    """

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('ANTHROPIC_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'ANTHROPIC_API_KEY'

    def _get_default_model(self) -> str:
        return 'claude-3-5-haiku-20241022'

    def _get_supported_models(self) -> List[str]:
        return ANTHROPIC_MODELS

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic>=0.18.0"
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)

        # Extract system message and user messages
        system_message = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_message,
            messages=user_messages,
            temperature=0.7
        )

        # Capture token usage from Claude API response
        self._last_input_tokens = response.usage.input_tokens
        self._last_output_tokens = response.usage.output_tokens

        return response.content[0].text


class GeminiBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using Google's Gemini models.

    Supports Gemini models across multiple generations:
    - Gemini 2.0: Latest generation (gemini-2.0-flash, including thinking modes)
    - Gemini 1.5: Stable and capable (gemini-1.5-pro, gemini-1.5-flash)
    - Gemini 1.0: Legacy models (gemini-1.0-pro)

    Default model: gemini-2.0-flash (latest Flash, excellent speed and quality)

    Cost tiers:
    - Budget: gemini-1.5-flash, gemini-2.0-flash (free tier available, ~$0.075/1M)
    - Standard: gemini-1.5-pro (~$1.25/1M input tokens)
    - Experimental: gemini-2.0-flash-thinking-exp (enhanced reasoning)

    Token limits:
    - Gemini 2.0: Up to 1M token context window
    - Gemini 1.5: Up to 2M token context window (Pro)
    - Gemini 1.0: 32K token context window

    Best use cases:
    - gemini-2.0-flash: Fast responses with good quality
    - gemini-1.5-pro: Complex reasoning and long context
    - gemini-2.0-flash-thinking-exp: Enhanced strategic thinking
    """

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('GOOGLE_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'GOOGLE_API_KEY'

    def _get_default_model(self) -> str:
        return 'gemini-2.0-flash'

    def _get_supported_models(self) -> List[str]:
        return GEMINI_MODELS

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai>=0.4.0"
            ) from exc

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        # Combine system and user messages for Gemini
        combined_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                combined_prompt += f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                combined_prompt += msg['content']

        response = model.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2000
            )
        )

        return response.text
