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
from reinforcetactics.game.llm_prompts import (
    DEFAULT_PROMPT,
    PROMPT_TWO_PHASE_PLAN,
    PROMPT_TWO_PHASE_EXECUTE,
    get_prompt,
    get_dynamic_prompt,
)

# Configure logging
logger = logging.getLogger(__name__)


# Supported models for each provider
OPENAI_MODELS = [
    # GPT-5.2 (flagship reasoning model)
    'gpt-5.2',
    # GPT-5 family
    'gpt-5-2025-08-07',
    'gpt-5-mini-2025-08-07',
    'gpt-5-nano-2025-08-07',
]

ANTHROPIC_MODELS = [
    # Claude Opus 4.6 (latest, most intelligent)
    'claude-opus-4-6',
    # Claude Sonnet 4.5 (best speed/intelligence balance)
    'claude-sonnet-4-5-20250929',
    # Claude Haiku 4.5 (fastest, near-frontier intelligence)
    'claude-haiku-4-5-20251001',
    # Claude Opus 4.5
    'claude-opus-4-5-20251101',
    # Claude Opus 4.1
    'claude-opus-4-1-20250805',
    # Claude Sonnet 4 / Opus 4
    'claude-sonnet-4-20250514',
    'claude-opus-4-20250514',
]

GEMINI_MODELS = [
    # Gemini 3.0 (latest generation, preview)
    'gemini-3-pro-preview',
    'gemini-3-flash-preview',
    # Gemini 2.5 (current production)
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
]


# Legacy alias - the actual prompt used by bots is DEFAULT_PROMPT from llm_prompts.py
# which includes all 8 unit types (W, M, C, A, K, R, S, B) with correct stats.
# See reinforcetactics/game/llm_prompts.py for the full prompt definitions.
SYSTEM_PROMPT = DEFAULT_PROMPT


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
                 should_reason: bool = False,
                 max_tokens: Optional[int] = 8_000,
                 temperature: Optional[float] = None,
                 system_prompt: Optional[str] = None,
                 two_phase_planning: bool = False):
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
            max_tokens: Maximum number of tokens for LLM response (default 8000).
                Set to 0 or None to not pass max_tokens to the LLM provider.
                If not specified, uses DEFAULT_MAX_TOKENS (8000).
            temperature: Temperature for LLM response (default None).
                Set to None to use the LLM provider's default temperature.
                Set to a value (e.g., 0, 0.5, 1.0) to override the default.
            system_prompt: Custom system prompt to use (default None uses DEFAULT_PROMPT).
                Can be a prompt string or a prompt name from llm_prompts (e.g., "strategic").
                See reinforcetactics.game.llm_prompts for available prompts.
            two_phase_planning: Enable two-phase planning mode (default False).
                When True, the bot first generates a strategic plan, then executes it.
                This encourages deeper strategic thinking about action sequences.
                Note: This doubles the number of API calls per turn.
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
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.two_phase_planning = two_phase_planning

        # Resolve system prompt - can be a name or a full prompt string
        if system_prompt is None:
            self.system_prompt = DEFAULT_PROMPT
        elif len(system_prompt) < 100 and '\n' not in system_prompt:
            # Looks like a prompt name, try to resolve it
            try:
                self.system_prompt = get_prompt(system_prompt)
            except ValueError:
                # Not a known name, treat as custom prompt
                self.system_prompt = system_prompt
        else:
            # Full prompt string
            self.system_prompt = system_prompt

        # Initialize conversation history for stateful mode
        self.conversation_history = []

        # Initialize token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Per-call token tracking (set by subclasses that support it)
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        # Per-call stop reason tracking (set by subclasses)
        self._last_stop_reason = ""

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

    @abstractmethod
    def _get_llm_sdk_version(self) -> str:
        """Get the version of the LLM SDK being used."""

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

    def get_token_usage(self) -> Dict[str, int]:
        """
        Get unified token usage statistics across all providers.

        Returns:
            Dictionary with total_input_tokens, total_output_tokens, and total_tokens.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def _get_effective_system_prompt(self) -> str:
        """
        Get the effective system prompt considering enabled units.

        If some units are disabled, appends a note to the system prompt
        informing the LLM about the restricted unit types.

        Returns:
            The system prompt with any disabled unit information appended.
        """
        # Get enabled units from game state
        enabled_units = getattr(self.game_state, 'enabled_units', None)

        # If all units are enabled or enabled_units is not set, use original prompt
        all_units = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']
        if enabled_units is None or set(enabled_units) == set(all_units):
            return self.system_prompt

        # Find disabled units
        disabled_units = [u for u in all_units if u not in enabled_units]

        if not disabled_units:
            return self.system_prompt

        # Get unit names for disabled units
        disabled_names = [UNIT_DATA[u]['name'] for u in disabled_units]

        # Append disabled units note to the system prompt
        disabled_note = (
            f"\n\nIMPORTANT - DISABLED UNITS:\n"
            f"The following unit types are DISABLED for this game and cannot be created: "
            f"{', '.join(disabled_names)} ({', '.join(disabled_units)}).\n"
            f"Do NOT attempt to create these units. Only the following units are available: "
            f"{', '.join([UNIT_DATA[u]['name'] for u in enabled_units])} ({', '.join(enabled_units)})."
        )

        return self.system_prompt + disabled_note

    def _log_conversation_to_json(self, system_prompt: str, user_prompt: str,
                                   assistant_response: str,
                                   input_tokens: int = 0,
                                   output_tokens: int = 0,
                                   stop_reason: str = "") -> None:
        """
        Log the conversation to a JSON file (single file per game).

        Only logs if log_conversations is True.
        Creates a single log file per game with all turns appended.

        Args:
            system_prompt: The system prompt sent to the LLM
            user_prompt: The user prompt (formatted game state)
            assistant_response: The LLM's response
            input_tokens: Number of input tokens used for this turn
            output_tokens: Number of output tokens used for this turn
            stop_reason: The stop/finish reason from the LLM API response
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

            # Build turn data with token usage and stop reason
            turn_data = {
                "turn_number": turn,
                "timestamp": timestamp.isoformat(),
                "user_prompt": user_prompt,
                "assistant_response": assistant_response
            }

            # Include stop reason if available
            if stop_reason:
                turn_data["stop_reason"] = stop_reason

            # Include token usage if tracked
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

                # Update cumulative token usage
                if input_tokens > 0 or output_tokens > 0:
                    log_data['total_token_usage'] = self.get_token_usage()
            else:
                # Create new log file with metadata
                log_data = {
                    "game_session_id": self.game_session_id,
                    "version": {
                        "reinforce_tactics": __version__,
                        "llm_sdk": self._get_llm_sdk_version()
                    },
                    "model": self.model,
                    "max_tokens": self.max_tokens if self.max_tokens is not None else "None",
                    "temperature": self.temperature,
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

                # Add cumulative token usage
                if input_tokens > 0 or output_tokens > 0:
                    log_data['total_token_usage'] = self.get_token_usage()

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
        2. Optionally runs a planning phase (if two_phase_planning is enabled)
        3. Calls the LLM API with retry logic (including conversation history if stateful)
        4. Parses the LLM response
        5. Validates and executes the suggested actions

        The method handles errors gracefully and will fall back to ending
        the turn if the LLM fails to respond or provides invalid actions.
        """
        logger.info("LLM Bot (Player %s) is thinking...", self.bot_player)

        # Serialize game state
        game_state_json = self._serialize_game_state()

        # Two-phase planning: first get a strategic plan, then execute
        strategic_plan = None
        if self.two_phase_planning:
            strategic_plan = self._run_planning_phase(game_state_json)
            if strategic_plan:
                logger.info("Strategic plan generated: %s",
                           strategic_plan.get('primary_objective', 'No objective'))

        # Format the user prompt (include plan if two-phase mode)
        user_prompt = self._format_prompt(game_state_json, strategic_plan=strategic_plan)

        # Determine which system prompt to use for execution
        if self.two_phase_planning and strategic_plan:
            # Use the execution prompt for phase 2
            execution_system_prompt = PROMPT_TWO_PHASE_EXECUTE.format(
                plan=json.dumps(strategic_plan, indent=2)
            )
        else:
            # Use the effective prompt that includes disabled unit information
            execution_system_prompt = self._get_effective_system_prompt()

        # Get LLM response with retries
        response_text = self._call_llm_with_retry(execution_system_prompt, user_prompt)

        if not response_text:
            logger.error("No response from LLM. Ending turn.")
            self.game_state.end_turn()
            return

        # Store conversation in history if stateful mode is enabled
        if self.stateful:
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response_text})

        # Log the conversation if enabled (include token usage and stop reason)
        self._log_conversation_to_json(
            execution_system_prompt, user_prompt, response_text,
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens,
            stop_reason=self._last_stop_reason
        )

        # Parse and execute actions
        self._execute_actions(response_text)

        # End turn (advance game state to next player, collect income, etc.)
        # Skip if game is already over (e.g., due to resignation)
        if not self.game_state.game_over:
            self.game_state.end_turn()

    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        Call LLM with retry logic and exponential backoff.

        Args:
            system_prompt: The system prompt to use
            user_prompt: The user prompt with game state

        Returns:
            The LLM response text, or None if all retries failed
        """
        response_text = None
        for attempt in range(self.max_retries):
            try:
                # Build messages list
                if self.stateful and self.conversation_history:
                    # In stateful mode, include full conversation history
                    messages = [{"role": "system", "content": system_prompt}]
                    messages.extend(self.conversation_history)
                    messages.append({"role": "user", "content": user_prompt})
                else:
                    # In stateless mode, only send current turn
                    messages = [
                        {"role": "system", "content": system_prompt},
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
                    logger.error("Max retries reached.")
                    return None

        return response_text

    def _run_planning_phase(self, game_state_json: Dict[str, Any]) -> Optional[Dict]:
        """
        Run the planning phase for two-phase planning mode.

        This phase asks the LLM to analyze the situation and create a strategic
        plan before deciding on specific actions. This encourages deeper thinking
        about multi-step tactical sequences.

        Args:
            game_state_json: The serialized game state

        Returns:
            The strategic plan as a dictionary, or None if planning failed
        """
        logger.info("Running planning phase...")

        # Format the planning prompt
        planning_prompt = f"""Analyze this game state and create a strategic plan:

{json.dumps(game_state_json, indent=2)}

Consider:
1. What buildings can be captured this turn?
2. Which enemies need to be killed to enable captures?
3. What order should units act?
4. Are there any threats to address?

Respond with your strategic plan in JSON format."""

        # Call LLM for planning (use planning prompt)
        response_text = self._call_llm_with_retry(PROMPT_TWO_PHASE_PLAN, planning_prompt)

        if not response_text:
            logger.warning("Planning phase failed, falling back to single-phase")
            return None

        # Log the planning conversation if enabled
        self._log_conversation_to_json(
            PROMPT_TWO_PHASE_PLAN, planning_prompt, response_text,
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens,
            stop_reason=self._last_stop_reason
        )

        # Parse the plan
        try:
            plan = self._extract_json(response_text)
            if plan:
                return plan
        except Exception as e:
            logger.warning("Failed to parse strategic plan: %s", e)

        return None

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
                    'type': unit.type,
                    'position': [orig_x, orig_y],
                    'hp': unit.health,
                    'max_hp': UNIT_DATA[unit.type]['health'],
                    'can_move': unit.can_move,
                    'can_attack': unit.can_attack,
                    'is_paralyzed': unit.is_paralyzed()
                }
                player_units.append(unit_data)
                unit_id_map[unit] = unit_id
                unit_id += 1

        # Serialize enemy units (less detail, convert to original coordinates)
        # With fog of war, only include visible enemy units
        enemy_units = []
        for unit in self.game_state.units:
            if unit.player != self.bot_player:
                # FOW: Skip enemies that are not visible
                if self.game_state.fog_of_war:
                    if not self.game_state.is_position_visible(unit.x, unit.y, self.bot_player):
                        continue

                orig_x, orig_y = self.game_state.padded_to_original_coords(unit.x, unit.y)
                enemy_data = {
                    'type': unit.type,
                    'position': [orig_x, orig_y],
                    'hp': unit.health,
                    'max_hp': UNIT_DATA[unit.type]['health']
                }
                enemy_units.append(enemy_data)

        # Serialize buildings (convert to original coordinates)
        # With fog of war, only include explored structures
        player_buildings = []
        enemy_buildings = []
        neutral_buildings = []

        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type in ['b', 'h', 't']:
                    # FOW: Skip structures that haven't been explored
                    if self.game_state.fog_of_war:
                        if not self.game_state.is_position_explored(tile.x, tile.y, self.bot_player):
                            continue

                    orig_x, orig_y = self.game_state.padded_to_original_coords(tile.x, tile.y)
                    building_info = {
                        'type': tile.type,
                        'position': [orig_x, orig_y],
                        'income': 150 if tile.type == 'h' else (100 if tile.type == 'b' else 50)
                    }

                    # FOW: For non-visible structures, don't show current ownership
                    if self.game_state.fog_of_war:
                        if not self.game_state.is_position_visible(tile.x, tile.y, self.bot_player):
                            # Mark as 'last_seen' to indicate outdated info
                            building_info['last_seen'] = True

                    if tile.player == self.bot_player:
                        player_buildings.append(building_info)
                    elif tile.player is not None:
                        enemy_buildings.append(building_info)
                    else:
                        neutral_buildings.append(building_info)

        # Format legal actions for the LLM
        formatted_legal_actions = self._format_legal_actions(legal_actions, unit_id_map)

        # Extract map name from file path
        map_name = "unknown"
        if self.game_state.map_file_used:
            map_name = Path(self.game_state.map_file_used).stem

        # Get enabled units (for informing LLM which units can be created)
        enabled_units = getattr(self.game_state, 'enabled_units', ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B'])

        # Build the state dictionary
        state = {
            'map_name': map_name,
            'map_width': self.game_state.original_map_width,
            'map_height': self.game_state.original_map_height,
            'turn_number': self.game_state.turn_number,
            'player_gold': self.game_state.player_gold[self.bot_player],
            'enabled_units': enabled_units,
            'player_units': player_units,
            'enemy_units': enemy_units,
            'player_buildings': player_buildings,
            'enemy_buildings': enemy_buildings,
            'neutral_buildings': neutral_buildings,
            'legal_actions': formatted_legal_actions
        }

        # FOW: Include fog of war status and hide enemy gold
        if self.game_state.fog_of_war:
            state['fog_of_war'] = True
            state['opponent_gold'] = 'hidden'  # Hide enemy gold in FOW mode
        else:
            state['fog_of_war'] = False
            state['opponent_gold'] = self.game_state.player_gold[
                1 if self.bot_player == 2 else 2
            ]

        return state

    def _compute_move_then_actions(self, unit, unit_id: int,
                                     reachable_positions: List[tuple]
                                     ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute actions that become available after moving to reachable positions.

        Args:
            unit: The unit to check
            unit_id: The unit's ID for the LLM
            reachable_positions: List of (x, y) positions the unit can move to

        Returns:
            Dict with move_then_attack, move_then_seize, etc. combinations
        """
        result = {
            'move_then_attack': [],
            'move_then_seize': [],
            'move_then_heal': [],
            'move_then_cure': [],
            'move_then_paralyze': []
        }

        # Get enemy units for attack calculations
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player]

        # Get ally units for heal/cure calculations (Cleric only)
        ally_units = [u for u in self.game_state.units
                      if u.player == self.bot_player and u != unit]

        for to_x, to_y in reachable_positions:
            # Convert to original coords for output
            orig_to_x, orig_to_y = self.game_state.padded_to_original_coords(to_x, to_y)

            # Check if moving here allows attacking enemies
            # Temporarily calculate what would be in range from this position
            tile = self.game_state.grid.get_tile(to_x, to_y)
            on_mountain = tile.type == 'm' if tile else False

            for enemy in enemy_units:
                # Calculate distance from potential new position
                distance = abs(to_x - enemy.x) + abs(to_y - enemy.y)

                # Check if enemy would be in attack range from new position
                min_range, max_range = unit.get_attack_range(on_mountain)
                if min_range <= distance <= max_range:
                    orig_enemy_x, orig_enemy_y = self.game_state.padded_to_original_coords(
                        enemy.x, enemy.y
                    )
                    result['move_then_attack'].append({
                        'unit_id': unit_id,
                        'move_to': [orig_to_x, orig_to_y],
                        'then_attack': [orig_enemy_x, orig_enemy_y]
                    })

                    # Mage can also paralyze when the target is within its valid range
                    if unit.type == 'M' and min_range <= distance <= max_range:
                        result['move_then_paralyze'].append({
                            'unit_id': unit_id,
                            'move_to': [orig_to_x, orig_to_y],
                            'then_paralyze': [orig_enemy_x, orig_enemy_y]
                        })

            # Check if moving here allows seizing a structure
            if tile and tile.is_capturable() and tile.player != self.bot_player:
                result['move_then_seize'].append({
                    'unit_id': unit_id,
                    'move_to': [orig_to_x, orig_to_y],
                    'then_seize': True
                })

            # Cleric-specific: check for heal/cure opportunities
            if unit.type == 'C':
                adjacent_positions = [
                    (to_x, to_y - 1), (to_x, to_y + 1),
                    (to_x - 1, to_y), (to_x + 1, to_y)
                ]

                for ally in ally_units:
                    if (ally.x, ally.y) in adjacent_positions:
                        orig_ally_x, orig_ally_y = self.game_state.padded_to_original_coords(
                            ally.x, ally.y
                        )
                        # Heal if damaged
                        if ally.health < ally.max_health:
                            result['move_then_heal'].append({
                                'unit_id': unit_id,
                                'move_to': [orig_to_x, orig_to_y],
                                'then_heal': [orig_ally_x, orig_ally_y]
                            })
                        # Cure if paralyzed
                        if ally.is_paralyzed():
                            result['move_then_cure'].append({
                                'unit_id': unit_id,
                                'move_to': [orig_to_x, orig_to_y],
                                'then_cure': [orig_ally_x, orig_ally_y]
                            })

        return result

    def _format_legal_actions(self, legal_actions: Dict[str, List[Any]],
                              unit_id_map: Dict) -> Dict[str, List[Dict[str, Any]]]:
        """Format legal actions for LLM consumption with original map coordinates."""
        formatted = {
            'create_unit': [],
            'move': [],
            'attack': [],
            'paralyze': [],
            'heal': [],
            'cure': [],
            'seize': [],
            # Move-then-action combinations
            'move_then_attack': [],
            'move_then_seize': [],
            'move_then_heal': [],
            'move_then_cure': [],
            'move_then_paralyze': []
        }

        # Create unit actions (convert coordinates)
        for action in legal_actions['create_unit']:
            orig_x, orig_y = self.game_state.padded_to_original_coords(
                action['x'], action['y']
            )
            formatted['create_unit'].append({
                'unit_type': action['unit_type'],
                'position': [orig_x, orig_y],
                'cost': UNIT_DATA[action['unit_type']]['cost']
            })

        # Move actions (convert coordinates)
        for action in legal_actions['move']:
            if action['unit'] in unit_id_map:
                from_x, from_y = self.game_state.padded_to_original_coords(
                    action['from_x'], action['from_y']
                )
                to_x, to_y = self.game_state.padded_to_original_coords(
                    action['to_x'], action['to_y']
                )
                formatted['move'].append({
                    'unit_id': unit_id_map[action['unit']],
                    'from': [from_x, from_y],
                    'to': [to_x, to_y]
                })

        # Attack actions (convert coordinates)
        for action in legal_actions['attack']:
            if action['attacker'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['attack'].append({
                    'unit_id': unit_id_map[action['attacker']],
                    'target_position': [target_x, target_y]
                })

        # Paralyze actions (convert coordinates)
        for action in legal_actions['paralyze']:
            if action['paralyzer'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['paralyze'].append({
                    'unit_id': unit_id_map[action['paralyzer']],
                    'target_position': [target_x, target_y]
                })

        # Heal actions (convert coordinates)
        for action in legal_actions['heal']:
            if action['healer'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['heal'].append({
                    'unit_id': unit_id_map[action['healer']],
                    'target_position': [target_x, target_y]
                })

        # Cure actions (convert coordinates)
        for action in legal_actions['cure']:
            if action['curer'] in unit_id_map:
                target_x, target_y = self.game_state.padded_to_original_coords(
                    action['target'].x, action['target'].y
                )
                formatted['cure'].append({
                    'unit_id': unit_id_map[action['curer']],
                    'target_position': [target_x, target_y]
                })

        # Seize actions (convert coordinates)
        for action in legal_actions['seize']:
            if action['unit'] in unit_id_map:
                tile_x, tile_y = self.game_state.padded_to_original_coords(
                    action['tile'].x, action['tile'].y
                )
                formatted['seize'].append({
                    'unit_id': unit_id_map[action['unit']],
                    'position': [tile_x, tile_y]
                })

        # Compute move-then-action combinations for units that can move
        # Group move actions by unit to get all reachable positions per unit
        unit_reachable_positions: Dict[Any, List[tuple]] = {}
        for action in legal_actions['move']:
            unit = action['unit']
            if unit not in unit_reachable_positions:
                unit_reachable_positions[unit] = []
            unit_reachable_positions[unit].append((action['to_x'], action['to_y']))

        # For each movable unit, compute what actions become available after moving
        for unit, positions in unit_reachable_positions.items():
            if unit in unit_id_map:
                unit_id = unit_id_map[unit]
                move_then_actions = self._compute_move_then_actions(unit, unit_id, positions)

                # Merge results into formatted output
                for key in ['move_then_attack', 'move_then_seize', 'move_then_heal',
                           'move_then_cure', 'move_then_paralyze']:
                    formatted[key].extend(move_then_actions[key])

        return formatted

    def _format_prompt(self, game_state_json: Dict[str, Any],
                       strategic_plan: Optional[Dict] = None) -> str:
        """
        Format the game state into a prompt for the LLM.

        Args:
            game_state_json: The serialized game state
            strategic_plan: Optional strategic plan from two-phase planning mode

        Returns:
            The formatted user prompt string
        """
        reasoning_line = (
            '    "reasoning": "Brief explanation of your strategy (1-2 sentences)",\n'
            if self.should_reason or strategic_plan
            else ""
        )

        # Include strategic plan context if provided
        plan_context = ""
        if strategic_plan:
            plan_context = f"""
STRATEGIC PLAN TO EXECUTE:
{json.dumps(strategic_plan, indent=2)}

Execute the actions according to this plan.

"""

        return f"""Current Game State:
{json.dumps(game_state_json, indent=2)}
{plan_context}
Respond with a JSON object in the following format:
{{
{reasoning_line}    "actions": [
        {{"type": "CREATE_UNIT", "unit_type": "W|M|C|A", "position": [x, y]}},
        {{"type": "MOVE", "unit_id": 0, "from": [x, y], "to": [x, y]}},
        {{"type": "ATTACK", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "PARALYZE", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "HEAL", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "CURE", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "SEIZE", "unit_id": 0}},
        {{"type": "END_TURN"}},
        {{"type": "RESIGN"}}
    ]
}}

Only include actions that are legal based on the legal_actions provided.
You can take multiple actions in one turn.
Use RESIGN only as a last resort when victory is impossible."""

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
                    elif action_type == 'RESIGN':
                        self._execute_resign()
                        return  # Exit immediately after resignation
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

    def _execute_resign(self):
        """Execute a RESIGN action - the bot concedes the game."""
        logger.info("LLM Bot (Player %s) has decided to resign.", self.bot_player)
        self.game_state.resign(self.bot_player)


class OpenAIBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using OpenAI's GPT models.

    Supports OpenAI GPT-5+ models:
    - GPT-5.2: Flagship reasoning model, most capable
    - GPT-5: gpt-5, gpt-5-mini, gpt-5-nano

    Default model: gpt-5-mini-2025-08-07 (good balance of cost and performance)

    Cost tiers:
    - Budget: gpt-5-nano, gpt-5-mini (~$0.15-0.50/1M input tokens)
    - Premium: gpt-5.2 (~$10-15/1M input tokens)
    """

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('OPENAI_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'OPENAI_API_KEY'

    def _get_default_model(self) -> str:
        return 'gpt-5-mini-2025-08-07'

    def _get_supported_models(self) -> List[str]:
        return OPENAI_MODELS

    def _get_llm_sdk_version(self) -> str:
        """Get the OpenAI SDK version."""
        try:
            import openai
            return f"openai=={openai.__version__}"
        except (ImportError, AttributeError):
            return "openai==unknown"

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Install with: pip install openai>=1.0.0"
            ) from exc

        client = openai.OpenAI(api_key=self.api_key)

        # Build request kwargs, conditionally including max_completion_tokens and temperature
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self.max_tokens is not None:
            request_kwargs["max_completion_tokens"] = self.max_tokens
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        response = client.chat.completions.create(**request_kwargs)

        # Capture token usage from OpenAI API response
        if response.usage:
            self._last_input_tokens = response.usage.prompt_tokens
            self._last_output_tokens = response.usage.completion_tokens

        # Capture finish reason from OpenAI API response
        if response.choices and response.choices[0].finish_reason:
            self._last_stop_reason = response.choices[0].finish_reason

        return response.choices[0].message.content


class ClaudeBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using Anthropic's Claude models.

    Supports Claude 4+ models:
    - Claude Opus 4.6: Most intelligent, exceptional coding/reasoning (claude-opus-4-6)
    - Claude Sonnet 4.5: Best speed/intelligence balance (claude-sonnet-4-5-20250929)
    - Claude Haiku 4.5: Fastest, near-frontier intelligence (claude-haiku-4-5-20251001)
    - Claude Opus 4.5/4.1/4: Previous generation Opus and Sonnet models

    Default model: claude-haiku-4-5-20251001 (fast, economical, near-frontier)

    Cost tiers:
    - Budget: claude-haiku-4-5 (~$1/1M input tokens)
    - Standard: claude-sonnet-4-5 (~$3/1M input tokens)
    - Premium: claude-opus-4-6, claude-opus-4-5 (~$5/1M input tokens)

    Token limits:
    - Claude Opus 4.6 / Sonnet 4.5: 200K (1M beta), 128K/64K max output
    - Claude Haiku 4.5: 200K context, 64K max output
    """

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('ANTHROPIC_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'ANTHROPIC_API_KEY'

    def _get_default_model(self) -> str:
        return 'claude-haiku-4-5-20251001'

    def _get_supported_models(self) -> List[str]:
        return ANTHROPIC_MODELS

    def _get_llm_sdk_version(self) -> str:
        """Get the Anthropic SDK version."""
        try:
            import anthropic
            return f"anthropic=={anthropic.__version__}"
        except (ImportError, AttributeError):
            return "anthropic==unknown"

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

        # Prefill assistant response with "{" to guide JSON output
        user_messages.append({"role": "assistant", "content": "{"})

        # Build request kwargs, conditionally including max_tokens and temperature
        request_kwargs = {
            "model": self.model,
            "system": system_message,
            "messages": user_messages,
        }
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature

        response = client.messages.create(**request_kwargs)

        # Capture token usage from Claude API response
        self._last_input_tokens = response.usage.input_tokens
        self._last_output_tokens = response.usage.output_tokens

        # Capture stop reason from Claude API response
        if response.stop_reason:
            self._last_stop_reason = response.stop_reason

        # Prepend the prefilled "{" to reconstruct the full JSON response
        return "{" + response.content[0].text


class GeminiBot(LLMBot):  # pylint: disable=too-few-public-methods
    """
    LLM bot using Google's Gemini models via the google-genai SDK.

    Supports Gemini 2.5+ models:
    - Gemini 3.0: Latest generation (gemini-3-pro-preview, gemini-3-flash-preview)
    - Gemini 2.5: Production models with thinking (gemini-2.5-pro, gemini-2.5-flash)

    Default model: gemini-2.5-flash (production Flash with thinking capabilities)

    Cost tiers:
    - Budget: gemini-2.5-flash-lite (~$0.075/1M input tokens)
    - Standard: gemini-2.5-flash, gemini-3-flash-preview (~$0.15/1M input tokens)
    - Premium: gemini-2.5-pro, gemini-3-pro-preview (~$1.25/1M input tokens)

    Token limits:
    - Gemini 3.0/2.5: Up to 1M token context window

    Best use cases:
    - gemini-3-flash-preview: Latest generation, fast, frontier-class
    - gemini-2.5-flash: Best production balance of speed and quality
    - gemini-2.5-pro: Complex reasoning tasks
    """

    def __init__(self, *args, **kwargs):
        """Initialize GeminiBot with optional chat session for stateful mode."""
        super().__init__(*args, **kwargs)
        self._chat_session = None
        self._client = None

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('GOOGLE_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'GOOGLE_API_KEY'

    def _get_default_model(self) -> str:
        return 'gemini-2.5-flash'

    def _get_supported_models(self) -> List[str]:
        return GEMINI_MODELS

    def _get_llm_sdk_version(self) -> str:
        """Get the Google GenAI SDK version."""
        try:
            from google import genai
            return f"google-genai=={genai.__version__}"
        except (ImportError, AttributeError):
            return "google-genai==unknown"

    def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Google Gemini API using the new google-genai SDK."""
        try:
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            ) from exc

        client = self._get_client()

        # Extract system instruction and build conversation contents
        system_instruction = None
        contents = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg["content"])]
                ))

        # Build generation config, conditionally including max_output_tokens and temperature
        config_kwargs = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
        }
        if self.max_tokens is not None:
            config_kwargs["max_output_tokens"] = self.max_tokens
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Track token usage from response metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                self._last_input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                self._last_output_tokens = getattr(usage, 'candidates_token_count', 0) or 0

            # Capture finish reason from Gemini API response
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    # Convert enum to string if necessary
                    finish_reason = candidate.finish_reason
                    self._last_stop_reason = str(finish_reason.name) if hasattr(finish_reason, 'name') else str(finish_reason)

            # Handle potential blocked responses
            if not response.text:
                # Check if response was blocked
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if hasattr(feedback, 'block_reason') and feedback.block_reason:
                        logger.warning("Gemini response blocked: %s", feedback.block_reason)
                        return '{"reasoning": "Response blocked by safety filter", "actions": [{"type": "END_TURN"}]}'

                logger.warning("Empty response from Gemini API")
                return '{"reasoning": "Empty response", "actions": [{"type": "END_TURN"}]}'

            return response.text

        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise
