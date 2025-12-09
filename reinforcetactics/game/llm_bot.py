"""
LLM-powered bots for playing Reinforce Tactics using OpenAI, Claude, and Gemini.
"""
import os
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from reinforcetactics.constants import UNIT_DATA

# Configure logging
logger = logging.getLogger(__name__)


# System prompt explaining the game rules
SYSTEM_PROMPT = """You are an expert player of Reinforce Tactics, a turn-based strategy game.

GAME OBJECTIVE:
- Win by capturing the enemy HQ or eliminating all enemy units and buildings
- Build units, move strategically, attack enemies, and capture structures

UNIT TYPES:
1. Warrior (W): Cost 200 gold, HP 15, Attack 10, Defense 6, Movement 3
   - Strong melee fighter
2. Mage (M): Cost 250 gold, HP 10, Attack 8 (adjacent) or 12 (range), Defense 4, Movement 2
   - Can PARALYZE enemies (disable them for turns)
3. Cleric (C): Cost 200 gold, HP 8, Attack 4, Defense 3, Movement 2
   - Can HEAL allies and CURE paralyzed units

BUILDING TYPES:
- HQ (h): Generates 1000 gold/turn, losing it means defeat
- Building (b): Generates 200 gold/turn, can recruit units
- Tower (t): Generates 100 gold/turn, defensive structure

AVAILABLE ACTIONS:
1. CREATE_UNIT: Spawn a unit at an owned building (costs gold)
2. MOVE: Move a unit to a reachable position (up to movement range)
3. ATTACK: Attack an adjacent enemy unit
4. PARALYZE: (Mage only) Paralyze an adjacent enemy unit
5. HEAL: (Cleric only) Heal an adjacent ally unit
6. CURE: (Cleric only) Remove paralysis from an adjacent ally
7. SEIZE: Capture a neutral/enemy structure by standing on it
8. END_TURN: Finish your turn

COMBAT RULES:
- Units can only attack adjacent enemies (orthogonally, not diagonally)
- Attacked units counter-attack if they can
- Paralyzed units cannot move or attack
- Units can move then attack, or attack then move (if they survive counter)

ECONOMY:
- You earn gold from buildings you control at the start of each turn
- Spend gold to create units at buildings
- Control more structures to generate more income

STRATEGY TIPS:
- Balance economy (capturing buildings) with military (building units)
- Protect your HQ at all costs
- Mages can disable key enemy units with paralyze
- Clerics keep your army healthy and mobile
- Position units to protect each other

Respond with a JSON object containing your reasoning and a list of actions to take this turn.
Be strategic and consider the game state carefully before deciding."""


class LLMBot(ABC):
    """Abstract base class for LLM-powered bots."""

    def __init__(self, game_state, player: int = 2, api_key: Optional[str] = None,
                 model: Optional[str] = None, max_retries: int = 3):
        """
        Initialize the LLM bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot (default 2)
            api_key: API key for the LLM provider (optional, uses env var if not provided)
            model: Model name to use (optional, uses default if not provided)
            max_retries: Maximum number of retries for API calls
        """
        self.game_state = game_state
        self.bot_player = player
        self.api_key = api_key or self._get_api_key_from_env()
        self.model = model or self._get_default_model()
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError(
                f"API key not provided. Set {self._get_env_var_name()} environment variable "
                f"or pass api_key parameter."
            )

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

    def take_turn(self):
        """Execute the bot's turn using LLM guidance."""
        logger.info(f"LLM Bot (Player {self.bot_player}) is thinking...")
        
        # Serialize game state
        game_state_json = self._serialize_game_state()
        
        # Get LLM response with retries
        response_text = None
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._format_prompt(game_state_json)}
                ]
                response_text = self._call_llm(messages)
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Ending turn.")
                    return
        
        if not response_text:
            logger.error("No response from LLM. Ending turn.")
            return
        
        # Parse and execute actions
        self._execute_actions(response_text)

    def _serialize_game_state(self) -> Dict[str, Any]:
        """
        Serialize the current game state to a dictionary.
        
        Returns:
            Dictionary containing game state information
        """
        # Get legal actions first
        legal_actions = self.game_state.get_legal_actions(self.bot_player)
        
        # Serialize player's units with IDs
        player_units = []
        unit_id = 0
        unit_id_map = {}  # Map unit objects to IDs for later reference
        
        for unit in self.game_state.units:
            if unit.player == self.bot_player:
                unit_data = {
                    'id': unit_id,
                    'type': unit.type,
                    'position': [unit.x, unit.y],
                    'hp': unit.health,
                    'max_hp': UNIT_DATA[unit.type]['health'],
                    'can_move': unit.can_move,
                    'can_attack': unit.can_attack,
                    'is_paralyzed': unit.is_paralyzed()
                }
                player_units.append(unit_data)
                unit_id_map[unit] = unit_id
                unit_id += 1
        
        # Serialize enemy units (less detail)
        enemy_units = []
        for unit in self.game_state.units:
            if unit.player != self.bot_player:
                enemy_data = {
                    'type': unit.type,
                    'position': [unit.x, unit.y],
                    'hp': unit.health,
                    'max_hp': UNIT_DATA[unit.type]['health']
                }
                enemy_units.append(enemy_data)
        
        # Serialize buildings
        player_buildings = []
        enemy_buildings = []
        neutral_buildings = []
        
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type in ['b', 'h', 't']:
                    building_info = {
                        'type': tile.type,
                        'position': [tile.x, tile.y],
                        'income': 1000 if tile.type == 'h' else (200 if tile.type == 'b' else 100)
                    }
                    
                    if tile.player == self.bot_player:
                        player_buildings.append(building_info)
                    elif tile.player is not None:
                        enemy_buildings.append(building_info)
                    else:
                        neutral_buildings.append(building_info)
        
        # Format legal actions for the LLM
        formatted_legal_actions = self._format_legal_actions(legal_actions, unit_id_map)
        
        return {
            'turn_number': self.game_state.turn_number,
            'player_gold': self.game_state.player_gold[self.bot_player],
            'opponent_gold': self.game_state.player_gold[
                1 if self.bot_player == 2 else 2
            ],
            'player_units': player_units,
            'enemy_units': enemy_units,
            'player_buildings': player_buildings,
            'enemy_buildings': enemy_buildings,
            'neutral_buildings': neutral_buildings,
            'legal_actions': formatted_legal_actions
        }

    def _format_legal_actions(self, legal_actions: Dict[str, List[Any]],
                              unit_id_map: Dict) -> Dict[str, List[Dict[str, Any]]]:
        """Format legal actions for LLM consumption."""
        formatted = {
            'create_unit': [],
            'move': [],
            'attack': [],
            'paralyze': [],
            'heal': [],
            'cure': [],
            'seize': []
        }
        
        # Create unit actions
        for action in legal_actions['create_unit']:
            formatted['create_unit'].append({
                'unit_type': action['unit_type'],
                'position': [action['x'], action['y']],
                'cost': UNIT_DATA[action['unit_type']]['cost']
            })
        
        # Move actions
        for action in legal_actions['move']:
            if action['unit'] in unit_id_map:
                formatted['move'].append({
                    'unit_id': unit_id_map[action['unit']],
                    'from': [action['from_x'], action['from_y']],
                    'to': [action['to_x'], action['to_y']]
                })
        
        # Attack actions
        for action in legal_actions['attack']:
            if action['attacker'] in unit_id_map:
                formatted['attack'].append({
                    'unit_id': unit_id_map[action['attacker']],
                    'target_position': [action['target'].x, action['target'].y]
                })
        
        # Paralyze actions
        for action in legal_actions['paralyze']:
            if action['paralyzer'] in unit_id_map:
                formatted['paralyze'].append({
                    'unit_id': unit_id_map[action['paralyzer']],
                    'target_position': [action['target'].x, action['target'].y]
                })
        
        # Heal actions
        for action in legal_actions['heal']:
            if action['healer'] in unit_id_map:
                formatted['heal'].append({
                    'unit_id': unit_id_map[action['healer']],
                    'target_position': [action['target'].x, action['target'].y]
                })
        
        # Cure actions
        for action in legal_actions['cure']:
            if action['curer'] in unit_id_map:
                formatted['cure'].append({
                    'unit_id': unit_id_map[action['curer']],
                    'target_position': [action['target'].x, action['target'].y]
                })
        
        # Seize actions
        for action in legal_actions['seize']:
            if action['unit'] in unit_id_map:
                formatted['seize'].append({
                    'unit_id': unit_id_map[action['unit']],
                    'position': [action['tile'].x, action['tile'].y]
                })
        
        return formatted

    def _format_prompt(self, game_state_json: Dict[str, Any]) -> str:
        """Format the game state into a prompt for the LLM."""
        return f"""Current Game State:
{json.dumps(game_state_json, indent=2)}

Respond with a JSON object in the following format:
{{
    "reasoning": "Brief explanation of your strategy (1-2 sentences)",
    "actions": [
        {{"type": "CREATE_UNIT", "unit_type": "W|M|C", "position": [x, y]}},
        {{"type": "MOVE", "unit_id": 0, "from": [x, y], "to": [x, y]}},
        {{"type": "ATTACK", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "PARALYZE", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "HEAL", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "CURE", "unit_id": 0, "target_position": [x, y]}},
        {{"type": "SEIZE", "unit_id": 0}},
        {{"type": "END_TURN"}}
    ]
}}

Only include actions that are legal based on the legal_actions provided.
You can take multiple actions in one turn."""

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
                logger.info(f"Bot reasoning: {response_json['reasoning']}")
            
            # Build unit ID to unit object mapping
            unit_map = self._get_unit_by_id()
            
            actions = response_json['actions']
            if not isinstance(actions, list):
                logger.error("Actions must be a list")
                return
            
            # Execute each action
            for action in actions:
                if not isinstance(action, dict) or 'type' not in action:
                    logger.warning(f"Skipping invalid action: {action}")
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
                        logger.warning(f"Unknown action type: {action_type}")
                except Exception as e:
                    logger.error(f"Error executing action {action}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing/executing LLM response: {e}")

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from response text, handling markdown code blocks."""
        # Try to parse the whole response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
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
        """Execute a CREATE_UNIT action."""
        unit_type = action.get('unit_type')
        position = action.get('position')
        
        if not unit_type or not position or len(position) != 2:
            logger.warning(f"Invalid CREATE_UNIT action: {action}")
            return
        
        x, y = position
        
        # Validate this is a legal action
        legal_actions = self.game_state.get_legal_actions(self.bot_player)
        is_legal = any(
            a['unit_type'] == unit_type and a['x'] == x and a['y'] == y
            for a in legal_actions['create_unit']
        )
        
        if not is_legal:
            logger.warning(f"Illegal CREATE_UNIT action: {action}")
            return
        
        self.game_state.create_unit(unit_type, x, y, self.bot_player)
        logger.info(f"Created {unit_type} at ({x}, {y})")

    def _execute_move(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a MOVE action."""
        unit_id = action.get('unit_id')
        to_pos = action.get('to')
        
        if unit_id not in unit_map or not to_pos or len(to_pos) != 2:
            logger.warning(f"Invalid MOVE action: {action}")
            return
        
        unit = unit_map[unit_id]
        to_x, to_y = to_pos
        
        # Validate this is a legal move
        if not unit.can_move:
            logger.warning(f"Unit {unit_id} cannot move")
            return
        
        self.game_state.move_unit(unit, to_x, to_y)
        logger.info(f"Moved unit {unit_id} to ({to_x}, {to_y})")

    def _execute_attack(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute an ATTACK action."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')
        
        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning(f"Invalid ATTACK action: {action}")
            return
        
        unit = unit_map[unit_id]
        target_x, target_y = target_pos
        
        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning(f"No unit at target position ({target_x}, {target_y})")
            return
        
        self.game_state.attack(unit, target)
        logger.info(f"Unit {unit_id} attacked enemy at ({target_x}, {target_y})")

    def _execute_paralyze(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a PARALYZE action."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')
        
        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning(f"Invalid PARALYZE action: {action}")
            return
        
        unit = unit_map[unit_id]
        target_x, target_y = target_pos
        
        if unit.type != 'M':
            logger.warning(f"Unit {unit_id} is not a Mage, cannot paralyze")
            return
        
        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning(f"No unit at target position ({target_x}, {target_y})")
            return
        
        self.game_state.paralyze(unit, target)
        logger.info(f"Unit {unit_id} paralyzed enemy at ({target_x}, {target_y})")

    def _execute_heal(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a HEAL action."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')
        
        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning(f"Invalid HEAL action: {action}")
            return
        
        unit = unit_map[unit_id]
        target_x, target_y = target_pos
        
        if unit.type != 'C':
            logger.warning(f"Unit {unit_id} is not a Cleric, cannot heal")
            return
        
        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning(f"No unit at target position ({target_x}, {target_y})")
            return
        
        self.game_state.heal(unit, target)
        logger.info(f"Unit {unit_id} healed ally at ({target_x}, {target_y})")

    def _execute_cure(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a CURE action."""
        unit_id = action.get('unit_id')
        target_pos = action.get('target_position')
        
        if unit_id not in unit_map or not target_pos or len(target_pos) != 2:
            logger.warning(f"Invalid CURE action: {action}")
            return
        
        unit = unit_map[unit_id]
        target_x, target_y = target_pos
        
        if unit.type != 'C':
            logger.warning(f"Unit {unit_id} is not a Cleric, cannot cure")
            return
        
        # Find target unit
        target = self.game_state.get_unit_at_position(target_x, target_y)
        if not target:
            logger.warning(f"No unit at target position ({target_x}, {target_y})")
            return
        
        self.game_state.cure(unit, target)
        logger.info(f"Unit {unit_id} cured ally at ({target_x}, {target_y})")

    def _execute_seize(self, action: Dict[str, Any], unit_map: Dict[int, Any]):
        """Execute a SEIZE action."""
        unit_id = action.get('unit_id')
        
        if unit_id not in unit_map:
            logger.warning(f"Invalid SEIZE action: {action}")
            return
        
        unit = unit_map[unit_id]
        self.game_state.seize(unit)
        logger.info(f"Unit {unit_id} is seizing structure at ({unit.x}, {unit.y})")


class OpenAIBot(LLMBot):
    """LLM bot using OpenAI's GPT models."""

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('OPENAI_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'OPENAI_API_KEY'

    def _get_default_model(self) -> str:
        return 'gpt-4o-mini'

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai>=1.0.0"
            )
        
        client = openai.OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content


class ClaudeBot(LLMBot):
    """LLM bot using Anthropic's Claude models."""

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('ANTHROPIC_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'ANTHROPIC_API_KEY'

    def _get_default_model(self) -> str:
        return 'claude-3-haiku-20240307'

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic>=0.18.0"
            )
        
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
        
        return response.content[0].text


class GeminiBot(LLMBot):
    """LLM bot using Google's Gemini models."""

    def _get_api_key_from_env(self) -> Optional[str]:
        return os.getenv('GOOGLE_API_KEY')

    def _get_env_var_name(self) -> str:
        return 'GOOGLE_API_KEY'

    def _get_default_model(self) -> str:
        return 'gemini-1.5-flash'

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call Google Gemini API."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai>=0.4.0"
            )
        
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
