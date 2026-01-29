"""
Kaggle Environments interpreter for Reinforce Tactics.

This module implements the kaggle-environments interface, bridging the
Kaggle agent evaluation framework with the Reinforce Tactics game engine.

Required exports for kaggle-environments:
    - interpreter(state, env): Core game logic called each step
    - renderer(state, env): ASCII text renderer
    - specification: JSON specification dict
    - html_renderer(): Optional HTML/JS renderer
    - agents: Dict of built-in agent functions
"""
import json
import logging
from os import path
from copy import deepcopy

import numpy as np

from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.unit import Unit
from reinforcetactics.constants import (
    UNIT_DATA, TileType, STARTING_GOLD,
    HEADQUARTERS_INCOME, BUILDING_INCOME, TOWER_INCOME,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level game state storage
# Kaggle's interpreter is called once per step; we need to persist the
# GameState object across calls within the same episode.
# ---------------------------------------------------------------------------
_games = {}


# ---------------------------------------------------------------------------
# Specification (loaded from JSON)
# ---------------------------------------------------------------------------
_dirpath = path.dirname(__file__)
_jsonpath = path.abspath(path.join(_dirpath, "reinforce_tactics.json"))
with open(_jsonpath, encoding="utf-8") as _f:
    specification = json.load(_f)


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
def interpreter(state, env):
    """
    Core game logic. Called once per step by the kaggle-environments engine.

    On the first call (``env.done == True``), this initialises the game.
    On subsequent calls it processes the active agent's actions, checks
    win/draw conditions, updates observations, and swaps the active player.

    Args:
        state: list of per-agent state structs. Each has:
            .action        - the action returned by the agent
            .reward        - read/write reward
            .status        - ACTIVE / INACTIVE / DONE / ERROR / INVALID / TIMEOUT
            .observation   - per-agent observation struct
        env: environment handle with:
            .configuration - merged configuration struct
            .done          - True on the initialisation call
            .steps         - list of all previous steps

    Returns:
        state (modified in-place)
    """
    key = id(env)

    # ------------------------------------------------------------------
    # Initialisation (first call after env.reset)
    # ------------------------------------------------------------------
    if env.done:
        game = _init_game(env.configuration)
        _games[key] = game
        _update_observations(state, game, env.configuration)
        state[0].status = "ACTIVE"
        state[1].status = "INACTIVE"
        return state

    game = _games.get(key)
    if game is None:
        # Safety fallback – should not happen in normal flow
        for agent in state:
            agent.status = "ERROR"
        return state

    # ------------------------------------------------------------------
    # Determine which agent is active
    # ------------------------------------------------------------------
    active_idx = _get_active_index(state)
    if active_idx is None:
        return state  # both done / error

    agent = state[active_idx]
    actions = agent.action if agent.action else []

    # Ensure actions is a list
    if not isinstance(actions, list):
        actions = [actions]

    # ------------------------------------------------------------------
    # Execute agent actions
    # ------------------------------------------------------------------
    game_player = active_idx + 1  # GameState uses 1-indexed players

    for action in actions:
        if not isinstance(action, dict):
            agent.status = "INVALID"
            agent.reward = -1
            state[1 - active_idx].reward = 1
            state[1 - active_idx].status = "DONE"
            agent.status = "DONE"
            return state

        action_type = action.get("type", "")
        if action_type == "end_turn":
            break

        success = _execute_action(game, action, game_player)
        if not success:
            # Invalid action – the agent loses
            agent.status = "INVALID"
            agent.reward = -1
            state[1 - active_idx].reward = 1
            state[1 - active_idx].status = "DONE"
            agent.status = "DONE"
            return state

        # Check if game ended mid-action (e.g. HQ captured or all units eliminated)
        if game.game_over:
            break

    # ------------------------------------------------------------------
    # End the turn (income, healing, status effects, etc.)
    # ------------------------------------------------------------------
    if not game.game_over:
        game.end_turn()

    # ------------------------------------------------------------------
    # Check win / draw conditions
    # ------------------------------------------------------------------
    if game.game_over:
        winner_idx = game.winner - 1  # convert to 0-indexed
        state[winner_idx].reward = 1
        state[winner_idx].status = "DONE"
        state[1 - winner_idx].reward = -1
        state[1 - winner_idx].status = "DONE"
        _update_observations(state, game, env.configuration)
        # Clean up stored game
        _games.pop(key, None)
        return state

    # Check max turns (draw)
    max_turns = env.configuration.episodeSteps
    if game.turn_number >= max_turns:
        for i in range(2):
            state[i].reward = 0
            state[i].status = "DONE"
        _update_observations(state, game, env.configuration)
        _games.pop(key, None)
        return state

    # ------------------------------------------------------------------
    # Update observations and swap active player
    # ------------------------------------------------------------------
    _update_observations(state, game, env.configuration)

    state[active_idx].status = "INACTIVE"
    state[1 - active_idx].status = "ACTIVE"

    return state


# ---------------------------------------------------------------------------
# Map Generation (inlined to avoid pygame dependency from utils package)
# ---------------------------------------------------------------------------
def _generate_map(width, height, num_players=2):
    """
    Generate a random map as a pandas DataFrame.

    This mirrors ``FileIO.generate_random_map`` but is self-contained so the
    kaggle adapter does not need to import the ``reinforcetactics.utils``
    package (which transitively pulls in pygame via ReplayPlayer).
    """
    import pandas as pd

    width = max(width, 20)
    height = max(height, 20)

    map_data = np.full((height, width), 'o', dtype=object)

    num_tiles = width * height

    # Forests (10%)
    for _ in range(num_tiles // 10):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        map_data[y, x] = 'f'

    # Mountains (5%)
    for _ in range(num_tiles // 20):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        map_data[y, x] = 'm'

    # Water (3%)
    for _ in range(num_tiles // 33):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        map_data[y, x] = 'w'

    # Player headquarters and buildings
    if num_players >= 1:
        map_data[1, 1] = 'h_1'
        map_data[1, 2] = 'b_1'
        map_data[2, 1] = 'b_1'

    if num_players >= 2:
        map_data[height - 2, width - 2] = 'h_2'
        map_data[height - 2, width - 3] = 'b_2'
        map_data[height - 3, width - 2] = 'b_2'

    if num_players >= 3:
        map_data[1, width - 2] = 'h_3'
        map_data[1, width - 3] = 'b_3'
        map_data[2, width - 2] = 'b_3'

    if num_players >= 4:
        map_data[height - 2, 1] = 'h_4'
        map_data[height - 2, 2] = 'b_4'
        map_data[height - 3, 1] = 'b_4'

    # Neutral towers in centre
    cx, cy = width // 2, height // 2
    for dx, dy in [(0, 0), (3, 0), (0, 3), (3, 3)]:
        x, y = cx + dx - 2, cy + dy - 2
        if 0 <= x < width and 0 <= y < height:
            if map_data[y, x] == 'p':
                map_data[y, x] = 't'

    return pd.DataFrame(map_data)


# ---------------------------------------------------------------------------
# Game Initialisation
# ---------------------------------------------------------------------------
def _init_game(config):
    """Create a new GameState from the Kaggle configuration."""
    width = config.mapWidth
    height = config.mapHeight
    seed = config.mapSeed

    if seed >= 0:
        np.random.seed(seed)

    map_data = _generate_map(width, height, num_players=2)

    enabled_units = [u.strip() for u in config.enabledUnits.split(",") if u.strip()]
    fog_of_war = bool(config.fogOfWar)

    game = GameState(
        map_data,
        num_players=2,
        max_turns=config.episodeSteps,
        enabled_units=enabled_units,
        fog_of_war=fog_of_war,
    )

    # Override starting gold if configured
    starting_gold = config.startingGold
    game.player_gold = {1: starting_gold, 2: starting_gold}

    return game


# ---------------------------------------------------------------------------
# Action Execution
# ---------------------------------------------------------------------------
def _execute_action(game, action, player):
    """
    Translate a single action dict into a GameState method call.

    Returns True on success, False on invalid action.
    """
    atype = action.get("type", "")

    try:
        if atype == "create_unit":
            unit_type = action.get("unit_type", "")
            x = int(action.get("x", -1))
            y = int(action.get("y", -1))
            if unit_type not in UNIT_DATA:
                return False
            result = game.create_unit(unit_type, x, y, player)
            return result is not None

        elif atype == "move":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            unit = game.get_unit_at_position(from_x, from_y)
            if unit is None or unit.player != player:
                return False
            return game.move_unit(unit, to_x, to_y)

        elif atype == "attack":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            attacker = game.get_unit_at_position(from_x, from_y)
            target = game.get_unit_at_position(to_x, to_y)
            if attacker is None or target is None:
                return False
            if attacker.player != player or target.player == player:
                return False
            game.attack(attacker, target)
            return True

        elif atype == "seize":
            x = int(action.get("x", -1))
            y = int(action.get("y", -1))
            unit = game.get_unit_at_position(x, y)
            if unit is None or unit.player != player:
                return False
            tile = game.grid.get_tile(x, y)
            if tile is None or not tile.is_capturable() or tile.player == player:
                return False
            game.seize(unit)
            return True

        elif atype == "heal":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            healer = game.get_unit_at_position(from_x, from_y)
            target = game.get_unit_at_position(to_x, to_y)
            if healer is None or target is None:
                return False
            if healer.player != player or healer.type != 'C':
                return False
            amount = game.heal(healer, target)
            return amount > 0

        elif atype == "cure":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            curer = game.get_unit_at_position(from_x, from_y)
            target = game.get_unit_at_position(to_x, to_y)
            if curer is None or target is None:
                return False
            if curer.player != player or curer.type != 'C':
                return False
            return game.cure(curer, target)

        elif atype == "paralyze":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            mage = game.get_unit_at_position(from_x, from_y)
            target = game.get_unit_at_position(to_x, to_y)
            if mage is None or target is None:
                return False
            if mage.player != player or mage.type != 'M':
                return False
            return game.paralyze(mage, target)

        elif atype == "haste":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            sorcerer = game.get_unit_at_position(from_x, from_y)
            target = game.get_unit_at_position(to_x, to_y)
            if sorcerer is None or target is None:
                return False
            if sorcerer.player != player or sorcerer.type != 'S':
                return False
            return game.haste(sorcerer, target)

        elif atype == "defence_buff":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            sorcerer = game.get_unit_at_position(from_x, from_y)
            target = game.get_unit_at_position(to_x, to_y)
            if sorcerer is None or target is None:
                return False
            if sorcerer.player != player or sorcerer.type != 'S':
                return False
            return game.defence_buff(sorcerer, target)

        elif atype == "attack_buff":
            from_x = int(action.get("from_x", -1))
            from_y = int(action.get("from_y", -1))
            to_x = int(action.get("to_x", -1))
            to_y = int(action.get("to_y", -1))
            sorcerer = game.get_unit_at_position(from_x, from_y)
            target = game.get_unit_at_position(to_x, to_y)
            if sorcerer is None or target is None:
                return False
            if sorcerer.player != player or sorcerer.type != 'S':
                return False
            return game.attack_buff(sorcerer, target)

        elif atype == "end_turn":
            return True

        else:
            # Unknown action type
            return False

    except Exception:
        logger.exception("Error executing action: %s", action)
        return False


# ---------------------------------------------------------------------------
# Observation Serialisation
# ---------------------------------------------------------------------------
def _update_observations(state, game, config):
    """Serialise the current GameState into each agent's observation."""
    board = _serialize_board(game)
    structures = _serialize_structures(game)
    gold = [game.player_gold.get(1, 0), game.player_gold.get(2, 0)]

    fog_of_war = bool(config.fogOfWar)

    for i in range(2):
        obs = state[i].observation
        obs.board = board
        obs.structures = structures
        obs.gold = gold
        obs.turnNumber = game.turn_number
        obs.mapWidth = game.grid.width
        obs.mapHeight = game.grid.height

        player = i + 1  # 1-indexed game player

        if fog_of_war:
            # Update visibility and filter units per player
            game.update_visibility(player)
            obs.units = _serialize_units(game, visible_for_player=player)
        else:
            obs.units = _serialize_units(game)


def _serialize_board(game):
    """Convert the game grid to a 2D array of terrain type codes."""
    board = []
    for y in range(game.grid.height):
        row = []
        for x in range(game.grid.width):
            tile = game.grid.get_tile(x, y)
            row.append(tile.type)
        board.append(row)
    return board


def _serialize_structures(game):
    """Convert capturable structures to a list of dicts."""
    structures = []
    for row in game.grid.tiles:
        for tile in row:
            if tile.is_capturable():
                structures.append({
                    "x": tile.x,
                    "y": tile.y,
                    "type": tile.type,
                    "owner": tile.player if tile.player else 0,
                    "hp": tile.health if tile.health is not None else 0,
                    "maxHp": tile.max_health if tile.max_health is not None else 0,
                })
    return structures


def _serialize_units(game, visible_for_player=None):
    """
    Convert units to a list of dicts.

    If ``visible_for_player`` is set and fog-of-war is enabled, only units
    visible to that player (own units + units in visible tiles) are included.
    """
    units = []
    for unit in game.units:
        # Fog of war filtering
        if visible_for_player is not None:
            if unit.player != visible_for_player:
                if not game.is_position_visible(unit.x, unit.y, visible_for_player):
                    continue

        units.append({
            "type": unit.type,
            "owner": unit.player,
            "x": unit.x,
            "y": unit.y,
            "hp": unit.health,
            "maxHp": unit.max_health,
            "canMove": unit.can_move,
            "canAttack": unit.can_attack,
            "paralyzedTurns": unit.paralyzed_turns,
            "isHasted": unit.is_hasted,
            "distanceMoved": unit.distance_moved,
            "defenceBuffTurns": unit.defence_buff_turns,
            "attackBuffTurns": unit.attack_buff_turns,
        })
    return units


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_active_index(state):
    """Return the index of the ACTIVE agent, or None if none."""
    for i in range(2):
        if state[i].status == "ACTIVE":
            return i
    return None


# ---------------------------------------------------------------------------
# Renderer (ASCII / ANSI)
# ---------------------------------------------------------------------------
def renderer(state, env):
    """Return an ASCII text representation of the current board."""
    if not state or len(state) < 2:
        return "No state available."

    obs = state[0].observation
    board = obs.board if hasattr(obs, "board") else []
    units_list = obs.units if hasattr(obs, "units") else []
    gold = obs.gold if hasattr(obs, "gold") else [0, 0]
    turn = obs.turnNumber if hasattr(obs, "turnNumber") else 0

    if not board:
        return "Board not initialised."

    # Build unit lookup
    unit_map = {}
    for u in units_list:
        unit_map[(u["x"], u["y"])] = u

    # Tile display characters
    tile_chars = {
        "p": ".", "w": "~", "m": "^", "f": "T",
        "r": "=", "b": "B", "h": "H", "t": "#", "o": "~",
    }

    lines = []
    lines.append(f"Turn {turn}  |  P1 Gold: {gold[0]}  |  P2 Gold: {gold[1]}")
    lines.append(f"P1 Status: {state[0].status}  |  P2 Status: {state[1].status}")
    lines.append("")

    for y, row in enumerate(board):
        line = ""
        for x, cell in enumerate(row):
            pos = (x, y)
            if pos in unit_map:
                u = unit_map[pos]
                # Show unit type with player indicator (lowercase=p1, uppercase=p2)
                ch = u["type"]
                line += ch.lower() if u["owner"] == 1 else ch.upper()
            else:
                line += tile_chars.get(cell, "?")
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in Agents
# ---------------------------------------------------------------------------
def _random_agent(observation, configuration):
    """Agent that always ends its turn immediately."""
    return [{"type": "end_turn"}]


def _aggressive_agent(observation, configuration):
    """
    Simple agent that creates warriors at available buildings,
    moves units toward the enemy, attacks when possible, and seizes
    structures.
    """
    actions = []
    player_idx = observation.player
    player = player_idx + 1  # 1-indexed
    gold = observation.gold[player_idx]
    my_units = [u for u in observation.units if u["owner"] == player]
    enemy_units = [u for u in observation.units if u["owner"] != player]

    # Find available buildings (structures owned by us that are buildings)
    structures = observation.structures if hasattr(observation, "structures") else []
    my_buildings = [
        s for s in structures
        if s["owner"] == player and s["type"] == "b"
    ]

    # Try to create warriors at buildings
    warrior_cost = UNIT_DATA["W"]["cost"]
    occupied = {(u["x"], u["y"]) for u in observation.units}
    for bldg in my_buildings:
        if gold >= warrior_cost and (bldg["x"], bldg["y"]) not in occupied:
            actions.append({
                "type": "create_unit",
                "unit_type": "W",
                "x": bldg["x"],
                "y": bldg["y"],
            })
            gold -= warrior_cost
            occupied.add((bldg["x"], bldg["y"]))

    actions.append({"type": "end_turn"})
    return actions


agents = {
    "random": _random_agent,
    "aggressive": _aggressive_agent,
}
