"""Saves and replays: GameState to and from JSON-ready dicts (review core-16).

Moved out of ``game_state.py`` so persistence can be read and changed
without the rules engine around it. ``GameState`` keeps its methods
(``to_dict``, ``from_dict``, ``saved_map_data``, ``save_to_file``,
``save_replay_to_file``, ``build_player_config``), which call these.
"""

from __future__ import annotations

import base64
import copy
import logging
import random
import struct
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from reinforcetactics.core.unit import Unit

if TYPE_CHECKING:
    from reinforcetactics.core.game_state import GameState

# Version of the save format written by ``GameState.to_dict``. Saves without
# the field are version 1 (everything before it existed) and still load.
# 2: adds the fields ``from_dict`` needs to resume a game exactly
# (winning_action_index, healing_totals, per-unit has_moved, fog-of-war
# attack snapshot and ambushed flag, the fog-of-war state). Early version 2
# saves also carry padding metadata (original_map_width/height,
# map_padding_offset_x/y, original_map_data); nothing ever set it, so it
# always described the saved grid itself, and ``from_dict`` ignores it
# (with a warning should the offsets not be zero). Later additions, optional
# on load: per-unit haste_refreshed and the fog-of-war pre-move view
# cancel_move restores.
SAVE_FORMAT_VERSION = 2

# Load warnings keep the logger they had while this code lived in
# game_state.py, so log filters set up for it still apply.
logger = logging.getLogger("reinforcetactics.core.game_state")


def rng_state_for_save(rng: Any) -> str | None:
    """The game RNG's exact position, compact enough for a JSON save.

    The seed alone only reproduces a game from turn 0; a mid-game save
    also needs the stream position so a reload rolls what the unsaved
    game would have rolled. None for a caller-supplied source whose
    state cannot be captured.
    """
    if not isinstance(rng, random.Random):
        return None
    try:
        version, internal, gauss_next = rng.getstate()
    except NotImplementedError:  # e.g. random.SystemRandom
        return None
    packed = base64.b64encode(struct.pack(f"<{len(internal)}I", *internal)).decode("ascii")
    return f"{version}:{gauss_next!r}:{packed}"


def rng_from_save(encoded: str) -> random.Random:
    """Rebuild the RNG ``rng_state_for_save`` captured."""
    version, gauss_next, packed = encoded.split(":", 2)
    raw = base64.b64decode(packed)
    internal = struct.unpack(f"<{len(raw) // 4}I", raw)
    rng = random.Random()
    rng.setstate((int(version), internal, None if gauss_next == "None" else float(gauss_next)))
    return rng


def unit_to_save_dict(unit: Unit) -> dict[str, Any]:
    """``unit.to_dict()`` plus, under fog of war, the view ``cancel_move`` restores.

    A unit whose move can still be cancelled carries its side's
    visibility map from before the move, so a game saved at that point
    can cancel it after loading, taking back what the move revealed.
    """
    data = unit.to_dict()
    if unit.pre_move_visibility is not None:
        data["pre_move_visibility"] = unit.pre_move_visibility.to_dict()
    return data


def game_to_dict(game: GameState) -> dict[str, Any]:
    """Convert game state to dictionary for serialization (``GameState.to_dict``).

    Records everything ``from_dict`` needs to resume the game exactly:
    ``from_dict(json(to_dict(g))).to_dict()`` equals ``json(to_dict(g))``,
    and both games offer every player the same legal actions (see
    tests/test_save_roundtrip_core.py). Not recorded: the legal-action
    caches and the UI-only ``Unit.selected``. Mutable containers are
    copied, so the result shares nothing with the live game.
    """
    return {
        "save_format_version": SAVE_FORMAT_VERSION,
        "timestamp": game.game_start_time.strftime("%Y-%m-%d %H-%M-%S"),
        "current_player": game.current_player,
        "num_players": game.num_players,
        "player_gold": dict(game.player_gold),
        "turn_number": game.turn_number,
        "game_over": game.game_over,
        "winner": game.winner,
        # from_dict has always read these two back; without them a
        # turn-limited game reloaded as unlimited and a finished game
        # lost how it ended.
        "end_reason": game.end_reason,
        "max_turns": game.max_turns,
        # The index of the action that ended the game, and the
        # game-lifetime auto-heal totals: both feed the replay's
        # integrity fields, which were wrong for continued games.
        "winning_action_index": game.game_over_action_index,
        "healing_totals": {p: dict(t) for p, t in game.healing_totals.items()},
        "map_file": game.map_file_used,
        # The exact tile codes the grid was built from (after any UI
        # padding), written for every save, not only map-file-less ones.
        # "tiles" below holds only capturable tiles, so without this a
        # random-map save could never be reloaded, and a save that
        # points at a map file breaks silently if that file is later
        # edited, renamed or padded differently: unit and structure
        # coordinates would land on different terrain. The cost is one
        # short string per tile: a fixed ~7 KB for the usual 24x24 padded
        # map at the save writer's indent=2, about a quarter of an
        # early-game save and a shrinking share as action_history grows.
        "map_data": [list(row) for row in game.initial_map_data],
        "player_configs": copy.deepcopy(game.player_configs),
        "enabled_units": list(game.enabled_units),
        "fog_of_war": game.fog_of_war,
        "fog_of_war_method": game.fog_of_war_method,
        # What each player has explored and remembers. Without it a
        # reloaded fog-of-war game re-fogged every tile out of current
        # sight and forgot every structure (critic-integration-11).
        # Keyed by str(player) so the dict is the same before and after
        # a JSON round trip.
        "fog_of_war_state": game.fog.to_dict(),
        # Persist the engine-constant overlay so a reloaded game runs under
        # the same balance (damage_model, structure HP, economy, unit cap)
        # it was saved under. Absent in pre-0.3.3 saves -> from_dict falls
        # back to {} (== module defaults), preserving backward-compat.
        "engine_overrides": game.engine_config.to_overrides(),
        # The combat RNG (review core-10): its seed, and its position so a
        # reloaded game rolls exactly what the unsaved game would have.
        "seed": game.seed,
        "rng_state": rng_state_for_save(game.rng),
        # Every seat's team and who is out (review core-4/core-7). Teams a
        # map declares are re-derived from map_data on load and must
        # agree; this also carries ones passed as GameState(teams=...).
        "teams": game.teams,
        "eliminated_players": sorted(game.eliminated_players),
        "units": [unit_to_save_dict(unit) for unit in game.units],
        "tiles": game.grid.to_dict()["tiles"],
        # Records are never edited once written, so a new list suffices.
        "action_history": list(game.action_history),
        # Restore the per-game unit-id counter on reload so newly
        # created units after load don't reuse retired ids
        # (which would let the replay v3 dispatch route an action
        # to the wrong unit -- exactly the brittleness this whole
        # schema bump is meant to eliminate).
        "next_unit_id": game._next_unit_id,
    }


def save_to_file(game: GameState, filepath: str | None = None) -> str | None:
    """
    Save game state to file.

    Args:
        game: The game to save
        filepath: Path to save file (auto-generated if None)

    Returns:
        Path to saved file
    """
    from reinforcetactics.utils.file_io import FileIO

    return FileIO.save_game(game, filepath)


def replay_player_type(config: dict[str, Any]) -> str:
    """
    Get the standardized player type for replay logs.

    Args:
        config: Player configuration dictionary

    Returns:
        Player type string: 'human', 'bot', 'llm', or 'rl'
    """
    if config.get("type") == "human":
        return "human"

    # Prefer the type already resolved by the app / tournament layers
    # (create_bots_from_config and the tournament runner both stamp it).
    resolved = config.get("player_type")
    if resolved:
        return resolved

    # Fallback for configs that never went through those layers.
    # Deferred import: the engine must not import the game layer at
    # module load (core stays self-contained); this only runs on the
    # save-replay path.
    from reinforcetactics.game.bot_registry import player_type

    return player_type(config.get("bot_type", ""))


def build_player_config(
    player_no: int, name: str, player_type: str, temperature: float | None = None, max_tokens: int | None = None
) -> dict[str, Any]:
    """
    Build a standardized player config for replay logs.

    Args:
        player_no: Player number (1, 2, etc.)
        name: Display name for the player/bot
        player_type: One of 'human', 'bot', 'llm', 'rl'
        temperature: LLM temperature (only for llm type)
        max_tokens: LLM max tokens (only for llm type)

    Returns:
        Standardized player config dictionary
    """
    config: dict[str, Any] = {"player_no": player_no, "type": player_type, "name": name}

    # Add LLM-specific fields
    if player_type == "llm":
        config["temperature"] = temperature
        config["max_tokens"] = max_tokens

    return config


def save_replay_to_file(game: GameState, filepath: str | None = None) -> str | None:
    """
    Save replay to file.

    Args:
        game: The game whose replay to save
        filepath: Path to replay file (auto-generated if None)

    Returns:
        Path to saved replay
    """
    from reinforcetactics.utils.file_io import FileIO

    # Build player_configs for replay
    # If already in standardized format (has 'player_no'), use directly
    # Otherwise, transform from old format for backward compatibility
    enhanced_player_configs = []

    for i, config in enumerate(game.player_configs):
        player_num = i + 1

        # Check if already in standardized format
        if "player_no" in config:
            enhanced_player_configs.append(config)
        else:
            # Transform from old format (player_name, player_type, bot_type, etc.)
            player_name = config.get("player_name", config.get("name", "Unknown"))

            # Always use replay_player_type to map old format types (e.g., 'computer' -> 'bot')
            player_type = replay_player_type(config)

            enhanced_config = {"player_no": player_num, "type": player_type, "name": player_name}

            # Add LLM-specific fields if applicable
            if player_type == "llm":
                enhanced_config["temperature"] = config.get("temperature", None)
                enhanced_config["max_tokens"] = config.get("max_tokens", None)

            enhanced_player_configs.append(enhanced_config)

    from reinforcetactics import __version__ as _rt_version

    # Final-state snapshot doubles as a replay-integrity checksum;
    # see runner._save_replay for the same fields.
    final_units_by_player: dict[int, list] = {}
    for u in game.units:
        final_units_by_player.setdefault(u.player, []).append(u)
    final_counts = {p: len(us) for p, us in final_units_by_player.items()}
    final_hp = {p: sum(u.health for u in us) for p, us in final_units_by_player.items()}

    game_info = {
        "num_players": game.num_players,
        "max_turns": game.max_turns,
        "total_turns": game.turn_number,
        "winner": game.winner,
        "game_over": game.game_over,
        "end_reason": game.end_reason,
        "winning_action_index": game.game_over_action_index,
        "start_time": game.game_start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "map_file": game.map_file_used,
        # The grid the actions' coordinates refer to (see record_action)
        "initial_map": game.initial_map_data,
        "player_configs": enhanced_player_configs,
        "enabled_units": game.enabled_units,
        "fog_of_war": game.fog_of_war,
        "fog_of_war_method": game.fog_of_war_method,
        # Seed of the combat RNG, so the game can be re-run (core-10).
        "seed": game.seed,
        "library_version": _rt_version,
        "replay_schema_version": 3,
        "final_unit_counts": final_counts,
        "final_hp_totals": final_hp,
        # Structure auto-heal economics (HP restored / gold spent per
        # player over the whole game). Queryable without re-simulating
        # the action log, and doubles as a replay-integrity checksum:
        # playback re-executes end_turn, so a faithful replay's
        # re-accumulated healing_totals must match these values.
        "healing_totals": {p: dict(t) for p, t in game.healing_totals.items()},
        # What the replay's GameState must be built with to play the log
        # back faithfully (see replay_actions.replay_game_state_kwargs):
        # the balance overlay (e.g. begin_first_turn gives Player 1 turn-0
        # income its first creates may spend) and the teams (explicit
        # ones are not in the map). eliminated_players is informational.
        "engine_overrides": game.engine_overrides,
        "begin_first_turn": game.begin_first_turn,
        "teams": game.teams,
        "eliminated_players": sorted(game.eliminated_players),
    }

    return FileIO.save_replay(game.action_history, game_info, filepath)


def saved_map_data(save_data: dict[str, Any]) -> pd.DataFrame | None:
    """Return the terrain a save recorded via ``to_dict``, or None.

    Saves written before the terrain was recorded return None; their
    callers fall back to reloading ``save_data["map_file"]``.
    """
    terrain = save_data.get("map_data")
    if not terrain:
        return None
    return pd.DataFrame(terrain)


def game_from_dict(cls: type[GameState], save_data: dict[str, Any], map_data=None) -> GameState:
    """
    Restore game state from dictionary (``GameState.from_dict``).

    Loads every save format version up to ``SAVE_FORMAT_VERSION``; a
    field an older save lacks takes the value a new game would have (a
    version 1 fog-of-war save, for one, gets fog rebuilt from the
    current board).

    Args:
        cls: The GameState class to build
        save_data: Dictionary with saved game data
        map_data: Map data (2D array). ``None`` rebuilds the grid from the
            terrain ``to_dict`` records under ``"map_data"``.

    Returns:
        Restored GameState instance

    Raises:
        ValueError: If ``map_data`` is None and the save predates recorded
            terrain (it only names a map file, or none for random maps).
    """
    if map_data is None:
        map_data = saved_map_data(save_data)
        if map_data is None:
            raise ValueError("Save has no recorded terrain ('map_data'); pass the map explicitly")

    # Nothing below depends on the version yet (every field falls back
    # to a new game's value), so an unknown one only earns a warning. A
    # hand-edited, non-numeric version must not make the load crash.
    raw_version = save_data.get("save_format_version", 1)
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        logger.warning("Save format version %r is not a number; loading what it recognises", raw_version)
    else:
        if version > SAVE_FORMAT_VERSION:
            logger.warning(
                "Save format version %s is newer than this version of the game (%s); loading what it recognises",
                version,
                SAVE_FORMAT_VERSION,
            )

    # Every container read from save_data is copied: the caller keeps its
    # dict, and a game must not share lists with it (or with the
    # class-level ALL_UNIT_TYPES) that either side could mutate.
    # Extract enabled_units from save data (default to all if not present for backward compatibility)
    saved_units = save_data.get("enabled_units")
    enabled_units = list(saved_units) if saved_units is not None else list(cls.ALL_UNIT_TYPES)

    # Extract fog_of_war from save data (default to False for backward compatibility)
    fog_of_war = save_data.get("fog_of_war", False)

    # Extract fog_of_war_method (default to 'simple_radius' if FOW enabled, 'none' otherwise)
    fog_of_war_method = save_data.get("fog_of_war_method", "simple_radius" if fog_of_war else "none")

    max_turns = save_data.get("max_turns")
    # Restore the engine-constant overlay (damage_model / structure HP /
    # economy / unit cap). Absent in pre-0.3.3 saves -> {} == module
    # defaults, byte-identical to the old load behaviour.
    engine_overrides = copy.deepcopy(save_data.get("engine_overrides") or {})
    # JSON turns the int player keys into strings. A save from before
    # teams existed has none and was played free-for-all, whatever team
    # codes its map carries: load it that way (map_teams=False).
    teams = {int(p): int(t) for p, t in (save_data.get("teams") or {}).items()} or None
    game = cls(
        map_data,
        save_data.get("num_players", 2),
        max_turns=max_turns,
        enabled_units=enabled_units,
        fog_of_war=fog_of_war,
        engine_overrides=engine_overrides,
        # Saves from before the seed was recorded get a fresh one.
        seed=save_data.get("seed"),
        teams=teams,
        map_teams="teams" in save_data,
    )
    if save_data.get("rng_state"):
        # Continue the saved stream exactly. The seed stays what the save
        # recorded (None for a caller-supplied rng), not a fresh draw.
        game.rng = rng_from_save(save_data["rng_state"])
        game.seed = save_data.get("seed")
    game.eliminated_players = {int(p) for p in save_data.get("eliminated_players", [])}

    # Restore the fog of war method
    game.fog.method = fog_of_war_method

    try:
        game.game_start_time = datetime.strptime(save_data["timestamp"], "%Y-%m-%d %H-%M-%S")
    except (KeyError, TypeError, ValueError):
        pass  # keep "now" for saves without a usable timestamp
    game.current_player = save_data.get("current_player", 1)
    game.turn_number = save_data.get("turn_number", 0)
    game.game_over = save_data.get("game_over", False)
    game.winner = save_data.get("winner")
    game.end_reason = save_data.get("end_reason")
    game.game_over_action_index = save_data.get("winning_action_index")
    # Restore unit-id counter. Old saves predate this field; ``from_dict``
    # for the units themselves leaves ``unit.unit_id = None`` in that
    # case and ``find_unit_by_id`` falls back to position-based lookup.
    game._next_unit_id = save_data.get("next_unit_id", 0)

    # Fix player_gold dictionary key type (JSON serializes as strings)
    saved_gold = save_data.get("player_gold", {})
    game.player_gold = {int(k): v for k, v in saved_gold.items()}

    # Game-lifetime auto-heal totals (version 2+; older saves restart at 0)
    for p, totals in (save_data.get("healing_totals") or {}).items():
        game.healing_totals[int(p)] = {"hp": int(totals.get("hp", 0)), "gold": int(totals.get("gold", 0))}

    game.map_file_used = save_data.get("map_file")

    # Early version 2 saves carry padding metadata, always zero offsets
    # (see SAVE_FORMAT_VERSION). Only a script calling the since-removed
    # GameState.set_map_metadata could have saved others, and then the
    # saved action_history is not on the saved grid.
    offsets = (save_data.get("map_padding_offset_x") or 0, save_data.get("map_padding_offset_y") or 0)
    if offsets != (0, 0):
        logger.warning(
            "Save records map padding offsets %s; its action history is not on its grid, so a replay "
            "saved from this game will misplace the actions before the save",
            offsets,
        )

    # Restore player_configs (backward compatible with old saves)
    game.player_configs = copy.deepcopy(save_data.get("player_configs", []))

    # Restore units, with this game's stats (engine_overrides may change
    # them) rather than the module defaults, so max_health and attack
    # match the game the save was made in.
    game.units = []
    for unit_data in save_data.get("units", []):
        unit = Unit.from_dict(unit_data, stats=game.unit_data[unit_data["type"]])
        if game.fog_of_war and unit_data.get("pre_move_visibility") is not None:
            unit.pre_move_visibility = game.fog.map_from_dict(unit_data["pre_move_visibility"], unit.player)
        game.units.append(unit)

    # Restore tile states
    for tile_data in save_data.get("tiles", []):
        x, y = tile_data["x"], tile_data["y"]
        if 0 <= x < game.grid.width and 0 <= y < game.grid.height:
            tile = game.grid.tiles[y][x]
            # Restore a recorded neutral owner too: an eliminated
            # player's structures turn neutral (review core-7).
            if "player" in tile_data:
                tile.player = tile_data["player"] or None
            if tile_data.get("health") is not None:
                tile.health = tile_data["health"]
            if tile_data.get("regenerating") is not None:
                tile.regenerating = tile_data["regenerating"]

    # Restore action history (for continuing replay recording from a loaded save)
    game.action_history = copy.deepcopy(save_data.get("action_history", []))

    # Fog of war: restore what each player had explored and remembers.
    # The maps __init__ computed describe the fresh map, not this game.
    # A version 1 save has no such state, so its fog is rebuilt from the
    # current board (anything explored before the save is lost).
    game.fog.restore(save_data.get("fog_of_war_state"))

    game._invalidate_cache()
    return game
