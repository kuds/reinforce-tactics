"""Shared utilities for save/load menus."""

import os
import re
from datetime import datetime
from typing import Any

from reinforcetactics.utils.language import get_language

# Seats the save/replay detail panels can draw (one colour and name row each).
MAX_DISPLAY_PLAYERS = 4


def extract_date_from_filename(filename: str) -> str:
    """Extract date from save/replay filename.

    Handles formats like "save_20251228_053412.json" or
    "game_20251228_053412_...".

    Args:
        filename: The filename to parse

    Returns:
        Formatted date string or "Unknown Date"
    """
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        date_part = match.group(1)
        time_part = match.group(2)
        try:
            dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return "Unknown Date"


def get_player_display_name(player_configs: list[dict], player_idx: int) -> str:
    """Get a display name for a player from config.

    Args:
        player_configs: List of player configuration dicts
        player_idx: Index of the player to get name for

    Returns:
        Human-readable player name
    """
    if player_idx >= len(player_configs):
        return f"Player {player_idx + 1}"

    config = player_configs[player_idx]
    player_type = config.get("type", "human")
    bot_type = config.get("bot_type", "")

    if player_type == "human":
        return "Human"
    elif player_type == "llm":
        name = config.get("name", "")
        if name:
            return name
        model = config.get("model", "")
        if model:
            return model
        return "LLM"
    elif player_type == "computer" or bot_type:
        if bot_type:
            return bot_type
        return "Bot"
    else:
        return player_type.title()


# The save and replay pickers read every JSON file in their folders when they
# open, and one hand-edited, truncated or foreign file used to crash the whole
# menu (and so the app). The helpers below coerce each field the pickers draw
# to the type the drawing code expects, falling back to a neutral default.


def mtime_or_zero(path: str) -> float:
    """Modification time for newest-first sorting; 0 if it can't be read.

    A file can vanish (or be a dangling symlink) between listing the folder
    and sorting it, which would otherwise raise out of the menu's __init__.
    """
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def as_int(value: Any, default: int) -> int:
    """Return ``value`` if it is an integer (JSON ``true``/``false`` aren't), else ``default``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def as_optional_int(value: Any) -> int | None:
    """Return ``value`` if it is an integer, else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def as_player_count(value: Any) -> int:
    """Return a player count the detail panels can draw (1-4), else 2."""
    count = as_int(value, 2)
    return count if 1 <= count <= MAX_DISPLAY_PLAYERS else 2


def as_list(value: Any) -> list:
    """Return ``value`` if it is a list, else an empty list."""
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict:
    """Return ``value`` if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def map_display_name(map_file: Any) -> str:
    """Label for a save's or replay's ``map_file``.

    Random-map games record ``"map_file": null``; they are labelled
    "Random Map" (``os.path.basename(None)`` used to crash the replay
    picker). A value of the wrong type is labelled "Unknown Map".
    """
    if map_file is None or map_file == "":
        return get_language().get("map_random", "Random Map")
    if not isinstance(map_file, str):
        return "Unknown Map"
    return os.path.basename(map_file).replace(".csv", "").replace("_", " ").title()


def safe_player_display_name(player_configs: Any, player_idx: int) -> str:
    """:func:`get_player_display_name` that tolerates malformed configs."""
    configs = as_list(player_configs)
    if player_idx < len(configs) and not isinstance(configs[player_idx], dict):
        return f"Player {player_idx + 1}"
    try:
        return str(get_player_display_name(configs, player_idx))
    except (AttributeError, TypeError):
        # e.g. a non-string "type" (get_player_display_name calls .title())
        return f"Player {player_idx + 1}"
