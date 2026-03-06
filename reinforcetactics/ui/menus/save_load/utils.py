"""Shared utilities for save/load menus."""

import re
from datetime import datetime
from typing import Dict, List


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


def get_player_display_name(player_configs: List[Dict], player_idx: int) -> str:
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
