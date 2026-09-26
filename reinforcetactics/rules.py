"""
Game rules: tile types, unit stats, economy, combat and status effects.

This is the data the headless engine (``core``, ``game``, ``rl``) plays by.
Rendering data (colours, sprite files, animation layout) lives in
:mod:`reinforcetactics.ui.assets`, and :mod:`reinforcetactics.constants`
re-exports both for older imports.
"""

from enum import Enum
from typing import Self


class TileType(Enum):
    """Enumeration of tile types in the game."""

    GRASS = "p"
    WATER = "w"
    MOUNTAIN = "m"
    FOREST = "f"
    ROAD = "r"
    BUILDING = "b"
    HEADQUARTERS = "h"
    TOWER = "t"
    OCEAN = "o"

    @classmethod
    def from_code(cls, code: str) -> Self:
        """Get TileType from single-letter code."""
        for tile_type in cls:
            if tile_type.value == code:
                return tile_type
        raise ValueError(f"Unknown tile code: {code}")

    def is_walkable(self) -> bool:
        """Check if this tile type can be walked on."""
        return self not in (TileType.WATER, TileType.OCEAN)

    def is_capturable(self) -> bool:
        """Check if this tile type can be captured."""
        return self in (TileType.TOWER, TileType.HEADQUARTERS, TileType.BUILDING)


# Tile type mapping (string code -> TileType name, e.g. "p" -> "GRASS")
TILE_TYPES = {tile_type.value: tile_type.name for tile_type in TileType}

# Map size
MIN_MAP_SIZE = 20
MIN_STRIP_SIZE = 6  # Minimum size to preserve when stripping water borders

# All available unit types — canonical ordering used by RL environments and bots.
# Must match the action space indices in StrategyGameEnv.
ALL_UNIT_TYPES = ["W", "M", "C", "A", "K", "R", "S", "B"]

# Mapping from unit type code to its index in ALL_UNIT_TYPES.
# Used by gym_env action masking and model_bot action translation.
UNIT_TYPE_TO_IDX = {ut: i for i, ut in enumerate(ALL_UNIT_TYPES)}

# Unit costs and stats. ``name`` is the display name bots and LLM prompts
# use; sprite files and colours are ``ui.assets.UNIT_ASSETS``, keyed by the
# same codes. engine_overrides["unit_data"] can change any of these fields
# per game (EngineConfig.from_overrides in core/engine_config.py).
UNIT_DATA = {
    "W": {
        "name": "Warrior",
        "cost": 200,
        "movement": 3,
        "health": 15,
        "attack": 10,
        "defence": 6,
    },
    "M": {
        "name": "Mage",
        "cost": 300,
        "movement": 2,
        "health": 10,
        "attack": {"adjacent": 8, "range": 12},
        "defence": 4,
    },
    "C": {
        "name": "Cleric",
        "cost": 200,
        "movement": 3,
        "health": 10,
        "attack": 2,
        "defence": 4,
    },
    "B": {
        "name": "Barbarian",
        "cost": 400,
        "movement": 5,
        "health": 20,
        "attack": 10,
        "defence": 2,
    },
    "A": {
        "name": "Archer",
        "cost": 250,
        "movement": 3,
        "health": 15,
        "attack": 5,
        "defence": 1,
    },
    "K": {
        "name": "Knight",
        "cost": 350,
        "movement": 4,
        "health": 18,
        "attack": 8,
        "defence": 5,
    },
    "R": {
        "name": "Rogue",
        "cost": 350,
        "movement": 4,
        "health": 12,
        "attack": 9,
        "defence": 3,
    },
    "S": {
        "name": "Sorcerer",
        "cost": 350,
        "movement": 2,
        "health": 12,
        "attack": {"adjacent": 6, "range": 8},
        "defence": 3,
    },
}

# Starting gold for each player
STARTING_GOLD = 250

# Income rates
HEADQUARTERS_INCOME = 150
BUILDING_INCOME = 100
TOWER_INCOME = 50

# Hard ceiling on how many units a single player may have on the board at
# once. Caps both the action-space balloon (move enumeration scales with
# army size) and the "convert-all-gold-to-permanent-free-units" economy --
# once at the cap, extra gold has no unit sink. Config-surfaced via
# engine_overrides["max_units_per_player"].
MAX_UNITS_PER_PLAYER = 50

# Structure health
TOWER_MAX_HEALTH = 30
BUILDING_MAX_HEALTH = 40
HEADQUARTERS_MAX_HEALTH = 50

# Structure regeneration rate (percentage of max HP per turn)
STRUCTURE_REGEN_RATE = 0.5

# Combat
COUNTER_ATTACK_MULTIPLIER = 0.8
DEFENCE_REDUCTION_PER_POINT = 0.05  # Each defence point reduces damage by 5%

# Special ability bonuses
CHARGE_BONUS = 0.5  # Knight: +50% damage if moved 3+ tiles
CHARGE_MIN_DISTANCE = 3  # Minimum tiles moved to trigger Charge
FLANK_BONUS = 0.5  # Rogue: +50% damage if enemy is adjacent to a friendly unit
ROGUE_EVADE_CHANCE = 0.15  # Rogue: 15% chance to dodge counter-attacks

# Status effects
# Durations count the affected unit's OWN turns, the same way for every
# status: a paralyzed unit loses its next PARALYZE_DURATION turns, and a buff
# covers SORCERER_BUFF_DURATION turns of its owner. The per-unit counters
# (``paralyzed_turns``, ``*_buff_turns``) instead count the owner's turn
# STARTS until the status ends -- they tick at the start of the owner's turn
# -- so a status applied outside its owner's turn is stored one higher (a
# paralysis is always cast on the victim's opponent's turn, so it is stored as
# PARALYZE_DURATION + 1 and wears off as the victim's first free turn starts).
# This is the pre-2026-09 rule unchanged (the constant used to be 3, the
# counter value, while the victim lost 2 turns).
PARALYZE_DURATION = 2
PARALYZE_COOLDOWN = 2  # Turns before Mage can use Paralyze again
PARALYZE_RANGE = 2  # Max Manhattan distance for Mage paralyze
HEAL_AMOUNT = 7
CLERIC_HEAL_RANGE = 3  # Max Manhattan distance for Cleric heal and cure-paralyze abilities
HASTE_COOLDOWN = 2  # Turns before Sorcerer can use Haste again
HASTE_RANGE = 2  # Max Manhattan distance for Sorcerer haste

# Rogue forest bonus
ROGUE_FOREST_EVADE_BONUS = 0.15  # Additional 15% dodge chance when in forest (15% + 15% = 30%)

# Sorcerer buff abilities
SORCERER_BUFF_DURATION = 3  # Own turns of the buffed unit the buff covers (see PARALYZE_DURATION)
SORCERER_BUFF_COOLDOWN = 2  # Turns before Sorcerer can use buff again
SORCERER_DEFENCE_BUFF_AMOUNT = 0.50  # 50% damage reduction
SORCERER_ATTACK_BUFF_AMOUNT = 0.50  # 50% damage increase
BUFF_RANGE = 2  # Max Manhattan distance for Sorcerer defence and attack buffs

# (min, max) Manhattan distance from caster to target for each targeted
# ability -- the one place the engine reads them from (the legal-action
# predicates and the mechanics that apply the ability both do). Min 1 means
# the caster cannot target itself; the buffs' min 0 lets a Sorcerer buff
# itself. Attack reach is per unit type: ``Unit.get_attack_range``.
ABILITY_RANGES = {
    "paralyze": (1, PARALYZE_RANGE),
    "heal": (1, CLERIC_HEAL_RANGE),
    "cure": (1, CLERIC_HEAL_RANGE),
    "haste": (1, HASTE_RANGE),
    "defence_buff": (0, BUFF_RANGE),
    "attack_buff": (0, BUFF_RANGE),
}
