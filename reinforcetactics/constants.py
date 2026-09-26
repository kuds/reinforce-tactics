"""
Compatibility facade for the game constants.

The constants are split by who reads them:

* :mod:`reinforcetactics.rules` -- tile types, unit stats, economy, combat
  and status effects. The headless engine (``core``, ``game``, ``rl``)
  imports these.
* :mod:`reinforcetactics.ui.assets` -- colours, sprite files and animation
  layout. Only the Pygame UI (and headless art tools) imports these.

Every name this module exported before the split is re-exported here, so
scripts, notebooks and older code keep working; new code should import from
the module that owns the value. The objects are the same ones, so editing
``UNIT_DATA`` in place still reaches the engine. ``UNIT_DATA`` holds the rule
fields only: the sprite paths and colour it used to carry per unit are
``ui.assets.UNIT_ASSETS``.
"""

from reinforcetactics.rules import (
    ALL_UNIT_TYPES,
    BUILDING_INCOME,
    BUILDING_MAX_HEALTH,
    CHARGE_BONUS,
    CHARGE_MIN_DISTANCE,
    CLERIC_HEAL_RANGE,
    COUNTER_ATTACK_MULTIPLIER,
    DEFENCE_REDUCTION_PER_POINT,
    FLANK_BONUS,
    HASTE_COOLDOWN,
    HEADQUARTERS_INCOME,
    HEADQUARTERS_MAX_HEALTH,
    HEAL_AMOUNT,
    MAX_UNITS_PER_PLAYER,
    MIN_MAP_SIZE,
    MIN_STRIP_SIZE,
    PARALYZE_COOLDOWN,
    PARALYZE_DURATION,
    ROGUE_EVADE_CHANCE,
    ROGUE_FOREST_EVADE_BONUS,
    SORCERER_ATTACK_BUFF_AMOUNT,
    SORCERER_BUFF_COOLDOWN,
    SORCERER_BUFF_DURATION,
    SORCERER_DEFENCE_BUFF_AMOUNT,
    STARTING_GOLD,
    STRUCTURE_REGEN_RATE,
    TILE_TYPES,
    TOWER_INCOME,
    TOWER_MAX_HEALTH,
    UNIT_DATA,
    UNIT_TYPE_TO_IDX,
    TileType,
)
from reinforcetactics.ui.assets import (
    ANIMATION_CONFIG,
    BASE_SPRITE_COLORS,
    FPS,
    NEUTRAL_STRUCTURE_PALETTE,
    PLAYER_COLORS,
    STRUCTURE_TILE_TYPES,
    TEAM_PALETTES,
    TILE_COLORS,
    TILE_IMAGES,
    TILE_SIZE,
    UNIT_ASSETS,
)

# Per-unit letter colour, now ``UNIT_ASSETS[code]["color"]``.
UNIT_COLORS = {code: art["color"] for code, art in UNIT_ASSETS.items()}

__all__ = [
    "ALL_UNIT_TYPES",
    "ANIMATION_CONFIG",
    "BASE_SPRITE_COLORS",
    "BUILDING_INCOME",
    "BUILDING_MAX_HEALTH",
    "CHARGE_BONUS",
    "CHARGE_MIN_DISTANCE",
    "CLERIC_HEAL_RANGE",
    "COUNTER_ATTACK_MULTIPLIER",
    "DEFENCE_REDUCTION_PER_POINT",
    "FLANK_BONUS",
    "FPS",
    "HASTE_COOLDOWN",
    "HEADQUARTERS_INCOME",
    "HEADQUARTERS_MAX_HEALTH",
    "HEAL_AMOUNT",
    "MAX_UNITS_PER_PLAYER",
    "MIN_MAP_SIZE",
    "MIN_STRIP_SIZE",
    "NEUTRAL_STRUCTURE_PALETTE",
    "PARALYZE_COOLDOWN",
    "PARALYZE_DURATION",
    "PLAYER_COLORS",
    "ROGUE_EVADE_CHANCE",
    "ROGUE_FOREST_EVADE_BONUS",
    "SORCERER_ATTACK_BUFF_AMOUNT",
    "SORCERER_BUFF_COOLDOWN",
    "SORCERER_BUFF_DURATION",
    "SORCERER_DEFENCE_BUFF_AMOUNT",
    "STARTING_GOLD",
    "STRUCTURE_REGEN_RATE",
    "STRUCTURE_TILE_TYPES",
    "TEAM_PALETTES",
    "TILE_COLORS",
    "TILE_IMAGES",
    "TILE_SIZE",
    "TILE_TYPES",
    "TOWER_INCOME",
    "TOWER_MAX_HEALTH",
    "TileType",
    "UNIT_COLORS",
    "UNIT_DATA",
    "UNIT_TYPE_TO_IDX",
]
