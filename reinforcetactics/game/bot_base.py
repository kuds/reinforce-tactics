"""
Shared bot foundations.

Provides:
  * ``BaseBot`` -- abstract base class all bot implementations conform to.
    A bot owns a reference to a ``GameState`` and a player number, and must
    implement ``take_turn()``. Concrete bots include the scripted hierarchy
    in :mod:`reinforcetactics.game.bot`, plus the model-driven and
    LLM-driven bots.

  * ``BotUnitMixin`` -- helper methods for bots that need to reason about
    enabled unit types, distances, heal-providing tiles, capture progress,
    and a small number of common per-unit ability flows (cleric heal/cure,
    mage paralyze). Designed to be mixed in alongside ``BaseBot``.

  * ``ABILITY_PROVIDERS`` -- single source of truth mapping a strategic
    ability name to the unit-type letter that provides it. Adding a new
    ability is a one-line table edit rather than a new predicate method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, cast

from reinforcetactics.constants import UNIT_DATA

# Strategic categories used by bot decision logic to bucket unit types by
# role. Kept as tuples so they're immutable shared constants.
MELEE_UNITS: Tuple[str, ...] = ("W", "K", "R", "B")
RANGED_UNITS: Tuple[str, ...] = ("A", "M", "S")
SUPPORT_UNITS: Tuple[str, ...] = ("C", "S")

# Maps a named strategic ability to the unit-type letter that provides it.
# Bots query this via ``has_units_with_ability(name)`` instead of hardcoding
# unit letters at the call site, so a unit-roster change touches one row
# here rather than every ``has_X_units`` predicate in the codebase.
ABILITY_PROVIDERS: Dict[str, str] = {
    "charge": "K",  # Knight charge bonus on long-distance approach
    "flank": "R",  # Rogue flank bonus when attacking from behind
    "buff": "S",  # Sorcerer haste / attack-buff / defence-buff
    "heal": "C",  # Cleric heal + cure
    "paralyze": "M",  # Mage paralyze
}


class BaseBot(ABC):
    """Abstract base for every bot that plays Reinforce Tactics.

    A bot is anything that, given a shared ``GameState`` and a player
    number, can execute a single turn's worth of actions via
    ``take_turn()``. The contract is:

      * ``take_turn()`` must terminate. It must call
        ``game_state.end_turn()`` (or return without acting once
        ``game_state.game_over`` is True).
      * ``self.game_state`` and ``self.bot_player`` must be set before
        ``take_turn()`` runs. ``BaseBot.__init__`` handles this; subclasses
        that override ``__init__`` should either call ``super().__init__``
        or set both attributes themselves.

    Used by tournament/runner, gym_env opponent loop, and the GUI's
    bot_factory -- all of which only depend on this minimal interface.
    """

    def __init__(self, game_state: Any, player: int = 2) -> None:
        self.game_state = game_state
        self.bot_player = player

    @abstractmethod
    def take_turn(self) -> None:
        """Execute one full turn for ``self.bot_player`` and end it."""


class BotUnitMixin:
    """Shared helpers for bots that reason about enabled unit types,
    distances, heal tiles, capture progress, and common ability flows.

    Designed to be mixed in alongside :class:`BaseBot`, which supplies
    ``self.game_state`` and ``self.bot_player``. The mixin does not
    subclass ``BaseBot`` so it can also be added to wrapper bots whose
    lifecycle is managed externally.
    """

    # Attribute promises -- supplied by BaseBot (or whichever class composes
    # this mixin).
    game_state: Any
    bot_player: int

    # Re-export the module-level categories as class attributes so existing
    # call sites (``self.MELEE_UNITS`` etc.) keep working.
    MELEE_UNITS = MELEE_UNITS
    RANGED_UNITS = RANGED_UNITS
    SUPPORT_UNITS = SUPPORT_UNITS

    # ------------------------------------------------------------------
    # Enabled-unit queries
    # ------------------------------------------------------------------
    def get_enabled_units(self) -> List[str]:
        """Get list of currently enabled unit types."""
        return self.game_state.enabled_units

    def is_unit_enabled(self, unit_type: str) -> bool:
        """Check if a specific unit type is enabled."""
        return self.game_state.is_unit_type_enabled(unit_type)

    def get_enabled_units_in(self, unit_types) -> List[str]:
        """Filter ``unit_types`` down to the ones currently enabled."""
        return [u for u in unit_types if self.is_unit_enabled(u)]

    def get_enabled_melee_units(self) -> List[str]:
        """Get enabled melee unit types (W, K, R, B)."""
        return self.get_enabled_units_in(MELEE_UNITS)

    def get_enabled_ranged_units(self) -> List[str]:
        """Get enabled ranged unit types (A, M, S)."""
        return self.get_enabled_units_in(RANGED_UNITS)

    def get_enabled_support_units(self) -> List[str]:
        """Get enabled support unit types (C, S)."""
        return self.get_enabled_units_in(SUPPORT_UNITS)

    # ------------------------------------------------------------------
    # Ability queries (data-driven via ABILITY_PROVIDERS)
    # ------------------------------------------------------------------
    def has_units_with_ability(self, ability: str) -> bool:
        """Return True iff the unit type providing ``ability`` is enabled.

        ``ability`` must be a key in :data:`ABILITY_PROVIDERS`. Unknown
        abilities return False rather than raising so call sites can opt
        into new abilities without breaking on older game states.
        """
        provider = ABILITY_PROVIDERS.get(ability)
        if provider is None:
            return False
        return self.is_unit_enabled(provider)

    def has_charge_units(self) -> bool:
        """Check if Knight (charge ability) is enabled."""
        return self.has_units_with_ability("charge")

    def has_flank_units(self) -> bool:
        """Check if Rogue (flank ability) is enabled."""
        return self.has_units_with_ability("flank")

    def has_buff_units(self) -> bool:
        """Check if Sorcerer (buff abilities) is enabled."""
        return self.has_units_with_ability("buff")

    def has_heal_units(self) -> bool:
        """Check if Cleric (heal ability) is enabled."""
        return self.has_units_with_ability("heal")

    def has_paralyze_units(self) -> bool:
        """Check if Mage (paralyze ability) is enabled."""
        return self.has_units_with_ability("paralyze")

    # ------------------------------------------------------------------
    # Geometry / movement helpers
    # ------------------------------------------------------------------
    def manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two points."""
        return abs(x1 - x2) + abs(y1 - y2)

    def get_reachable(self, unit):
        """Get all reachable positions for a unit on the current grid."""
        return unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units, moving_unit=unit, is_destination=False
            ),
        )

    # Heal amounts mirror GameState.heal_units_on_structures: tower=+1,
    # HQ/building=+2 at the start of the owner's next turn.
    _STRUCTURE_HEAL_AMOUNTS = {"t": 1, "h": 2, "b": 2}

    def heal_amount_at(self, x: int, y: int) -> int:
        """Return the per-turn HP a bot-owned unit would heal on tile (x, y)."""
        tile = self.game_state.grid.get_tile(x, y)
        if tile is None or tile.player != self.bot_player:
            return 0
        return self._STRUCTURE_HEAL_AMOUNTS.get(tile.type, 0)

    def is_on_heal_tile(self, unit) -> bool:
        """True if the unit currently stands on one of our heal-providing tiles."""
        return self.heal_amount_at(unit.x, unit.y) > 0

    def count_enemy_units_by_type(self) -> Dict[str, int]:
        """Tally living enemy units by type (e.g. ``{'W': 3, 'A': 2}``).

        Used by counter-composition logic; SimpleBot does not call this so
        purchasing remains static at that tier."""
        counts: Dict[str, int] = {}
        for u in self.game_state.units:
            if u.player == self.bot_player or u.player is None:
                continue
            if u.health <= 0:
                continue
            counts[u.type] = counts.get(u.type, 0) + 1
        return counts

    def is_actively_capturing(self, unit) -> bool:
        """True if ``unit`` stands on a capturable enemy/neutral tile that
        has already been damaged (i.e. it is mid-seize). Used to lock such
        units out of the multi-unit coordination passes that would
        otherwise pull them off the structure to attack a killable enemy
        and forfeit the capture progress."""
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if tile is None or not tile.is_capturable():
            return False
        if tile.player == self.bot_player:
            return False
        return tile.health < tile.max_health

    def continue_active_seizes(self, units) -> None:
        """Seize-in-place for any unit that is mid-capture, before the
        multi-unit coordination passes (coordinate_attacks etc) get a chance
        to drag them off. Mirrors the first check in act_with_unit /
        act_with_unit_enhanced; consolidating it here keeps the per-unit
        logic and the multi-unit logic agreeing on what counts as committed
        capture progress.
        """
        for unit in units:
            if not (unit.can_move or unit.can_attack):
                continue
            if self.is_actively_capturing(unit):
                self.game_state.seize(unit)

    def find_best_move_position(self, unit, target_x, target_y):
        """Find the best position to move towards a target."""
        reachable = self.get_reachable(unit)

        if not reachable:
            return None

        best_pos = None
        best_distance = float("inf")

        for pos in reachable:
            distance = self.manhattan_distance(pos[0], pos[1], target_x, target_y)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos

        return best_pos

    def _is_capturing_us(self, enemy) -> bool:
        """True if ``enemy`` stands on a capturable tile we want back."""
        tile = self.game_state.grid.get_tile(enemy.x, enemy.y)
        return tile.is_capturable() and tile.player != self.bot_player and tile.health < tile.max_health

    # ------------------------------------------------------------------
    # Per-unit ability flows (used by SimpleBot+ via composition)
    # ------------------------------------------------------------------
    def try_cleric_abilities(self, unit) -> bool:
        """Cure paralyzed allies, then heal damaged ones.

        Returns True if an ability was used (the caller is responsible for
        any haste re-entry). Heal priority: most-damaged frontline (W/B/K)
        first, falling back to the lowest-HP healable ally.
        """
        if unit.type != "C" or not unit.can_attack:
            return False

        curable = self.game_state.mechanics.get_curable_allies(unit, self.game_state.units)
        if curable:
            self.game_state.cure(unit, curable[0])
            return True

        healable = self.game_state.mechanics.get_healable_allies(unit, self.game_state.units)
        if not healable:
            return False

        frontline = [a for a in healable if a.type in ("W", "B", "K")]
        target = min(frontline or healable, key=lambda a: a.health)
        self.game_state.heal(unit, target)
        return True

    def try_mage_paralyze(self, unit) -> bool:
        """Paralyze a worthwhile in-range enemy.

        Returns True if paralyze was used. Priorities:
          1. Enemy currently capturing one of our structures (lock it).
          2. Highest-cost enemy that we can't one-shot from current position.
        """
        if unit.type != "M" or not unit.can_attack or not unit.can_use_paralyze():
            return False

        enemies = [e for e in self.game_state.units if e.player != self.bot_player and e.health > 0 and not e.is_paralyzed()]
        if not enemies:
            return False

        in_range = [e for e in enemies if 1 <= self.manhattan_distance(unit.x, unit.y, e.x, e.y) <= 2]
        if not in_range:
            return False

        # UNIT_DATA values are heterogeneous (the ``attack`` field is a
        # dict for ranged casters) so mypy widens the lookup to ``object``;
        # cast to int for the cost field, which is always int.
        def _unit_cost(e: Any) -> int:
            return cast(int, UNIT_DATA[e.type]["cost"])

        capturing = [e for e in in_range if self._is_capturing_us(e)]
        if capturing:
            target = max(capturing, key=_unit_cost)
            self.game_state.paralyze(unit, target)
            return True

        # Skip paralyze if a normal attack would already kill the best target.
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        on_mountain = tile.type == "m"

        def survives_attack(enemy):
            return unit.get_attack_damage(enemy.x, enemy.y, on_mountain) < enemy.health

        worth_paralyzing = [e for e in in_range if survives_attack(e)]
        if not worth_paralyzing:
            return False

        target = max(worth_paralyzing, key=_unit_cost)
        self.game_state.paralyze(unit, target)
        return True
