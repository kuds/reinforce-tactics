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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from reinforcetactics.core.actions import ACTOR_KEYS
from reinforcetactics.core.mechanics import same_side
from reinforcetactics.rules import ABILITY_RANGES, UNIT_DATA

# Strategic categories used by bot decision logic to bucket unit types by
# role. Kept as tuples so they're immutable shared constants.
MELEE_UNITS: tuple[str, ...] = ("W", "K", "R", "B")
RANGED_UNITS: tuple[str, ...] = ("A", "M", "S")
SUPPORT_UNITS: tuple[str, ...] = ("C", "S")

# Maps a named strategic ability to the unit-type letter that provides it.
# Bots query this via ``has_units_with_ability(name)`` instead of hardcoding
# unit letters at the call site, so a unit-roster change touches one row
# here rather than every ``has_X_units`` predicate in the codebase.
ABILITY_PROVIDERS: dict[str, str] = {
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
        ``game_state.game_over`` is True), unless it raises because the bot
        can't play at all: an LLM bot raises ``LLMBotError`` when its API
        is unreachable or misconfigured, leaving the turn un-ended for the
        caller. The tournament runner then records an errored game; the
        GUI hands the seat to SimpleBot.
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
        """Execute one full turn for ``self.bot_player`` and end it (see the class contract for when it may raise)."""

    # ------------------------------------------------------------------
    # Capability telemetry
    #
    # Per-game tally of how often each scripted capability fired (e.g.
    # ``knight_charge``, ``sorcerer_haste``, ``suicide_eval_rejected``,
    # ``buy_W``). Lazily created so subclasses that bypass
    # ``BaseBot.__init__`` (most of the scripted hierarchy sets fields
    # directly) still work. Tournament runner snapshots this into the
    # replay's ``game_info`` so the balance notebook can correlate
    # decision frequency with outcome -- the missing link in
    # ``endstate_per_game.csv``, which records *what happened in the
    # game state* but not *which heuristic the bot fired*.
    # ------------------------------------------------------------------
    def _record(self, name: str, n: int = 1) -> None:
        """Increment ``capabilities_fired[name]`` by ``n``.

        Lazy-creates the counter so subclasses that don't call
        ``super().__init__`` still get telemetry. Safe to call from
        anywhere on the bot; no-op if ``n == 0``.
        """
        if n == 0:
            return
        counters: dict[str, int] | None = getattr(self, "capabilities_fired", None)
        if counters is None:
            counters = {}
            self.capabilities_fired = counters
        counters[name] = counters.get(name, 0) + n

    def get_capabilities_fired(self) -> dict[str, int]:
        """Return per-game capability counters (empty if nothing recorded)."""
        return dict(getattr(self, "capabilities_fired", {}) or {})


class BotUnitMixin:
    """Shared helpers for bots that reason about enabled unit types,
    distances, heal tiles, capture progress, and common ability flows.

    Designed to be mixed in alongside :class:`BaseBot`, which supplies
    ``self.game_state`` and ``self.bot_player``. The mixin does not
    subclass ``BaseBot`` so it can also be added to wrapper bots whose
    lifecycle is managed externally.
    """

    # Attribute promises -- supplied by BaseBot (or whichever class composes
    # this mixin). ``_record`` is provided by BaseBot via MRO; the mixin
    # methods below call ``self._record(...)`` and rely on every concrete
    # bot subclassing BaseBot as well. The TYPE_CHECKING-only declaration
    # below tells mypy that ``_record`` is callable on a BotUnitMixin
    # instance without shadowing BaseBot's real implementation at runtime
    # (the mixin appears first in the MRO).
    game_state: Any
    bot_player: int
    if TYPE_CHECKING:

        def _record(self, name: str, n: int = 1) -> None: ...

    # Optional rng for stochastic tiebreaking. ``None`` (default) means
    # fully deterministic: every game from the same starting state plays
    # out identically. When set (a ``random.Random``), ``_maybe_shuffle``
    # randomises iteration order before sort / best-tracking loops, so
    # actions tied on the bot's scoring heuristic resolve to different
    # picks across episodes. Scoring logic is unchanged -- the bot still
    # only ever picks among its top-rated options -- so the bot's
    # strategic quality is preserved while episode-level diversity is
    # restored.
    _rng: Any = None

    def _maybe_shuffle(self, items: list[Any]) -> list[Any]:
        """Shuffle ``items`` in place when ``self._rng`` is set.

        Returns ``items`` (the same list) for chained use. With
        ``self._rng = None`` this is a no-op and the bot retains the
        deterministic, insertion-order tiebreak behaviour. Wrap the
        input of any sort / best-tracking site to randomise ties
        without touching the scoring logic.
        """
        if getattr(self, "_rng", None) is not None:
            self._rng.shuffle(items)
        return items

    # Re-export the module-level categories as class attributes so existing
    # call sites (``self.MELEE_UNITS`` etc.) keep working.
    MELEE_UNITS = MELEE_UNITS
    RANGED_UNITS = RANGED_UNITS
    SUPPORT_UNITS = SUPPORT_UNITS

    # ------------------------------------------------------------------
    # Sides (review core-4): a teammate is an ally, never a target
    # ------------------------------------------------------------------
    def _teams(self) -> Any:
        """The game's player -> team map (None for a stand-in state without one)."""
        return getattr(self.game_state, "teams", None)

    def _is_friendly(self, player: int | None) -> bool:
        """``player`` is this bot or a teammate. A neutral owner (None) is not."""
        return same_side(self.bot_player, player, self._teams())

    def _is_enemy(self, player: int | None) -> bool:
        """``player`` is on another team. A neutral owner (None) is not an enemy either."""
        return player is not None and not self._is_friendly(player)

    # ------------------------------------------------------------------
    # Enabled-unit queries
    # ------------------------------------------------------------------
    def get_enabled_units(self) -> list[str]:
        """Get list of currently enabled unit types."""
        return self.game_state.enabled_units

    def is_unit_enabled(self, unit_type: str) -> bool:
        """Check if a specific unit type is enabled."""
        return self.game_state.is_unit_type_enabled(unit_type)

    def get_enabled_units_in(self, unit_types) -> list[str]:
        """Filter ``unit_types`` down to the ones currently enabled."""
        return [u for u in unit_types if self.is_unit_enabled(u)]

    def get_enabled_melee_units(self) -> list[str]:
        """Get enabled melee unit types (W, K, R, B)."""
        return self.get_enabled_units_in(MELEE_UNITS)

    def get_enabled_ranged_units(self) -> list[str]:
        """Get enabled ranged unit types (A, M, S)."""
        return self.get_enabled_units_in(RANGED_UNITS)

    def get_enabled_support_units(self) -> list[str]:
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
        """Tiles ``unit`` may legally end a move on now (``GameState.get_move_destinations``).

        Empty when the unit is out of play (a counter-attack killed it earlier
        in the turn), paralyzed, or has spent its move, so every tile listed
        is a move the engine accepts. This used to be every tile a path can
        cross, friends' tiles included, whether or not the unit could still
        move: over half of SimpleBot's moves were refused and its
        armies jammed behind their own units (review rulebots-1). Every bot
        caller wants destinations; path semantics (tiles a unit can pass
        through) are ``GameState.get_reachable_positions``, which MasterBot's
        threat map asks for its enemies directly.
        """
        if not unit.can_move or unit.is_paralyzed() or unit not in self.game_state.units:
            return []
        return self.game_state.get_move_destinations(unit)

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

    def count_enemy_units_by_type(self) -> dict[str, int]:
        """Tally living enemy units by type (e.g. ``{'W': 3, 'A': 2}``).

        Used by counter-composition logic; SimpleBot does not call this so
        purchasing remains static at that tier."""
        counts: dict[str, int] = {}
        for u in self.game_state.units:
            if not self._is_enemy(u.player):
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
        # A teammate's structure is not ours to seize (the engine refuses it).
        if self._is_friendly(tile.player):
            return False
        return tile.health < tile.max_health

    def _capture_assignments(self) -> set[tuple[int, int]]:
        """Per-turn set of structure positions already claimed by another
        unit's capture priority. Reset by the tiers' take_turn; lazily
        created for callers that act without one."""
        claimed: set[tuple[int, int]] | None = getattr(self, "_capture_assigned", None)
        if claimed is None:
            claimed = set()
            self._capture_assigned = claimed
        return claimed

    def continue_active_seizes(self, units) -> None:
        """Seize-in-place for any unit that is mid-capture, before the
        multi-unit coordination passes (coordinate_attacks etc) get a chance
        to drag them off. Mirrors the first check in act_with_unit /
        act_with_unit_enhanced; consolidating it here keeps the per-unit
        logic and the multi-unit logic agreeing on what counts as committed
        capture progress. The seized tile is claimed, so no sibling picks
        it as its own capture target this turn.
        """
        for unit in units:
            if self.game_state.game_over:
                return
            if self.is_actively_capturing(unit) and self.try_seize(unit):
                self._capture_assignments().add((unit.x, unit.y))

    # ------------------------------------------------------------------
    # Acting through the engine (review rulebots-1/6/8/19)
    #
    # The engine refuses an illegal action and returns a falsy result
    # (``move_unit`` False, ``attack`` damage 0, ...). The bots used to plan
    # with their own reach and range checks and ignore those results, so
    # most of their moves and up to half their attacks were refused: a unit
    # stayed put while the bot carried on as if it had moved, claimed
    # captures it never reached, recursed on units with nothing left to do
    # and counted abilities that never happened. These helpers ask the
    # engine's own rules first, so a bot only sends actions the engine
    # accepts, and say whether the action happened.
    # ------------------------------------------------------------------
    def live_enemies(self) -> list[Any]:
        """Living units of the other teams, in ``game_state.units`` order."""
        return [u for u in self.game_state.units if self._is_enemy(u.player) and u.health > 0]

    def in_attack_reach(self, unit, enemy) -> bool:
        """Whether ``unit`` could attack ``enemy`` from where it stands.

        The engine's attack rule without its turn and action-slot gates, so
        planners can also ask it for a tile ``unit`` could move to (by
        setting ``unit.x``/``unit.y`` there for the question): a living enemy
        in reach and, under fog of war, one the unit may attack
        (``GameState.is_enemy_attackable_by_unit``). A unit that has not
        moved yet takes its attack snapshot from what its side sees when it
        moves, so an enemy hidden now stays out of reach after the move:
        moving to discover an enemy does not let the unit hit it.
        """
        gs = self.game_state
        return (
            self._is_enemy(enemy.player)
            and enemy.health > 0
            and gs.mechanics.can_reach(unit, enemy.x, enemy.y, gs.grid)
            and gs.is_enemy_attackable_by_unit(unit, enemy)
        )

    def attackable_enemies(self, unit, enemies=None) -> list[Any]:
        """The enemies (default: every living one) ``unit`` could attack from where it stands."""
        if enemies is None:
            enemies = self.live_enemies()
        return [e for e in enemies if self.in_attack_reach(unit, e)]

    def try_move(self, unit, x: int, y: int) -> bool:
        """Move ``unit`` to ``(x, y)`` if it is one of its legal destinations; True if it moved.

        Under fog of war a hidden unit on the path can cut the move short
        (the engine's ambush rule); that still counts as a move, so callers
        read ``unit.x``/``unit.y`` for where it ended up.
        """
        if self.game_state.game_over or (x, y) not in self.get_reachable(unit):
            return False
        return bool(self.game_state.move_unit(unit, x, y))

    def try_action(self, kind: str, actor, target) -> bool:
        """Carry out the targeted action ``kind`` (``attack`` or an ability) if the engine allows it.

        ``GameState.is_legal`` answers with the rule the action method
        validates by (range, side, cooldown, fog of war, the actor's turn
        and unspent action), so nothing is sent that the engine would
        refuse. True if the action happened.
        """
        action = {ACTOR_KEYS[kind]: actor, "target": target}
        if not self.game_state.is_legal(kind, action):
            return False
        return self.game_state.apply_action(kind, action).accepted

    def try_seize(self, unit) -> bool:
        """Seize the structure ``unit`` stands on if the engine allows it; True if it did."""
        action = {"unit": unit}
        if not self.game_state.is_legal("seize", action):
            return False
        return self.game_state.apply_action("seize", action).accepted

    def finish_unit_action(self, unit, act: Callable[..., Any], depth: int) -> None:
        """End ``unit``'s action; if the engine grants it another, act again with ``act(unit, depth + 1)``.

        The one continuation for the per-unit act methods, called once the
        unit has acted or has nothing more to do this action (review
        rulebots-6/8). ``GameState.end_unit_turn`` knows whether a next
        action exists: right after an action that haste refreshed
        (``haste_refreshed``) it keeps the extra action, and a hasted unit
        that ends its action without acting spends its haste on a fresh one
        (the GUI's Wait); any other unit is done for the turn. Re-entering on
        ``can_move or can_attack`` instead fired after every move
        (``can_attack`` stays set until the unit acts), so the bot re-ran its
        whole decision for a unit that could only be refused, and it fired
        for an attacker a counter-attack had just killed.
        """
        gs = self.game_state
        if gs.game_over or unit not in gs.units:
            return
        if gs.end_unit_turn(unit):
            act(unit, depth + 1)

    def find_best_move_position(self, unit, target_x, target_y):
        """Find the best position to move towards a target."""
        reachable = self.get_reachable(unit)

        if not reachable:
            return None

        # Shuffle so equidistant reachable tiles tiebreak randomly under
        # stochastic mode -- the strict ``<`` below otherwise hard-prefers
        # the first-visited candidate, which is the most-hit decision site
        # in the bot (every move-toward-target call). Without this, two
        # equally-good landing tiles produce identical games every run.
        reachable_list = list(reachable)
        self._maybe_shuffle(reachable_list)

        best_pos = None
        best_distance = float("inf")

        for pos in reachable_list:
            distance = self.manhattan_distance(pos[0], pos[1], target_x, target_y)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos

        return best_pos

    def _is_capturing_us(self, enemy) -> bool:
        """True if ``enemy`` stands on a capturable tile we want back."""
        tile = self.game_state.grid.get_tile(enemy.x, enemy.y)
        return tile.is_capturable() and not self._is_friendly(tile.player) and tile.health < tile.max_health

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

        curable = self.game_state.mechanics.get_curable_allies(unit, self.game_state.units, self._teams())
        if curable and self.try_action("cure", unit, curable[0]):
            self._record("cleric_cure")
            return True

        healable = self.game_state.mechanics.get_healable_allies(unit, self.game_state.units, self._teams())
        if not healable:
            return False

        frontline = [a for a in healable if a.type in ("W", "B", "K")]
        # Shuffle so equal-HP allies tiebreak randomly. ``min()`` returns
        # the first item on ties.
        pool = list(frontline or healable)
        self._maybe_shuffle(pool)
        target = min(pool, key=lambda a: a.health)
        if not self.try_action("heal", unit, target):
            return False
        self._record("cleric_heal")
        return True

    def try_mage_paralyze(self, unit) -> bool:
        """Paralyze a worthwhile in-range enemy.

        Returns True if paralyze was used. Priorities:
          1. Enemy currently capturing one of our structures (lock it).
          2. Highest-cost enemy that we can't one-shot from current position.
        """
        if unit.type != "M" or not unit.can_attack or not unit.can_use_paralyze():
            return False

        enemies = [e for e in self.live_enemies() if not e.is_paralyzed()]
        if not enemies:
            return False

        # Only enemies the engine lets this Mage paralyze: under fog of war
        # that excludes an enemy its side could not see when its action
        # began, which the range check alone let through.
        in_range = [
            e
            for e in self.game_state.mechanics.units_in_range(unit, enemies, *ABILITY_RANGES["paralyze"])
            if self.game_state.is_legal("paralyze", {"paralyzer": unit, "target": e})
        ]
        if not in_range:
            return False

        # UNIT_DATA values are heterogeneous (the ``attack`` field is a
        # dict for ranged casters) so mypy widens the lookup to ``object``;
        # cast to int for the cost field, which is always int.
        def _unit_cost(e: Any) -> int:
            return cast(int, UNIT_DATA[e.type]["cost"])

        capturing = [e for e in in_range if self._is_capturing_us(e)]
        if capturing:
            # Equal-cost enemies tiebreak randomly under stochastic mode.
            self._maybe_shuffle(capturing)
            target = max(capturing, key=_unit_cost)
            if not self.try_action("paralyze", unit, target):
                return False
            self._record("mage_paralyze")
            return True

        # Skip paralyze if a normal attack would already kill the best target.
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        on_mountain = tile.type == "m"

        def survives_attack(enemy):
            return unit.get_attack_damage(enemy.x, enemy.y, on_mountain) < enemy.health

        worth_paralyzing = [e for e in in_range if survives_attack(e)]
        if not worth_paralyzing:
            return False

        self._maybe_shuffle(worth_paralyzing)
        target = max(worth_paralyzing, key=_unit_cost)
        if not self.try_action("paralyze", unit, target):
            return False
        self._record("mage_paralyze")
        return True
