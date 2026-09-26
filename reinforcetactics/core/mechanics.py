"""
Core game mechanics including combat, movement, income, and structure capture.
"""

import random
from collections.abc import Mapping

from reinforcetactics.constants import (
    ABILITY_RANGES,
    BUILDING_INCOME,
    CHARGE_BONUS,
    CHARGE_MIN_DISTANCE,
    COUNTER_ATTACK_MULTIPLIER,
    DEFENCE_REDUCTION_PER_POINT,
    FLANK_BONUS,
    HASTE_COOLDOWN,
    HEADQUARTERS_INCOME,
    HEAL_AMOUNT,
    PARALYZE_COOLDOWN,
    PARALYZE_DURATION,
    ROGUE_EVADE_CHANCE,
    ROGUE_FOREST_EVADE_BONUS,
    SORCERER_ATTACK_BUFF_AMOUNT,
    SORCERER_BUFF_COOLDOWN,
    SORCERER_BUFF_DURATION,
    SORCERER_DEFENCE_BUFF_AMOUNT,
    STRUCTURE_REGEN_RATE,
    TOWER_INCOME,
)


def same_side(player_a, player_b, teams: Mapping[int, int] | None = None) -> bool:
    """Whether two owners are on the same side: the same player, or teammates.

    The one definition of "friendly" every hostility rule uses (``GameState``
    wraps it as ``are_allies``). ``teams`` maps player -> team id
    (``GameState.teams``); ``None`` means free-for-all, where only a player
    is its own ally -- the behaviour every helper below had before teams
    existed, so callers that do not pass ``teams`` are unchanged. ``None``
    as an owner (a neutral structure) is nobody's ally.
    """
    if player_a is None or player_b is None:
        return False
    if player_a == player_b:
        return True
    return teams is not None and player_a in teams and player_b in teams and teams[player_a] == teams[player_b]


class GameMechanics:
    """Handles core game mechanics and rules.

    Every helper that compares owners takes an optional ``teams`` mapping
    (see :func:`same_side`): ``GameState`` passes its own, so teammates count
    as allies (pass through, heal, buff, flank; never attacked, paralyzed or
    seized). Without it, only a unit's own player is friendly.
    """

    @staticmethod
    def can_move_to_position(x, y, grid, units, moving_unit=None, is_destination=False, teams=None):
        """
        Check if a position is valid for unit movement.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            grid: TileGrid instance
            units: List of Unit instances
            moving_unit: The unit that is moving (optional, for team checking)
            is_destination: If True, blocks all units. If False (pathfinding),
                           only blocks enemy units (default: False)

        Returns:
            True if position is valid for movement
        """
        if not (0 <= x < grid.width and 0 <= y < grid.height):
            return False

        tile = grid.get_tile(x, y)
        if not tile.is_walkable():
            return False

        # Check if another unit is already there. This scans every unit, so
        # searches build ``movement_blockers`` once instead of calling this
        # per tile (review core-20); it stays for callers checking one tile.
        for unit in units:
            if unit.x == x and unit.y == y:
                # A destination must be empty; a path may pass through friends.
                if is_destination or GameMechanics.blocks_movement(unit, moving_unit, teams):
                    return False

        return True

    @staticmethod
    def blocks_movement(unit, moving_unit, teams=None):
        """Whether ``unit`` stops ``moving_unit`` from passing through its tile.

        The one definition of who blocks whom: units of another side do,
        friendly units -- teammates' included (see ``same_side``) -- do not
        (though no unit may end its move on an occupied tile). With no
        ``moving_unit`` every unit blocks (the legacy rule).
        """
        return moving_unit is None or not same_side(unit.player, moving_unit.player, teams)

    @staticmethod
    def movement_blockers(units, moving_unit=None, teams=None):
        """The tiles ``moving_unit`` may not pass through, as a set of ``(x, y)``.

        Built once per search, so the search's per-tile test is a set lookup
        instead of a scan of every unit (review core-20). A search for a
        player's restricted view of the board (e.g. hidden enemies under fog
        of war) filters this set before searching.
        """
        return {(u.x, u.y) for u in units if GameMechanics.blocks_movement(u, moving_unit, teams)}

    @staticmethod
    def passability(grid, blocked):
        """``(x, y) -> bool`` for ``Unit.find_paths``: walkable and not in ``blocked``.

        Bounds are checked by the search itself.
        """
        tiles = grid.tiles
        return lambda x, y: (x, y) not in blocked and tiles[y][x].is_walkable()

    # ------------------------------------------------------------------
    # Range scans and per-target rules
    # ------------------------------------------------------------------
    # Every "who is within reach" question is ``in_range`` / ``units_in_range``
    # with a range from one place: ``Unit.get_attack_range`` for attacks,
    # ``constants.ABILITY_RANGES`` for the targeted abilities. Each ``is_*``
    # predicate is the single definition of "may this caster target this
    # unit"; the matching ``get_*`` list helper filters with it, and
    # ``GameState`` both enumerates legal actions through the list helpers and
    # validates a requested action with the predicate, so the mask and the
    # engine cannot disagree.

    @staticmethod
    def in_range(unit, target, lo, hi):
        """Whether ``target`` is ``lo..hi`` tiles (Manhattan distance) from ``unit``."""
        return lo <= abs(unit.x - target.x) + abs(unit.y - target.y) <= hi

    @staticmethod
    def units_in_range(center, units, lo, hi, predicate=None):
        """The units ``lo..hi`` tiles (Manhattan distance) from ``center`` that pass ``predicate``.

        In ``units`` order, which the orderings built from it (legal
        actions, bot tiebreaks) rely on. ``center`` itself is included when
        ``lo`` is 0 and it is in ``units``.
        """
        return [u for u in units if GameMechanics.in_range(center, u, lo, hi) and (predicate is None or predicate(u))]

    @staticmethod
    def in_ability_range(ability, caster, target):
        """Whether ``target`` is within ``caster``'s reach for ``ability`` (see ``ABILITY_RANGES``)."""
        lo, hi = ABILITY_RANGES[ability]
        return GameMechanics.in_range(caster, target, lo, hi)

    @staticmethod
    def _on_mountain(unit, grid):
        """Whether ``unit`` stands on a mountain (the Archer's +1 range); False without a grid."""
        if not grid:
            return False
        tile = grid.get_tile(unit.x, unit.y)
        return tile is not None and tile.type == "m"

    @staticmethod
    def get_attackable_enemies(unit, units, grid, teams=None):
        """
        Get list of enemy units within the given unit's attack range.

        Args:
            unit: The unit to check attack range for
            units: List of all units
            grid: TileGrid instance (for checking mountain tiles)
            teams: Optional player -> team map (see :func:`same_side`)

        Returns:
            List of enemy units within attack range
        """
        lo, hi = unit.get_attack_range(GameMechanics._on_mountain(unit, grid))
        return GameMechanics.units_in_range(
            unit, units, lo, hi, lambda enemy: not same_side(enemy.player, unit.player, teams) and enemy.health > 0
        )

    @staticmethod
    def is_enemy_flanked(attacker, target, units, teams=None):
        """
        Check if the target enemy is flanked (adjacent to at least one of attacker's allies).

        A teammate's unit counts as an ally here, like it does for healing.

        Args:
            attacker: The attacking unit
            target: The target enemy unit
            units: List of all units
            teams: Optional player -> team map (see :func:`same_side`)

        Returns:
            True if target is adjacent to at least one of attacker's allies (excluding attacker)
        """
        return bool(
            GameMechanics.units_in_range(
                target,
                units,
                1,
                1,
                lambda unit: unit is not attacker and same_side(unit.player, attacker.player, teams) and unit.health > 0,
            )
        )

    @staticmethod
    def _is_ally_in_ability_range(ability, caster, unit, teams=None):
        """A living unit on ``caster``'s side (see :func:`same_side`) within ``ability``'s range."""
        return (
            same_side(unit.player, caster.player, teams)
            and unit.health > 0
            and GameMechanics.in_ability_range(ability, caster, unit)
        )

    @staticmethod
    def is_healable_ally(cleric, ally, teams=None):
        """A damaged living ally (a teammate's unit too; not the Cleric itself) within the heal range."""
        return GameMechanics._is_ally_in_ability_range("heal", cleric, ally, teams) and ally.health < ally.max_health

    @staticmethod
    def get_healable_allies(cleric, units, teams=None):
        """
        Get damaged friendly units within the Cleric's heal range (1..CLERIC_HEAL_RANGE).
        """
        return [ally for ally in units if GameMechanics.is_healable_ally(cleric, ally, teams)]

    @staticmethod
    def is_curable_ally(cleric, ally, teams=None):
        """A paralyzed living ally (a teammate's unit too; not the Cleric itself) within the cure range."""
        return GameMechanics._is_ally_in_ability_range("cure", cleric, ally, teams) and ally.is_paralyzed()

    @staticmethod
    def get_curable_allies(cleric, units, teams=None):
        """
        Get paralyzed friendly units within the Cleric's cure range (1..CLERIC_HEAL_RANGE).
        """
        return [ally for ally in units if GameMechanics.is_curable_ally(cleric, ally, teams)]

    @staticmethod
    def is_hasteable_ally(sorcerer, unit):
        """A living, unparalyzed unit of the Sorcerer's own player within the haste range, not already hasted.

        Own units only, not a teammate's (no ``teams``, so only the
        Sorcerer's player is on its side): haste is an extra action *this*
        turn, and a teammate's unit cannot act on the Sorcerer's turn (its
        haste would be cleared when its owner's turn starts). A paralyzed
        unit cannot act at all, so hasting it would only waste the cast.
        """
        return (
            GameMechanics._is_ally_in_ability_range("haste", sorcerer, unit) and not unit.is_paralyzed() and not unit.is_hasted
        )

    @staticmethod
    def is_defence_buffable_ally(sorcerer, unit, teams=None):
        """A living ally (a teammate's unit too; the Sorcerer itself too) within the buff range without a defence buff."""
        return GameMechanics._is_ally_in_ability_range("defence_buff", sorcerer, unit, teams) and not unit.has_defence_buff()

    @staticmethod
    def is_attack_buffable_ally(sorcerer, unit, teams=None):
        """A living ally (a teammate's unit too; the Sorcerer itself too) within the buff range without an attack buff."""
        return GameMechanics._is_ally_in_ability_range("attack_buff", sorcerer, unit, teams) and not unit.has_attack_buff()

    @staticmethod
    def get_hasteable_allies(sorcerer, units):
        """
        Get list of friendly units that can receive Haste from the Sorcerer.

        Args:
            sorcerer: The Sorcerer unit
            units: List of all units

        Returns:
            List of allied units (excluding sorcerer) within the haste range that haven't been hasted
        """
        return [unit for unit in units if GameMechanics.is_hasteable_ally(sorcerer, unit)]

    @staticmethod
    def get_defence_buffable_allies(sorcerer, units, teams=None):
        """
        Get list of friendly units that can receive Defence Buff from the Sorcerer.

        Args:
            sorcerer: The Sorcerer unit
            units: List of all units

        Returns:
            List of allied units within the buff range that don't have defence buff
        """
        return [unit for unit in units if GameMechanics.is_defence_buffable_ally(sorcerer, unit, teams)]

    @staticmethod
    def get_attack_buffable_allies(sorcerer, units, teams=None):
        """
        Get list of friendly units that can receive Attack Buff from the Sorcerer.

        Args:
            sorcerer: The Sorcerer unit
            units: List of all units

        Returns:
            List of allied units within the buff range that don't have attack buff
        """
        return [unit for unit in units if GameMechanics.is_attack_buffable_ally(sorcerer, unit, teams)]

    @staticmethod
    def apply_defence_reduction(base_damage, target_defence):
        """
        Apply defence reduction to damage using percentage reduction.

        Each point of defence reduces damage by 5%.

        Args:
            base_damage: The raw damage before defence
            target_defence: The target's defence stat

        Returns:
            Reduced damage as integer: 0 when ``base_damage <= 0`` (no hit,
            e.g. the attacker is out of range), otherwise at least 1.
        """
        # The min-1 floor below guarantees a *hit* always does something;
        # applied to a zero base it turned "cannot reach" into 1 phantom
        # damage (review core-3 / prior-11).
        if base_damage <= 0:
            return 0
        reduction = target_defence * DEFENCE_REDUCTION_PER_POINT
        # Cap reduction at 90% to ensure some damage always gets through
        reduction = min(reduction, 0.9)
        reduced_damage = base_damage * (1 - reduction)
        return max(1, int(reduced_damage))

    @staticmethod
    def _hp_damage_scale(unit, damage_model):
        """Multiplier applied to a unit's outgoing damage under ``damage_model``.

        ``"flat"`` (default/legacy) returns 1.0 -- damage is HP-independent.
        ``"hp_scaled"`` returns the unit's current HP fraction so a wounded
        unit deals proportionally less (the engine-side analog of seize,
        which already scales with ``unit.health``). Guards against a zero /
        missing ``max_health`` by falling back to 1.0.
        """
        if damage_model == "hp_scaled" and getattr(unit, "max_health", 0):
            return unit.health / unit.max_health
        return 1.0

    @staticmethod
    def can_reach(unit, target_x, target_y, grid=None):
        """Whether ``unit`` can hit ``(target_x, target_y)`` from where it stands.

        The one definition of attack reach: ``GameState`` uses it to
        enumerate and validate attacks, and ``attack_unit`` uses it to
        decide whether the defender can counter. Reach is "the unit's
        range damage there is positive" (``Unit.get_attack_damage``, which
        is 0 outside ``Unit.get_attack_range``), which includes the
        Archer's +1 range on a mountain.
        """
        return unit.get_attack_damage(target_x, target_y, GameMechanics._on_mountain(unit, grid)) > 0

    @staticmethod
    def _calculate_counter_damage(unit, target_x, target_y, grid, damage_model="flat"):
        """
        Calculate counter-attack damage for a unit.

        Args:
            unit: The unit that would counter-attack
            target_x: X coordinate of the target
            target_y: Y coordinate of the target
            grid: TileGrid instance (optional, for checking mountain tiles)
            damage_model: "flat" or "hp_scaled" (see ``attack_unit``). Under
                "hp_scaled" the counter is scaled by the counter-attacker's
                HP fraction -- and because counters are computed *after* the
                unit has taken the incoming hit, a freshly-wounded defender
                counters weaker, which is what makes focus-fire pay off.

        Returns:
            Counter-attack damage as integer
        """
        base = unit.get_attack_damage(target_x, target_y, GameMechanics._on_mountain(unit, grid)) * COUNTER_ATTACK_MULTIPLIER
        base = base * GameMechanics._hp_damage_scale(unit, damage_model)
        return int(base)

    @staticmethod
    def attack_unit(attacker, target, grid=None, units=None, damage_model="flat", rng=None, teams=None):
        """
        Execute an attack from attacker to target.

        Args:
            attacker: The attacking unit
            target: The target unit
            grid: TileGrid instance (optional, for checking mountain tiles)
            units: List of all units (optional, for flanking checks)
            damage_model: "flat" (HP-independent, legacy) or "hp_scaled"
                (outgoing damage scaled by the attacker's current HP
                fraction). Applies symmetrically to the counter-attack.
            rng: Random source exposing ``random()`` (e.g. a
                ``random.Random`` instance) used for the Rogue evade roll —
                the only stochastic outcome in combat. ``GameState.attack``
                always forwards the game's own seeded ``GameState.rng``.
                ``None`` (direct callers only) rolls with a fresh, unseeded
                ``random.Random``; the module-global ``random`` is never
                read, so no caller's seeding can leak into combat or be
                disturbed by it.
            teams: Optional player -> team map, so a teammate standing next
                to the target counts toward the Rogue's flank.

        Returns:
            dict with 'attacker_alive', 'target_alive', 'damage', 'counter_damage',
            and bonus info ('charge_bonus', 'flank_bonus')
        """
        # Check if attacker is on mountain for range calculation
        attacker_on_mountain = GameMechanics._on_mountain(attacker, grid)

        # An attacker that cannot reach the target does not hit it, and so
        # provokes no counter either. ``GameState.attack`` rejects such an
        # attack before it gets here; this keeps a direct caller of the
        # mechanics layer from dealing the old phantom 1 damage each way.
        range_damage = attacker.get_attack_damage(target.x, target.y, attacker_on_mountain)
        if range_damage <= 0:
            return {
                "attacker_alive": True,
                "target_alive": True,
                "damage": 0,
                "counter_damage": 0,
                "charge_bonus": False,
                "flank_bonus": False,
                "evade": False,
                "attack_buff": False,
                "defence_buff": False,
            }

        # Calculate base attack damage. Under "hp_scaled" the base is reduced
        # by the attacker's current HP fraction *before* ability bonuses and
        # defence reduction, so a wounded unit hits proportionally weaker
        # (makes focus-fire decisive and discourages the even-attrition
        # stalemate flat damage produces). "flat" scales by 1.0 (legacy).
        base_attack_damage = range_damage * GameMechanics._hp_damage_scale(attacker, damage_model)

        # Apply special ability bonuses
        charge_applied = False
        flank_applied = False
        evade_applied = False
        attack_buff_applied = False
        defence_buff_applied = False

        # Knight's Charge: +50% damage if moved 3+ tiles
        if attacker.type == "K" and attacker.distance_moved >= CHARGE_MIN_DISTANCE:
            base_attack_damage = int(base_attack_damage * (1 + CHARGE_BONUS))
            charge_applied = True

        # Rogue's Flank: +50% damage if enemy is adjacent to another friendly unit
        if attacker.type == "R" and units:
            if GameMechanics.is_enemy_flanked(attacker, target, units, teams):
                base_attack_damage = int(base_attack_damage * (1 + FLANK_BONUS))
                flank_applied = True

        # Sorcerer's Attack Buff: +50% damage if attacker has attack buff
        if attacker.has_attack_buff():
            base_attack_damage = int(base_attack_damage * (1 + SORCERER_ATTACK_BUFF_AMOUNT))
            attack_buff_applied = True

        # Apply defence reduction to attack damage. The attacker reaches, so
        # this is a hit and lands for at least 1: under "hp_scaled" a bonus's
        # int() can truncate a badly wounded unit's base to 0, which
        # apply_defence_reduction (rightly, for a miss) would turn into 0.
        # Callers rely on it: an executed attack always has damage >= 1.
        attack_damage = max(1, GameMechanics.apply_defence_reduction(base_attack_damage, target.defence))

        # Sorcerer's Defence Buff: -50% damage taken if target has defence buff
        if target.has_defence_buff():
            attack_damage = max(1, int(attack_damage * (1 - SORCERER_DEFENCE_BUFF_AMOUNT)))
            defence_buff_applied = True

        target_alive = target.take_damage(attack_damage)

        attacker_alive = True
        counter_damage = 0

        # Counter-attack logic with Archer restrictions
        if target_alive and not target.is_paralyzed():
            # Determine if counter-attack is allowed
            can_counter = True

            # If attacker is an Archer, only Archers, Mages, and Sorcerers can counter
            if attacker.type == "A":
                if target.type not in ["A", "M", "S"]:
                    can_counter = False

            # A defender that cannot reach the attacker cannot counter (a
            # Warrior hit by a Mage at range 2, an Archer hit point-blank).
            # Checked before the Rogue evade roll: with no counter coming
            # there is nothing to evade, so no roll is spent or reported.
            if can_counter and not GameMechanics.can_reach(target, attacker.x, attacker.y, grid):
                can_counter = False

            # Rogue's Evade: ROGUE_EVADE_CHANCE to dodge counter-attacks,
            # plus ROGUE_FOREST_EVADE_BONUS when the Rogue stands in forest.
            if can_counter and attacker.type == "R":
                evade_chance = ROGUE_EVADE_CHANCE
                # Check if Rogue is in forest for bonus evade chance
                if grid:
                    rogue_tile = grid.get_tile(attacker.x, attacker.y)
                    if rogue_tile.type == "f":  # Forest tile
                        evade_chance += ROGUE_FOREST_EVADE_BONUS
                evade_rng = rng if rng is not None else random.Random()
                if evade_rng.random() < evade_chance:
                    can_counter = False
                    evade_applied = True

            if can_counter:
                # Calculate base counter damage
                base_counter_damage = GameMechanics._calculate_counter_damage(
                    target, attacker.x, attacker.y, grid, damage_model=damage_model
                )

                # Apply attack buff to counter-attacker (target)
                if target.has_attack_buff():
                    base_counter_damage = int(base_counter_damage * (1 + SORCERER_ATTACK_BUFF_AMOUNT))

                # Apply defence reduction to counter damage. The defender
                # reaches, so this is a hit: floor at 1 as for the primary
                # (the int() in _calculate_counter_damage can truncate a
                # wounded or weak defender's base to 0).
                counter_damage = max(1, GameMechanics.apply_defence_reduction(base_counter_damage, attacker.defence))

                # Apply defence buff to attacker receiving counter damage
                if attacker.has_defence_buff():
                    counter_damage = max(1, int(counter_damage * (1 - SORCERER_DEFENCE_BUFF_AMOUNT)))

                if counter_damage > 0:
                    attacker_alive = attacker.take_damage(counter_damage)

        return {
            "attacker_alive": attacker_alive,
            "target_alive": target_alive,
            "damage": attack_damage,
            "counter_damage": counter_damage,
            "charge_bonus": charge_applied,
            "flank_bonus": flank_applied,
            "evade": evade_applied,
            "attack_buff": attack_buff_applied,
            "defence_buff": defence_buff_applied,
        }

    @staticmethod
    def paralyze_unit(paralyzer, target, teams=None):
        """Mage paralyzes the target unit: it loses its next PARALYZE_DURATION turns."""
        if paralyzer.type != "M":
            return False

        if paralyzer.paralyze_cooldown > 0:
            return False

        if same_side(target.player, paralyzer.player, teams):
            return False

        # The range the legal-action mask uses too (``ABILITY_RANGES``, via
        # ``GameState._can_paralyze_target``). Kept symmetric with the
        # heal/cure/buff checks so the mask and execution can't drift and
        # reopen the heal-spam loop pattern: a legal-but-unexecutable action
        # that, under a deterministic policy, traps the legal-actions cache
        # (which only invalidates on successful mutations).
        if not GameMechanics.in_ability_range("paralyze", paralyzer, target):
            return False

        # The counter ticks at the start of each of the victim's turns and
        # frees it on reaching 0 (tick_statuses). A paralysis is cast
        # on the victim's opponent's turn, so the victim's next turn start
        # ticks it before that lost turn is played: +1 makes it cost exactly
        # PARALYZE_DURATION of the victim's own turns, and keeps it paralyzed
        # (no counter-attacks, no re-paralysis) until its first free turn
        # starts. See the constant's comment in constants.py.
        target.paralyzed_turns = PARALYZE_DURATION + 1
        paralyzer.paralyze_cooldown = PARALYZE_COOLDOWN
        return True

    @staticmethod
    def heal_unit(healer, target, teams=None):
        """
        Healer heals the target unit.

        Args:
            healer: The unit doing the healing (must be Cleric)
            target: The target unit to heal

        Returns:
            int: Actual amount healed, or -1 if heal failed
        """
        if healer.type != "C":
            return -1

        if not same_side(target.player, healer.player, teams):
            return -1

        # Check distance: the heal range ``is_healable_ally`` (the mask's
        # rule) reads too. A tighter bound here produced mask/execution
        # drift -- the mask advertised heal actions whose execution
        # returned -1 with no state change, and under a deterministic policy
        # that looped the legal_actions cache forever (eval signature:
        # thousands of consecutive invalid heal actions until max_steps
        # truncation).
        if not GameMechanics.in_ability_range("heal", healer, target):
            return -1

        if target.health >= target.max_health:
            return -1

        old_health = target.health
        target.health = min(target.health + HEAL_AMOUNT, target.max_health)
        return target.health - old_health

    @staticmethod
    def cure_unit(curer, target, teams=None):
        """Cleric cures the target unit's paralysis."""
        if curer.type != "C":
            return False

        if not same_side(target.player, curer.player, teams):
            return False

        # Check distance: the cure range ``is_curable_ally`` reads too, or
        # the mask/execution disagreement re-opens the heal-spam loop on a
        # paralyzed ally (see heal_unit).
        if not GameMechanics.in_ability_range("cure", curer, target):
            return False

        if not target.is_paralyzed():
            return False

        target.paralyzed_turns = 0
        target.can_move = True
        target.can_attack = True
        return True

    @staticmethod
    def haste_unit(sorcerer, target):
        """
        Sorcerer grants Haste to target unit, allowing an extra action.

        Only marks the target (``is_hasted``) and starts the cooldown; the
        extra action itself is granted by ``GameState`` when the target's
        action is spent (``GameState._consume_action``), or at once if it
        already is. This used to re-arm ``can_move``/``can_attack`` here,
        which gave a unit that had not acted yet nothing and one that had
        acted a refresh on top of the one ``end_unit_turn`` later granted
        (review core-8).

        Args:
            sorcerer: The Sorcerer unit using Haste
            target: The target friendly unit to receive Haste

        Returns:
            bool: True if Haste was successfully applied
        """
        if sorcerer.type != "S":
            return False

        if sorcerer.haste_cooldown > 0:
            return False

        if not GameMechanics.is_hasteable_ally(sorcerer, target):
            return False

        # Apply haste to target
        target.is_hasted = True

        # Set cooldown on sorcerer
        sorcerer.haste_cooldown = HASTE_COOLDOWN

        return True

    @staticmethod
    def defence_buff_unit(sorcerer, target, teams=None):
        """
        Sorcerer grants Defence Buff to target unit, reducing damage taken by
        SORCERER_DEFENCE_BUFF_AMOUNT (50%) for SORCERER_BUFF_DURATION turns.

        Args:
            sorcerer: The Sorcerer unit using Defence Buff
            target: The target friendly unit to receive the buff

        Returns:
            bool: True if Defence Buff was successfully applied
        """
        if sorcerer.type != "S":
            return False

        if sorcerer.defence_buff_cooldown > 0:
            return False

        if not same_side(target.player, sorcerer.player, teams):
            return False

        if target.has_defence_buff():
            return False

        # Check distance (min 0: can buff self)
        if not GameMechanics.in_ability_range("defence_buff", sorcerer, target):
            return False

        # Apply defence buff to target (see _buff_counter for the teammate case)
        target.defence_buff_turns = GameMechanics._buff_counter(sorcerer, target)

        # Set cooldown on sorcerer
        sorcerer.defence_buff_cooldown = SORCERER_BUFF_COOLDOWN

        return True

    @staticmethod
    def attack_buff_unit(sorcerer, target, teams=None):
        """
        Sorcerer grants Attack Buff to target unit, increasing damage dealt by
        SORCERER_ATTACK_BUFF_AMOUNT (50%) for SORCERER_BUFF_DURATION turns.

        Args:
            sorcerer: The Sorcerer unit using Attack Buff
            target: The target friendly unit to receive the buff

        Returns:
            bool: True if Attack Buff was successfully applied
        """
        if sorcerer.type != "S":
            return False

        if sorcerer.attack_buff_cooldown > 0:
            return False

        if not same_side(target.player, sorcerer.player, teams):
            return False

        if target.has_attack_buff():
            return False

        # Check distance (min 0: can buff self)
        if not GameMechanics.in_ability_range("attack_buff", sorcerer, target):
            return False

        # Apply attack buff to target (see _buff_counter for the teammate case)
        target.attack_buff_turns = GameMechanics._buff_counter(sorcerer, target)

        # Set cooldown on sorcerer
        sorcerer.attack_buff_cooldown = SORCERER_BUFF_COOLDOWN

        return True

    @staticmethod
    def _buff_counter(sorcerer, target):
        """Counter value for a buff covering SORCERER_BUFF_DURATION of ``target``'s own turns.

        Buff counters tick at the start of the buffed unit's owner's turn
        (tick_statuses). Cast on the Sorcerer's own unit, the cast turn is
        one of the covered turns and is not ticked, so the counter is the
        duration itself (unchanged behaviour). A teammate's
        unit is buffed outside its owner's turn and would lose one covered
        turn to its owner's next turn start, so it is stored one higher --
        the same rule a paralysis follows (see constants.PARALYZE_DURATION).
        """
        return SORCERER_BUFF_DURATION + (0 if target.player == sorcerer.player else 1)

    # The per-unit counters ``tick_statuses`` runs down, each with the unit
    # type it applies to (None: any unit). Statuses (paralysis, the buffs)
    # count their unit's turn starts until they end, cooldowns the caster's
    # turn starts until it may cast again; see the Status effects comment in
    # constants.py for why a status applied outside its owner's turn is
    # stored one higher (PARALYZE_DURATION + 1, _buff_counter).
    TURN_START_COUNTERS = (
        ("paralyzed_turns", None),
        ("paralyze_cooldown", "M"),
        ("haste_cooldown", "S"),
        ("defence_buff_cooldown", "S"),
        ("attack_buff_cooldown", "S"),
        ("defence_buff_turns", None),
        ("attack_buff_turns", None),
    )

    @staticmethod
    def tick_statuses(units, player):
        """Tick ``player``'s status and cooldown counters as its turn starts.

        Each positive ``TURN_START_COUNTERS`` counter on each of the
        player's units (of the counter's unit type, where it has one) drops
        by one. Only the player whose turn is starting ticks, so every
        duration is counted in its unit's own turns.

        Args:
            units: List of all units
            player: Player number whose turn is starting

        Returns:
            ``(unit, attr)`` for every counter that ran out (reached 0) on
            this tick: a unit freed from paralysis, a buff that expired, a
            caster whose ability came off cooldown.
        """
        expired = []
        for unit in units:
            if unit.player != player:
                continue
            for attr, unit_type in GameMechanics.TURN_START_COUNTERS:
                if unit_type is not None and unit.type != unit_type:
                    continue
                value = getattr(unit, attr)
                if value > 0:
                    setattr(unit, attr, value - 1)
                    if value == 1:
                        expired.append((unit, attr))
        return expired

    @staticmethod
    def seize_structure(unit, tile, teams=None):
        """
        Unit seizes a structure (tower, building, or HQ).

        Args:
            unit: The unit seizing
            tile: The structure tile
            teams: Optional player -> team map; a teammate's structure is
                friendly and cannot be seized.

        Returns:
            dict with 'captured' boolean, 'game_over' boolean, and
            'structure_type' (single-letter tile code) so callers can break
            capture counts down by structure type without having to
            re-look-up the tile.
        """
        if not tile.is_capturable():
            return {"captured": False, "game_over": False, "structure_type": tile.type}

        if same_side(tile.player, unit.player, teams):
            return {"captured": False, "game_over": False, "structure_type": tile.type}

        if tile.regenerating:
            tile.regenerating = False

        damage = unit.health
        tile.health -= damage

        captured = False
        game_over = False

        if tile.health <= 0:
            tile.health = tile.max_health
            tile.player = unit.player
            tile.regenerating = False
            captured = True

            if tile.type == "h":
                game_over = True

        return {
            "captured": captured,
            "game_over": game_over,
            "damage": damage,
            "remaining_hp": tile.health,
            "structure_type": tile.type,
        }

    @staticmethod
    def reset_structure_if_vacated(tile, units):
        """Reset structure HP if no unit is on it."""
        if not tile.is_capturable():
            return False

        # Check if any unit is on this tile
        for unit in units:
            if unit.x == tile.x and unit.y == tile.y:
                return False

        if tile.health < tile.max_health:
            tile.health = tile.max_health
            tile.regenerating = False
            return True

        return False

    @staticmethod
    def regenerate_structures(grid, units):
        """Regenerate HP for structures that are marked for regeneration."""
        regenerated = []
        for row in grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.regenerating:
                    # Check if there's a unit on this tile
                    unit_on_tile = False
                    for unit in units:
                        if unit.x == tile.x and unit.y == tile.y:
                            unit_on_tile = True
                            tile.regenerating = False
                            break

                    if not unit_on_tile:
                        regen_amount = int(tile.max_health * STRUCTURE_REGEN_RATE)
                        old_health = tile.health
                        tile.health = min(tile.health + regen_amount, tile.max_health)

                        if tile.health >= tile.max_health:
                            tile.regenerating = False

                        regenerated.append({"tile": tile, "amount": tile.health - old_health})

        return regenerated

    @staticmethod
    def calculate_income(player, grid, income_rates=None):
        """Calculate income for a player based on controlled structures.

        Args:
            player: Player number to compute income for.
            grid: The tile grid.
            income_rates: Optional ``{"headquarters", "building", "tower"}``
                per-structure rate overrides. When ``None`` the module
                constants are used, preserving behaviour for every legacy
                caller. ``GameState`` passes its per-game resolved rates so
                engine-override (economy) sweeps flow through here without
                mutating the shared module constants.
        """
        if income_rates is None:
            income_rates = {
                "headquarters": HEADQUARTERS_INCOME,
                "building": BUILDING_INCOME,
                "tower": TOWER_INCOME,
            }
        headquarters_count = 0
        building_count = 0
        tower_count = 0

        for row in grid.tiles:
            for tile in row:
                if tile.player == player:
                    if tile.type == "h":
                        headquarters_count += 1
                    elif tile.type == "b":
                        building_count += 1
                    elif tile.type == "t":
                        tower_count += 1

        total_income = (
            headquarters_count * income_rates["headquarters"]
            + building_count * income_rates["building"]
            + tower_count * income_rates["tower"]
        )

        return {"total": total_income, "headquarters": headquarters_count, "buildings": building_count, "towers": tower_count}
