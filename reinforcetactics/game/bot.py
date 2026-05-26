"""
AI bots for computer opponents with support for all unit types.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from reinforcetactics.constants import (
    CHARGE_BONUS,
    CHARGE_MIN_DISTANCE,
    COUNTER_ATTACK_MULTIPLIER,
    FLANK_BONUS,
    ROGUE_EVADE_CHANCE,
    ROGUE_FOREST_EVADE_BONUS,
    UNIT_DATA,
)
from reinforcetactics.game.bot_base import BaseBot, BotUnitMixin

# Maximum recursion depth for haste-triggered re-actions
MAX_RECURSION_DEPTH = 10


class NoopBot(BotUnitMixin, BaseBot):
    """Opponent that takes no actions and immediately ends its turn.

    Useful as a curriculum stage-0 / sanity check for RL agents: with no
    enemy units, no defenders, and a stationary HQ, even a random policy
    will eventually walk a unit onto the enemy HQ and seize it. If an
    agent fails to win against ``NoopBot``, the issue is in the policy
    or reward signal — not the opponent strength or map difficulty.
    """

    def __init__(self, game_state, player=2):
        self.game_state = game_state
        self.bot_player = player

    def take_turn(self):
        """Do nothing; just end the turn."""
        if not self.game_state.game_over:
            self.game_state.end_turn()


class RandomBot(BotUnitMixin, BaseBot):
    """AI bot that picks uniformly random legal actions each turn.

    Useful as a weak training opponent for RL agents and for sanity-checking
    environment dynamics. Picks actions uniformly from the set of currently
    legal non-end-turn actions, then ends its turn.
    """

    # Action keys to sample from (all legal action types except end_turn).
    _SAMPLE_ACTION_KEYS = (
        "create_unit",
        "move",
        "attack",
        "seize",
        "paralyze",
        "heal",
        "cure",
        "haste",
        "defence_buff",
        "attack_buff",
    )

    def __init__(self, game_state, player=2, max_actions: int = 20, rng=None):
        """
        Initialize the bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
            max_actions: Maximum number of random actions to attempt per turn
            rng: Optional ``random.Random`` instance for reproducibility. If
                ``None``, the global ``random`` module is used (non-deterministic).
        """
        self.game_state = game_state
        self.bot_player = player
        self.max_actions = max_actions
        # Both ``random`` and ``random.Random()`` instances expose ``.choice``.
        self._rng = rng if rng is not None else random

    def take_turn(self):
        """Execute random legal actions, then end the turn."""
        for _ in range(self.max_actions):
            if self.game_state.game_over:
                break

            legal_actions = self.game_state.get_legal_actions(player=self.bot_player)

            all_actions = []
            for action_key in self._SAMPLE_ACTION_KEYS:
                for action in legal_actions.get(action_key, []):
                    all_actions.append((action_key, action))

            if not all_actions:
                break  # Only end_turn available

            action_key, action = self._rng.choice(all_actions)
            try:
                self._execute(action_key, action)
            except Exception:
                continue  # Skip failed actions, try another

        if not self.game_state.game_over:
            self.game_state.end_turn()

    def _execute(self, action_key: str, action: Dict[str, Any]) -> None:
        """Dispatch a sampled action to the appropriate game-state method."""
        if action_key == "create_unit":
            self.game_state.create_unit(action["unit_type"], action["x"], action["y"], player=self.bot_player)
        elif action_key == "move":
            self.game_state.move_unit(action["unit"], action["to_x"], action["to_y"])
        elif action_key == "attack":
            self.game_state.attack(action["attacker"], action["target"])
        elif action_key == "seize":
            self.game_state.seize(action["unit"])
        elif action_key == "paralyze":
            self.game_state.paralyze(action["paralyzer"], action["target"])
        elif action_key == "heal":
            self.game_state.heal(action["healer"], action["target"])
        elif action_key == "cure":
            self.game_state.cure(action["curer"], action["target"])
        elif action_key == "haste":
            self.game_state.haste(action["sorcerer"], action["target"])
        elif action_key == "defence_buff":
            self.game_state.defence_buff(action["sorcerer"], action["target"])
        elif action_key == "attack_buff":
            self.game_state.attack_buff(action["sorcerer"], action["target"])


class BalancedRandomBot(RandomBot):
    """Random opponent whose action throughput scales with army size.

    Each turn:
      1. Optionally build one random unit (if a ``create_unit`` action is
         legal -- i.e. the bot can afford to spawn one and there is a free
         building tile).
      2. For every owned unit, pick one random legal action involving that
         unit and execute it (move / attack / seize / heal / buff). Units
         with no legal actions sit out the turn.
      3. End the turn.

    Compared to ``RandomBot(max_actions=1)``, this baseline keeps applying
    pressure even after losing units (it can still build) and makes the
    pressure proportional to the bot's current unit count instead of
    capping at one action regardless of army size. That avoids the
    pathological case where the agent kills the bot's only unit and the
    bot then "stalemates" itself, picking the build action only ~1 in
    ``len(_SAMPLE_ACTION_KEYS)`` of the time.

    Use as a curriculum stepping stone between ``NoopBot`` (zero actions
    per turn) and ``RandomBot`` (up to ``max_actions=20`` actions per
    turn) -- see configs/ppo/bootstrap.yaml.
    """

    # Map action_key -> attribute on the action dict that identifies the
    # actor unit. ``create_unit`` is handled out-of-band because it has no
    # actor unit.
    _ACTOR_FIELDS: Dict[str, str] = {
        "move": "unit",
        "attack": "attacker",
        "seize": "unit",
        "heal": "healer",
        "cure": "curer",
        "paralyze": "paralyzer",
        "haste": "sorcerer",
        "defence_buff": "sorcerer",
        "attack_buff": "sorcerer",
    }

    def __init__(self, game_state, player: int = 2, rng=None) -> None:
        # Skip RandomBot.__init__'s ``max_actions`` arg; it doesn't apply
        # here. Wire up the same fields it sets.
        self.game_state = game_state
        self.bot_player = player
        self._rng = rng if rng is not None else random

    def take_turn(self) -> None:
        if self.game_state.game_over:
            return

        legal_actions = self.game_state.get_legal_actions(player=self.bot_player)

        # Step 1: maybe build one unit. Recompute legal actions afterwards
        # because the new unit may shift positions / affordability.
        creates = legal_actions.get("create_unit", [])
        if creates:
            try:
                self._execute("create_unit", self._rng.choice(creates))
            except Exception:
                pass
            if self.game_state.game_over:
                return
            legal_actions = self.game_state.get_legal_actions(player=self.bot_player)

        # Step 2: bucket non-create legal actions by the unit that
        # performs them, then pick one per unit. Iterating ``self.game_state.units``
        # in order keeps turn-to-turn behaviour deterministic given the RNG
        # seed (RandomBot relies on the same property).
        actions_by_unit: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
        for action_key, actor_field in self._ACTOR_FIELDS.items():
            for action in legal_actions.get(action_key, []):
                actor = action.get(actor_field)
                if actor is None:
                    continue
                actions_by_unit.setdefault(id(actor), []).append((action_key, action))

        for unit in list(self.game_state.units):
            if getattr(unit, "player", None) != self.bot_player:
                continue
            unit_actions = actions_by_unit.get(id(unit))
            if not unit_actions:
                continue
            action_key, action = self._rng.choice(unit_actions)
            try:
                self._execute(action_key, action)
            except Exception:
                continue
            if self.game_state.game_over:
                return

        if not self.game_state.game_over:
            self.game_state.end_turn()


class SimpleBot(BotUnitMixin, BaseBot):
    """Simple AI bot for player 2 with basic unit type awareness."""

    # Unit purchase priorities (lower = higher priority)
    UNIT_PRIORITIES = {
        "W": 1,  # Warrior - cheap, good for capturing
        "B": 2,  # Barbarian - fast, good mobility
        "A": 3,  # Archer - safe ranged damage
        "K": 4,  # Knight - heavy hitter
        "R": 5,  # Rogue - flanking potential
        "M": 6,  # Mage - ranged + paralyze
        "C": 7,  # Cleric - healing support
        "S": 8,  # Sorcerer - buff support
    }

    # Cap Warrior share of the army. Without this, the (priority=1,
    # cost=cheapest) sort always picks Warrior -- the baseline tournament
    # showed SimpleBot building 100% Warriors. Once the army has at least
    # ``WARRIOR_CAP_MIN_UNITS`` units AND Warrior share is at or above
    # ``WARRIOR_SHARE_CAP``, Warriors drop out of the affordable set and
    # the next priority unit fills the slot. The bot still defaults to
    # Warrior on the very first buy and any time it's the only thing in
    # range, so weak early-game tempo is preserved.
    WARRIOR_SHARE_CAP = 0.5
    WARRIOR_CAP_MIN_UNITS = 3

    def __init__(self, game_state, player=2, rng=None):
        """
        Initialize the bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
            rng: Optional ``random.Random`` instance enabling stochastic
                tiebreaking at every sort / best-tracking decision site.
                ``None`` (default) keeps the bot fully deterministic --
                identical starting states produce identical games. When
                set, ties on the scoring heuristic resolve randomly,
                giving distinct episode trajectories from the same
                opponent while preserving strategic quality.
        """
        self.game_state = game_state
        self.bot_player = player
        self._rng = rng

    def take_turn(self):
        """Execute the bot's turn."""
        # Phase 1: Purchase units
        self.purchase_units()

        # Phase 2: Move and act with units
        self.move_and_act_units()

        # Phase 3: End turn
        self.game_state.end_turn()

    def purchase_units(self):
        """Purchase units based on priority from enabled types."""
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions["create_unit"]
            # Note: legal_actions already filters by enabled_units

            if not create_actions:
                break

            available_gold = self.game_state.player_gold[self.bot_player]

            # Filter to affordable units
            affordable = [a for a in create_actions if UNIT_DATA[a["unit_type"]]["cost"] <= available_gold]
            if not affordable:
                break

            # Apply Warrior composition cap. Counts live units (newly built
            # ones from earlier iterations of this loop are included), so the
            # ratio updates as we go.
            my_units = [u for u in self.game_state.units if u.player == self.bot_player]
            total = len(my_units)
            if total >= self.WARRIOR_CAP_MIN_UNITS:
                w_count = sum(1 for u in my_units if u.type == "W")
                if total > 0 and (w_count / total) >= self.WARRIOR_SHARE_CAP:
                    non_w = [a for a in affordable if a["unit_type"] != "W"]
                    if non_w:
                        affordable = non_w
                        self._record("warrior_cap_hit")

            # Sort by priority (lower = buy first), then by cost (cheaper first).
            # Shuffle first so equal-priority/equal-cost units (e.g. two
            # affordable warriors on different spawn tiles) tiebreak randomly
            # when stochastic mode is enabled.
            self._maybe_shuffle(affordable)
            affordable.sort(key=lambda a: (self.UNIT_PRIORITIES.get(a["unit_type"], 99), UNIT_DATA[a["unit_type"]]["cost"]))

            action = affordable[0]
            self.game_state.create_unit(action["unit_type"], action["x"], action["y"], self.bot_player)
            self._record(f"buy_{action['unit_type']}")

    def move_and_act_units(self):
        """Move and act with all bot units."""
        bot_units = [
            u
            for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack) and not u.is_paralyzed()
        ]

        for unit in bot_units:
            self.act_with_unit(unit)

    def act_with_unit(self, unit, _depth=0):
        """Determine and execute best action for a single unit."""
        if _depth >= MAX_RECURSION_DEPTH:
            return
        # Stop issuing actions once a prior action (typically a kill or HQ
        # capture) flipped game_over -- otherwise we keep appending cosmetic
        # moves to action_history, which makes replays look like the game
        # was truncated mid-turn.
        if self.game_state.game_over:
            return

        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if tile.is_capturable() and tile.player != self.bot_player and tile.health < tile.max_health:
            self.game_state.seize(unit)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
            return

        # Stay put if wounded and already standing on one of our heal tiles --
        # leaving would forfeit next turn's heal. This is the only retreat
        # behaviour SimpleBot has; routing wounded units to heal tiles is a
        # MediumBot+ feature.
        if unit.health < unit.max_health * 0.5 and self.is_on_heal_tile(unit):
            unit.end_unit_turn()
            return

        # Cleric: try to heal damaged allies or cure paralyzed allies
        if unit.type == "C" and unit.can_attack:
            if self.try_cleric_abilities(unit):
                if unit.can_move or unit.can_attack:
                    self.act_with_unit(unit, _depth + 1)
                return

        # Mage: try to paralyze high-value enemies before normal attack
        if unit.type == "M" and unit.can_attack:
            if self.try_mage_paralyze(unit):
                if unit.can_move or unit.can_attack:
                    self.act_with_unit(unit, _depth + 1)
                return

        # Find best target
        target = self.find_best_target(unit)

        if target:
            target_type, target_obj, _ = target

            if target_type == "enemy_unit":
                self.attack_enemy(unit, target_obj, _depth)
            elif target_type in ["enemy_tower", "enemy_building", "enemy_hq"]:
                self.move_to_and_seize(unit, target_obj, _depth)
        else:
            can_still_act = unit.end_unit_turn()
            if can_still_act:
                self.act_with_unit(unit, _depth + 1)

    def find_best_target(self, unit):
        """Find the best target for a unit (enemy unit or structure)."""
        # Find enemy units
        enemy_units = [
            (u, self.manhattan_distance(unit.x, unit.y, u.x, u.y))
            for u in self.game_state.units
            if u.player != self.bot_player
        ]

        # Find capturable structures (enemy-owned and neutral). The enemy HQ is
        # always a valid target -- seizing it wins the game, so we don't gate
        # it on the opponent having no other bases/units.
        enemy_structures = []
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.player != self.bot_player:
                    dist = self.manhattan_distance(unit.x, unit.y, tile.x, tile.y)
                    if tile.type == "t":
                        enemy_structures.append(("enemy_tower", tile, dist))
                    elif tile.type == "b":
                        enemy_structures.append(("enemy_building", tile, dist))
                    elif tile.type == "h" and tile.player is not None:
                        enemy_structures.append(("enemy_hq", tile, dist))

        # Combine targets
        all_targets = [("enemy_unit", u, d) for u, d in enemy_units]
        all_targets.extend(enemy_structures)

        if not all_targets:
            return None

        # Sort by distance, prioritize buildings/towers
        def sort_key(target):
            target_type, _target_obj, distance = target
            priority = 0 if target_type in ["enemy_building", "enemy_tower", "enemy_hq"] else 1
            return (distance, priority)

        # Shuffle pre-sort so equidistant equal-priority targets (very
        # common on small maps) resolve to a random pick instead of
        # whichever happened to be first in the source iteration order.
        self._maybe_shuffle(all_targets)
        all_targets.sort(key=sort_key)
        return all_targets[0]

    def attack_enemy(self, unit, enemy, _depth=0):
        """Attack an enemy unit with unit-type awareness."""
        distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)

        # Handle Archer (range 2-3, cannot attack adjacent)
        if unit.type == "A":
            self._attack_as_archer(unit, enemy, distance)
            return

        # Handle Mage/Sorcerer (range 1-2, prefer ranged)
        if unit.type in ["M", "S"]:
            self._attack_as_ranged_caster(unit, enemy, distance)
            return

        # Standard melee attack (W, K, R, B, C)
        if distance == 1:
            self.game_state.attack(unit, enemy)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
        else:
            target_pos = self.find_best_move_position(unit, enemy.x, enemy.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                new_distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
                if new_distance == 1:
                    self.game_state.attack(unit, enemy)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit(unit, _depth + 1)
                else:
                    can_still_act = unit.end_unit_turn()
                    if can_still_act:
                        self.act_with_unit(unit, _depth + 1)
            else:
                can_still_act = unit.end_unit_turn()
                if can_still_act:
                    self.act_with_unit(unit, _depth + 1)

    def _attack_as_archer(self, unit, enemy, distance):
        """Handle Archer attacks (range 2-3, cannot attack at distance 1)."""
        # Check if on mountain for extended range (2-4)
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        max_range = 4 if tile.type == "m" else 3
        min_range = 2

        # Already in valid range
        if min_range <= distance <= max_range:
            self.game_state.attack(unit, enemy)
            return

        # Need to move to valid range
        target_pos = self._find_ranged_attack_position(unit, enemy, min_range, max_range)
        if target_pos:
            self.game_state.move_unit(unit, target_pos[0], target_pos[1])
            new_distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
            if min_range <= new_distance <= max_range:
                self.game_state.attack(unit, enemy)
            else:
                unit.end_unit_turn()
        else:
            unit.end_unit_turn()

    def _attack_as_ranged_caster(self, unit, enemy, distance):
        """Handle Mage/Sorcerer attacks (range 1-2, prefer distance 2)."""
        # Can attack at distance 1 or 2
        if 1 <= distance <= 2:
            self.game_state.attack(unit, enemy)
            return

        # Need to move into range
        target_pos = self._find_ranged_attack_position(unit, enemy, 1, 2)
        if target_pos:
            self.game_state.move_unit(unit, target_pos[0], target_pos[1])
            new_distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
            if 1 <= new_distance <= 2:
                self.game_state.attack(unit, enemy)
            else:
                unit.end_unit_turn()
        else:
            unit.end_unit_turn()

    def _find_ranged_attack_position(self, unit, enemy, min_range: int, max_range: int) -> Optional[Tuple[int, int]]:
        """Find a position from which unit can attack enemy at valid range."""
        reachable = self.get_reachable(unit)

        if not reachable:
            return None

        # Find positions within attack range of enemy
        valid_positions = []
        for pos in reachable:
            dist = self.manhattan_distance(pos[0], pos[1], enemy.x, enemy.y)
            if min_range <= dist <= max_range:
                valid_positions.append((pos, dist))

        if not valid_positions:
            # No valid attack position, move closer
            return self.find_best_move_position(unit, enemy.x, enemy.y)

        # Prefer positions at max range (safer). Shuffle so multiple
        # tiles at the same max distance resolve to a random pick.
        self._maybe_shuffle(valid_positions)
        valid_positions.sort(key=lambda x: -x[1])
        return valid_positions[0][0]

    def move_to_and_seize(self, unit, structure, _depth=0):
        """Move towards and seize a structure."""
        if unit.x == structure.x and unit.y == structure.y:
            self.game_state.seize(unit)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
        else:
            target_pos = self.find_best_move_position(unit, structure.x, structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                if unit.x == structure.x and unit.y == structure.y:
                    self.game_state.seize(unit)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit(unit, _depth + 1)
                else:
                    can_still_act = unit.end_unit_turn()
                    if can_still_act:
                        self.act_with_unit(unit, _depth + 1)
            else:
                can_still_act = unit.end_unit_turn()
                if can_still_act:
                    self.act_with_unit(unit, _depth + 1)


class MediumBot(BotUnitMixin, BaseBot):
    """Medium difficulty AI bot with improved strategic decision-making."""

    # Unit purchase priorities for MediumBot (lower = higher priority)
    # Priorities consider tactical value and cost efficiency
    UNIT_PRIORITIES = {
        "W": (0, "capture"),  # Warrior - cheap, good HP for capturing
        "B": (1, "mobility"),  # Barbarian - fast capturing
        "A": (2, "ranged"),  # Archer - safe ranged damage
        "K": (3, "damage"),  # Knight - heavy damage with charge
        "R": (4, "flank"),  # Rogue - flanking potential
        "M": (5, "control"),  # Mage - ranged + paralyze
        "C": (6, "support"),  # Cleric - healing
        "S": (7, "buff"),  # Sorcerer - buff support
    }

    # Warrior composition cap, same shape as SimpleBot but with a higher
    # threshold so MediumBot still goes Warrior-heavy in the early game.
    # The (priority=0, cost=cheapest) sort otherwise produces a Warrior
    # monoculture identical to SimpleBot's; smoke runs showed MediumBot
    # building ~7 Warriors per game and zero of anything else even though
    # A/K are enabled. With the cap, once the army has at least
    # ``WARRIOR_CAP_MIN_UNITS`` units AND Warrior share is at or above
    # ``WARRIOR_SHARE_CAP``, Warriors drop out of the affordable set and
    # the next priority unit (Barbarian, then Archer) fills the slot.
    # ``warrior_cap_hit`` is recorded each time the filter activates so
    # the cap's frequency is observable in capabilities_per_bot.
    WARRIOR_SHARE_CAP = 0.6
    WARRIOR_CAP_MIN_UNITS = 4

    def __init__(self, game_state, player=2, rng=None):
        """
        Initialize the bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
            rng: Optional ``random.Random`` enabling stochastic
                tiebreaking. See SimpleBot.__init__ for details.
        """
        self.game_state = game_state
        self.bot_player = player
        self._rng = rng

    def take_turn(self):
        """Execute the bot's turn with improved strategy."""
        # Per-turn set of contested-structure enemy ids already being handled,
        # so multiple units don't all converge on the same interrupt target.
        self._interrupt_assigned = set()
        # Per-turn set of structure positions already claimed by another unit's
        # capture priority. Reset every turn so abandoned targets are reusable.
        self._capture_assigned = set()

        # Phase 1: Purchase units - maximize unit production
        self.purchase_units()

        # Phase 2: Move and act with units using coordinated strategy
        self.move_and_act_units()

        # Phase 3: End turn
        self.game_state.end_turn()

    def find_our_hq(self):
        """
        Locate the bot's headquarters.

        Returns:
            Tuple of (x, y) for HQ location, or None if not found
        """
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type == "h" and tile.player == self.bot_player:
                    return (tile.x, tile.y)
        return None

    # Retreat to a friendly structure when below half HP. Subclasses tune this.
    RETREAT_HEALTH_THRESHOLD = 0.5

    def should_retreat_to_heal(self, unit) -> bool:
        """Whether this unit's current HP warrants a heal retreat."""
        return unit.health < unit.max_health * self.RETREAT_HEALTH_THRESHOLD

    def find_retreat_tile(self, unit):
        """Pick a reachable owned heal tile, preferring the highest heal amount
        and breaking ties by closeness to the unit."""
        reachable = self.get_reachable(unit)
        if not reachable:
            return None

        # Standing still counts as a candidate -- if we're already on a heal
        # tile, ending the turn there is the best move.
        candidates = [(unit.x, unit.y)] + list(reachable)

        best = None
        best_score = (0, float("inf"))  # (heal_amount, -distance) maximised
        # Shuffle candidates so multiple equally-good heal tiles
        # resolve to a random pick instead of the first-visited one.
        self._maybe_shuffle(candidates)
        for x, y in candidates:
            heal = self.heal_amount_at(x, y)
            if heal == 0:
                continue
            distance = self.manhattan_distance(unit.x, unit.y, x, y)
            score = (heal, -distance)
            if score > best_score:
                best_score = score
                best = (x, y)
        return best

    def try_retreat_to_heal(self, unit) -> bool:
        """Move a wounded unit onto our nearest heal tile and end its turn.

        Returns True if a retreat was committed. Caller is expected to skip
        this when an interrupt or a finishing blow is available.
        """
        if not self.should_retreat_to_heal(unit):
            return False

        target = self.find_retreat_tile(unit)
        if target is None:
            return False

        if (unit.x, unit.y) != target:
            self.game_state.move_unit(unit, target[0], target[1])
        unit.end_unit_turn()
        self._record("retreat_to_heal")
        return True

    def get_structure_priority(self, structure, unit=None):
        """
        Score structures for capture priority. Lower is better.

        When ``unit`` is supplied, the unit's Manhattan distance to the
        structure dominates -- a Warrior next to a tower picks that tower,
        not whatever happens to be closest to HQ. HQ distance, income, and
        the neutral bias act as tiebreakers within the same unit-distance
        bucket. When ``unit`` is None we fall back to the legacy HQ-only
        formula so callers that don't yet thread a unit (e.g. ad-hoc
        scoring in tests) still work.
        """
        income_weights = {"h": 100, "b": 100, "t": 50}
        income_bonus = income_weights.get(structure.type, 0)
        neutral_bonus = 80 if structure.player is None else 0

        our_hq = self.find_our_hq()
        if our_hq is None:
            distance_to_hq = self.manhattan_distance(0, 0, structure.x, structure.y)
        else:
            distance_to_hq = self.manhattan_distance(our_hq[0], our_hq[1], structure.x, structure.y)

        # Tiebreaker score: HQ proximity, structure income, neutral-undefended
        # bias. Same shape as the previous formula -- preserved so callers
        # without a unit still see consistent ordering.
        secondary = distance_to_hq - (income_bonus / 10.0) - (neutral_bonus / 10.0)

        if unit is None:
            return secondary

        # With a unit, dominate by unit-distance so each unit picks its own
        # closest target. We multiply secondary by 1/100 so a 100-tile gap
        # in HQ-distance can't outweigh a single tile of unit-distance.
        unit_distance = self.manhattan_distance(unit.x, unit.y, structure.x, structure.y)
        return unit_distance + secondary / 100.0

    def _capture_assignments(self):
        """Per-turn set of structure positions already claimed by another
        unit's capture priority. Lazily created in take_turn or on first
        access."""
        if not hasattr(self, "_capture_assigned"):
            self._capture_assigned = set()
        return self._capture_assigned

    def pick_capture_target(self, unit):
        """Return the best capturable structure for ``unit`` that isn't
        already claimed by a sibling this turn, or None if none remain."""
        claimed = self._capture_assignments()
        candidates = []
        for row in self.game_state.grid.tiles:
            for structure in row:
                if not structure.is_capturable():
                    continue
                if structure.player == self.bot_player:
                    continue
                if (structure.x, structure.y) in claimed:
                    continue
                candidates.append((structure, self.get_structure_priority(structure, unit)))

        if not candidates:
            return None
        # Shuffle so equal-priority structures (e.g. two same-distance
        # neutral towers) resolve to a random pick across episodes.
        self._maybe_shuffle(candidates)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    # Single hard-counter rule: when the enemy has at least
    # COUNTER_TRIGGER_THRESHOLD of one unit type, MediumBot rotates one slot
    # of priority to the named counter. AdvancedBot uses a smoother weighted
    # matrix instead (see FULL_COUNTER_MATRIX).
    SINGLE_COUNTER_TARGETS = {
        "W": "A",
        "B": "A",
        "K": "A",
        "R": "A",  # melee → kite with Archer
        "A": "K",  # Archers → close gap with Knight
        "M": "K",
        "S": "A",  # ranged casters → close or outrange
        "C": "M",  # Cleric → paralyze
    }
    COUNTER_TRIGGER_THRESHOLD = 3

    def get_counter_unit(self) -> Optional[str]:
        """Return the unit type that should be prioritised based on the
        single most common enemy unit, or None when nothing crosses the
        trigger threshold."""
        counts = self.count_enemy_units_by_type()
        if not counts:
            return None
        # Pick the most common enemy type; tie-break by alphabetical so
        # the choice is deterministic across calls.
        dominant_type = max(counts, key=lambda k: (counts[k], -ord(k[0])))
        if counts[dominant_type] < self.COUNTER_TRIGGER_THRESHOLD:
            return None
        return self.SINGLE_COUNTER_TARGETS.get(dominant_type)

    def purchase_units(self):
        """Purchase units based on priority from all enabled types."""
        # Keep buying units until we can't afford any more
        counter_unit = self.get_counter_unit()
        if counter_unit is not None:
            self._record("counter_unit_rule_fired")
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions["create_unit"]
            # Note: legal_actions already filters by enabled_units

            if not create_actions:
                break

            # Available gold
            available_gold = self.game_state.player_gold[self.bot_player]

            # Find affordable units
            affordable_actions = [
                action for action in create_actions if UNIT_DATA[action["unit_type"]]["cost"] <= available_gold
            ]

            if not affordable_actions:
                break

            # Warrior composition cap. Same shape as SimpleBot but with a
            # higher threshold (0.6) so MediumBot is allowed to go heavier
            # on Warriors before the cap intervenes. The counter-unit rule
            # always overrides the cap -- if the enemy fielded enough W/B
            # to trigger the counter, we want to buy the Archer counter
            # regardless of our current W share.
            my_units = [u for u in self.game_state.units if u.player == self.bot_player]
            total = len(my_units)
            if total >= self.WARRIOR_CAP_MIN_UNITS:
                w_count = sum(1 for u in my_units if u.type == "W")
                if total > 0 and (w_count / total) >= self.WARRIOR_SHARE_CAP:
                    non_w = [a for a in affordable_actions if a["unit_type"] != "W"]
                    if non_w:
                        affordable_actions = non_w
                        self._record("warrior_cap_hit")

            # Sort by priority (uses UNIT_PRIORITIES), then by cost. When a
            # counter unit applies, it leaps to the front of the queue;
            # other types keep their relative ordering behind it.
            def unit_priority(action):
                unit_type = action["unit_type"]
                cost = UNIT_DATA[unit_type]["cost"]
                priority = self.UNIT_PRIORITIES.get(unit_type, (99, "unknown"))[0]
                if counter_unit is not None and unit_type == counter_unit:
                    priority = -1
                return (priority, cost)

            # Shuffle so multiple affordable units at the same priority
            # tiebreak randomly across episodes.
            self._maybe_shuffle(affordable_actions)
            affordable_actions.sort(key=unit_priority)

            # Buy the top priority affordable unit
            action = affordable_actions[0]
            self.game_state.create_unit(action["unit_type"], action["x"], action["y"], self.bot_player)
            self._record(f"buy_{action['unit_type']}")
            if counter_unit is not None and action["unit_type"] == counter_unit:
                self._record("counter_unit_bought")

    def move_and_act_units(self):
        """Move and act with all bot units using coordinated strategy."""
        # Get all bot units that can act
        bot_units = [
            u
            for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack) and not u.is_paralyzed()
        ]

        # Lock in any in-progress captures before coordinate_attacks gets to
        # pull units off their tower. seize() flips can_move/can_attack to
        # False, so the subsequent steps will skip those units automatically.
        self.continue_active_seizes(bot_units)

        # First, coordinate attacks to kill targets
        self.coordinate_attacks(bot_units)

        # Then, act with remaining units
        for unit in bot_units:
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit)

    def find_killable_targets(self, available_units):
        """
        Identify enemies that can be killed this turn with coordinated attacks.

        Args:
            available_units: List of units that can still act

        Returns:
            List of (enemy, attackers) tuples where attackers can kill the enemy
        """
        killable = []
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        for enemy in enemy_units:
            # Find all units that can attack this enemy and the damage they
            # would deal (so we can pick the minimum set of biggest hitters).
            # Apply Knight charge bonus when the move covers the threshold so
            # we don't underestimate Knights moving in.
            potential = []  # list of (attacker, damage)
            for unit in available_units:
                if not (unit.can_move or unit.can_attack):
                    continue

                # Direct attack: zero move distance, no charge bonus.
                attackable = self.game_state.mechanics.get_attackable_enemies(unit, [enemy], self.game_state.grid)
                if enemy in attackable:
                    tile = self.game_state.grid.get_tile(unit.x, unit.y)
                    damage = unit.get_attack_damage(enemy.x, enemy.y, tile.type == "m")
                    potential.append((unit, damage))
                    continue

                # Move-then-attack: pick the position with highest projected damage.
                best_damage = 0
                reachable = self.get_reachable(unit)
                for pos in reachable:
                    move_distance = self.manhattan_distance(unit.x, unit.y, pos[0], pos[1])
                    old_x, old_y = unit.x, unit.y
                    unit.x, unit.y = pos[0], pos[1]
                    attackable_from_pos = self.game_state.mechanics.get_attackable_enemies(unit, [enemy], self.game_state.grid)
                    if enemy in attackable_from_pos:
                        from_tile = self.game_state.grid.get_tile(pos[0], pos[1])
                        d = unit.get_attack_damage(enemy.x, enemy.y, from_tile.type == "m")
                        if unit.type == "K" and self.has_charge_units() and move_distance >= CHARGE_MIN_DISTANCE:
                            d = int(d * (1 + CHARGE_BONUS))
                        if d > best_damage:
                            best_damage = d
                    unit.x, unit.y = old_x, old_y

                if best_damage > 0:
                    potential.append((unit, best_damage))

            if not potential:
                continue

            # Pick minimal kill set: biggest hitters first. Shuffle
            # so attackers with equal projected damage queue up in
            # random order across episodes.
            #
            # NOTE: this is more than cosmetic tiebreaking -- the loop
            # below stops accumulating once damage_so_far >= enemy.health,
            # so when two equal-damage attackers tie, the shuffle
            # decides which one is COMMITTED to the kill and which is
            # left free to act elsewhere this turn. That cascades into
            # who takes the counterattack damage, who is spent vs.
            # available, and (for MasterBot.coordinate_attacks) the
            # HP-asc swing order. The bot still only picks among
            # equally-good options, but the *set* of committed
            # attackers changes per episode -- intended for diversity,
            # but worth flagging for readers expecting a strictly-
            # cosmetic shuffle.
            self._maybe_shuffle(potential)
            potential.sort(key=lambda ad: ad[1], reverse=True)
            total_damage = sum(d for _, d in potential)
            if total_damage < enemy.health:
                continue

            attackers_needed = []
            damage_so_far = 0
            for attacker, damage in potential:
                attackers_needed.append(attacker)
                damage_so_far += damage
                if damage_so_far >= enemy.health:
                    break

            killable.append((enemy, attackers_needed))

        return killable

    def coordinate_attacks(self, bot_units):
        """
        Plan and execute multi-unit attacks on single targets.

        Re-plans after every kill attempt: ``find_killable_targets`` builds
        each (enemy, attackers) group independently, so the same attacker
        can appear in multiple groups -- and when a counterattack kills or
        exhausts an attacker mid-turn, later groups silently lose it and
        deal only partial damage. The replan loop fixes that: pick the
        single highest-priority killable target on the *current* state,
        execute the kill, then recompute against whatever is still alive
        and still able to act. ``attempted`` keeps us from spinning on
        targets we tried but couldn't finish.

        Args:
            bot_units: List of bot units that can act
        """

        def target_priority(item):
            enemy, attackers = item
            tile = self.game_state.grid.get_tile(enemy.x, enemy.y)
            if tile.is_capturable() and tile.player != self.bot_player and tile.health < tile.max_health:
                return (0,)
            cost = UNIT_DATA[enemy.type]["cost"]
            return (1, -cost, len(attackers))

        attempted: set = set()
        while True:
            if self.game_state.game_over:
                return
            available = [u for u in bot_units if u.health > 0 and (u.can_move or u.can_attack)]
            if not available:
                return
            killable = self.find_killable_targets(available)
            killable = [(e, a) for e, a in killable if e.health > 0 and id(e) not in attempted]
            if not killable:
                return
            # Equal-priority killable targets (e.g. two enemies with the
            # same value-tier and equal HP) tiebreak randomly under
            # stochastic mode.
            self._maybe_shuffle(killable)
            killable.sort(key=target_priority)
            enemy, attackers = killable[0]
            attempted.add(id(enemy))
            self._execute_focus_fire(enemy, attackers)

    def _execute_focus_fire(self, enemy, attackers):
        """Run one focus-fire group: each attacker swings at ``enemy`` until
        the enemy drops or the attackers are spent. Caller is responsible
        for filtering enemies/attackers freshly between calls."""
        for attacker in attackers:
            if self.game_state.game_over:
                return
            # Skip attackers killed by an earlier counterattack -- the
            # engine doesn't refuse ``attack()`` on a dead unit, so a
            # stale reference here would land a phantom hit live but
            # no-op on replay (replay looks up the attacker by position
            # and finds nothing), making the rebuilt state diverge.
            if attacker.health <= 0:
                continue
            if not (attacker.can_move or attacker.can_attack):
                continue
            if enemy.health <= 0:
                return

            attackable = self.game_state.mechanics.get_attackable_enemies(attacker, [enemy], self.game_state.grid)
            if enemy in attackable:
                self.game_state.attack(attacker, enemy)
                continue

            # Move-then-attack. The engine applies the Knight charge bonus
            # automatically based on distance_moved set by move_unit.
            target_pos = self.find_best_move_position(attacker, enemy.x, enemy.y)
            if target_pos is None:
                continue
            self.game_state.move_unit(attacker, target_pos[0], target_pos[1])
            attackable_after = self.game_state.mechanics.get_attackable_enemies(attacker, [enemy], self.game_state.grid)
            if enemy in attackable_after and enemy.health > 0:
                self.game_state.attack(attacker, enemy)

    def find_contested_structures(self):
        """
        Find structures currently being captured by enemies.

        Returns:
            List of (structure, enemy_unit, capture_progress) tuples
        """
        contested = []

        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.player != self.bot_player:
                    # Check if health is below max (being captured)
                    if tile.health < tile.max_health:
                        # Find enemy unit on this structure
                        enemy_on_structure = None
                        for unit in self.game_state.units:
                            if unit.player != self.bot_player and unit.x == tile.x and unit.y == tile.y:
                                enemy_on_structure = unit
                                break

                        if enemy_on_structure:
                            # Calculate capture progress (0 to 1)
                            progress = 1.0 - (tile.health / tile.max_health)
                            contested.append((tile, enemy_on_structure, progress))

        return contested

    def calculate_attack_value(self, attacker, target, move_distance: Optional[int] = None):
        """
        Evaluate attack efficiency considering damage dealt, received, and abilities.

        Args:
            attacker: Unit that would attack
            target: Enemy unit to attack
            move_distance: Distance moved this turn before attacking (for Knight
                charge). If None, falls back to ``attacker.distance_moved``,
                which the engine maintains across moves -- so callers that
                already moved the unit get charge accounting for free.

        Returns:
            Value score (higher is better)
        """
        if move_distance is None:
            move_distance = getattr(attacker, "distance_moved", 0)

        # Calculate damage attacker would deal
        attacker_tile = self.game_state.grid.get_tile(attacker.x, attacker.y)
        on_mountain = attacker_tile.type == "m"
        damage_dealt = attacker.get_attack_damage(target.x, target.y, on_mountain)

        # Apply Knight charge bonus if applicable
        if attacker.type == "K" and self.has_charge_units():
            if move_distance >= CHARGE_MIN_DISTANCE:
                damage_dealt = int(damage_dealt * (1 + CHARGE_BONUS))

        # Apply Rogue flank bonus if applicable
        if attacker.type == "R" and self.has_flank_units():
            if self._can_flank(attacker, target):
                damage_dealt = int(damage_dealt * (1 + FLANK_BONUS))

        # Check if this kills the target (no counter-attack)
        if damage_dealt >= target.health:
            # Killing is very valuable - no counter-attack
            return 1000 + damage_dealt

        # Calculate counter-attack damage if target survives
        counter_damage = 0
        # Check if target can counter-attack
        target_tile = self.game_state.grid.get_tile(target.x, target.y)
        target_on_mountain = target_tile.type == "m"
        target_damage = target.get_attack_damage(attacker.x, attacker.y, target_on_mountain)

        # Archers can't be counter-attacked by melee units
        distance = self.manhattan_distance(attacker.x, attacker.y, target.x, target.y)
        if distance > 1 and target.type not in ["M", "A", "S"]:
            # Target can't counter-attack ranged attacker
            counter_damage = 0
        elif target_damage > 0:
            # Counter-attacks deal reduced damage
            counter_damage = int(target_damage * COUNTER_ATTACK_MULTIPLIER)

            # Rogue evade reduces expected counter-damage
            if attacker.type == "R" and self.has_flank_units():
                evade_chance = ROGUE_EVADE_CHANCE
                if attacker_tile.type == "f":  # Forest
                    evade_chance += ROGUE_FOREST_EVADE_BONUS
                counter_damage = int(counter_damage * (1 - evade_chance))

        # Value = damage dealt - damage received
        # Also consider unit costs
        attacker_cost = int(UNIT_DATA[attacker.type]["cost"])
        target_cost = int(UNIT_DATA[target.type]["cost"])

        # Suicide guard: don't die to land a non-killing hit. Prior versions
        # of this function let value go positive when raw damage was high
        # enough to offset the cost penalty (a 200g Warrior trading 100
        # damage for 80 counter and dying still scored +4). The kill_confirm
        # path above already short-circuits the lethal case, so reaching
        # here with ``counter_damage >= attacker.health`` always means a
        # fatal trade that leaves the target alive. Treat those as strictly
        # worse than skipping the attack -- callers gate on ``best_value
        # > 0``, so a strongly negative score reliably routes the unit to
        # another action.
        #
        # ``suicide_eval_rejected`` (not ``suicide_blocked``) because this
        # records one event per *evaluation* that triggered the guard,
        # not one per attack we declined to take. Multiple candidates
        # within a single attack-selection pass can trip it; the chosen
        # attack might separately be a positive-EV non-suicidal target,
        # in which case "blocked" overstates the guard's behavioural
        # impact. Counting evaluations is still useful (it tells you
        # *that* the guard fired, and which bots see suicidal options
        # the most often) -- just don't read the column as "attacks
        # prevented."
        if counter_damage >= attacker.health and damage_dealt < target.health:
            self._record("suicide_eval_rejected")
            return -1000.0 - counter_damage

        # Prefer favorable trades
        value = damage_dealt - counter_damage
        # Bonus for attacking high-value targets
        value += target_cost / 100.0
        # Penalty for risking high-value units
        value -= (counter_damage * attacker_cost) / 1000.0

        return value

    def _can_flank(self, attacker, target) -> bool:
        """Check if attacker can flank target (target adjacent to a friendly unit)."""
        if attacker.type != "R":
            return False

        # Check if any friendly unit is adjacent to target
        for unit in self.game_state.units:
            if unit.player == self.bot_player and unit != attacker:
                dist = self.manhattan_distance(unit.x, unit.y, target.x, target.y)
                if dist == 1:
                    return True
        return False

    def _find_flank_targets(self, rogue) -> List:
        """Find enemies that can be flanked by the Rogue."""
        if rogue.type != "R" or not self.has_flank_units():
            return []

        flankable = []
        enemies = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        for enemy in enemies:
            if self._can_flank(rogue, enemy):
                flankable.append(enemy)

        return flankable

    def act_with_unit(self, unit, _depth=0):
        """Execute actions for a single unit based on strategic priorities."""
        if _depth >= MAX_RECURSION_DEPTH:
            return
        # Game already won/lost (typically by a focus-fire kill earlier this
        # turn) -- don't queue extra cosmetic actions on top.
        if self.game_state.game_over:
            return

        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if tile.is_capturable() and tile.player != self.bot_player and tile.health < tile.max_health:
            self.game_state.seize(unit)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
            return

        # Cleric heal/cure and Mage paralyze before regular attacks.
        if unit.type == "C" and unit.can_attack and self.try_cleric_abilities(unit):
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
            return
        if unit.type == "M" and unit.can_attack and self.try_mage_paralyze(unit):
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
            return

        if not hasattr(self, "_interrupt_assigned"):
            self._interrupt_assigned = set()
        interrupt_assigned = self._interrupt_assigned

        # Priority 1: Interrupt enemy captures (skip enemies another unit
        # already committed to interrupt this turn).
        contested = self.find_contested_structures()
        if contested:
            # Sort by capture progress (higher progress = higher priority).
            # Shuffle first so multiple contests at the same progress
            # level resolve in random order across episodes.
            self._maybe_shuffle(contested)
            contested.sort(key=lambda x: x[2], reverse=True)

            for _, enemy_unit, __ in contested:
                if id(enemy_unit) in interrupt_assigned:
                    continue
                # Check if we can attack this enemy
                attackable = self.game_state.mechanics.get_attackable_enemies(unit, [enemy_unit], self.game_state.grid)

                if enemy_unit in attackable:
                    interrupt_assigned.add(id(enemy_unit))
                    self.game_state.attack(unit, enemy_unit)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit(unit, _depth + 1)
                    return

                # Try to move towards enemy and attack
                target_pos = self.find_best_move_position(unit, enemy_unit.x, enemy_unit.y)
                if target_pos:
                    move_distance = self.manhattan_distance(unit.x, unit.y, target_pos[0], target_pos[1])
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    # Check if can attack after moving
                    attackable_after = self.game_state.mechanics.get_attackable_enemies(
                        unit, [enemy_unit], self.game_state.grid
                    )
                    if enemy_unit in attackable_after:
                        interrupt_assigned.add(id(enemy_unit))
                        self.game_state.attack(unit, enemy_unit)
                        if unit.can_move or unit.can_attack:
                            self.act_with_unit(unit, _depth + 1)
                        return
                    # Move was committed but we can't reach: don't unwind, fall
                    # through. (move_distance computed for the curious; bonus
                    # accounting belongs in the value calc, not here.)
                    del move_distance

        # Priority 1.5: Retreat to a heal tile when wounded, unless we can
        # finish off an enemy in range right now (a killing blow is worth
        # more than one turn of healing).
        if self.should_retreat_to_heal(unit):
            attackable_now = self.game_state.mechanics.get_attackable_enemies(
                unit,
                [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0],
                self.game_state.grid,
            )
            on_mountain = self.game_state.grid.get_tile(unit.x, unit.y).type == "m"
            can_finish = any(unit.get_attack_damage(e.x, e.y, on_mountain) >= e.health for e in attackable_now)
            if not can_finish and self.try_retreat_to_heal(unit):
                return

        # Priority 2: Attack enemies with good value trades
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]
        if enemy_units:
            # Evaluate all possible attacks
            best_value = -1000
            best_target = None

            for enemy in enemy_units:
                attackable = self.game_state.mechanics.get_attackable_enemies(unit, [enemy], self.game_state.grid)

                if enemy in attackable:
                    value = self.calculate_attack_value(unit, enemy)
                    if value > best_value:
                        best_value = value
                        best_target = enemy

            # Attack if value is positive
            if best_target and best_value > 0:
                self.game_state.attack(unit, best_target)
                if unit.can_move or unit.can_attack:
                    self.act_with_unit(unit, _depth + 1)
                return

        # Priority 3: Capture structures. Each unit picks its own closest
        # unclaimed structure (see pick_capture_target); siblings can't
        # double-up on the same target this turn.
        target_structure = self.pick_capture_target(unit)
        if target_structure is not None:
            self._capture_assignments().add((target_structure.x, target_structure.y))

            # Check if already on structure
            if unit.x == target_structure.x and unit.y == target_structure.y:
                self.game_state.seize(unit)
                # Check if unit can act again (haste)
                if unit.can_move or unit.can_attack:
                    self.act_with_unit(unit, _depth + 1)
                return

            # Move towards structure
            target_pos = self.find_best_move_position(unit, target_structure.x, target_structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                # Check if reached structure
                if unit.x == target_structure.x and unit.y == target_structure.y:
                    self.game_state.seize(unit)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit(unit, _depth + 1)
                    return

        # Fallback: End turn
        can_still_act = unit.end_unit_turn()
        if can_still_act:
            self.act_with_unit(unit, _depth + 1)


class MixedBot(BotUnitMixin, BaseBot):
    """Curriculum bridge between two scripted bots.

    On construction, samples one of two bots (``easy`` or ``hard``) using
    ``p_hard`` (probability of choosing ``hard``, in ``[0, 1]``) and
    delegates ``take_turn()`` to that instance for the lifetime of this
    MixedBot. The env reconstructs its opponent on every ``reset()``
    (gym_env.py:reset), so the choice is effectively resampled per
    episode -- one episode is fully ``easy`` or fully ``hard``, never a
    mid-episode switch -- which preserves multi-turn strategy adaptation
    on the episodes where the harder bot plays.

    Bot type names: ``simple`` (SimpleBot), ``medium`` (MediumBot),
    ``advanced`` (AdvancedBot). Defaults bridge ``simple`` -> ``medium``.

    Use as a curriculum stepping stone via ``opponent_kwargs`` in
    configs/ppo/bootstrap.yaml, e.g.
    ``{easy: simple, hard: medium, p_hard: 0.5}`` for the simple->medium
    bridge or ``{easy: medium, hard: advanced, p_hard: 0.5}`` for the
    medium->advanced bridge.
    """

    _BOT_NAMES = ("simple", "medium", "advanced", "master", "random", "balanced_random")

    # Inner-bot constructor args MixedBot supplies itself. Forbidden in
    # easy_kwargs / hard_kwargs because forwarding them would either trip
    # Python's "got multiple values for keyword argument" TypeError
    # (rng, player are passed positionally / explicitly to ``_build_inner``)
    # or let a YAML config override the live game_state mid-curriculum.
    # Surfaced at construction time so a typo in opponent_kwargs fails on
    # the very first env.reset() rather than at some random Nth episode
    # when the coin flip happens to pick the side carrying the bad kwarg.
    _RESERVED_INNER_KWARGS = frozenset({"rng", "player", "game_state"})

    def __init__(
        self,
        game_state,
        player: int = 2,
        easy: str = "simple",
        hard: str = "medium",
        p_hard: float = 0.5,
        rng=None,
        easy_kwargs: Optional[Dict[str, Any]] = None,
        hard_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.game_state = game_state
        self.bot_player = player
        self.easy = easy
        self.hard = hard
        self.p_hard = p_hard
        # Reject reserved keys up front -- pre-PR these would crash with
        # an opaque "got multiple values for keyword argument 'rng'" only
        # on the episodes whose coin flip picks the side carrying the
        # collision, hiding the misconfig until mid-curriculum.
        for label, kw in (("easy_kwargs", easy_kwargs), ("hard_kwargs", hard_kwargs)):
            if kw:
                conflicts = self._RESERVED_INNER_KWARGS & set(kw)
                if conflicts:
                    raise ValueError(
                        f"MixedBot {label} cannot contain reserved keys {sorted(conflicts)}; "
                        f"MixedBot supplies these to the inner bot itself."
                    )
        # Both ``random`` and ``random.Random()`` instances expose ``.random``.
        self._rng = rng if rng is not None else random
        self.use_hard = self._rng.random() < p_hard
        chosen = hard if self.use_hard else easy
        chosen_kwargs = (hard_kwargs if self.use_hard else easy_kwargs) or {}
        # Forward the rng into the chosen inner bot so its stochastic
        # tiebreaking activates. Without this the inner bot's _rng is
        # None and every MixedBot episode that lands on the same inner
        # choice plays the byte-identical game -- defeating the purpose
        # of passing rng to MixedBot in the first place. The inner bot
        # consumes the same rng as the coin flip above; that's fine
        # because the coin flip happens once at construction and the
        # inner then takes over for the rest of the episode.
        self._inner = self._build_inner(chosen, game_state, player, rng=self._rng, **chosen_kwargs)

    @classmethod
    def _build_inner(cls, name: str, game_state, player: int, rng=None, **kwargs):
        # The ``random`` / ``balanced_random`` cases exist because the
        # curriculum's ``intermediate_mixed_random_simple`` / ``skirmish_mixed_*`` /
        # ``corner_points_mixed_*`` bridge stages pass these as the ``easy``
        # branch (see configs/ppo/bootstrap.yaml). Without them the env
        # crashes at reset whenever the coin flip picks the easy side,
        # which blocks the curriculum from progressing past random_20
        # into the harder simple/medium opponents.
        if name == "simple":
            return SimpleBot(game_state, player=player, rng=rng, **kwargs)
        if name == "medium":
            return MediumBot(game_state, player=player, rng=rng, **kwargs)
        if name == "advanced":
            return AdvancedBot(game_state, player=player, rng=rng, **kwargs)
        if name == "master":
            return MasterBot(game_state, player=player, rng=rng, **kwargs)
        if name == "random":
            return RandomBot(game_state, player=player, rng=rng, **kwargs)
        if name == "balanced_random":
            return BalancedRandomBot(game_state, player=player, rng=rng, **kwargs)
        raise ValueError(f"MixedBot: unknown bot type {name!r}; expected one of: {', '.join(cls._BOT_NAMES)}")

    def take_turn(self):
        self._inner.take_turn()


class AdvancedBot(MediumBot):
    """Advanced AI bot extending MediumBot with map analysis and enhanced tactics."""

    # Full composition targets for all 8 unit types
    # These will be dynamically adjusted based on enabled units
    FULL_COMPOSITION_TARGETS = {
        "W": 0.25,  # Warriors - capturing, frontline
        "A": 0.20,  # Archers - ranged damage
        "M": 0.15,  # Mages - ranged + paralyze
        "K": 0.10,  # Knights - heavy charge damage
        "R": 0.10,  # Rogues - flanking assassin
        "B": 0.08,  # Barbarians - fast mobility
        "C": 0.07,  # Clerics - healing support
        "S": 0.05,  # Sorcerers - buff support
    }

    # Per-archetype retreat thresholds. Frontliners take hits and benefit
    # from staying engaged longer; ranged/support are squishier and worth
    # pulling back earlier. Cleric is highest -- a dead Cleric loses the
    # whole heal economy.
    RETREAT_THRESHOLDS = {
        "W": 0.45,
        "K": 0.45,
        "B": 0.45,  # frontline
        "R": 0.40,  # flanker
        "A": 0.55,
        "M": 0.55,
        "S": 0.55,  # ranged
        "C": 0.65,  # support
    }

    # Per-enemy-unit weight bumps applied to FULL_COMPOSITION_TARGETS. For
    # every observed enemy unit of type ``key``, each counter in the inner
    # dict gets its target ratio bumped by that amount before
    # renormalisation. Smoother than MediumBot's single-counter rule:
    # mixed enemy comps produce mixed responses.
    FULL_COUNTER_MATRIX = {
        "W": {"A": 0.04, "M": 0.02},
        "B": {"A": 0.04, "M": 0.02},
        "K": {"A": 0.06, "M": 0.02},
        "R": {"A": 0.04},
        "A": {"K": 0.06, "B": 0.02},
        "M": {"K": 0.04, "A": 0.02},
        "C": {"M": 0.06, "R": 0.02},
        "S": {"M": 0.04, "A": 0.02},
    }

    # Game phase identifiers. EXPAND prioritises tempo (Warrior spam, willing
    # to march multi-turn for captures, refuse chip-trade combat). CONSOLIDATE
    # restores the existing balanced behaviour. CONQUER kicks in once nearly
    # all neutrals are claimed: only the enemy HQ/buildings remain to be taken,
    # so the bot accepts even bad trades to push through to the win condition.
    # Hysteresis (see take_turn) needs PHASE_TRANSITION_TURNS consecutive
    # turns of agreement before flipping, so the bot doesn't oscillate when
    # the ratio hovers around a threshold.
    PHASE_EXPAND = "EXPAND"
    PHASE_CONSOLIDATE = "CONSOLIDATE"
    PHASE_CONQUER = "CONQUER"

    # Phase thresholds on the fraction of capturable structures still neutral:
    #   neutral_pct > EXPAND_NEUTRAL_THRESHOLD -> EXPAND
    #   neutral_pct < CONQUER_NEUTRAL_THRESHOLD -> CONQUER
    #   otherwise                              -> CONSOLIDATE
    # 0.45 (vs the original 0.30) is calibrated against beginner.csv where
    # 4 of 10 capturable tiles are neutral (40%): below 0.45 so beginner
    # stays in CONSOLIDATE and the strict EXPAND attack threshold doesn't
    # refuse the close-range fights that small map demands.
    EXPAND_NEUTRAL_THRESHOLD = 0.45
    CONQUER_NEUTRAL_THRESHOLD = 0.10
    PHASE_TRANSITION_TURNS = 2

    # Floor for the Warrior target ratio while in EXPAND. Stops the matrix
    # from buying expensive ranged comp during the early-game tempo race
    # which MediumBot dominates with cheap Warrior spam.
    EXPAND_WARRIOR_FLOOR = 0.50

    # Attack-value gates for Priority 7 (move-to-attack). EXPAND demands a
    # genuinely good trade (we'd rather walk past and capture); CONSOLIDATE
    # accepts moderately bad trades to break stalls; CONQUER accepts almost
    # any trade because the alternative is the timeout-draw stalemate seen
    # on crossroads / last_stand when neutrals dry up.
    EXPAND_ATTACK_THRESHOLD = 200
    CONSOLIDATE_ATTACK_THRESHOLD = -500
    CONQUER_ATTACK_THRESHOLD = -1000

    def __init__(self, game_state, player=2, rng=None):
        """
        Initialize the AdvancedBot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
            rng: Optional ``random.Random`` enabling stochastic
                tiebreaking. Forwarded to MediumBot via super().
        """
        super().__init__(game_state, player, rng=rng)

        # Map analysis cache
        self.map_analyzed = False
        self.hq_positions = {}
        self.defensive_positions = []  # Mountains
        self.forest_positions = []  # Forests (for Rogue evade bonus)
        self.turn_count = 0

        # Game-phase state. ``phase`` is None until take_turn runs the first
        # time so direct callers (tests, REPL probing) see the unbiased
        # composition; the live bot updates it every turn.
        self.phase: Optional[str] = None
        self._pending_phase: Optional[str] = None
        self._pending_phase_streak = 0

    def compute_target_phase(self) -> str:
        """Phase the *current game state* suggests we should be in.

        EXPAND when plenty of neutrals remain (early/mid economy race);
        CONQUER once nearly all neutrals are claimed (push enemy HQ);
        CONSOLIDATE in between. Hysteresis is applied by update_phase.
        """
        total_capturable = 0
        neutral_capturable = 0
        for row in self.game_state.grid.tiles:
            for tile in row:
                if not tile.is_capturable():
                    continue
                total_capturable += 1
                if tile.player is None:
                    neutral_capturable += 1
        if total_capturable == 0:
            return self.PHASE_CONSOLIDATE
        neutral_pct = neutral_capturable / total_capturable
        if neutral_pct > self.EXPAND_NEUTRAL_THRESHOLD:
            return self.PHASE_EXPAND
        if neutral_pct < self.CONQUER_NEUTRAL_THRESHOLD:
            return self.PHASE_CONQUER
        return self.PHASE_CONSOLIDATE

    def update_phase(self) -> None:
        """Advance the phase state machine using ``compute_target_phase`` and
        a hysteresis counter so a one-turn fluctuation can't flip phases."""
        target = self.compute_target_phase()
        if self.phase is None:
            # Cold start: adopt the suggested phase immediately so the first
            # turn's purchases see the right bias.
            self.phase = target
            self._pending_phase = None
            self._pending_phase_streak = 0
            return
        if target == self.phase:
            self._pending_phase = None
            self._pending_phase_streak = 0
            return
        if target == self._pending_phase:
            self._pending_phase_streak += 1
            if self._pending_phase_streak >= self.PHASE_TRANSITION_TURNS:
                self.phase = target
                self._pending_phase = None
                self._pending_phase_streak = 0
        else:
            self._pending_phase = target
            self._pending_phase_streak = 1

    def take_turn(self):
        """Execute the bot's turn with enhanced strategy."""
        self.turn_count += 1

        # Per-turn dedup set for contested-structure interrupts.
        self._interrupt_assigned = set()
        # Per-turn capture-target dedup; see MediumBot.pick_capture_target.
        self._capture_assigned = set()
        # Phase update before any purchase/movement decisions read self.phase.
        self.update_phase()

        # Phase 1: Analyze map on first turn
        if not self.map_analyzed:
            self.analyze_map()
            self.map_analyzed = True

        # Phase 2: Use enhanced purchase strategy
        self.purchase_units_enhanced()

        # Phase 3: Enhanced unit actions with special abilities and better tactics
        self.move_and_act_units_enhanced()

        # Phase 4: End turn
        self.game_state.end_turn()

    def analyze_map(self):
        """Pre-compute strategic map features on first turn."""
        grid = self.game_state.grid

        # Identify HQ positions
        for row in grid.tiles:
            for tile in row:
                if tile.type == "h" and tile.player:
                    self.hq_positions[tile.player] = (tile.x, tile.y)

        # Identify defensive positions (mountains for Archer range bonus)
        self.defensive_positions = []
        # Identify forest positions (for Rogue evade bonus)
        self.forest_positions = []

        for row in grid.tiles:
            for tile in row:
                if tile.type == "m":  # Mountains
                    self.defensive_positions.append((tile.x, tile.y))
                elif tile.type == "f":  # Forests
                    self.forest_positions.append((tile.x, tile.y))

    def get_dynamic_composition_targets(self) -> Dict[str, float]:
        """Calculate target composition based on enabled units, then bias
        toward counters of the observed enemy composition."""
        # Filter to only enabled units
        enabled = self.get_enabled_units()
        enabled_targets = {k: v for k, v in self.FULL_COMPOSITION_TARGETS.items() if k in enabled}

        # Apply counter-composition bumps. For each enemy unit observed, walk
        # FULL_COUNTER_MATRIX and add the per-counter bump to our target.
        # Counters that aren't enabled (e.g. opponent locked us out of "K")
        # are silently skipped -- the renormalisation below absorbs them.
        enemy_counts = self.count_enemy_units_by_type()
        if enemy_counts:
            for enemy_type, count in enemy_counts.items():
                bumps = self.FULL_COUNTER_MATRIX.get(enemy_type, {})
                for counter_type, weight in bumps.items():
                    if counter_type in enabled_targets:
                        enabled_targets[counter_type] += weight * count

        # Renormalise to 1.0 first so any phase bias below operates on
        # comparable units.
        if enabled_targets:
            total_enabled = sum(enabled_targets.values())
            if total_enabled > 0:
                enabled_targets = {k: v / total_enabled for k, v in enabled_targets.items()}

        # EXPAND phase: floor the Warrior ratio so we win the early tempo
        # race. Other types are scaled proportionally to absorb the shift.
        # Skipped when phase is None (e.g. tests poking the bot before any
        # take_turn) or W is disabled.
        if self.phase == self.PHASE_EXPAND and "W" in enabled_targets:
            current_w = enabled_targets["W"]
            if current_w < self.EXPAND_WARRIOR_FLOOR:
                others_total = sum(v for k, v in enabled_targets.items() if k != "W")
                if others_total > 0:
                    scale = (1.0 - self.EXPAND_WARRIOR_FLOOR) / others_total
                    enabled_targets = {
                        k: (self.EXPAND_WARRIOR_FLOOR if k == "W" else v * scale) for k, v in enabled_targets.items()
                    }
                else:
                    enabled_targets = {"W": 1.0}

        return enabled_targets

    def should_retreat_to_heal(self, unit) -> bool:
        """Per-archetype retreat thresholds (frontline holds longer than ranged)."""
        threshold = self.RETREAT_THRESHOLDS.get(unit.type, self.RETREAT_HEALTH_THRESHOLD)
        return unit.health < unit.max_health * threshold

    def find_retreat_tile(self, unit):
        """Pick a heal tile that maximises HP recovered while minimising the
        number of enemies that can attack it next turn. Falls back to the
        MediumBot policy (heal amount, then proximity) when no enemies are
        within striking range of any candidate."""
        reachable = self.get_reachable(unit)
        if not reachable:
            return None

        candidates = [(unit.x, unit.y)] + list(reachable)
        enemies = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        best = None
        # Maximise (heal_amount, -threats_in_range, -distance).
        best_score = (0, 1, float("inf"))
        # Shuffle so equally-good heal positions (same heal, threats,
        # distance) tiebreak randomly under stochastic mode.
        self._maybe_shuffle(candidates)
        for x, y in candidates:
            heal = self.heal_amount_at(x, y)
            if heal == 0:
                continue
            # Approximate threat: enemies whose Manhattan distance to the tile
            # is within their max attack range. Cheap and avoids the cost of
            # simulating each enemy's reachable set.
            threats = 0
            for e in enemies:
                e_on_mountain = self.game_state.grid.get_tile(e.x, e.y).type == "m"
                _, e_max_range = e.get_attack_range(on_mountain=e_on_mountain)
                if self.manhattan_distance(x, y, e.x, e.y) <= e_max_range:
                    threats += 1
            distance = self.manhattan_distance(unit.x, unit.y, x, y)
            score = (heal, -threats, -distance)
            if score > best_score:
                best_score = score
                best = (x, y)
        return best

    def purchase_units_enhanced(self):
        """Enhanced unit purchasing with dynamic composition for all enabled units."""
        # Get dynamic composition targets based on enabled units
        target_ratios = self.get_dynamic_composition_targets()

        # Count existing unit types (only enabled ones)
        my_units = [u for u in self.game_state.units if u.player == self.bot_player]
        enabled = self.get_enabled_units()
        unit_counts = {ut: 0 for ut in enabled}
        for unit in my_units:
            if unit.type in unit_counts:
                unit_counts[unit.type] += 1

        total_units = len(my_units)

        # Enhanced composition: buy units to match target ratios
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions["create_unit"]
            # Note: legal_actions already filters by enabled_units

            if not create_actions:
                break

            available_gold = self.game_state.player_gold[self.bot_player]
            affordable_actions = [a for a in create_actions if UNIT_DATA[a["unit_type"]]["cost"] <= available_gold]

            if not affordable_actions:
                break

            # Find unit type most below its target ratio
            best_action = None
            best_priority = -float("inf")

            for action in affordable_actions:
                unit_type = action["unit_type"]
                if unit_type not in target_ratios:
                    continue

                # Calculate how far below target we are
                current_ratio = unit_counts.get(unit_type, 0) / max(1, total_units + 1)
                target_ratio = target_ratios[unit_type]
                priority = target_ratio - current_ratio

                # Support units (C, S) need at least 3 units before buying
                if unit_type in ["C", "S"] and total_units < 3:
                    priority = -1

                if priority > best_priority:
                    best_priority = priority
                    best_action = action

            if best_action and best_priority > -1:
                self.game_state.create_unit(best_action["unit_type"], best_action["x"], best_action["y"], self.bot_player)
                self._record(f"buy_{best_action['unit_type']}")
                unit_counts[best_action["unit_type"]] = unit_counts.get(best_action["unit_type"], 0) + 1
                total_units += 1
            else:
                # Fallback: buy any affordable unit
                if affordable_actions:
                    # Prefer cheaper units for economy. Shuffle so units
                    # at the same cost tier tiebreak randomly.
                    self._maybe_shuffle(affordable_actions)
                    affordable_actions.sort(key=lambda a: UNIT_DATA[a["unit_type"]]["cost"])
                    action = affordable_actions[0]
                    self.game_state.create_unit(action["unit_type"], action["x"], action["y"], self.bot_player)
                    self._record(f"buy_{action['unit_type']}")
                    self._record("buy_fallback")
                    unit_counts[action["unit_type"]] = unit_counts.get(action["unit_type"], 0) + 1
                    total_units += 1
                else:
                    break

    def move_and_act_units_enhanced(self):
        """Enhanced version of MediumBot's unit movement with special abilities."""
        # Get all bot units that can act
        bot_units = [
            u
            for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack) and not u.is_paralyzed()
        ]

        # Lock in any in-progress captures before any of the coordination
        # passes (Knight charge / focus-fire / per-unit) can drag those
        # units off their tower and forfeit the half-finished seize.
        self.continue_active_seizes(bot_units)

        # Step 1: Knights charge first so the bonus isn't burned on a chip-kill
        # in coordinated focus fire.
        if self.has_charge_units():
            for unit in bot_units:
                if unit.type != "K" or not (unit.can_move or unit.can_attack):
                    continue
                self._try_knight_charge(unit)

        # Step 2: Coordinated focus fire on remaining killable enemies.
        remaining = [u for u in bot_units if (u.can_move or u.can_attack)]
        self.coordinate_attacks(remaining)

        # Step 3: Per-unit enhanced acting for the rest.
        for unit in bot_units:
            if unit.can_move or unit.can_attack:
                self.act_with_unit_enhanced(unit)

    def act_with_unit_enhanced(self, unit, _depth=0):
        """Enhanced version of MediumBot's act_with_unit with superior tactics."""
        if _depth >= MAX_RECURSION_DEPTH:
            return
        if self.game_state.game_over:
            return

        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if tile.is_capturable() and tile.player != self.bot_player and tile.health < tile.max_health:
            self.game_state.seize(unit)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit_enhanced(unit, _depth + 1)
            return

        # Try special abilities first (Cleric heal, Mage paralyze, Sorcerer buffs)
        if self.try_use_special_ability(unit):
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit_enhanced(unit, _depth + 1)
            return

        # PRIORITY 1: Knight charge attack (move 3+ tiles for +50% damage)
        if unit.type == "K" and self.has_charge_units():
            if self._try_knight_charge(unit):
                return

        # PRIORITY 2: Rogue flank attack (+50% damage when target adjacent to ally)
        if unit.type == "R" and self.has_flank_units():
            if self._try_rogue_flank(unit):
                return

        # PRIORITY 3: Position Rogues in forests for evade bonus
        if unit.type == "R" and self.has_flank_units() and tile.type != "f":
            if self._try_rogue_forest_position(unit):
                return

        # PRIORITY 4: Interrupt enemy captures (don't double-up; coordinate
        # across units like MediumBot does).
        if not hasattr(self, "_interrupt_assigned"):
            self._interrupt_assigned = set()
        interrupt_assigned = self._interrupt_assigned
        contested = self.find_contested_structures()
        if contested:
            # Shuffle pre-sort so equal-progress interrupts tiebreak
            # randomly across episodes.
            self._maybe_shuffle(contested)
            contested.sort(key=lambda x: x[2], reverse=True)
            for _, enemy_unit, __ in contested:
                if id(enemy_unit) in interrupt_assigned:
                    continue
                # Direct attack first
                attackable = self.game_state.mechanics.get_attackable_enemies(unit, [enemy_unit], self.game_state.grid)
                if enemy_unit in attackable:
                    interrupt_assigned.add(id(enemy_unit))
                    self.game_state.attack(unit, enemy_unit)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit_enhanced(unit, _depth + 1)
                    return
                # Move-then-attack
                target_pos = self.find_best_move_position(unit, enemy_unit.x, enemy_unit.y)
                if target_pos:
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    attackable = self.game_state.mechanics.get_attackable_enemies(unit, [enemy_unit], self.game_state.grid)
                    if enemy_unit in attackable:
                        interrupt_assigned.add(id(enemy_unit))
                        self.game_state.attack(unit, enemy_unit)
                        if unit.can_move or unit.can_attack:
                            self.act_with_unit_enhanced(unit, _depth + 1)
                        return

        # PRIORITY 4.5: Retreat to heal when wounded, unless a finishing blow
        # is in range. Uses per-archetype thresholds and safety-aware tile
        # scoring (see find_retreat_tile override).
        if self.should_retreat_to_heal(unit):
            attackable_now = self.game_state.mechanics.get_attackable_enemies(
                unit,
                [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0],
                self.game_state.grid,
            )
            on_mountain = self.game_state.grid.get_tile(unit.x, unit.y).type == "m"
            can_finish = any(unit.get_attack_damage(e.x, e.y, on_mountain) >= e.health for e in attackable_now)
            if not can_finish and self.try_retreat_to_heal(unit):
                return

        # PRIORITY 4.6: Free capture. If a capture target is reachable this
        # turn (already standing on it, or can move-and-seize without a
        # multi-turn march), do it before combat. Captures compound -- each
        # adds income next turn -- and the previous Priority 8-only policy
        # let units sit trading attacks while free buildings sat next door,
        # which is the dominant AdvancedBot stalemate failure mode.
        #
        # In EXPAND phase we additionally accept multi-turn marches: walking
        # toward a distant capture beats parking next to one enemy and chip-
        # trading attacks while the rest of the map gets snapped up.
        target_structure = self.pick_capture_target(unit)
        if target_structure is not None:
            structure_pos = (target_structure.x, target_structure.y)
            already_on = (unit.x, unit.y) == structure_pos
            reachable_this_turn = already_on or structure_pos in set(self.get_reachable(unit))
            if reachable_this_turn:
                self._capture_assignments().add(structure_pos)
                if not already_on:
                    self.game_state.move_unit(unit, structure_pos[0], structure_pos[1])
                if (unit.x, unit.y) == structure_pos:
                    self.game_state.seize(unit)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit_enhanced(unit, _depth + 1)
                    return
            elif self.phase in (self.PHASE_EXPAND, self.PHASE_CONQUER):
                # EXPAND: walk the map to grab distant neutrals before the
                # enemy does. CONQUER: with neutrals exhausted, the only
                # remaining capture target is the enemy HQ/buildings -- a
                # multi-turn march toward them is exactly the push we want.
                self._capture_assignments().add(structure_pos)
                target_pos = self.find_best_move_position(unit, structure_pos[0], structure_pos[1])
                if target_pos is not None:
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    if (unit.x, unit.y) == structure_pos:
                        self.game_state.seize(unit)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit_enhanced(unit, _depth + 1)
                    return

        # PRIORITY 5: Position on mountains for attack bonus. Pre-evaluate
        # every reachable mountain (without committing the move) and pick
        # the one that yields an in-range enemy; otherwise skip entirely.
        if unit.type in ["W", "B", "A", "M", "K", "S"] and tile.type != "m":
            enemy_units = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]
            enemy_close = any(self.manhattan_distance(unit.x, unit.y, e.x, e.y) <= 4 for e in enemy_units)
            if enemy_close and enemy_units:
                reachable_mountains = [
                    pos for pos in self.get_reachable(unit) if self.game_state.grid.get_tile(pos[0], pos[1]).type == "m"
                ]
                # Find a mountain that puts an enemy in attack range without
                # committing the move first.
                chosen_mountain = None
                old_x, old_y = unit.x, unit.y
                try:
                    for pos in reachable_mountains:
                        unit.x, unit.y = pos[0], pos[1]
                        if self.game_state.mechanics.get_attackable_enemies(unit, enemy_units, self.game_state.grid):
                            chosen_mountain = pos
                            break
                finally:
                    unit.x, unit.y = old_x, old_y

                if chosen_mountain is not None:
                    self.game_state.move_unit(unit, chosen_mountain[0], chosen_mountain[1])
                    if self.try_ranged_attack(unit):
                        if unit.can_move or unit.can_attack:
                            self.act_with_unit_enhanced(unit, _depth + 1)
                        return

        # PRIORITY 6: Ranged attacks (Archers/Mages/Sorcerers should attack from range)
        if unit.type in self.get_enabled_ranged_units() and self.try_ranged_attack(unit):
            if unit.can_move or unit.can_attack:
                self.act_with_unit_enhanced(unit, _depth + 1)
            return

        # PRIORITY 7: Move to attack range and attack
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        if enemy_units:
            # Find closest attackable enemy. Shuffle the candidate
            # iteration order so equal-scoring targets tiebreak
            # randomly across episodes (the strict > below means the
            # first-visited equal-value enemy would otherwise always
            # win the tie).
            #
            # NOTE: the rebound ``enemy_units`` propagates further than
            # the strict-> loop -- the ``min(enemy_units, ...)`` call
            # below for the move-toward-nearest fallback also iterates
            # the shuffled list (Python min() returns the first item
            # on ties, so equidistant enemies tiebreak randomly too),
            # and the ``get_attackable_enemies(unit, enemy_units, ...)``
            # call after a move receives a shuffle-ordered input. Both
            # are consistent with the stochastic-tiebreak philosophy
            # but broader than the immediate sort below.
            enemy_units = self._maybe_shuffle(list(enemy_units))
            best_target = None
            best_score = -float("inf")

            for enemy in enemy_units:
                attackable = self.game_state.mechanics.get_attackable_enemies(unit, [enemy], self.game_state.grid)

                if enemy in attackable:
                    value = self.calculate_attack_value(unit, enemy)
                    # Bonus for killing
                    on_mountain = self.game_state.grid.get_tile(unit.x, unit.y).type == "m"
                    damage = unit.get_attack_damage(enemy.x, enemy.y, on_mountain)
                    if damage >= enemy.health:
                        value += 500

                    if value > best_score:
                        best_score = value
                        best_target = enemy

            if self.phase == self.PHASE_EXPAND:
                attack_threshold = self.EXPAND_ATTACK_THRESHOLD
            elif self.phase == self.PHASE_CONQUER:
                attack_threshold = self.CONQUER_ATTACK_THRESHOLD
            else:
                attack_threshold = self.CONSOLIDATE_ATTACK_THRESHOLD
            if best_target and best_score > attack_threshold:
                self.game_state.attack(unit, best_target)
                if unit.can_move or unit.can_attack:
                    self.act_with_unit_enhanced(unit, _depth + 1)
                return

            # Try to move towards nearest enemy
            nearest_enemy = min(enemy_units, key=lambda e: self.manhattan_distance(unit.x, unit.y, e.x, e.y))
            target_pos = self.find_best_move_position(unit, nearest_enemy.x, nearest_enemy.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                attackable_after = self.game_state.mechanics.get_attackable_enemies(unit, enemy_units, self.game_state.grid)
                if attackable_after:
                    # Python's max() returns the first item on ties --
                    # shuffle the candidate list so equal-value enemies
                    # tiebreak randomly across episodes.
                    best_after_move = max(
                        self._maybe_shuffle(list(attackable_after)),
                        key=lambda e: self.calculate_attack_value(unit, e),
                    )
                    self.game_state.attack(unit, best_after_move)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit_enhanced(unit, _depth + 1)
                    return

        # PRIORITY 8: Capture structures (fallback). Per-unit assignment via
        # pick_capture_target so each unit picks its closest unclaimed target.
        target_structure = self.pick_capture_target(unit)
        if target_structure is not None:
            self._capture_assignments().add((target_structure.x, target_structure.y))

            if unit.x == target_structure.x and unit.y == target_structure.y:
                self.game_state.seize(unit)
                # Check if unit can act again (haste)
                if unit.can_move or unit.can_attack:
                    self.act_with_unit_enhanced(unit, _depth + 1)
                return

            target_pos = self.find_best_move_position(unit, target_structure.x, target_structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                if unit.x == target_structure.x and unit.y == target_structure.y:
                    self.game_state.seize(unit)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit_enhanced(unit, _depth + 1)
                    return

        # Fallback: End turn
        can_still_act = unit.end_unit_turn()
        if can_still_act:
            self.act_with_unit_enhanced(unit, _depth + 1)

    def _try_knight_charge(self, unit) -> bool:
        """Attempt Knight charge attack for +50% damage (requires 3+ tile move)."""
        if unit.type != "K" or not self.has_charge_units():
            return False
        if self.game_state.game_over:
            return False

        enemies = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        if not enemies:
            return False

        # Get all reachable positions
        reachable = self.get_reachable(unit)

        best_charge = None
        best_value = -float("inf")

        for pos in reachable:
            # Calculate move distance
            move_distance = self.manhattan_distance(unit.x, unit.y, pos[0], pos[1])
            if move_distance < CHARGE_MIN_DISTANCE:
                continue  # Not a valid charge

            # Check if we can attack any enemy from this position
            for enemy in enemies:
                attack_distance = self.manhattan_distance(pos[0], pos[1], enemy.x, enemy.y)
                if attack_distance == 1:  # Knight is melee
                    # Calculate value with charge bonus
                    value = self.calculate_attack_value(unit, enemy, move_distance)
                    if value > best_value:
                        best_value = value
                        best_charge = (pos, enemy, move_distance)

        if best_charge and best_value > 0:
            pos, enemy, _ = best_charge
            self.game_state.move_unit(unit, pos[0], pos[1])
            self.game_state.attack(unit, enemy)
            self._record("knight_charge")
            return True

        return False

    def _try_rogue_flank(self, unit) -> bool:
        """Attempt Rogue flank attack for +50% damage (target adjacent to ally)."""
        if unit.type != "R" or not self.has_flank_units():
            return False

        # Find enemies that can be flanked
        flankable = self._find_flank_targets(unit)
        if not flankable:
            return False

        # Check if we can attack any flankable target directly
        attackable = self.game_state.mechanics.get_attackable_enemies(unit, flankable, self.game_state.grid)

        if attackable:
            # Attack the highest value flankable target. Shuffle so
            # equal-value flank candidates tiebreak randomly under
            # stochastic mode (max() picks the first on ties).
            best_target = max(
                self._maybe_shuffle(list(attackable)),
                key=lambda e: self.calculate_attack_value(unit, e),
            )
            self.game_state.attack(unit, best_target)
            self._record("rogue_flank")
            return True

        # Try to move to flank position
        reachable = self.get_reachable(unit)

        best_flank_pos = None
        best_target = None
        best_value = -float("inf")

        for pos in reachable:
            for enemy in flankable:
                # Check if we can attack enemy from this position (distance 1 for melee)
                if self.manhattan_distance(pos[0], pos[1], enemy.x, enemy.y) == 1:
                    value = self.calculate_attack_value(unit, enemy)
                    # Bonus for forest positions (evade bonus)
                    tile = self.game_state.grid.get_tile(pos[0], pos[1])
                    if tile.type == "f":
                        value += 50  # Forest bonus

                    if value > best_value:
                        best_value = value
                        best_flank_pos = pos
                        best_target = enemy

        if best_flank_pos and best_target:
            self.game_state.move_unit(unit, best_flank_pos[0], best_flank_pos[1])
            self.game_state.attack(unit, best_target)
            self._record("rogue_flank")
            return True

        return False

    def _try_rogue_forest_position(self, unit) -> bool:
        """Try to position Rogue in a forest for evade bonus."""
        if unit.type != "R" or not self.has_flank_units():
            return False

        # Already in forest
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if tile.type == "f":
            return False

        # Check if there are nearby enemies (only position if combat expected)
        enemies = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]
        if not enemies:
            return False

        nearest_enemy_dist = min(self.manhattan_distance(unit.x, unit.y, e.x, e.y) for e in enemies)
        if nearest_enemy_dist > 5:  # Only position if enemy is close
            return False

        # Find reachable forest positions
        reachable = self.get_reachable(unit)

        # Find best forest position (closest to enemies)
        best_forest = None
        best_dist = float("inf")

        for pos in reachable:
            forest_tile = self.game_state.grid.get_tile(pos[0], pos[1])
            if forest_tile.type == "f":
                # Find distance to nearest enemy from this forest
                min_enemy_dist = min(self.manhattan_distance(pos[0], pos[1], e.x, e.y) for e in enemies)
                if min_enemy_dist < best_dist:
                    best_dist = min_enemy_dist
                    best_forest = pos

        if best_forest:
            self.game_state.move_unit(unit, best_forest[0], best_forest[1])
            self._record("rogue_forest_position")
            # Try to attack after moving to forest. Shuffle so
            # equal-value attackable enemies tiebreak randomly.
            attackable = self.game_state.mechanics.get_attackable_enemies(unit, enemies, self.game_state.grid)
            if attackable:
                best_target = max(
                    self._maybe_shuffle(list(attackable)),
                    key=lambda e: self.calculate_attack_value(unit, e),
                )
                self.game_state.attack(unit, best_target)
            return True

        return False

    def try_use_special_ability(self, unit):
        """Try to use unit special abilities effectively."""
        # Mage Paralyze: prefer locking down a unit currently capturing one of
        # our structures; fall back to the shared mage paralyze heuristic.
        if unit.type == "M" and self.has_paralyze_units() and unit.can_attack and unit.can_use_paralyze():
            # First-match-wins iteration over self.game_state.units biases
            # toward earlier-spawned enemies. Shuffle so among multiple
            # in-range capturing enemies, the paralyze pick tiebreaks
            # randomly under stochastic mode.
            candidates = self._maybe_shuffle(list(self.game_state.units))
            for enemy in candidates:
                if enemy.player == self.bot_player or enemy.health <= 0 or enemy.is_paralyzed():
                    continue
                if not self._is_capturing_us(enemy):
                    continue
                dist = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
                if 1 <= dist <= 2:
                    self.game_state.paralyze(unit, enemy)
                    return True
            if self.try_mage_paralyze(unit):
                return True

        # Cleric Heal (only if Clerics enabled). Use the shared helper which
        # also cures paralysis and prefers frontline targets.
        if unit.type == "C" and self.has_heal_units():
            if self.try_cleric_abilities(unit):
                return True

        # Sorcerer abilities (only if Sorcerers enabled)
        if unit.type == "S" and self.has_buff_units():
            if self._try_sorcerer_abilities(unit):
                return True

        return False

    def _try_sorcerer_abilities(self, unit) -> bool:
        """Use Sorcerer abilities strategically (Haste, Attack Buff, Defence Buff)."""
        if unit.type != "S" or not self.has_buff_units():
            return False

        allies = [u for u in self.game_state.units if u.player == self.bot_player and u != unit]
        enemies = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        if not allies:
            return False

        # Check range (Sorcerer buffs have range 0-2)
        def in_buff_range(target):
            dist = self.manhattan_distance(unit.x, unit.y, target.x, target.y)
            return dist <= 2

        allies_in_range = [a for a in allies if in_buff_range(a)]
        if not allies_in_range:
            return False

        can_haste = unit.can_use_haste()
        can_attack_buff = unit.can_use_attack_buff()
        can_defence_buff = unit.can_use_defence_buff()

        # Priority 1: Haste a Knight that can charge (if Knight is enabled)
        if can_haste and self.has_charge_units():
            knights_in_range = [a for a in allies_in_range if a.type == "K" and a.can_attack and not a.is_hasted]
            for knight in knights_in_range:
                # Check if knight has a potential charge target
                for enemy in enemies:
                    dist = self.manhattan_distance(knight.x, knight.y, enemy.x, enemy.y)
                    if CHARGE_MIN_DISTANCE <= dist <= knight.movement_range + 1:
                        self.game_state.haste(unit, knight)
                        self._record("sorcerer_haste")
                        self._record("sorcerer_haste_knight_charge")
                        return True

        # Priority 2: Haste a Rogue for flank opportunity (if Rogue is enabled)
        if can_haste and self.has_flank_units():
            rogues_in_range = [a for a in allies_in_range if a.type == "R" and a.can_attack and not a.is_hasted]
            for rogue in rogues_in_range:
                if self._find_flank_targets(rogue):
                    self.game_state.haste(unit, rogue)
                    self._record("sorcerer_haste")
                    self._record("sorcerer_haste_rogue_flank")
                    return True

        # Priority 3: Attack buff on frontline unit about to engage
        if can_attack_buff and enemies:
            frontline_in_range = [
                a for a in allies_in_range if a.type in ("W", "K", "B", "R") and a.can_attack and not a.has_attack_buff()
            ]
            if frontline_in_range:
                # Equidistant frontline allies tiebreak randomly under
                # stochastic mode.
                self._maybe_shuffle(frontline_in_range)
                best_frontline = min(
                    frontline_in_range,
                    key=lambda a: min(self.manhattan_distance(a.x, a.y, e.x, e.y) for e in enemies),
                )
                self.game_state.attack_buff(unit, best_frontline)
                self._record("sorcerer_attack_buff")
                return True

        # Priority 4: Defence buff on unit capturing contested structure.
        # Shuffle so multiple contested-capturing allies tiebreak randomly.
        if can_defence_buff:
            for ally in self._maybe_shuffle(list(allies_in_range)):
                tile = self.game_state.grid.get_tile(ally.x, ally.y)
                if tile.is_capturable() and tile.health < tile.max_health and not ally.has_defence_buff():
                    self.game_state.defence_buff(unit, ally)
                    self._record("sorcerer_defence_buff")
                    self._record("sorcerer_defence_buff_capturing")
                    return True

            # Priority 5: Defence buff on low-health frontline unit
            low_health_frontline = [
                a
                for a in allies_in_range
                if a.type in ("W", "K", "B") and a.health < a.max_health * 0.5 and not a.has_defence_buff()
            ]
            if low_health_frontline:
                self._maybe_shuffle(low_health_frontline)
                target = min(low_health_frontline, key=lambda a: a.health)
                self.game_state.defence_buff(unit, target)
                self._record("sorcerer_defence_buff")
                self._record("sorcerer_defence_buff_low_hp")
                return True

        return False

    def try_ranged_attack(self, unit):
        """Try to use ranged attacks to minimize counter-attack damage."""
        # Ranged units: Archer (A), Mage (M), Sorcerer (S)
        if unit.type not in ["A", "M", "S"]:
            return False

        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        attackable = self.game_state.mechanics.get_attackable_enemies(unit, enemy_units, self.game_state.grid)

        if not attackable:
            return False

        # Prioritize melee units (they can't counter-attack ranged attacks)
        # Ranged units that can counter: A, M, S
        melee_targets = [e for e in attackable if e.type not in ["A", "M", "S"]]

        if melee_targets:
            # Attack the one with lowest health (finish off). Equal-HP
            # targets tiebreak randomly under stochastic mode.
            self._maybe_shuffle(melee_targets)
            target = min(melee_targets, key=lambda e: e.health)
            self.game_state.attack(unit, target)
            return True

        # Otherwise attack any target
        attackable_list = self._maybe_shuffle(list(attackable))
        target = min(attackable_list, key=lambda e: e.health)
        self.game_state.attack(unit, target)
        return True


class MasterBot(AdvancedBot):
    """Master-tier scripted bot built on AdvancedBot's strategy layer.

    MasterBot keeps every AdvancedBot heuristic (game-phase state machine,
    weighted counter matrix, per-archetype retreat thresholds, Knight
    charge / Rogue flank / mountain positioning / Sorcerer ability
    orchestration) and layers targeted upgrades on top:

    1. Threat map (``_compute_threat_map``). Per-tile aggregate of expected
       incoming damage if we stood there next turn -- the sum of attack
       values across enemies whose own move-and-attack reach covers the
       tile. Cached per turn. Used by retreat-tile selection
       (``find_retreat_tile`` override) and Knight charge landing
       selection (``_try_knight_charge`` override) so we don't pull back
       into a crossfire or leap onto a suicide tile.

       Crucially, the threat map does NOT bias forward movement
       (``find_best_move_position``) or attack value
       (``calculate_attack_value``). Earlier iterations did and the bot
       refused to engage -- every tile near enemies looks ''threatened'',
       so adding a safety penalty everywhere just biased away from
       fighting. The fix is to use threat only for decisions where there
       is a genuine choice between equivalent tiles (retreat, charge
       landing) rather than for go/no-go decisions about attacking.

    2. HP-ascending focus fire (``coordinate_attacks`` override). The
       parent picks ``attackers_needed`` biggest-hitter-first to confirm
       a kill, then sends them in that order. Master keeps the same
       selection but executes lowest-HP first: a wounded attacker that
       can still land one hit should use it now, before a target's
       counter would have killed it on a later swing of the chain.

    3. HQ-snipe priority (``pick_capture_target`` override). In CONQUER
       phase the enemy HQ takes priority over a same-distance tower so
       every unit actively pushes the win condition rather than fighting
       on the way.

    4. Haste followthrough (``move_and_act_units_enhanced`` post-pass +
       expanded ``_try_sorcerer_abilities``). The base flow leaves haste
       *unused* whenever its target took action via an early-return
       branch (seize, attack, charge, flank), because those branches
       zero ``can_move``/``can_attack`` but don't consume ``is_hasted``
       -- only the fallback ``end_unit_turn`` call site does. Master
       explicitly re-runs ``act_with_unit_enhanced`` on still-hasted
       allies after the main pass, enabling:
        - **Double capture**: warrior seizes structure A, then the haste
          refresh moves it to and seizes structure B in the same turn.
        - **Cross-map mobility**: Barbarian (movement=5) hasted -> two
          5-tile moves -> reaches a distant neutral that would
          otherwise take two turns to claim.
       ``_try_sorcerer_abilities`` is extended to deliberately set up
       these patterns: it hastes (a) Knights with a kill-confirming
       charge, (b) allies with two reachable capturables this turn, and
       (c) high-mobility units with a capture target inside the
       haste-doubled envelope.

    Full ability coverage is inherited from AdvancedBot's
    ``try_use_special_ability`` and ``move_and_act_units_enhanced``:
    Knight charge (Step 1 pre-pass + ``_try_knight_charge``), Rogue flank
    (``_try_rogue_flank``) and forest evade (``_try_rogue_forest_position``),
    Archer mountain range (PRIORITY 5 mountain positioning +
    ``try_ranged_attack``), Mage paralyze (``try_use_special_ability`` +
    ``try_mage_paralyze``), Cleric heal/cure (``try_cleric_abilities``),
    Sorcerer haste/attack-buff/defence-buff (``_try_sorcerer_abilities``),
    and Barbarian high-mobility capture (the EXPAND-phase multi-turn-march
    branch in ``act_with_unit_enhanced`` Priority 4.6).
    """

    # Cap on the number of attackers to consider reordering per target.
    # We don't permute all orderings (parent's biggest-hitter-first is
    # usually optimal); we just check the reverse as a safety swap.
    MAX_PERMUTE_ATTACKERS = 6

    def __init__(self, game_state, player=2, rng=None):
        super().__init__(game_state, player, rng=rng)
        # Per-turn cached threat map. Rebuilt in take_turn before any move
        # decisions. dict[(x, y)] -> float damage.
        self._threat_map: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Threat map
    # ------------------------------------------------------------------
    def _enemy_base_damage(self, enemy) -> float:
        """Best-case attack damage this enemy could deal (used as the
        threat contribution to tiles it can reach-and-attack). For ranged
        casters with split adjacent/range values we take the larger."""
        ad = enemy.attack_data
        if isinstance(ad, dict):
            return float(max(ad.get("adjacent", 0), ad.get("range", 0)))
        return float(ad)

    def _enemy_reachable_positions(self, enemy) -> List[Tuple[int, int]]:
        """Reachable positions for an *enemy* unit. Mirrors get_reachable
        but uses the enemy as the moving unit so the move-validity check
        is correct (our units block, theirs pass through allies, etc.)."""
        return enemy.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units, moving_unit=enemy, is_destination=False
            ),
        )

    def _compute_threat_map(self) -> Dict[Tuple[int, int], float]:
        """Build a tile -> incoming damage map.

        For each living, non-paralyzed enemy unit, we union every tile it
        could attack this turn (move to any reachable position, then
        strike any tile within its attack range). Each such tile gets that
        enemy's base damage added once -- summing across enemies so a tile
        in three Archers' range scores 3x.
        """
        threat: Dict[Tuple[int, int], float] = {}
        width = self.game_state.grid.width
        height = self.game_state.grid.height

        enemies = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0 and not u.is_paralyzed()]

        for enemy in enemies:
            base_damage = self._enemy_base_damage(enemy)
            if base_damage <= 0:
                continue
            positions: List[Tuple[int, int]] = [(enemy.x, enemy.y)]
            positions.extend(self._enemy_reachable_positions(enemy))
            attackable_tiles: set = set()
            for ex, ey in positions:
                tile = self.game_state.grid.get_tile(ex, ey)
                on_mountain = tile is not None and tile.type == "m"
                min_r, max_r = enemy.get_attack_range(on_mountain=on_mountain)
                for dx in range(-max_r, max_r + 1):
                    abs_dx = abs(dx)
                    if abs_dx > max_r:
                        continue
                    for dy in range(-max_r, max_r + 1):
                        dist = abs_dx + abs(dy)
                        if not (min_r <= dist <= max_r):
                            continue
                        tx, ty = ex + dx, ey + dy
                        if 0 <= tx < width and 0 <= ty < height:
                            attackable_tiles.add((tx, ty))
            for pos in attackable_tiles:
                threat[pos] = threat.get(pos, 0.0) + base_damage

        return threat

    def threat_at(self, x: int, y: int, exclude_enemy=None) -> float:
        """Aggregate incoming damage at (x, y).

        ``exclude_enemy`` lets two-ply attack value subtract the target's
        own contribution -- we already count the target's direct counter
        in ``calculate_attack_value``, so we don't want to double-count it
        as neighbour retaliation. Cheap to recompute without that enemy
        because we just rerun the inner loop on it.
        """
        base = self._threat_map.get((x, y), 0.0)
        if exclude_enemy is None or exclude_enemy.health <= 0:
            return base
        # Remove ``exclude_enemy``'s contribution if it could hit (x, y).
        base_damage = self._enemy_base_damage(exclude_enemy)
        if base_damage <= 0:
            return base
        positions: List[Tuple[int, int]] = [(exclude_enemy.x, exclude_enemy.y)]
        positions.extend(self._enemy_reachable_positions(exclude_enemy))
        for ex, ey in positions:
            tile = self.game_state.grid.get_tile(ex, ey)
            on_mountain = tile is not None and tile.type == "m"
            min_r, max_r = exclude_enemy.get_attack_range(on_mountain=on_mountain)
            d = self.manhattan_distance(ex, ey, x, y)
            if min_r <= d <= max_r:
                return max(0.0, base - base_damage)
        return base

    # ------------------------------------------------------------------
    # take_turn: rebuild threat map at the top of every turn
    # ------------------------------------------------------------------
    def take_turn(self):
        # Build the threat map *before* purchase/movement so every override
        # below sees a fresh snapshot. We rebuild once per turn -- not after
        # every move -- because the cost would dominate otherwise. Slight
        # staleness is fine: our own movement only changes who *we* threaten,
        # not who threatens us, and enemy positions are fixed during our turn.
        self._threat_map = self._compute_threat_map()
        super().take_turn()

    # ------------------------------------------------------------------
    # Threat-aware retreat tile selection
    # ------------------------------------------------------------------
    def find_retreat_tile(self, unit):
        """Pick a heal tile that maximises HP recovered while minimising the
        *actual damage* an enemy could land on it next turn.

        AdvancedBot's version counts the *number* of enemies in striking
        range. Master's threat map already has aggregate expected damage
        per tile, so we use that directly -- a tile in range of a single
        Knight (10 dmg) is worse than one in range of two Clerics (4
        dmg combined). Same scoring shape as the parent so behaviour stays
        comparable when no heal-tile candidates differ in threat.
        """
        reachable = self.get_reachable(unit)
        if not reachable:
            return None

        candidates = [(unit.x, unit.y)] + list(reachable)

        best = None
        # Maximise (heal_amount, -threat_damage, -distance).
        best_score = (0, 1, float("inf"))
        # Shuffle so equally-good heal tiles tiebreak randomly under
        # stochastic mode (strict > below otherwise hard-prefers the
        # first-visited candidate).
        self._maybe_shuffle(candidates)
        for x, y in candidates:
            heal = self.heal_amount_at(x, y)
            if heal == 0:
                continue
            threat = self._threat_map.get((x, y), 0.0)
            distance = self.manhattan_distance(unit.x, unit.y, x, y)
            score = (heal, -threat, -distance)
            if score > best_score:
                best_score = score
                best = (x, y)
        return best

    # ------------------------------------------------------------------
    # Focus fire: send the lowest-HP attacker first so a chipped unit
    # uses its remaining HP for its hit rather than wasting it absorbing
    # the first counter-attack.
    # ------------------------------------------------------------------
    def coordinate_attacks(self, bot_units):
        """Confirm kills, then send the squishiest attacker in first.

        Same replan-after-each-kill loop as MediumBot.coordinate_attacks
        (see that method for why), but executes each focus-fire group
        with attackers sorted ascending by current HP so a chipped unit
        spends its remaining HP on a hit rather than absorbing the first
        counter-attack.
        """

        def target_priority(item):
            enemy, attackers = item
            tile = self.game_state.grid.get_tile(enemy.x, enemy.y)
            if tile.is_capturable() and tile.player != self.bot_player and tile.health < tile.max_health:
                return (0,)
            cost = UNIT_DATA[enemy.type]["cost"]
            return (1, -cost, len(attackers))

        attempted: set = set()
        while True:
            if self.game_state.game_over:
                return
            available = [u for u in bot_units if u.health > 0 and (u.can_move or u.can_attack)]
            if not available:
                return
            killable = self.find_killable_targets(available)
            killable = [(e, a) for e, a in killable if e.health > 0 and id(e) not in attempted]
            if not killable:
                return
            # Equal-priority killable targets tiebreak randomly under
            # stochastic mode.
            self._maybe_shuffle(killable)
            killable.sort(key=target_priority)
            enemy, attackers = killable[0]
            attempted.add(id(enemy))
            # Hp-asc execution: chipped attackers swing first.
            # Shuffle so equal-HP attackers queue up in random order
            # across episodes.
            live_attackers = [a for a in attackers if a.health > 0]
            self._maybe_shuffle(live_attackers)
            ordered = sorted(live_attackers, key=lambda a: a.health)
            self._execute_focus_fire(enemy, ordered)

    # ------------------------------------------------------------------
    # HQ-snipe / capture targeting refinement
    # ------------------------------------------------------------------
    def pick_capture_target(self, unit):
        """Same shape as MediumBot.pick_capture_target but with an
        enemy-HQ bonus in CONQUER phase. The HQ is the only structure that
        wins the game outright, so when neutrals are exhausted we want
        every unit pushing toward it instead of fighting on the way."""
        target = super().pick_capture_target(unit)
        if target is None or self.phase != self.PHASE_CONQUER:
            return target

        # In CONQUER we also explicitly check the enemy HQ; if it's
        # reachable to *this unit's path planner* we prefer it over a
        # closer tower. AdvancedBot's distance-dominant scoring usually
        # picks it anyway, but only when it's literally closest.
        enemy_hq = None
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type == "h" and tile.player is not None and tile.player != self.bot_player:
                    enemy_hq = tile
                    break
            if enemy_hq:
                break
        if enemy_hq is None:
            return target
        if (enemy_hq.x, enemy_hq.y) in self._capture_assignments():
            return target
        # Only steer the swap if the unit is closer to HQ than to its
        # current pick (so a unit guarding a flank doesn't suddenly march
        # across the map).
        d_hq = self.manhattan_distance(unit.x, unit.y, enemy_hq.x, enemy_hq.y)
        d_pick = self.manhattan_distance(unit.x, unit.y, target.x, target.y)
        if d_hq <= d_pick + 2:
            self._record("hq_snipe")
            return enemy_hq
        return target

    # ------------------------------------------------------------------
    # move_and_act post-pass: consume leftover haste.
    #
    # The base flow leaves haste *unused* when its target took action via
    # any early-return branch (seize, attack, charge, flank ...). Each of
    # those branches zeros ``can_move``/``can_attack`` but leaves
    # ``is_hasted=True``; the fallback ``unit.end_unit_turn()`` is the
    # only call site that consumes haste and refreshes action flags. So
    # without intervention, a hasted unit that captured / killed / etc.
    # silently loses its second action when the turn ends and the engine
    # wipes ``is_hasted`` (see GameState.end_turn).
    #
    # We mop that up by re-running ``act_with_unit_enhanced`` on every
    # still-hasted ally after the main pass. This enables two patterns
    # that are otherwise impossible:
    #   - **Double capture**: Warrior moves+seizes structure A on action
    #     1; post-pass refreshes flags; action 2 moves+seizes structure B.
    #   - **Cross-map mobility**: Barbarian (movement=5) hasted -> two
    #     5-tile moves in one turn, e.g. reaching a distant neutral that
    #     would otherwise take two turns to claim.
    # ------------------------------------------------------------------
    def move_and_act_units_enhanced(self):
        super().move_and_act_units_enhanced()
        if self.game_state.game_over:
            return
        # ``list(...)`` because act_with_unit_enhanced may invoke seize()
        # which mutates the tile state read by other passes; the unit list
        # itself is stable mid-turn but we snapshot for safety.
        for unit in list(self.game_state.units):
            if self.game_state.game_over:
                return
            if unit.player != self.bot_player:
                continue
            if not unit.is_hasted:
                continue
            # end_unit_turn(force_end=False) sees is_hasted=True, refreshes
            # can_move/can_attack to True, sets is_hasted=False, and
            # returns True. If the caller already consumed haste (e.g. a
            # recursion path inside the parent), is_hasted is already
            # False and we wouldn't have entered this branch.
            if unit.end_unit_turn(force_end=False):
                self._record("haste_followthrough")
                self.act_with_unit_enhanced(unit)

    # ------------------------------------------------------------------
    # Sorcerer: extend AdvancedBot's combo list with three more haste
    # targets the parent policy misses entirely:
    #
    # 1. Knight charge-and-kill (existing, kept).
    # 2. Double-capture chain: an ally that can reach two distinct
    #    capturable structures this turn -- one to seize now (via the
    #    parent priority ladder), one to seize after the haste refresh.
    # 3. Long-range Barbarian (or other mover) -- a unit with a capture
    #    target outside its one-turn reach but inside its haste-doubled
    #    reach (~2 * movement_range tiles). Lets us snap distant
    #    neutrals in one turn that would otherwise take two.
    # ------------------------------------------------------------------
    def _try_sorcerer_abilities(self, unit) -> bool:
        if super()._try_sorcerer_abilities(unit):
            return True
        if unit.type != "S" or not self.has_buff_units() or not unit.can_use_haste():
            return False

        # Candidates: living, controllable, unhasted allies in haste range.
        candidates = [
            a
            for a in self.game_state.units
            if a.player == self.bot_player
            and a is not unit
            and a.health > 0
            and a.can_move
            and not a.is_hasted
            and not a.is_paralyzed()
            and self.manhattan_distance(unit.x, unit.y, a.x, a.y) <= 2
        ]
        if not candidates:
            return False

        # Combo 1: Knight charge-and-kill (kept from previous version).
        for k in [a for a in candidates if a.type == "K" and a.can_attack]:
            reachable = self.get_reachable(k)
            for pos in reachable:
                if self.manhattan_distance(k.x, k.y, pos[0], pos[1]) < CHARGE_MIN_DISTANCE:
                    continue
                for e in self.game_state.units:
                    if e.player == self.bot_player or e.health <= 0:
                        continue
                    if self.manhattan_distance(pos[0], pos[1], e.x, e.y) != 1:
                        continue
                    a_tile = self.game_state.grid.get_tile(pos[0], pos[1])
                    on_mountain = a_tile is not None and a_tile.type == "m"
                    damage = int(k.get_attack_damage(e.x, e.y, on_mountain) * (1 + CHARGE_BONUS))
                    if damage >= e.health:
                        self.game_state.haste(unit, k)
                        self._record("sorcerer_haste")
                        self._record("master_haste_combo_knight_kill")
                        return True

        # Combo 2: double-capture. Find an ally that can move-and-seize
        # one capturable now, and whose post-seize position has at least
        # one *other* unclaimed capturable inside its movement range so
        # the haste refresh enables a second seize on the same turn.
        capturables = [
            tile
            for row in self.game_state.grid.tiles
            for tile in row
            if tile.is_capturable() and tile.player != self.bot_player
        ]
        if len(capturables) >= 2:
            for ally in candidates:
                reachable = set(self.get_reachable(ally))
                reachable.add((ally.x, ally.y))
                # First capture: pick the cheapest reachable capturable.
                # Equidistant capturables tiebreak randomly under
                # stochastic mode.
                first_options = [c for c in capturables if (c.x, c.y) in reachable]
                if not first_options:
                    continue
                self._maybe_shuffle(first_options)
                first = min(
                    first_options,
                    key=lambda t: self.manhattan_distance(ally.x, ally.y, t.x, t.y),
                )
                # After seizing ``first``, the ally stands on ``first``'s
                # tile with fresh movement. A second capture must be
                # within the ally's movement_range from ``first``.
                for second in capturables:
                    if second is first:
                        continue
                    if self.manhattan_distance(first.x, first.y, second.x, second.y) <= ally.movement_range:
                        self.game_state.haste(unit, ally)
                        self._record("sorcerer_haste")
                        self._record("master_haste_combo_double_capture")
                        return True

        # Combo 3: long-mobility ally with a distant capture. Pick a
        # capturable that the ally can't reach this turn but CAN reach
        # with a haste-doubled move. Prioritise high-movement units
        # (Barbarian=5, Knight=4, Rogue=4) where the doubled reach is
        # most game-changing.
        if capturables:
            # Shuffle pre-sort so allies with the same movement_range
            # tiebreak randomly across episodes.
            mobility_candidates = [a for a in candidates if a.movement_range >= 3]
            self._maybe_shuffle(mobility_candidates)
            high_mobility = sorted(
                mobility_candidates,
                key=lambda a: -a.movement_range,
            )
            for ally in high_mobility:
                reachable_now = set(self.get_reachable(ally))
                for tile in capturables:
                    pos = (tile.x, tile.y)
                    if pos in reachable_now:
                        continue
                    # Within the haste-doubled envelope (approximated as
                    # 2x movement_range -- the actual reachable set after
                    # one move would require simulating the move, which
                    # is too expensive here; this is a conservative
                    # upper bound that the post-pass will validate).
                    if self.manhattan_distance(ally.x, ally.y, tile.x, tile.y) <= 2 * ally.movement_range:
                        self.game_state.haste(unit, ally)
                        self._record("sorcerer_haste")
                        self._record("master_haste_combo_distant_capture")
                        return True

        return False
