"""Tests for GameState class, specifically win conditions."""

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.unit import Unit


@pytest.fixture
def simple_game():
    """Create a simple game state for testing."""
    map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[0][0] = "h_1"  # HQ for player 1
    map_data[9][9] = "h_2"  # HQ for player 2
    return GameState(map_data, num_players=2)


@pytest.fixture
def game_with_building():
    """Create a game state with a building for unit creation tests."""
    map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[0][0] = "h_1"  # HQ for player 1
    map_data[1][1] = "b_1"  # Building for player 1
    map_data[9][9] = "h_2"  # HQ for player 2
    return GameState(map_data, num_players=2)


class TestUnitCreationRestrictions:
    """Test that unit creation is restricted to Buildings only, not HQ."""

    def test_hq_cannot_create_units(self, simple_game):
        """Test that HQ (headquarters) cannot create units."""
        # Player 1 has HQ at (0, 0) with enough gold
        simple_game.player_gold[1] = 500

        # Get legal actions for player 1
        legal_actions = simple_game.get_legal_actions(player=1)

        # No create_unit actions should be available (no Buildings, only HQ)
        assert len(legal_actions["create_unit"]) == 0

    def test_building_can_create_units(self, game_with_building):
        """Test that Buildings can create units."""
        # Player 1 has Building at (1, 1) with enough gold
        game_with_building.player_gold[1] = 500

        # Get legal actions for player 1
        legal_actions = game_with_building.get_legal_actions(player=1)

        # Create unit actions should be available at the Building
        assert len(legal_actions["create_unit"]) > 0

        # All create_unit actions should be at the Building location (1, 1), not HQ (0, 0)
        for action in legal_actions["create_unit"]:
            assert action["x"] == 1 and action["y"] == 1, "Unit creation should only be at Building (1,1), not HQ (0,0)"

    def test_hq_not_in_legal_create_actions(self, game_with_building):
        """Test that HQ location is never in create_unit legal actions."""
        # Player 1 has both HQ at (0, 0) and Building at (1, 1)
        game_with_building.player_gold[1] = 500

        # Get legal actions
        legal_actions = game_with_building.get_legal_actions(player=1)

        # Check that no action has HQ coordinates
        hq_actions = [action for action in legal_actions["create_unit"] if action["x"] == 0 and action["y"] == 0]
        assert len(hq_actions) == 0, "HQ should not allow unit creation"


class TestLegalActionSanity:
    """Legal-action enumeration should not emit nonsensical actions.

    Covers two fixes: (1) paralyze must not target an already-paralyzed
    enemy, and (2) dead units (health <= 0) must never act nor be targeted,
    even if they linger in ``self.units``.
    """

    def _ready(self, unit):
        """Clear summoning sickness so a freshly created unit can act."""
        unit.can_move = True
        unit.can_attack = True

    def _fund(self, game):
        """Give both players enough gold to create any unit."""
        for p in game.player_gold:
            game.player_gold[p] = 1000

    def test_paralyze_not_offered_for_already_paralyzed_enemy(self, simple_game):
        game = simple_game
        game.current_player = 1
        self._fund(game)
        mage = game.create_unit("M", 5, 5, player=1)
        enemy = game.create_unit("W", 5, 6, player=2)  # adjacent (distance 1)
        self._ready(mage)
        game._invalidate_cache()

        # Baseline: an un-paralyzed enemy in range yields both attack + paralyze.
        actions = game.get_legal_actions(player=1)
        assert any(a["target"] is enemy for a in actions["attack"])
        assert any(a["target"] is enemy for a in actions["paralyze"])

        # Once paralyzed, the enemy is still attackable but must NOT be
        # offered as a paralyze target (re-cast is a wasteful near no-op).
        enemy.paralyzed_turns = 2
        game._invalidate_cache()
        actions = game.get_legal_actions(player=1)
        assert any(a["target"] is enemy for a in actions["attack"])
        assert all(a["target"] is not enemy for a in actions["paralyze"])

    def test_dead_target_not_offered_as_attack(self, simple_game):
        game = simple_game
        game.current_player = 1
        self._fund(game)
        attacker = game.create_unit("W", 5, 5, player=1)
        enemy = game.create_unit("W", 5, 6, player=2)
        self._ready(attacker)

        enemy.health = 0  # corpse left in self.units
        game._invalidate_cache()
        actions = game.get_legal_actions(player=1)
        assert all(a["target"] is not enemy for a in actions["attack"])

    def test_dead_unit_generates_no_actions(self, simple_game):
        game = simple_game
        game.current_player = 1
        self._fund(game)
        unit = game.create_unit("W", 5, 5, player=1)
        self._ready(unit)

        unit.health = 0  # corpse left in self.units
        game._invalidate_cache()
        actions = game.get_legal_actions(player=1)
        assert all(a["unit"] is not unit for a in actions["move"])
        assert all(a["attacker"] is not unit for a in actions["attack"])


class TestUnitEliminationWinCondition:
    """Test that eliminating all enemy units results in a win."""

    def test_game_ends_when_target_player_loses_last_unit(self, simple_game):
        """Test game ends when target is killed and they have no remaining units."""
        # Create attacker for player 1 and a single target for player 2
        attacker = simple_game.create_unit("W", 5, 5, player=1)
        target = simple_game.create_unit("C", 6, 5, player=2)  # Cleric: 10 HP, 4 def
        # Pre-damage so the warrior's 8 dmg one-shots. Post-buff cleric
        # would otherwise survive a single warrior hit (10 - 8 = 2 HP).
        target.health = 8

        # Verify initial state
        assert simple_game.game_over is False
        assert simple_game.winner is None
        assert len([u for u in simple_game.units if u.player == 2]) == 1

        # Attack - Warrior does 8 damage (10 atk - 4 def reduction), kills Cleric.
        result = simple_game.attack(attacker, target)

        # Verify target is killed
        assert result["target_alive"] is False
        assert target not in simple_game.units

        # Verify game is over and player 1 won
        assert simple_game.game_over is True
        assert simple_game.winner == 1

    def test_game_ends_when_attacker_player_loses_last_unit(self, simple_game):
        """Test game ends when attacker dies via counter-attack and they have no remaining units."""
        # Create a weak attacker for player 1 and a strong defender for player 2
        attacker = simple_game.create_unit("C", 5, 5, player=1)  # Cleric has 8 HP, 2 attack
        defender = simple_game.create_unit("W", 6, 5, player=2)  # Warrior has 15 HP, 10 attack

        # Pre-damage the Cleric so it will die from counter-attack
        # Warrior counter does 6 damage (10 * 0.8 counter * 0.8 defense reduction)
        # Cleric needs <= 6 HP to die from counter
        attacker.health = 5

        # Verify initial state
        assert simple_game.game_over is False
        assert simple_game.winner is None
        assert len([u for u in simple_game.units if u.player == 1]) == 1

        # Attack - Cleric attacks Warrior, Warrior counter-attacks and kills Cleric
        result = simple_game.attack(attacker, defender)

        # Verify attacker is killed by counter
        assert result["attacker_alive"] is False
        assert attacker not in simple_game.units

        # Verify game is over and player 2 won
        assert simple_game.game_over is True
        assert simple_game.winner == 2

    def test_game_continues_when_player_has_remaining_units(self, simple_game):
        """Test game does NOT end when a unit is killed but player still has units."""
        # Give player 2 enough gold for two units
        simple_game.player_gold[2] = 500

        # Create attacker for player 1 and two units for player 2
        attacker = simple_game.create_unit("W", 5, 5, player=1)
        target = simple_game.create_unit("C", 6, 5, player=2)  # Will be killed
        # Pre-damage post-buff cleric (10 HP) so warrior's 8 dmg one-shots.
        target.health = 8
        _survivor = simple_game.create_unit("W", 7, 7, player=2)  # Will survive

        # Verify initial state
        assert simple_game.game_over is False
        assert len([u for u in simple_game.units if u.player == 2]) == 2

        # Attack - Warrior kills Cleric
        result = simple_game.attack(attacker, target)

        # Verify target is killed
        assert result["target_alive"] is False
        assert target not in simple_game.units

        # Verify game is NOT over because player 2 still has a unit
        assert simple_game.game_over is False
        assert simple_game.winner is None
        assert len([u for u in simple_game.units if u.player == 2]) == 1

    def test_both_players_die_simultaneously_attacker_wins(self, simple_game):
        """Test when both units die (mutual kill), check the remaining player wins."""
        # Give player 1 enough gold for attacker
        simple_game.player_gold[1] = 300

        # Create two units - a weak attacker and strong defender to ensure counter-kill
        attacker = simple_game.create_unit("C", 5, 5, player=1)  # 8 HP, 2 attack, 3 defense
        defender = simple_game.create_unit("W", 6, 5, player=2)  # 15 HP, 10 attack, 6 defense

        # Damage attacker so counter-attack will kill it
        # Warrior counter: 10 attack - 3 defense = 7 damage * 0.9 = 6.3 -> 6 damage
        attacker.health = 6  # Will die from 6 counter damage
        # Defender needs to survive first attack to counter, then die from damage
        # Cleric does 2 damage - defender needs 3 HP to survive and counter
        defender.health = 3  # Survives 2 damage (3-2=1), then counter, but dies after

        # Verify initial state
        assert simple_game.game_over is False
        assert len([u for u in simple_game.units if u.player == 1]) == 1
        assert len([u for u in simple_game.units if u.player == 2]) == 1

        # Attack - defender survives first hit with 1 HP, counters and kills attacker
        result = simple_game.attack(attacker, defender)

        # Attacker should die from counter, but defender survives with 1 HP
        assert result["target_alive"] is True  # Defender survives with 1 HP
        assert result["attacker_alive"] is False  # Attacker dies from counter

        # Only one player has no units (player 1)
        assert simple_game.game_over is True
        assert simple_game.winner == 2  # Player 2 wins

    def test_unit_elimination_with_hq_still_owned(self, simple_game):
        """Test that losing all units ends the game even if player owns HQ."""
        # Player 2 owns HQ at (9,9) but will lose their only unit
        attacker = simple_game.create_unit("W", 5, 5, player=1)
        target = simple_game.create_unit("C", 6, 5, player=2)
        # Pre-damage post-buff cleric (10 HP) so warrior's 8 dmg one-shots.
        target.health = 8

        # Verify player 2 owns HQ
        hq_tile = simple_game.grid.get_tile(9, 9)
        assert hq_tile.type == "h"
        assert hq_tile.player == 2

        # Kill player 2's only unit
        result = simple_game.attack(attacker, target)

        # Game should end even though player 2 still owns their HQ
        assert result["target_alive"] is False
        assert simple_game.game_over is True
        assert simple_game.winner == 1


class _ScriptedRng:
    """Random source returning a fixed value; counts how often it's read."""

    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.value


class TestEngineRngInjection:
    """``GameState.rng`` must drive the Rogue evade roll and survive reset().

    Regression tests for the evade roll reading the module-global
    ``random`` regardless of seeding -- which made seeded episodes
    non-reproducible whenever a Rogue attacked into a counter.
    """

    @staticmethod
    def _attack_with_rng(rng):
        map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
        map_data[0][0] = "h_1"
        map_data[9][9] = "h_2"
        gs = GameState(map_data, num_players=2, rng=rng)
        rogue = Unit("R", 5, 5, 1)
        target = Unit("W", 6, 5, 2)
        gs.units.extend([rogue, target])
        return gs, gs.attack(rogue, target)

    def test_injected_rng_forces_evade(self):
        rng = _ScriptedRng(0.0)  # below any evade threshold
        _gs, result = self._attack_with_rng(rng)
        assert rng.calls == 1, "the evade roll must come from the injected rng"
        assert result["evade"] is True
        assert result["counter_damage"] == 0

    def test_injected_rng_suppresses_evade(self):
        rng = _ScriptedRng(0.999)  # above any evade threshold
        _gs, result = self._attack_with_rng(rng)
        assert rng.calls == 1
        assert result["evade"] is False
        assert result["counter_damage"] > 0

    def test_rng_preserved_across_reset(self):
        rng = _ScriptedRng(0.0)
        gs, _result = self._attack_with_rng(rng)
        gs.reset(np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object))
        assert gs.rng is rng

    def test_default_rng_is_none_module_global_fallback(self):
        # No rng -> legacy behaviour (module-global random); the attribute
        # must exist and be None so mechanics falls back cleanly.
        map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
        gs = GameState(map_data, num_players=2)
        assert gs.rng is None


class TestStructureAutoHealAccumulator:
    """``heal_units_on_structures`` accumulates game-lifetime totals into
    ``healing_totals`` -- the only surviving record of the auto-heal
    economy, since callers routinely discard ``end_turn()``'s return
    (bots call ``end_turn`` internally; the gym env drops the value)."""

    def test_healing_totals_accumulate_across_turns(self, game_with_building):
        gs = game_with_building
        gs.player_gold[1] = 500
        wounded = Unit("W", 1, 1, 1)  # parked on player 1's building
        wounded.health = 10  # 5 below the Warrior's 15 max
        gs.units.append(wounded)

        stats = gs.heal_units_on_structures(1)
        # Building heals 2 HP at (heal/max_hp) * unit_cost gold.
        expected_cost = round(2 * 200 / 15)
        assert stats["total_healed"] == 2
        assert stats["total_cost"] == expected_cost
        assert gs.healing_totals[1] == {"hp": 2, "gold": expected_cost}
        assert gs.healing_totals[2] == {"hp": 0, "gold": 0}

        gs.heal_units_on_structures(1)
        assert gs.healing_totals[1]["hp"] == 4
        assert gs.healing_totals[1]["gold"] == 2 * expected_cost

    def test_no_gold_means_no_heal_and_no_accumulation(self, game_with_building):
        gs = game_with_building
        gs.player_gold[1] = 0  # cannot afford even a 1 HP partial heal
        wounded = Unit("W", 1, 1, 1)
        wounded.health = 10
        gs.units.append(wounded)

        stats = gs.heal_units_on_structures(1)
        assert stats["total_healed"] == 0
        assert gs.healing_totals[1] == {"hp": 0, "gold": 0}
