"""Tests for GameState class, specifically win conditions."""
import pytest
import numpy as np
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.unit import Unit


@pytest.fixture
def simple_game():
    """Create a simple game state for testing."""
    map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[0][0] = 'h_1'  # HQ for player 1
    map_data[9][9] = 'h_2'  # HQ for player 2
    return GameState(map_data, num_players=2)


class TestUnitEliminationWinCondition:
    """Test that eliminating all enemy units results in a win."""

    def test_game_ends_when_target_player_loses_last_unit(self, simple_game):
        """Test game ends when target is killed and they have no remaining units."""
        # Create attacker for player 1 and a single target for player 2
        attacker = simple_game.create_unit('W', 5, 5, player=1)
        target = simple_game.create_unit('C', 6, 5, player=2)  # Cleric has 8 HP
        
        # Verify initial state
        assert simple_game.game_over is False
        assert simple_game.winner is None
        assert len([u for u in simple_game.units if u.player == 2]) == 1
        
        # Attack - Warrior does 10 damage, should kill Cleric
        result = simple_game.attack(attacker, target)
        
        # Verify target is killed
        assert result['target_alive'] is False
        assert target not in simple_game.units
        
        # Verify game is over and player 1 won
        assert simple_game.game_over is True
        assert simple_game.winner == 1

    def test_game_ends_when_attacker_player_loses_last_unit(self, simple_game):
        """Test game ends when attacker dies via counter-attack and they have no remaining units."""
        # Create a weak attacker for player 1 and a strong defender for player 2
        attacker = simple_game.create_unit('C', 5, 5, player=1)  # Cleric has 8 HP, 2 attack
        defender = simple_game.create_unit('W', 6, 5, player=2)  # Warrior has 15 HP, 10 attack
        
        # Verify initial state
        assert simple_game.game_over is False
        assert simple_game.winner is None
        assert len([u for u in simple_game.units if u.player == 1]) == 1
        
        # Attack - Cleric attacks Warrior, Warrior counter-attacks and kills Cleric
        result = simple_game.attack(attacker, defender)
        
        # Verify attacker is killed by counter
        assert result['attacker_alive'] is False
        assert attacker not in simple_game.units
        
        # Verify game is over and player 2 won
        assert simple_game.game_over is True
        assert simple_game.winner == 2

    def test_game_continues_when_player_has_remaining_units(self, simple_game):
        """Test game does NOT end when a unit is killed but player still has units."""
        # Give player 2 enough gold for two units
        simple_game.player_gold[2] = 500
        
        # Create attacker for player 1 and two units for player 2
        attacker = simple_game.create_unit('W', 5, 5, player=1)
        target = simple_game.create_unit('C', 6, 5, player=2)  # Will be killed
        survivor = simple_game.create_unit('W', 7, 7, player=2)  # Will survive
        
        # Verify initial state
        assert simple_game.game_over is False
        assert len([u for u in simple_game.units if u.player == 2]) == 2
        
        # Attack - Warrior kills Cleric
        result = simple_game.attack(attacker, target)
        
        # Verify target is killed
        assert result['target_alive'] is False
        assert target not in simple_game.units
        
        # Verify game is NOT over because player 2 still has a unit
        assert simple_game.game_over is False
        assert simple_game.winner is None
        assert len([u for u in simple_game.units if u.player == 2]) == 1

    def test_both_players_die_simultaneously_attacker_wins(self, simple_game):
        """Test when both units die (mutual kill), check the remaining player wins."""
        # Create two units - a weak attacker and strong defender to ensure counter-kill
        attacker = Unit('C', 5, 5, player=1)  # 8 HP, 2 attack, 3 defense
        defender = Unit('W', 6, 5, player=2)  # 15 HP, 10 attack, 6 defense
        
        # Manually add them to game (bypass gold check)
        simple_game.units.append(attacker)
        simple_game.units.append(defender)
        simple_game._invalidate_cache()
        
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
        assert result['target_alive'] is True  # Defender survives with 1 HP
        assert result['attacker_alive'] is False  # Attacker dies from counter
        
        # Only one player has no units (player 1)
        assert simple_game.game_over is True
        assert simple_game.winner == 2  # Player 2 wins

    def test_unit_elimination_with_hq_still_owned(self, simple_game):
        """Test that losing all units ends the game even if player owns HQ."""
        # Player 2 owns HQ at (9,9) but will lose their only unit
        attacker = simple_game.create_unit('W', 5, 5, player=1)
        target = simple_game.create_unit('C', 6, 5, player=2)
        
        # Verify player 2 owns HQ
        hq_tile = simple_game.grid.get_tile(9, 9)
        assert hq_tile.type == 'h'
        assert hq_tile.player == 2
        
        # Kill player 2's only unit
        result = simple_game.attack(attacker, target)
        
        # Game should end even though player 2 still owns their HQ
        assert result['target_alive'] is False
        assert simple_game.game_over is True
        assert simple_game.winner == 1
