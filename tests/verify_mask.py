"""Test module for verifying action masking in the gym environment."""
import os
import sys
import unittest

import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.core.unit import Unit

class TestActionMasking(unittest.TestCase):
    """Test action masking correctness."""

    def setUp(self):
        """Create environment for testing."""
        self.env = StrategyGameEnv(map_file=None, opponent='bot', render_mode=None)

    def tearDown(self):
        self.env.close()

    def test_cache_invalidation(self):
        """Test that executing an action invalidates the cache."""
        self.env.reset()
        self.env.game_state.units = []
        self.env.game_state._invalidate_cache()

        # Unit at 0,0
        unit = Unit('W', 0, 0, player=1)
        unit.can_move = True
        self.env.game_state.units.append(unit)
        self.env.game_state.grid.get_tile(0,0).type = 'p'
        self.env.game_state.grid.get_tile(1,0).type = 'p'
        self.env.game_state.current_player = 1

        # 1. Check initial mask (Move to 1,0 is valid)
        mask1 = self.env._get_action_mask()
        area = self.env.grid_width * self.env.grid_height
        idx_move_1_0 = (1 * area) + (0 * self.env.grid_width + 1)
        self.assertEqual(mask1[idx_move_1_0], 1.0)

        # 2. Execute move to 1,0 (directly via game state to simulate action)
        self.env.game_state.move_unit(unit, 1, 0)

        # 3. Check mask again.
        # Unit moved to 1,0. It can no longer move (has_moved=True).
        # So "Move" layer should be all zeros (or at least 1,0 is no longer a valid dest from 0,0, and unit can't move).
        mask2 = self.env._get_action_mask()

        # Check if the mask changed (it should have)
        self.assertFalse(np.array_equal(mask1, mask2), "Mask should change after action")

        # Specifically, unit exhausted movement, so NO move actions should be valid for this unit.
        # Since it's the only unit, ALL move actions should be 0.
        move_layer_start = 1 * area
        move_layer_end = 2 * area
        self.assertTrue(np.all(mask2[move_layer_start:move_layer_end] == 0.0), "No moves should be valid after unit moves")


    def test_cure_masking_and_execution(self):
        """Test that Cure action is correctly masked and executed."""
        self.env.reset()
        self.env.game_state.units = []
        self.env.game_state._invalidate_cache()

        # Setup: Cleric (Player 1) and Paralyzed Ally (Player 1)
        cleric = Unit('C', 5, 5, player=1)
        cleric.can_attack = True # Enable unit to act
        ally = Unit('W', 5, 6, player=1)
        ally.paralyzed_turns = 2 # Paralyzed

        self.env.game_state.units.append(cleric)
        self.env.game_state.units.append(ally)
        self.env.game_state._invalidate_cache()

        # 1. Check Mask
        mask = self.env._get_action_mask()
        area = self.env.grid_width * self.env.grid_height

        # Index for Heal/Cure action (type 4) at ally position (5,6)
        heal_idx = (4 * area) + (6 * self.env.grid_width + 5)

        self.assertEqual(mask[heal_idx], 1.0, "Cure action should be masked as valid")

        # 2. Execute Cure
        # Action: Type 4 (Heal/Cure), Cleric, From(5,5), To(5,6)
        action_dict = {
            'action_type': 4,
            'unit_type': 'C',
            'from_pos': (5, 5),
            'to_pos': (5, 6)
        }

        reward, is_valid = self.env._execute_action(action_dict)

        self.assertTrue(is_valid, "Cure action should be valid")
        self.assertFalse(ally.is_paralyzed(), "Ally should be cured (paralyzed_turns=0)")
        self.assertTrue(reward > 0, "Should receive reward for curing")

if __name__ == '__main__':
    unittest.main()
