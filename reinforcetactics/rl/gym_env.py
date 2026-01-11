"""
Gymnasium environment for Reinforce Tactics
Supports both flat and hierarchical RL training
"""
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.constants import UNIT_DATA


class StrategyGameEnv(gym.Env):
    """
    Gymnasium environment for turn-based strategy game.

    Observation Space:
        Dict with:
        - 'grid': (H, W, 3) - terrain type, owner, structure HP
        - 'units': (H, W, 3) - unit type, owner, HP
        - 'global': (6,) - gold_p1, gold_p2, turn, num_units_p1, num_units_p2, current_player
        - 'action_mask': (action_space_size,) - binary mask of valid actions

    Action Space:
        MultiDiscrete with 6 dimensions:
        - action_type: [0=create_unit, 1=move, 2=attack, 3=seize, 4=heal, 5=end_turn]
        - unit_type: [0=W, 1=M, 2=C] (for create_unit)
        - from_x: [0, grid_width)
        - from_y: [0, grid_height)
        - to_x: [0, grid_width)
        - to_y: [0, grid_height)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(
        self,
        map_file: Optional[str] = None,
        opponent: str = 'bot',  # 'bot', 'random', 'self', or None
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        reward_config: Optional[Dict[str, float]] = None,
        hierarchical: bool = False,  # Enable for HRL
        goal_space_size: int = 64  # For HRL goal space
    ):
        """
        Initialize environment.

        Args:
            map_file: Path to map CSV. If None, generates random map
            opponent: Type of opponent ('bot', 'random', 'self', None for manual)
            render_mode: 'human' or 'rgb_array' or None
            max_steps: Maximum steps per episode
            reward_config: Dict of reward weights
            hierarchical: Whether to use hierarchical action space
            goal_space_size: Size of goal space for HRL
        """
        super().__init__()

        # Load or generate map
        if map_file:
            map_data = FileIO.load_map(map_file)
        else:
            map_data = FileIO.generate_random_map(20, 20, num_players=2)

        self.initial_map_data = map_data
        self.game_state = GameState(map_data, num_players=2)
        self.opponent_type = opponent
        self.opponent = None
        self.max_steps = max_steps
        self.current_step = 0
        self.hierarchical = hierarchical
        self.goal_space_size = goal_space_size

        # Reward configuration
        self.reward_config = reward_config or {
            'win': 1000.0,
            'loss': -1000.0,
            'income_diff': 0.1,
            'unit_diff': 1.0,
            'structure_control': 5.0,
            'invalid_action': -10.0,
            'turn_penalty': -0.1
        }

        # Grid dimensions
        self.grid_height = self.game_state.grid.height
        self.grid_width = self.game_state.grid.width

        # Define observation space
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=255,
                shape=(self.grid_height, self.grid_width, 3),
                dtype=np.float32
            ),
            'units': spaces.Box(
                low=0, high=255,
                shape=(self.grid_height, self.grid_width, 3),
                dtype=np.float32
            ),
            'global_features': spaces.Box(
                low=0, high=10000,
                shape=(6,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self._get_action_space_size(),),
                dtype=np.float32
            )
        })

        # Define action space
        if hierarchical:
            # HRL: Manager outputs goals, worker outputs primitive actions
            self.action_space = spaces.Dict({
                'goal': spaces.Discrete(goal_space_size),  # Manager action
                'primitive': spaces.MultiDiscrete([
                    6,  # action_type
                    3,  # unit_type (for create)
                    self.grid_width,  # from_x
                    self.grid_height,  # from_y
                    self.grid_width,  # to_x
                    self.grid_height   # to_y
                ])
            })
        else:
            # Flat RL: Direct primitive actions
            self.action_space = spaces.MultiDiscrete([
                6,  # action_type
                3,  # unit_type (for create)
                self.grid_width,  # from_x
                self.grid_height,  # from_y
                self.grid_width,  # to_x
                self.grid_height   # to_y
            ])

        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == 'human':
            from reinforcetactics.ui.renderer import Renderer
            self.renderer = Renderer(self.game_state)

        # Episode statistics
        self.episode_stats = {
            'reward': 0.0,
            'length': 0,
            'winner': None,
            'invalid_actions': 0
        }

    def _get_action_space_size(self) -> int:
        """Calculate total action space size for masking."""
        # Simplified: num_action_types * grid_size^2 (approximate)
        return 6 * self.grid_width * self.grid_height

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Convert game state to numpy arrays
        state_arrays = self.game_state.to_numpy()

        # Get action mask
        action_mask = self._get_action_mask()

        obs = {
            'grid': state_arrays['grid'].astype(np.float32),
            'units': state_arrays['units'].astype(np.float32),
            'global_features': np.array([
                self.game_state.player_gold[1],
                self.game_state.player_gold[2],
                self.game_state.turn_number,
                len([u for u in self.game_state.units if u.player == 1]),
                len([u for u in self.game_state.units if u.player == 2]),
                self.game_state.current_player
            ], dtype=np.float32),
            'action_mask': action_mask
        }

        return obs

    def _get_action_mask(self) -> np.ndarray:
        """
        Get binary mask of valid actions for the current player.

        The action mask corresponds to the flattened action space of size 6 * W * H.
        Segments:
        0: Create Unit (at pos) - Valid if pos is a building and we can afford unit
        1: Move (to pos) - Valid if ANY unit can move to pos
        2: Attack (target at pos) - Valid if ANY unit can attack unit at pos
        3: Seize (at pos) - Valid if unit at pos can seize
        4: Heal (target at pos) - Valid if ANY unit can heal unit at pos
        5: End Turn (any pos) - Always valid (usually just mapped to one index or all)

        Note: This is a "valid target" mask. It tells the agent *where* something can happen,
        but implies *someone* can do it. The agent then picks (ActionType, UnitType, From, To).
        Strictly speaking, for a MultiDiscrete space, masking is complex.
        Here we map to the flattened intention: "Can I perform Action X at Target Y?".
        """
        # Get legal actions from game state (uses cache)
        # Note: gym env manages current player, but game_state also knows it.
        legal_actions = self.game_state.get_legal_actions(player=self.game_state.current_player)

        # Create mask (all zeros initially)
        # Size: 6 * W * H
        mask = np.zeros(self._get_action_space_size(), dtype=np.float32)

        width = self.grid_width
        height = self.grid_height
        area = width * height

        # Helper to set mask bit
        def set_mask(action_type_idx, x, y):
            idx = (action_type_idx * area) + (y * width + x)
            if 0 <= idx < mask.size:
                mask[idx] = 1.0

        # 0: Create Unit
        # legal_actions['create_unit'] contains list of dicts: {unit_type, x, y}
        for action in legal_actions.get('create_unit', []):
            set_mask(0, action['x'], action['y'])

        # 1: Move
        # legal_actions['move'] contains list of dicts: {unit, from_x, from_y, to_x, to_y}
        for action in legal_actions.get('move', []):
            set_mask(1, action['to_x'], action['to_y'])

        # 2: Attack
        # legal_actions['attack'] contains list of dicts: {attacker, target}
        for action in legal_actions.get('attack', []):
            target = action['target']
            set_mask(2, target.x, target.y)

        # 3: Seize
        # legal_actions['seize'] contains list of dicts: {unit, tile}
        for action in legal_actions.get('seize', []):
            tile = action['tile']
            set_mask(3, tile.x, tile.y)

        # 4: Heal (includes Cure)
        for action in legal_actions.get('heal', []):
            target = action['target']
            set_mask(4, target.x, target.y)

        # Map 'cure' to the same action type as 'heal' (4)
        # If a cleric selects action type 4 on a target, they will Heal OR Cure depending on condition
        for action in legal_actions.get('cure', []):
            target = action['target']
            set_mask(4, target.x, target.y)

        # 5: End Turn
        # Always allow end turn. We can map it to (0,0) or everywhere.
        # Usually EndTurn doesn't need parameters.
        # Let's enable it everywhere to be safe, or just index 0.
        # Enabling everywhere allows the agent to easier "find" the action.
        start_idx = 5 * area
        end_idx = 6 * area
        mask[start_idx:end_idx] = 1.0

        return mask

    def _encode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Encode action array into game action.

        Args:
            action: [action_type, unit_type, from_x, from_y, to_x, to_y]

        Returns:
            Dict with action details
        """
        action_type = int(action[0])
        unit_type_idx = int(action[1])
        from_x, from_y = int(action[2]), int(action[3])
        to_x, to_y = int(action[4]), int(action[5])

        unit_types = ['W', 'M', 'C']
        unit_type = unit_types[unit_type_idx % 3]

        return {
            'action_type': action_type,
            'unit_type': unit_type,
            'from_pos': (from_x, from_y),
            'to_pos': (to_x, to_y)
        }

    def _execute_action(self, action_dict: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Execute encoded action in game.

        Returns:
            (reward, is_valid)
        """
        action_type = action_dict['action_type']
        from_pos = action_dict['from_pos']
        to_pos = action_dict['to_pos']

        reward = 0.0
        is_valid = True

        try:
            if action_type == 0:  # Create unit
                unit_type = action_dict['unit_type']
                unit = self.game_state.create_unit(
                    unit_type, to_pos[0], to_pos[1], player=1
                )
                if unit:
                    reward += 2.0
                else:
                    is_valid = False

            elif action_type == 1:  # Move
                unit = self.game_state.get_unit_at_position(*from_pos)
                if unit and unit.player == 1 and unit.can_move:
                    if self.game_state.move_unit(unit, to_pos[0], to_pos[1]):
                        reward += 0.1
                    else:
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 2:  # Attack
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.player == 1 and target.player != 1:
                    result = self.game_state.attack(unit, target)
                    reward += result['damage'] / 5.0
                    if not result['target_alive']:
                        reward += 10.0
                else:
                    is_valid = False

            elif action_type == 3:  # Seize
                unit = self.game_state.get_unit_at_position(*from_pos)
                if unit and unit.player == 1:
                    result = self.game_state.seize(unit)
                    reward += 1.0
                    if result['captured']:
                        reward += 20.0
                else:
                    is_valid = False

            elif action_type == 4:  # Heal/Cure (Cleric)
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.type == 'C':
                    # Priority: Cure if paralyzed, otherwise Heal
                    # Or check what's possible
                    
                    # Try to cure first if target is paralyzed
                    action_performed = False
                    if target.is_paralyzed():
                        result = self.game_state.cure(unit, target)
                        if result:
                            reward += 5.0 # Reward for curing
                            action_performed = True
                    
                    # If not cured (or not paralyzed), try to heal
                    if not action_performed:
                        heal_amount = self.game_state.heal(unit, target)
                        if heal_amount > 0:
                            reward += heal_amount / 2.0
                            action_performed = True
                    
                    if not action_performed:
                        # If neither worked (e.g. full health and not paralyzed), action failed
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 5:  # End turn
                self.game_state.end_turn()
                reward += self.reward_config['turn_penalty']
                # Opponent plays
                if self.opponent:
                    self._opponent_turn()
                    self.game_state.end_turn()

        except Exception as e:
            print(f"Error executing action: {e}")
            is_valid = False

        return reward, is_valid

    def _opponent_turn(self):
        """Execute opponent's turn."""
        if self.opponent_type == 'bot':
            self.opponent.take_turn()
        elif self.opponent_type == 'random':
            # Random valid action
            legal_actions = self.game_state.get_legal_actions(player=2)
            if legal_actions and legal_actions.get('end_turn'):
                # Just end turn for random opponent
                pass
        elif self.opponent_type == 'self':
            # Self-play (will be handled by training script)
            pass

    def _calculate_reward(self, action_reward: float, is_valid: bool) -> float:
        """Calculate total reward including shaping terms."""
        reward = action_reward

        if not is_valid:
            reward += self.reward_config['invalid_action']
            self.episode_stats['invalid_actions'] += 1

        # Dense reward shaping
        if self.reward_config['income_diff'] > 0:
            income_data_p1 = self.game_state.mechanics.calculate_income(1, self.game_state.grid)
            income_data_p2 = self.game_state.mechanics.calculate_income(2, self.game_state.grid)
            income_diff = income_data_p1['total'] - income_data_p2['total']
            reward += income_diff * self.reward_config['income_diff']

        if self.reward_config['unit_diff'] > 0:
            units_p1 = len([u for u in self.game_state.units if u.player == 1])
            units_p2 = len([u for u in self.game_state.units if u.player == 2])
            unit_diff = units_p1 - units_p2
            reward += unit_diff * self.reward_config['unit_diff']

        if self.reward_config['structure_control'] > 0:
            structures_p1 = len(self.game_state.grid.get_capturable_tiles(player=1))
            structures_p2 = len(self.game_state.grid.get_capturable_tiles(player=2))
            structure_diff = structures_p1 - structures_p2
            reward += structure_diff * self.reward_config['structure_control']

        return reward

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Decode and execute action
        action_dict = self._encode_action(action)
        action_reward, is_valid = self._execute_action(action_dict)

        # Calculate total reward
        reward = self._calculate_reward(action_reward, is_valid)
        self.episode_stats['reward'] += reward

        # Check termination
        terminated = self.game_state.game_over
        truncated = self.current_step >= self.max_steps

        if terminated:
            if self.game_state.winner == 1:
                reward += self.reward_config['win']
                self.episode_stats['winner'] = 1
            else:
                reward += self.reward_config['loss']
                self.episode_stats['winner'] = 2

        # Get observation
        obs = self._get_obs()

        # Info dict
        info = {
            'episode_stats': self.episode_stats.copy() if terminated or truncated else {},
            'game_over': terminated,
            'winner': self.game_state.winner if terminated else None,
            'turn': self.game_state.turn_number,
            'valid_action': is_valid
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset game state
        self.game_state = GameState(self.initial_map_data, num_players=2)
        self.current_step = 0

        # Reset opponent
        if self.opponent_type == 'bot':
            self.opponent = SimpleBot(self.game_state, player=2)

        # Reset renderer
        if self.render_mode == 'human' and self.renderer:
            from reinforcetactics.ui.renderer import Renderer
            self.renderer = Renderer(self.game_state)

        # Reset episode stats
        self.episode_stats = {
            'reward': 0.0,
            'length': 0,
            'winner': None,
            'invalid_actions': 0
        }

        obs = self._get_obs()
        info = {}

        return obs, info

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            if self.renderer:
                self.renderer.render()
        elif self.render_mode == 'rgb_array':
            if self.renderer:
                return self.renderer.get_rgb_array()

    def close(self):
        """Clean up."""
        if self.renderer:
            self.renderer.close()