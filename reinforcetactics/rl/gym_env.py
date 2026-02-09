"""
Gymnasium environment for Reinforce Tactics
Supports both flat and hierarchical RL training
"""
import logging
import random
import traceback
from typing import Dict, Any, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils.file_io import FileIO

logger = logging.getLogger(__name__)


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
        - action_type: [0=create_unit, 1=move, 2=attack, 3=seize, 4=heal, 5=end_turn,
                        6=paralyze, 7=haste, 8=defence_buff, 9=attack_buff]
        - unit_type: [0=W, 1=M, 2=C, 3=A, 4=K, 5=R, 6=S, 7=B] (for create_unit)
        - from_x: [0, grid_width)
        - from_y: [0, grid_height)
        - to_x: [0, grid_width)
        - to_y: [0, grid_height)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    # All available unit types
    ALL_UNIT_TYPES = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    def __init__(
        self,
        map_file: Optional[str] = None,
        opponent: str = 'bot',  # 'bot', 'random', 'self', or None
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        reward_config: Optional[Dict[str, float]] = None,
        hierarchical: bool = False,  # Enable for HRL
        goal_space_size: int = 64,  # For HRL goal space
        enabled_units: Optional[List[str]] = None,  # List of enabled unit types
        fog_of_war: bool = False  # Enable fog of war
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
            enabled_units: List of enabled unit types (default all)
            fog_of_war: Enable fog of war for partial observability (default False)
        """
        super().__init__()

        # Load or generate map
        if map_file:
            map_data = FileIO.load_map(map_file)
        else:
            map_data = FileIO.generate_random_map(20, 20, num_players=2)

        self.initial_map_data = map_data
        # Store enabled units (default to all if not specified)
        self.enabled_units = enabled_units if enabled_units is not None else self.ALL_UNIT_TYPES.copy()
        # Fog of war setting
        self.fog_of_war = fog_of_war
        self.game_state = GameState(map_data, num_players=2, enabled_units=self.enabled_units, fog_of_war=fog_of_war)

        # Initialize visibility at game start
        if fog_of_war:
            self.game_state.update_visibility()
        self.opponent_type = opponent
        self.opponent = None
        self.max_steps = max_steps
        self.current_step = 0
        self.hierarchical = hierarchical
        self.goal_space_size = goal_space_size

        # Which player the RL agent controls (1 or 2). SelfPlayEnv may set to 2.
        self.agent_player = 1

        # Reward configuration with defaults
        default_reward_config = {
            'win': 1000.0,
            'loss': -1000.0,
            'draw': 0.0,
            'income_diff': 0.1,
            'unit_diff': 1.0,
            'structure_control': 5.0,
            'invalid_action': -10.0,
            'turn_penalty': -0.1,
            # Action rewards (moved from hardcoded values for tunability)
            'create_unit': 2.0,
            'move': 0.1,
            'damage_scale': 0.2,       # reward per damage point (was damage/5.0)
            'kill': 10.0,
            'seize_progress': 1.0,
            'capture': 20.0,
            'cure': 5.0,
            'heal_scale': 0.5,         # reward per HP healed (was heal/2.0)
            'paralyze': 8.0,
            'haste': 6.0,
            'defence_buff': 5.0,
            'attack_buff': 5.0,
        }
        if reward_config:
            default_reward_config.update(reward_config)
        self.reward_config = default_reward_config

        # Previous potential for potential-based reward shaping (Phi(s) tracking)
        self._prev_potential = 0.0

        # Grid dimensions
        self.grid_height = self.game_state.grid.height
        self.grid_width = self.game_state.grid.width

        # Define observation space
        obs_dict = {
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
        }

        # Add visibility layer when fog of war is enabled
        if fog_of_war:
            obs_dict['visibility'] = spaces.Box(
                low=0, high=2,  # 0=unexplored, 1=shrouded, 2=visible
                shape=(self.grid_height, self.grid_width),
                dtype=np.uint8
            )

        self.observation_space = spaces.Dict(obs_dict)

        # Define action space
        if hierarchical:
            # HRL: Manager outputs goals, worker outputs primitive actions
            self.action_space = spaces.Dict({
                'goal': spaces.Discrete(goal_space_size),  # Manager action
                'primitive': spaces.MultiDiscrete([
                    10,  # action_type (0-9)
                    8,  # unit_type (for create): W, M, C, A, K, R, S, B
                    self.grid_width,  # from_x
                    self.grid_height,  # from_y
                    self.grid_width,  # to_x
                    self.grid_height   # to_y
                ])
            })
        else:
            # Flat RL: Direct primitive actions
            self.action_space = spaces.MultiDiscrete([
                10,  # action_type (0-9)
                8,  # unit_type (for create): W, M, C, A, K, R, S, B
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
        # 10 action types: create, move, attack, seize, heal, end_turn, paralyze, haste, defence_buff, attack_buff
        return 10 * self.grid_width * self.grid_height

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation from the agent's perspective."""
        ap = self.agent_player
        opp = 3 - ap
        # Convert game state to numpy arrays
        # When fog of war is enabled, filter observation for the agent's player
        if self.fog_of_war:
            state_arrays = self.game_state.to_numpy(for_player=ap)
        else:
            state_arrays = self.game_state.to_numpy()

        # Get action mask
        action_mask = self._get_action_mask()

        # When FOW is enabled, hide enemy gold
        if self.fog_of_war:
            global_features = np.array([
                self.game_state.player_gold[ap],
                0,  # Hide enemy gold
                self.game_state.turn_number,
                len([u for u in self.game_state.units if u.player == ap]),
                # Count only visible enemy units
                len([u for u in self.game_state.units
                     if u.player == opp and self.game_state.is_position_visible(u.x, u.y, ap)]),
                self.game_state.current_player
            ], dtype=np.float32)
        else:
            global_features = np.array([
                self.game_state.player_gold[ap],
                self.game_state.player_gold[opp],
                self.game_state.turn_number,
                len([u for u in self.game_state.units if u.player == ap]),
                len([u for u in self.game_state.units if u.player == opp]),
                self.game_state.current_player
            ], dtype=np.float32)

        obs = {
            'grid': state_arrays['grid'].astype(np.float32),
            'units': state_arrays['units'].astype(np.float32),
            'global_features': global_features,
            'action_mask': action_mask
        }

        # Add visibility layer when FOW is enabled
        if self.fog_of_war and 'visibility' in state_arrays:
            obs['visibility'] = state_arrays['visibility']

        return obs

    # Mapping from action key → (action_type_idx, source_key, target_key)
    # source_key/target_key name the dict keys or object attrs for from/to positions.
    _ACTION_KEY_MAP = {
        # key            idx  from_fields            to_fields
        'create_unit':   (0,  None,                  ('x', 'y')),
        'move':          (1,  ('from_x', 'from_y'),  ('to_x', 'to_y')),
        'attack':        (2,  'attacker',             'target'),
        'seize':         (3,  'unit',                 'tile'),
        'heal':          (4,  'healer',               'target'),
        'cure':          (4,  'healer',               'target'),
        'paralyze':      (6,  'paralyzer',            'target'),
        'haste':         (7,  'sorcerer',             'target'),
        'defence_buff':  (8,  'sorcerer',             'target'),
        'attack_buff':   (9,  'sorcerer',             'target'),
    }

    def _build_masks(self) -> Tuple[
        np.ndarray,  # flat mask  (10*W*H,)
        np.ndarray, np.ndarray,  # action_type (10,), unit_type (8,)
        np.ndarray, np.ndarray,  # from_x (W,), from_y (H,)
        np.ndarray, np.ndarray,  # to_x (W,), to_y (H,)
    ]:
        """
        Compute both the flat target-based mask and per-dimension masks from
        a single ``get_legal_actions`` call.

        Returns:
            (flat_mask, action_type_mask, unit_type_mask,
             from_x_mask, from_y_mask, to_x_mask, to_y_mask)
        """
        legal_actions = self.game_state.get_legal_actions(
            player=self.game_state.current_player
        )

        width = self.grid_width
        height = self.grid_height
        area = width * height

        # Flat target mask: size 10 * W * H
        flat_mask = np.zeros(self._get_action_space_size(), dtype=np.float32)

        # Per-dimension masks for MaskablePPO
        at_mask = np.zeros(10, dtype=bool)
        ut_mask = np.zeros(8, dtype=bool)
        fx_mask = np.zeros(width, dtype=bool)
        fy_mask = np.zeros(height, dtype=bool)
        tx_mask = np.zeros(width, dtype=bool)
        ty_mask = np.zeros(height, dtype=bool)

        unit_type_to_idx = {'W': 0, 'M': 1, 'C': 2, 'A': 3, 'K': 4, 'R': 5, 'S': 6, 'B': 7}

        def _pos(obj_or_dict, fields):
            """Extract (x, y) from an object (.x/.y) or a dict (fields tuple)."""
            if isinstance(fields, str):
                # fields is the name of an object attribute with .x, .y
                o = obj_or_dict[fields]
                return o.x, o.y
            # fields is a tuple of dict keys like ('to_x', 'to_y')
            return obj_or_dict[fields[0]], obj_or_dict[fields[1]]

        for key, (at_idx, src_fields, tgt_fields) in self._ACTION_KEY_MAP.items():
            for action in legal_actions.get(key, []):
                at_mask[at_idx] = True

                # Target position — used for both flat and per-dim masks
                tx, ty = _pos(action, tgt_fields)
                tx_mask[tx] = True
                ty_mask[ty] = True

                # Flat mask: set bit at (action_type, target_x, target_y)
                flat_idx = at_idx * area + ty * width + tx
                if 0 <= flat_idx < flat_mask.size:
                    flat_mask[flat_idx] = 1.0

                # Source position — per-dim only
                if src_fields is not None:
                    sx, sy = _pos(action, src_fields)
                    fx_mask[sx] = True
                    fy_mask[sy] = True
                else:
                    # create_unit: no source, mark building pos for from
                    fx_mask[tx] = True
                    fy_mask[ty] = True

                # unit_type for create_unit
                if key == 'create_unit':
                    ut_mask[unit_type_to_idx.get(action['unit_type'], 0)] = True

        # 5: End Turn — always valid
        at_mask[5] = True
        flat_mask[5 * area: 6 * area] = 1.0
        fx_mask[0] = True
        fy_mask[0] = True
        tx_mask[0] = True
        ty_mask[0] = True

        # Ensure unit_type mask has at least one valid option
        if not ut_mask.any():
            if self.enabled_units:
                ut_mask[unit_type_to_idx.get(self.enabled_units[0], 0)] = True
            else:
                ut_mask[0] = True

        return flat_mask, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask

    def _get_action_mask(self) -> np.ndarray:
        """
        Get binary mask of valid actions for the current player.

        The action mask corresponds to the flattened action space of size 10 * W * H.
        Each segment maps an action type to target positions on the grid.

        Note: This is a "valid target" mask. It tells the agent *where* something can
        happen, but implies *someone* can do it. The agent then picks
        (ActionType, UnitType, From, To).
        """
        flat_mask, *_ = self._build_masks()
        return flat_mask

    def action_masks(self) -> Tuple[np.ndarray, ...]:
        """
        Get action masks for MaskablePPO (sb3-contrib).

        For MultiDiscrete action space [action_type, unit_type, from_x, from_y, to_x, to_y],
        returns a tuple of boolean arrays, one per dimension.

        Since action dimensions are interdependent (e.g., valid to_x/to_y depends on
        action_type and from_x/from_y), we compute the UNION of all valid values
        for each dimension. This is an over-approximation that still helps training
        by eliminating clearly invalid options.

        Returns:
            Tuple of 6 boolean numpy arrays for each action dimension
        """
        _, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask = self._build_masks()
        return (at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask)

    def get_action_mask_flat(self) -> np.ndarray:
        """
        Get flattened action mask for compatibility with some algorithms.

        Returns the original target-based mask of size (10 * W * H,).
        """
        return self._get_action_mask()

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

        unit_types = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']
        unit_type = unit_types[unit_type_idx % 8]

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
        rc = self.reward_config
        ap = self.agent_player

        reward = 0.0
        is_valid = True

        try:
            if action_type == 0:  # Create unit
                unit_type = action_dict['unit_type']
                unit = self.game_state.create_unit(
                    unit_type, to_pos[0], to_pos[1], player=ap
                )
                if unit:
                    reward += rc.get('create_unit', 2.0)
                else:
                    is_valid = False

            elif action_type == 1:  # Move
                unit = self.game_state.get_unit_at_position(*from_pos)
                if unit and unit.player == ap and unit.can_move:
                    if self.game_state.move_unit(unit, to_pos[0], to_pos[1]):
                        reward += rc.get('move', 0.1)
                    else:
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 2:  # Attack
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.player == ap and target.player != ap:
                    result = self.game_state.attack(unit, target)
                    reward += result['damage'] * rc.get('damage_scale', 0.2)
                    if not result['target_alive']:
                        reward += rc.get('kill', 10.0)
                else:
                    is_valid = False

            elif action_type == 3:  # Seize
                unit = self.game_state.get_unit_at_position(*from_pos)
                if unit and unit.player == ap:
                    result = self.game_state.seize(unit)
                    if result.get('damage', 0) > 0:
                        reward += rc.get('seize_progress', 1.0)
                        if result['captured']:
                            reward += rc.get('capture', 20.0)
                    else:
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 4:  # Heal/Cure (Cleric)
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.type == 'C' and unit.player == ap:
                    action_performed = False
                    if target.is_paralyzed():
                        result = self.game_state.cure(unit, target)
                        if result:
                            reward += rc.get('cure', 5.0)
                            action_performed = True

                    if not action_performed:
                        heal_amount = self.game_state.heal(unit, target)
                        if heal_amount > 0:
                            reward += heal_amount * rc.get('heal_scale', 0.5)
                            action_performed = True

                    if not action_performed:
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 5:  # End turn
                self.game_state.end_turn()
                reward += self.reward_config['turn_penalty']
                # Opponent plays (dispatch on opponent_type, not opponent object)
                if self.opponent_type and self.opponent_type != 'self':
                    if not self.game_state.game_over:
                        self._opponent_turn()
                        if not self.game_state.game_over:
                            self.game_state.end_turn()

            elif action_type == 6:  # Paralyze (Mage/Sorcerer)
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.type in ['M', 'S'] and target.player != ap:
                    result = self.game_state.paralyze(unit, target)
                    if result:
                        reward += rc.get('paralyze', 8.0)
                    else:
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 7:  # Haste (Sorcerer only)
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.type == 'S' and target.player == ap:
                    result = self.game_state.haste(unit, target)
                    if result:
                        reward += rc.get('haste', 6.0)
                    else:
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 8:  # Defence Buff (Sorcerer only)
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.type == 'S' and target.player == ap:
                    result = self.game_state.defence_buff(unit, target)
                    if result:
                        reward += rc.get('defence_buff', 5.0)
                    else:
                        is_valid = False
                else:
                    is_valid = False

            elif action_type == 9:  # Attack Buff (Sorcerer only)
                unit = self.game_state.get_unit_at_position(*from_pos)
                target = self.game_state.get_unit_at_position(*to_pos)
                if unit and target and unit.type == 'S' and target.player == ap:
                    result = self.game_state.attack_buff(unit, target)
                    if result:
                        reward += rc.get('attack_buff', 5.0)
                    else:
                        is_valid = False
                else:
                    is_valid = False

        except (ValueError, KeyError, IndexError) as e:
            logger.debug("Game action failed (type=%s): %s", action_type, e)
            is_valid = False
        except (TypeError, AttributeError) as e:
            # Programming errors should propagate so they are not silently ignored
            raise
        except Exception as e:
            logger.error("Unexpected error executing action (type=%s): %s\n%s",
                         action_type, e, traceback.format_exc())
            is_valid = False

        return reward, is_valid

    def _opponent_turn(self):
        """Execute opponent's turn."""
        if self.opponent_type == 'bot':
            if self.opponent:
                self.opponent.take_turn()
        elif self.opponent_type == 'random':
            self._random_opponent_turn()
        # 'self' mode is handled externally by the training script

    def _random_opponent_turn(self, max_actions: int = 20):
        """Execute random valid actions for opponent player 2."""
        for _ in range(max_actions):
            if self.game_state.game_over:
                break
            legal_actions = self.game_state.get_legal_actions(player=2)

            # Collect all non-end-turn actions
            all_actions = []
            for action_key in ['create_unit', 'move', 'attack', 'seize',
                               'paralyze', 'heal', 'cure', 'haste',
                               'defence_buff', 'attack_buff']:
                for action in legal_actions.get(action_key, []):
                    all_actions.append((action_key, action))

            if not all_actions:
                break  # Only end_turn available, stop

            # Pick a random action and execute it
            action_key, action = random.choice(all_actions)
            try:
                if action_key == 'create_unit':
                    self.game_state.create_unit(
                        action['unit_type'], action['x'], action['y'], player=2)
                elif action_key == 'move':
                    self.game_state.move_unit(
                        action['unit'], action['to_x'], action['to_y'])
                elif action_key == 'attack':
                    self.game_state.attack(action['attacker'], action['target'])
                elif action_key == 'seize':
                    self.game_state.seize(action['unit'])
                elif action_key == 'paralyze':
                    self.game_state.paralyze(action['paralyzer'], action['target'])
                elif action_key == 'heal':
                    self.game_state.heal(action['healer'], action['target'])
                elif action_key == 'cure':
                    self.game_state.cure(action['curer'], action['target'])
                elif action_key == 'haste':
                    self.game_state.haste(action['sorcerer'], action['target'])
                elif action_key == 'defence_buff':
                    self.game_state.defence_buff(action['sorcerer'], action['target'])
                elif action_key == 'attack_buff':
                    self.game_state.attack_buff(action['sorcerer'], action['target'])
            except Exception:
                continue  # Skip failed actions, try another

    def _compute_potential(self) -> float:
        """
        Compute potential function Phi(s) for potential-based reward shaping.

        Using potential-based shaping (Ng et al., 1999) preserves optimal policy:
        shaping = gamma * Phi(s') - Phi(s)
        Since gamma ~= 1.0 in practice, we use: shaping = Phi(s') - Phi(s)
        """
        potential = 0.0
        ap = self.agent_player
        opp = 3 - ap

        if self.reward_config.get('income_diff', 0) > 0:
            income_agent = self.game_state.mechanics.calculate_income(ap, self.game_state.grid)
            income_opp = self.game_state.mechanics.calculate_income(opp, self.game_state.grid)
            potential += (income_agent['total'] - income_opp['total']) * self.reward_config['income_diff']

        if self.reward_config.get('unit_diff', 0) > 0:
            units_agent = sum(1 for u in self.game_state.units if u.player == ap)
            units_opp = sum(1 for u in self.game_state.units if u.player == opp)
            potential += (units_agent - units_opp) * self.reward_config['unit_diff']

        if self.reward_config.get('structure_control', 0) > 0:
            structures_agent = len(self.game_state.grid.get_capturable_tiles(player=ap))
            structures_opp = len(self.game_state.grid.get_capturable_tiles(player=opp))
            potential += (structures_agent - structures_opp) * self.reward_config['structure_control']

        return potential

    def _calculate_reward(self, action_reward: float, is_valid: bool) -> float:
        """Calculate total reward including potential-based shaping terms."""
        reward = action_reward

        if not is_valid:
            reward += self.reward_config['invalid_action']
            self.episode_stats['invalid_actions'] += 1

        # Potential-based reward shaping: reward += Phi(s') - Phi(s)
        # This only rewards CHANGES in advantage, not maintaining a lead
        current_potential = self._compute_potential()
        reward += current_potential - self._prev_potential
        self._prev_potential = current_potential

        return reward

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        self.episode_stats['length'] = self.current_step

        # In hierarchical mode, extract the primitive action from the Dict
        if self.hierarchical and isinstance(action, dict):
            action = action['primitive']

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
            if self.game_state.winner == self.agent_player:
                reward += self.reward_config['win']
                self.episode_stats['winner'] = self.agent_player
            elif self.game_state.winner is None:
                # Draw (e.g. max_turns reached)
                reward += self.reward_config.get('draw', 0.0)
                self.episode_stats['winner'] = None
            else:
                reward += self.reward_config['loss']
                self.episode_stats['winner'] = self.game_state.winner

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

        # Reset game state (preserving enabled_units and fog_of_war configuration)
        self.game_state = GameState(
            self.initial_map_data,
            num_players=2,
            enabled_units=self.enabled_units,
            fog_of_war=self.fog_of_war
        )
        self.current_step = 0
        self._prev_potential = 0.0

        # Initialize visibility at game start
        if self.fog_of_war:
            self.game_state.update_visibility()

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
