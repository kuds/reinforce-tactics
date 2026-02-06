"""
Monte Carlo Tree Search (MCTS) for AlphaZero-style planning.

Uses a neural network to guide tree search with the PUCT selection formula.
Handles the multi-action-per-turn nature of Reinforce Tactics by treating
each individual action (including end_turn) as a tree edge.

Key design decisions:
- Actions are represented as flat indices into the 10*W*H action mask space.
  When multiple legal actions map to the same flat index (e.g. two units can
  attack the same target), the first matching legal action is used.
- The neural network provides prior probabilities over flat action indices.
- Dirichlet noise is added at the root for exploration.
- Game states are cloned via deepcopy for simulation.
"""

import copy
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MCTSNode:
    """
    A node in the MCTS tree.

    Each node represents a game state. Edges to children represent actions.
    Statistics are stored on the edges (parent -> child).
    """

    __slots__ = [
        'parent', 'action', 'children', 'visit_count', 'value_sum',
        'prior', 'game_state', 'player', 'is_terminal', '_legal_actions_cache',
        '_action_info',
    ]

    def __init__(
        self,
        game_state,
        parent: Optional['MCTSNode'] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.parent = parent
        self.action = action  # flat action index that led to this node
        self.children: Dict[int, 'MCTSNode'] = {}  # flat_action -> child node
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.game_state = game_state
        self.player: int = game_state.current_player if game_state else 0
        self.is_terminal: bool = game_state.game_over if game_state else False
        self._legal_actions_cache = None

    @property
    def q_value(self) -> float:
        """Mean action value Q = W / N."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def get_legal_flat_actions(self, grid_width: int, grid_height: int) -> Dict[int, dict]:
        """
        Get legal actions mapped to flat action indices.

        Returns:
            Dict mapping flat_index -> first legal action dict for that index.
            Each action dict has 'key' (action type string) and 'action' (details).
        """
        if self._legal_actions_cache is not None:
            return self._legal_actions_cache

        legal_actions = self.game_state.get_legal_actions(player=self.player)
        area = grid_width * grid_height
        flat_map = {}

        def flat_idx(action_type_idx: int, x: int, y: int) -> int:
            return action_type_idx * area + y * grid_width + x

        # Map structured actions to flat indices
        # 0: Create unit
        for action in legal_actions.get('create_unit', []):
            idx = flat_idx(0, action['x'], action['y'])
            if idx not in flat_map:
                flat_map[idx] = {'key': 'create_unit', 'action': action}

        # 1: Move
        for action in legal_actions.get('move', []):
            idx = flat_idx(1, action['to_x'], action['to_y'])
            if idx not in flat_map:
                flat_map[idx] = {'key': 'move', 'action': action}

        # 2: Attack
        for action in legal_actions.get('attack', []):
            target = action['target']
            idx = flat_idx(2, target.x, target.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'attack', 'action': action}

        # 3: Seize
        for action in legal_actions.get('seize', []):
            tile = action['tile']
            idx = flat_idx(3, tile.x, tile.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'seize', 'action': action}

        # 4: Heal / Cure
        for action in legal_actions.get('heal', []):
            target = action['target']
            idx = flat_idx(4, target.x, target.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'heal', 'action': action}

        for action in legal_actions.get('cure', []):
            target = action['target']
            idx = flat_idx(4, target.x, target.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'cure', 'action': action}

        # 5: End turn — map to a single canonical index (0, 0)
        idx = flat_idx(5, 0, 0)
        flat_map[idx] = {'key': 'end_turn', 'action': {}}

        # 6: Paralyze
        for action in legal_actions.get('paralyze', []):
            target = action['target']
            idx = flat_idx(6, target.x, target.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'paralyze', 'action': action}

        # 7: Haste
        for action in legal_actions.get('haste', []):
            target = action['target']
            idx = flat_idx(7, target.x, target.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'haste', 'action': action}

        # 8: Defence buff
        for action in legal_actions.get('defence_buff', []):
            target = action['target']
            idx = flat_idx(8, target.x, target.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'defence_buff', 'action': action}

        # 9: Attack buff
        for action in legal_actions.get('attack_buff', []):
            target = action['target']
            idx = flat_idx(9, target.x, target.y)
            if idx not in flat_map:
                flat_map[idx] = {'key': 'attack_buff', 'action': action}

        self._legal_actions_cache = flat_map
        return flat_map


def _execute_action_on_state(game_state, action_key: str, action_data: dict) -> None:
    """Execute a structured action on a game state (mutates in place)."""
    if action_key == 'create_unit':
        game_state.create_unit(
            action_data['unit_type'],
            action_data['x'],
            action_data['y'],
            player=game_state.current_player,
        )
    elif action_key == 'move':
        game_state.move_unit(action_data['unit'], action_data['to_x'], action_data['to_y'])
    elif action_key == 'attack':
        game_state.attack(action_data['attacker'], action_data['target'])
    elif action_key == 'seize':
        game_state.seize(action_data['unit'])
    elif action_key == 'heal':
        game_state.heal(action_data['healer'], action_data['target'])
    elif action_key == 'cure':
        game_state.cure(action_data['curer'], action_data['target'])
    elif action_key == 'end_turn':
        game_state.end_turn()
    elif action_key == 'paralyze':
        game_state.paralyze(action_data['paralyzer'], action_data['target'])
    elif action_key == 'haste':
        game_state.haste(action_data['sorcerer'], action_data['target'])
    elif action_key == 'defence_buff':
        game_state.defence_buff(action_data['sorcerer'], action_data['target'])
    elif action_key == 'attack_buff':
        game_state.attack_buff(action_data['sorcerer'], action_data['target'])


def _obs_from_game_state(game_state, grid_width: int, grid_height: int,
                         num_action_types: int = 10):
    """
    Extract observation tensors from a GameState for neural network evaluation.

    Returns:
        (grid, units, global_features, action_mask) as numpy arrays.
    """
    state_arrays = game_state.to_numpy()
    grid = state_arrays['grid'].astype(np.float32)
    units = state_arrays['units'].astype(np.float32)

    global_features = np.array([
        game_state.player_gold.get(1, 0),
        game_state.player_gold.get(2, 0),
        game_state.turn_number,
        sum(1 for u in game_state.units if u.player == 1),
        sum(1 for u in game_state.units if u.player == 2),
        game_state.current_player,
    ], dtype=np.float32)

    # Build flat action mask
    area = grid_width * grid_height
    mask_size = num_action_types * area
    mask = np.zeros(mask_size, dtype=np.float32)

    legal_actions = game_state.get_legal_actions(player=game_state.current_player)

    def set_mask(action_type_idx, x, y):
        idx = action_type_idx * area + y * grid_width + x
        if 0 <= idx < mask_size:
            mask[idx] = 1.0

    for action in legal_actions.get('create_unit', []):
        set_mask(0, action['x'], action['y'])
    for action in legal_actions.get('move', []):
        set_mask(1, action['to_x'], action['to_y'])
    for action in legal_actions.get('attack', []):
        set_mask(2, action['target'].x, action['target'].y)
    for action in legal_actions.get('seize', []):
        set_mask(3, action['tile'].x, action['tile'].y)
    for action in legal_actions.get('heal', []):
        set_mask(4, action['target'].x, action['target'].y)
    for action in legal_actions.get('cure', []):
        set_mask(4, action['target'].x, action['target'].y)
    # End turn: always valid at canonical position (0,0)
    set_mask(5, 0, 0)
    for action in legal_actions.get('paralyze', []):
        set_mask(6, action['target'].x, action['target'].y)
    for action in legal_actions.get('haste', []):
        set_mask(7, action['target'].x, action['target'].y)
    for action in legal_actions.get('defence_buff', []):
        set_mask(8, action['target'].x, action['target'].y)
    for action in legal_actions.get('attack_buff', []):
        set_mask(9, action['target'].x, action['target'].y)

    return grid, units, global_features, mask


class MCTS:
    """
    Monte Carlo Tree Search guided by a neural network.

    Implements the AlphaZero variant:
    - Selection via PUCT (Polynomial Upper Confidence Trees)
    - Expansion & evaluation via the neural network (no rollouts)
    - Dirichlet noise at the root for exploration
    """

    def __init__(
        self,
        network: 'torch.nn.Module',
        grid_width: int = 20,
        grid_height: int = 20,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = 'cpu',
    ):
        """
        Args:
            network: AlphaZeroNet instance for policy & value prediction.
            grid_width: Width of the game grid.
            grid_height: Height of the game grid.
            num_simulations: Number of MCTS simulations per move.
            c_puct: Exploration constant in PUCT formula.
            dirichlet_alpha: Alpha parameter for Dirichlet noise at root.
            dirichlet_epsilon: Weight of Dirichlet noise vs. network prior.
            device: Torch device ('cpu' or 'cuda').
        """
        self.network = network
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

    @torch.no_grad()
    def _evaluate(self, game_state) -> Tuple[np.ndarray, float]:
        """
        Evaluate a game state with the neural network.

        Returns:
            (policy_probs, value) where policy_probs is over the flat action space
            and value is from the current player's perspective.
        """
        grid, units, global_features, mask = _obs_from_game_state(
            game_state, self.grid_width, self.grid_height
        )

        # To tensors with batch dim
        grid_t = torch.tensor(grid, device=self.device).unsqueeze(0)
        units_t = torch.tensor(units, device=self.device).unsqueeze(0)
        gf_t = torch.tensor(global_features, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, device=self.device).unsqueeze(0)

        policy_probs, value = self.network.predict(grid_t, units_t, gf_t, mask_t)

        return policy_probs.squeeze(0).cpu().numpy(), value.item()

    def search(self, game_state, add_noise: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run MCTS from the given game state.

        Args:
            game_state: Current GameState to search from.
            add_noise: Whether to add Dirichlet noise at root (True during
                       self-play, False during evaluation).

        Returns:
            (action_probs, root_value) where action_probs is a distribution
            over the flat action space based on visit counts, and root_value
            is the mean value estimate at the root.
        """
        # Create root node with a deep copy so simulations don't mutate the real state
        root = MCTSNode(game_state=copy.deepcopy(game_state))

        # Evaluate root
        policy_probs, root_value = self._evaluate(root.game_state)

        # Expand root
        legal_flat = root.get_legal_flat_actions(self.grid_width, self.grid_height)
        self._expand_node(root, policy_probs, legal_flat)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children)
            )
            for i, child in enumerate(root.children.values()):
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * noise[i]
                )

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using PUCT until we reach a leaf
            while node.is_expanded and not node.is_terminal:
                action_idx, node = self._select_child(node)
                search_path.append(node)

            # Evaluate leaf
            if node.is_terminal:
                # Terminal value from the perspective of the node's player
                value = self._terminal_value(node, root.player)
            else:
                # Neural network evaluation
                policy_probs, value = self._evaluate(node.game_state)
                legal_flat = node.get_legal_flat_actions(
                    self.grid_width, self.grid_height
                )
                if legal_flat:
                    self._expand_node(node, policy_probs, legal_flat)
                # Value is from the node's current player's perspective.
                # We need it from root player's perspective for backup.
                if node.player != root.player:
                    value = -value

            # Backup: propagate value up the search path
            self._backup(search_path, value, root.player)

        # Build action probability distribution from visit counts
        action_space_size = self.grid_width * self.grid_height * 10
        action_probs = np.zeros(action_space_size, dtype=np.float32)
        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits > 0:
            for action_idx, child in root.children.items():
                action_probs[action_idx] = child.visit_count / total_visits

        return action_probs, root.q_value

    def _expand_node(self, node: MCTSNode, policy_probs: np.ndarray,
                     legal_flat: Dict[int, dict]) -> None:
        """Expand a node by creating child nodes for all legal actions."""
        for flat_idx, action_info in legal_flat.items():
            prior = policy_probs[flat_idx] if flat_idx < len(policy_probs) else 0.0
            # We don't create the child game state yet (lazy expansion).
            # It will be created when the child is first visited during selection.
            child = MCTSNode(
                game_state=None,
                parent=node,
                action=flat_idx,
                prior=float(prior),
            )
            # Store the action info for later execution
            child._action_info = action_info  # noqa: SLF001
            node.children[flat_idx] = child

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """Select the child with highest PUCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent = math.sqrt(node.visit_count + 1)

        for action_idx, child in node.children.items():
            # PUCT formula
            if child.visit_count > 0:
                q = child.q_value
                # Flip value if child is opponent's turn
                if child.player != node.player and child.game_state is not None:
                    q = -q
            else:
                q = 0.0

            exploration = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + exploration

            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        # Lazy state creation: if child doesn't have a game state yet, create it
        if best_child is not None and best_child.game_state is None:
            best_child.game_state = copy.deepcopy(node.game_state)
            action_info = best_child._action_info  # noqa: SLF001
            try:
                _execute_action_on_state(
                    best_child.game_state,
                    action_info['key'],
                    action_info['action'],
                )
            except Exception:
                # If action fails, treat as terminal loss
                logger.debug("MCTS action execution failed for %s", action_info['key'])
                best_child.is_terminal = True

            best_child.player = best_child.game_state.current_player
            best_child.is_terminal = best_child.game_state.game_over

        return best_action, best_child

    def _backup(self, search_path: List[MCTSNode], value: float,
                root_player: int) -> None:
        """Propagate the value back up the search path."""
        for node in search_path:
            node.visit_count += 1
            # Value is always stored from the root player's perspective
            if node.player == root_player:
                node.value_sum += value
            else:
                node.value_sum -= value

    def _terminal_value(self, node: MCTSNode, root_player: int) -> float:
        """Get the terminal value from the root player's perspective."""
        gs = node.game_state
        if gs.winner is None:
            return 0.0  # Draw
        if gs.winner == root_player:
            return 1.0
        return -1.0

    def select_action(self, game_state, temperature: float = 1.0,
                      add_noise: bool = True) -> Tuple[int, np.ndarray]:
        """
        Run MCTS and select an action.

        Args:
            game_state: Current GameState.
            temperature: Controls exploration vs. exploitation.
                         >0 for proportional to visit counts,
                         0 for greedy (argmax).
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            (selected_flat_action, action_probs) where action_probs is the
            full MCTS policy distribution.
        """
        action_probs, _ = self.search(game_state, add_noise=add_noise)

        if temperature == 0:
            # Greedy selection
            action = int(np.argmax(action_probs))
        else:
            # Temperature-adjusted sampling
            if temperature != 1.0:
                # Raise visit counts to 1/temperature
                probs = action_probs ** (1.0 / temperature)
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                else:
                    probs = action_probs
            else:
                probs = action_probs

            # Sample from distribution
            nonzero = np.nonzero(probs)[0]
            if len(nonzero) == 0:
                # Fallback: pick end_turn
                action = 5 * self.grid_width * self.grid_height
            else:
                action = int(np.random.choice(len(probs), p=probs))

        return action, action_probs

    def get_action_info(self, game_state, flat_action: int) -> Optional[dict]:
        """
        Resolve a flat action index to a structured action using legal actions.

        Args:
            game_state: Current GameState.
            flat_action: Flat action index.

        Returns:
            Dict with 'key' and 'action', or None if not found.
        """
        node = MCTSNode(game_state=game_state)
        legal_flat = node.get_legal_flat_actions(self.grid_width, self.grid_height)
        return legal_flat.get(flat_action)
