"""
AlphaZero training pipeline for Reinforce Tactics.

Implements the full AlphaZero training loop:
1. Self-play: generate training data by having the network play against itself
   using MCTS to select moves.
2. Training: optimize the network on self-play data using a combined
   policy + value loss.
3. Evaluation: periodically pit the new network against the previous best
   to decide whether to accept the update.

The replay buffer stores (observation, mcts_policy, game_outcome) tuples
from recent self-play games.
"""

import copy
import json
import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from reinforcetactics.core.game_state import GameState
from reinforcetactics.rl.alphazero_net import AlphaZeroNet
from reinforcetactics.rl.mcts import MCTS, _execute_action_on_state, _obs_from_game_state
from reinforcetactics.utils.file_io import FileIO

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Fixed-size replay buffer that stores self-play training examples.

    Each example is a tuple of:
    - grid: (H, W, 3) numpy array
    - units: (H, W, 3) numpy array
    - global_features: (6,) numpy array
    - action_mask: (action_space_size,) numpy array
    - mcts_policy: (action_space_size,) numpy array of visit-count probabilities
    - value_target: float, game outcome from current player's perspective
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, examples: List[tuple]) -> None:
        """Add a list of training examples from a single game."""
        self.buffer.extend(examples)

    def sample(self, batch_size: int) -> List[tuple]:
        """Sample a random batch of examples."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


def self_play_game(
    mcts: MCTS,
    map_data: list,
    max_steps: int = 400,
    temperature_threshold: int = 30,
    enabled_units: Optional[list] = None,
) -> Tuple[List[tuple], int]:
    """
    Play a complete game using MCTS for both players.

    For the first `temperature_threshold` actions, temperature=1.0 is used
    to encourage exploration. After that, temperature=0 (greedy) is used.

    Args:
        mcts: MCTS instance with the current network.
        map_data: Map data to create the game.
        max_steps: Maximum actions before truncating.
        temperature_threshold: Number of actions before switching to greedy.
        enabled_units: Optional list of enabled unit types.

    Returns:
        (examples, winner) where examples is a list of
        (grid, units, global_features, action_mask, mcts_policy, current_player)
        tuples, and winner is the game winner (1, 2, or None for draw).
    """
    game_state = GameState(map_data, num_players=2, enabled_units=enabled_units)
    examples = []
    step = 0

    while not game_state.game_over and step < max_steps:
        temperature = 1.0 if step < temperature_threshold else 0.0

        # Get observation for the current state
        grid, units, global_features, action_mask = _obs_from_game_state(
            game_state,
            mcts.grid_width,
            mcts.grid_height,
        )

        # Run MCTS to get action probabilities
        flat_action, action_probs = mcts.select_action(
            game_state,
            temperature=temperature,
            add_noise=True,
        )

        # Store training example (value target will be filled in after game ends)
        examples.append((
            grid, units, global_features, action_mask,
            action_probs, game_state.current_player,
        ))

        # Execute the selected action
        action_info = mcts.get_action_info(game_state, flat_action)
        if action_info is None:
            # Fallback: end turn
            game_state.end_turn()
        else:
            try:
                _execute_action_on_state(game_state, action_info['key'], action_info['action'])
            except Exception:
                logger.debug("Self-play action failed, ending turn")
                game_state.end_turn()

        step += 1

    # Determine game outcome
    winner = game_state.winner if game_state.game_over else None

    # Assign value targets based on game outcome
    training_examples = []
    for grid, units, gf, mask, policy, player in examples:
        if winner is None:
            value_target = 0.0  # Draw
        elif winner == player:
            value_target = 1.0  # Win
        else:
            value_target = -1.0  # Loss

        training_examples.append((grid, units, gf, mask, policy, value_target))

    return training_examples, winner


class AlphaZeroTrainer:
    """
    Full AlphaZero training loop.

    Alternates between self-play data generation and network training.
    Optionally evaluates new networks against the previous best.
    """

    def __init__(
        self,
        map_file: Optional[str] = None,
        grid_height: int = 20,
        grid_width: int = 20,
        num_res_blocks: int = 6,
        channels: int = 128,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        replay_buffer_size: int = 100_000,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_iterations: int = 100,
        games_per_iteration: int = 25,
        epochs_per_iteration: int = 10,
        max_game_steps: int = 400,
        temperature_threshold: int = 30,
        eval_games: int = 20,
        eval_win_threshold: float = 0.55,
        checkpoint_dir: str = 'checkpoints/alphazero',
        device: str = 'cpu',
        enabled_units: Optional[list] = None,
    ):
        self.map_file = map_file
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_iterations = num_iterations
        self.games_per_iteration = games_per_iteration
        self.epochs_per_iteration = epochs_per_iteration
        self.max_game_steps = max_game_steps
        self.temperature_threshold = temperature_threshold
        self.eval_games = eval_games
        self.eval_win_threshold = eval_win_threshold
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.enabled_units = enabled_units

        # Load map data
        if map_file:
            self.map_data = FileIO.load_map(map_file)
        else:
            self.map_data = FileIO.generate_random_map(grid_width, grid_height, num_players=2)

        # Infer grid dimensions from map
        temp_gs = GameState(self.map_data, num_players=2, enabled_units=enabled_units)
        self.grid_height = temp_gs.grid.height
        self.grid_width = temp_gs.grid.width

        # Create network
        self.network = AlphaZeroNet(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            num_res_blocks=num_res_blocks,
            channels=channels,
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler: reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

        # MCTS config
        self.mcts_kwargs = dict(
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            device=device,
        )

        # Training history
        self.history = {
            'iteration': [],
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'self_play_wins_p1': [],
            'self_play_wins_p2': [],
            'self_play_draws': [],
        }

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict:
        """
        Run the full AlphaZero training loop.

        Returns:
            Training history dict.
        """
        logger.info(
            "Starting AlphaZero training: %d iterations, %d games/iter, "
            "%d simulations/move, grid=%dx%d",
            self.num_iterations, self.games_per_iteration,
            self.num_simulations, self.grid_width, self.grid_height,
        )

        best_network_state = copy.deepcopy(self.network.state_dict())

        for iteration in range(1, self.num_iterations + 1):
            iter_start = time.time()
            logger.info("=== Iteration %d/%d ===", iteration, self.num_iterations)

            # Phase 1: Self-play
            self.network.eval()
            examples, stats = self._self_play_phase()
            self.replay_buffer.push(examples)

            logger.info(
                "Self-play: %d games, P1 wins=%d, P2 wins=%d, draws=%d, "
                "%d examples (buffer=%d)",
                self.games_per_iteration,
                stats['p1_wins'], stats['p2_wins'], stats['draws'],
                len(examples), len(self.replay_buffer),
            )

            # Phase 2: Training
            self.network.train()
            train_stats = self._training_phase()

            logger.info(
                "Training: policy_loss=%.4f, value_loss=%.4f, total=%.4f",
                train_stats['policy_loss'], train_stats['value_loss'],
                train_stats['total_loss'],
            )

            # Phase 3: Evaluation (optional)
            if self.eval_games > 0 and iteration % 5 == 0:
                win_rate = self._evaluation_phase(best_network_state)
                logger.info("Evaluation: new network win rate = %.2f", win_rate)
                if win_rate >= self.eval_win_threshold:
                    logger.info("Accepting new network (%.2f >= %.2f)",
                                win_rate, self.eval_win_threshold)
                    best_network_state = copy.deepcopy(self.network.state_dict())
                else:
                    logger.info("Rejecting new network (%.2f < %.2f), reverting",
                                win_rate, self.eval_win_threshold)
                    self.network.load_state_dict(best_network_state)

            # Record history
            self.history['iteration'].append(iteration)
            self.history['policy_loss'].append(train_stats['policy_loss'])
            self.history['value_loss'].append(train_stats['value_loss'])
            self.history['total_loss'].append(train_stats['total_loss'])
            self.history['self_play_wins_p1'].append(stats['p1_wins'])
            self.history['self_play_wins_p2'].append(stats['p2_wins'])
            self.history['self_play_draws'].append(stats['draws'])

            # Checkpoint
            if iteration % 10 == 0 or iteration == self.num_iterations:
                self._save_checkpoint(iteration)

            elapsed = time.time() - iter_start
            logger.info("Iteration %d complete in %.1fs", iteration, elapsed)

        # Save final model
        self._save_checkpoint(self.num_iterations, is_final=True)
        self._save_history()

        return self.history

    def _self_play_phase(self) -> Tuple[List[tuple], Dict]:
        """Generate self-play training data."""
        mcts = MCTS(network=self.network, **self.mcts_kwargs)
        all_examples = []
        stats = {'p1_wins': 0, 'p2_wins': 0, 'draws': 0}

        for game_idx in range(self.games_per_iteration):
            # Optionally randomize map for diversity
            if self.map_file is None:
                map_data = FileIO.generate_random_map(
                    self.grid_width, self.grid_height, num_players=2,
                )
            else:
                map_data = self.map_data

            examples, winner = self_play_game(
                mcts=mcts,
                map_data=map_data,
                max_steps=self.max_game_steps,
                temperature_threshold=self.temperature_threshold,
                enabled_units=self.enabled_units,
            )

            all_examples.extend(examples)

            if winner == 1:
                stats['p1_wins'] += 1
            elif winner == 2:
                stats['p2_wins'] += 1
            else:
                stats['draws'] += 1

            if (game_idx + 1) % 5 == 0:
                logger.debug("Self-play game %d/%d complete",
                             game_idx + 1, self.games_per_iteration)

        return all_examples, stats

    def _training_phase(self) -> Dict[str, float]:
        """Train the network on replay buffer data."""
        if len(self.replay_buffer) < self.batch_size:
            logger.info("Buffer too small (%d < %d), skipping training",
                        len(self.replay_buffer), self.batch_size)
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(self.epochs_per_iteration):
            batch = self.replay_buffer.sample(self.batch_size)

            # Unpack batch
            grids = torch.tensor(
                np.array([ex[0] for ex in batch]), dtype=torch.float32, device=self.device,
            )
            units_batch = torch.tensor(
                np.array([ex[1] for ex in batch]), dtype=torch.float32, device=self.device,
            )
            global_features = torch.tensor(
                np.array([ex[2] for ex in batch]), dtype=torch.float32, device=self.device,
            )
            masks = torch.tensor(
                np.array([ex[3] for ex in batch]), dtype=torch.float32, device=self.device,
            )
            target_policies = torch.tensor(
                np.array([ex[4] for ex in batch]), dtype=torch.float32, device=self.device,
            )
            target_values = torch.tensor(
                np.array([ex[5] for ex in batch]), dtype=torch.float32, device=self.device,
            ).unsqueeze(1)

            # Forward pass
            policy_logits, values = self.network(grids, units_batch, global_features)

            # Policy loss: cross-entropy between MCTS policy and network output
            # Apply mask before log_softmax
            masked_logits = policy_logits.clone()
            masked_logits[masks == 0] = -1e8
            log_probs = F.log_softmax(masked_logits, dim=-1)
            policy_loss = -torch.sum(target_policies * log_probs, dim=-1).mean()

            # Value loss: MSE between predicted value and game outcome
            value_loss = F.mse_loss(values, target_values)

            # Combined loss
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        avg_policy = total_policy_loss / max(num_batches, 1)
        avg_value = total_value_loss / max(num_batches, 1)

        # Step LR scheduler
        self.scheduler.step(avg_policy + avg_value)

        return {
            'policy_loss': avg_policy,
            'value_loss': avg_value,
            'total_loss': avg_policy + avg_value,
        }

    def _evaluation_phase(self, best_state_dict: dict) -> float:
        """
        Evaluate the current network against the previous best.

        Plays eval_games games, alternating which player uses which network.

        Returns:
            Win rate of the current network (0.0 to 1.0).
        """
        # Create opponent network with the best weights
        opponent_net = AlphaZeroNet(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        ).to(self.device)
        opponent_net.load_state_dict(best_state_dict)
        opponent_net.eval()

        current_mcts = MCTS(network=self.network, **self.mcts_kwargs)
        opponent_mcts = MCTS(network=opponent_net, **self.mcts_kwargs)

        wins = 0
        losses = 0
        draws = 0

        for game_idx in range(self.eval_games):
            # Alternate who plays as player 1
            current_is_p1 = (game_idx % 2 == 0)
            p1_mcts = current_mcts if current_is_p1 else opponent_mcts
            p2_mcts = opponent_mcts if current_is_p1 else current_mcts

            winner = self._play_eval_game(p1_mcts, p2_mcts)

            if winner is None:
                draws += 1
            elif (winner == 1 and current_is_p1) or (winner == 2 and not current_is_p1):
                wins += 1
            else:
                losses += 1

        total_decided = wins + losses
        if total_decided == 0:
            return 0.5
        return wins / total_decided

    def _play_eval_game(self, p1_mcts: MCTS, p2_mcts: MCTS) -> Optional[int]:
        """Play a single evaluation game. Returns winner (1, 2, or None)."""
        game_state = GameState(
            self.map_data, num_players=2, enabled_units=self.enabled_units,
        )

        for _ in range(self.max_game_steps):
            if game_state.game_over:
                break

            mcts = p1_mcts if game_state.current_player == 1 else p2_mcts

            flat_action, _ = mcts.select_action(
                game_state, temperature=0, add_noise=False,
            )

            action_info = mcts.get_action_info(game_state, flat_action)
            if action_info is None:
                game_state.end_turn()
            else:
                try:
                    _execute_action_on_state(
                        game_state, action_info['key'], action_info['action'],
                    )
                except Exception:
                    game_state.end_turn()

        return game_state.winner if game_state.game_over else None

    def _save_checkpoint(self, iteration: int, is_final: bool = False) -> None:
        """Save a training checkpoint."""
        suffix = 'final' if is_final else f'iter_{iteration:04d}'
        path = self.checkpoint_dir / f'alphazero_{suffix}.pt'

        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': {
                'grid_height': self.grid_height,
                'grid_width': self.grid_width,
                'num_simulations': self.num_simulations,
            },
        }, path)

        logger.info("Saved checkpoint: %s", path)

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = self.checkpoint_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info("Saved training history: %s", path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu', **kwargs) -> 'AlphaZeroTrainer':
        """
        Load a trainer from a checkpoint.

        Args:
            path: Path to the .pt checkpoint file.
            device: Torch device.
            **kwargs: Additional kwargs to override trainer config.

        Returns:
            AlphaZeroTrainer instance with loaded weights.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        config.update(kwargs)
        config['device'] = device

        trainer = cls(**config)
        trainer.network.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'history' in checkpoint:
            trainer.history = checkpoint['history']

        logger.info("Loaded checkpoint from iteration %d: %s",
                     checkpoint.get('iteration', 0), path)
        return trainer
