"""
AlphaZero bot that uses a trained network + MCTS for game play.

Integrates with the existing bot interface so it can be used in
tournaments, human-vs-AI games, and evaluation.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from reinforcetactics.rl.alphazero_net import AlphaZeroNet
from reinforcetactics.rl.mcts import MCTS, _execute_action_on_state

logger = logging.getLogger(__name__)


class AlphaZeroBot:
    """
    Bot that uses a trained AlphaZero network with MCTS for decision-making.

    Compatible with the existing bot interface (take_turn method).
    """

    def __init__(
        self,
        game_state,
        player: int = 2,
        model_path: Optional[str] = None,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: str = 'cpu',
    ):
        """
        Args:
            game_state: GameState instance.
            player: Player number for this bot.
            model_path: Path to a .pt checkpoint file.
            num_simulations: MCTS simulations per move.
            c_puct: Exploration constant.
            temperature: Action selection temperature (0 = greedy).
            device: Torch device.
        """
        self.game_state = game_state
        self.bot_player = player
        self.temperature = temperature
        self.device = device
        self.num_simulations = num_simulations

        grid_width = game_state.grid.width
        grid_height = game_state.grid.height

        # Load or create network
        if model_path:
            self.network, self.mcts = self._load_model(
                model_path, grid_width, grid_height, num_simulations, c_puct, device,
            )
        else:
            self.network = AlphaZeroNet(
                grid_height=grid_height,
                grid_width=grid_width,
            ).to(device)
            self.network.eval()
            self.mcts = MCTS(
                network=self.network,
                grid_width=grid_width,
                grid_height=grid_height,
                num_simulations=num_simulations,
                c_puct=c_puct,
                device=device,
            )

    def _load_model(self, model_path: str, grid_width: int, grid_height: int,
                    num_simulations: int, c_puct: float, device: str):
        """Load a trained AlphaZero model from checkpoint."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(str(path), map_location=device, weights_only=False)

        # Get config from checkpoint or use defaults
        config = checkpoint.get('config', {})
        net_height = config.get('grid_height', grid_height)
        net_width = config.get('grid_width', grid_width)

        network = AlphaZeroNet(
            grid_height=net_height,
            grid_width=net_width,
        ).to(device)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()

        mcts = MCTS(
            network=network,
            grid_width=grid_width,
            grid_height=grid_height,
            num_simulations=num_simulations,
            c_puct=c_puct,
            device=device,
        )

        logger.info("Loaded AlphaZero model from %s (iteration %d)",
                     path, checkpoint.get('iteration', 0))
        return network, mcts

    def take_turn(self) -> None:
        """Execute the bot's turn using MCTS-guided decisions."""
        if self.game_state.current_player != self.bot_player:
            return

        max_actions = 50  # Safety limit per turn
        actions_taken = 0

        while actions_taken < max_actions:
            if self.game_state.game_over:
                break
            if self.game_state.current_player != self.bot_player:
                break

            flat_action, _ = self.mcts.select_action(
                self.game_state,
                temperature=self.temperature,
                add_noise=False,
            )

            action_info = self.mcts.get_action_info(self.game_state, flat_action)

            if action_info is None:
                # No valid action found, end turn
                self.game_state.end_turn()
                break

            if action_info['key'] == 'end_turn':
                self.game_state.end_turn()
                break

            try:
                _execute_action_on_state(
                    self.game_state,
                    action_info['key'],
                    action_info['action'],
                )
            except Exception as e:
                logger.debug("AlphaZero action failed: %s, ending turn", e)
                self.game_state.end_turn()
                break

            actions_taken += 1

        # Ensure turn is ended
        if self.game_state.current_player == self.bot_player:
            self.game_state.end_turn()
