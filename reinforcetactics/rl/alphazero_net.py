"""
AlphaZero neural network for Reinforce Tactics.

Dual-head architecture:
- Shared residual CNN backbone that processes the spatial game state
- Policy head: outputs log-probabilities over the flat action space (10 * W * H)
- Value head: outputs a scalar in [-1, 1] estimating the expected outcome

The network takes the same observation format as StrategyGameEnv:
- grid: (H, W, 3) - terrain type, owner, structure HP
- units: (H, W, 3) - unit type, owner, HP
- global_features: (6,) - gold, turn, unit counts, current player
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-activation residual block with batch normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style dual-head network for Reinforce Tactics.

    Architecture:
        Input -> Conv -> N ResidualBlocks -> (PolicyHead, ValueHead)

    The policy head outputs logits over the flat action space of size
    num_action_types * grid_width * grid_height. During MCTS, these are
    masked to legal actions and converted to probabilities.

    The value head outputs a scalar in [-1, 1] representing the expected
    game outcome from the current player's perspective.
    """

    def __init__(
        self,
        grid_height: int = 20,
        grid_width: int = 20,
        num_action_types: int = 10,
        num_res_blocks: int = 6,
        channels: int = 128,
        global_features_dim: int = 6,
    ):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_action_types = num_action_types
        self.action_space_size = num_action_types * grid_width * grid_height

        # Input: 6 spatial channels (grid=3 + units=3) + global features
        # broadcast to spatial dims
        n_input_channels = 6 + global_features_dim
        self.global_features_dim = global_features_dim

        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(32 * grid_height * grid_width, self.action_space_size)

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(grid_height * grid_width, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, grid: torch.Tensor, units: torch.Tensor,
                global_features: torch.Tensor):
        """
        Forward pass.

        Args:
            grid: (B, H, W, 3) terrain/owner/structure tensor
            units: (B, H, W, 3) unit type/owner/HP tensor
            global_features: (B, 6) scalar features

        Returns:
            policy_logits: (B, action_space_size) raw logits
            value: (B, 1) scalar value in [-1, 1]
        """
        batch_size = grid.shape[0]

        # Combine spatial features: (B, H, W, 6) -> (B, 6, H, W)
        spatial = torch.cat([grid, units], dim=-1).permute(0, 3, 1, 2)

        # Broadcast global features to spatial dims: (B, 6) -> (B, 6, H, W)
        global_expanded = global_features.unsqueeze(-1).unsqueeze(-1)
        global_expanded = global_expanded.expand(
            batch_size, self.global_features_dim, self.grid_height, self.grid_width
        )

        # Concatenate: (B, 12, H, W)
        x = torch.cat([spatial, global_expanded], dim=1)

        # Shared backbone
        x = self.input_conv(x)
        x = self.res_blocks(x)

        # Policy head
        p = self.policy_head(x)
        p = p.reshape(batch_size, -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_head(x)
        v = v.reshape(batch_size, -1)
        value = self.value_fc(v)

        return policy_logits, value

    def predict(self, grid: torch.Tensor, units: torch.Tensor,
                global_features: torch.Tensor, action_mask: torch.Tensor):
        """
        Predict policy probabilities and value, applying action mask.

        Args:
            grid: (B, H, W, 3)
            units: (B, H, W, 3)
            global_features: (B, 6)
            action_mask: (B, action_space_size) binary mask of legal actions

        Returns:
            policy_probs: (B, action_space_size) masked & normalized probabilities
            value: (B, 1) scalar value in [-1, 1]
        """
        policy_logits, value = self.forward(grid, units, global_features)

        # Mask illegal actions with large negative value before softmax
        masked_logits = policy_logits.clone()
        masked_logits[action_mask == 0] = -1e8

        policy_probs = F.softmax(masked_logits, dim=-1)
        return policy_probs, value
