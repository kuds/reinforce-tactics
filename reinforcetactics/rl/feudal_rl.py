"""
Feudal Reinforcement Learning Architecture
Manager-Worker hierarchy for strategy games
"""
from typing import Tuple, Dict, Optional
import torch
from torch import nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatialFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN-based feature extractor for spatial game state.
    Processes grid channels, unit channels, and global features
    (gold, turn number, unit counts, current player).
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        # Assuming observation is dict with 'grid' and 'units'
        # grid: (H, W, 3), units: (H, W, 3)
        n_input_channels = 6  # 3 for grid + 3 for units

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Determine global_features size if present in observation space
        if 'global_features' in observation_space.spaces:
            self.n_global = observation_space['global_features'].shape[0]
        else:
            self.n_global = 0

        # Compute CNN output shape by doing one forward pass
        with torch.no_grad():
            sample_obs = observation_space.sample()
            grid = torch.as_tensor(sample_obs['grid']).float()
            units = torch.as_tensor(sample_obs['units']).float()
            combined = torch.cat([grid, units], dim=-1).permute(2, 0, 1).unsqueeze(0)
            n_flatten = self.cnn(combined).flatten(1).shape[1]

        # Linear projection: CNN output + global features -> features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.n_global, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Dict with 'grid', 'units', and optionally 'global_features'

        Returns:
            Feature tensor of shape (batch, features_dim)
        """
        grid = observations['grid']  # (B, H, W, 3)
        units = observations['units']  # (B, H, W, 3)

        # Combine and permute to (B, C, H, W)
        combined = torch.cat([grid, units], dim=-1)  # (B, H, W, 6)
        combined = combined.permute(0, 3, 1, 2)  # (B, 6, H, W)

        # CNN forward
        features = self.cnn(combined)  # (B, 64, H, W)
        features = features.flatten(1)  # (B, 64*H*W)

        # Concatenate global features (gold, turn, unit counts, current player)
        if self.n_global > 0 and 'global_features' in observations:
            global_feat = observations['global_features']  # (B, 6)
            features = torch.cat([features, global_feat], dim=1)  # (B, 64*H*W + 6)

        # Linear projection
        features = self.linear(features)  # (B, features_dim)

        return features


class ManagerNetwork(nn.Module):
    """
    Manager network for high-level goal generation.
    Outputs spatial goals: (goal_x, goal_y, goal_type)
    """

    def __init__(
        self,
        feature_dim: int = 512,
        grid_width: int = 20,
        grid_height: int = 20,
        num_goal_types: int = 4  # attack, defend, capture, expand
    ):
        super().__init__()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_goal_types = num_goal_types

        # Process features
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Goal heads
        self.goal_x_head = nn.Linear(256, grid_width)
        self.goal_y_head = nn.Linear(256, grid_height)
        self.goal_type_head = nn.Linear(256, num_goal_types)

        # Value head for critic
        self.value_head = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, feature_dim)

        Returns:
            goal_x_logits: (batch, grid_width)
            goal_y_logits: (batch, grid_height)
            goal_type_logits: (batch, num_goal_types)
            value: (batch, 1)
        """
        x = self.mlp(features)

        goal_x_logits = self.goal_x_head(x)
        goal_y_logits = self.goal_y_head(x)
        goal_type_logits = self.goal_type_head(x)
        value = self.value_head(x)

        return goal_x_logits, goal_y_logits, goal_type_logits, value

    def sample_goal(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a goal from the policy.

        Returns:
            goal: (batch, 3) - [goal_x, goal_y, goal_type]
            log_prob: (batch,) - log probability of sampled goal
        """
        goal_x_logits, goal_y_logits, goal_type_logits, _ = self.forward(features)

        # Sample from categorical distributions
        goal_x_dist = torch.distributions.Categorical(logits=goal_x_logits)
        goal_y_dist = torch.distributions.Categorical(logits=goal_y_logits)
        goal_type_dist = torch.distributions.Categorical(logits=goal_type_logits)

        goal_x = goal_x_dist.sample()
        goal_y = goal_y_dist.sample()
        goal_type = goal_type_dist.sample()

        # Compute log probability
        log_prob = (
            goal_x_dist.log_prob(goal_x) +
            goal_y_dist.log_prob(goal_y) +
            goal_type_dist.log_prob(goal_type)
        )

        goal = torch.stack([goal_x, goal_y, goal_type], dim=1)

        return goal, log_prob

    def evaluate_goal(self, features: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a given goal.

        Args:
            features: (batch, feature_dim)
            goal: (batch, 3) - [goal_x, goal_y, goal_type]

        Returns:
            log_prob: (batch,)
            entropy: (batch,)
            value: (batch, 1)
        """
        goal_x_logits, goal_y_logits, goal_type_logits, value = self.forward(features)

        goal_x_dist = torch.distributions.Categorical(logits=goal_x_logits)
        goal_y_dist = torch.distributions.Categorical(logits=goal_y_logits)
        goal_type_dist = torch.distributions.Categorical(logits=goal_type_logits)

        goal_x, goal_y, goal_type = goal[:, 0].long(), goal[:, 1].long(), goal[:, 2].long()

        log_prob = (
            goal_x_dist.log_prob(goal_x) +
            goal_y_dist.log_prob(goal_y) +
            goal_type_dist.log_prob(goal_type)
        )

        entropy = (
            goal_x_dist.entropy() +
            goal_y_dist.entropy() +
            goal_type_dist.entropy()
        )

        return log_prob, entropy, value


class WorkerNetwork(nn.Module):
    """
    Worker network for low-level action execution.
    Conditioned on manager's goal.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        goal_embedding_dim: int = 64,
        action_space_dims: list = [10, 8, 20, 20, 20, 20]  # 10 action types, 8 unit types
    ):
        super().__init__()

        self.action_space_dims = action_space_dims

        # Goal embedding
        self.goal_embedding = nn.Sequential(
            nn.Linear(3, goal_embedding_dim),  # goal is (x, y, type)
            nn.ReLU(),
            nn.Linear(goal_embedding_dim, goal_embedding_dim),
            nn.ReLU()
        )

        # Combined processing
        combined_dim = feature_dim + goal_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Action heads (one for each dimension)
        self.action_heads = nn.ModuleList([
            nn.Linear(256, dim) for dim in action_space_dims
        ])

        # Value head
        self.value_head = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, goal: torch.Tensor) -> Tuple[list, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, feature_dim)
            goal: (batch, 3) - [goal_x, goal_y, goal_type]

        Returns:
            action_logits: List of (batch, action_dim) tensors
            value: (batch, 1)
        """
        # Embed goal
        goal_emb = self.goal_embedding(goal.float())

        # Combine features and goal
        combined = torch.cat([features, goal_emb], dim=1)
        x = self.mlp(combined)

        # Compute action logits
        action_logits = [head(x) for head in self.action_heads]

        # Compute value
        value = self.value_head(x)

        return action_logits, value

    def sample_action(self, features: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action.

        Returns:
            action: (batch, len(action_space_dims))
            log_prob: (batch,)
        """
        action_logits, _ = self.forward(features, goal)

        # Sample from each dimension
        actions = []
        log_probs = []

        for logits in action_logits:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            actions.append(a)
            log_probs.append(dist.log_prob(a))

        action = torch.stack(actions, dim=1)
        log_prob = torch.stack(log_probs, dim=1).sum(dim=1)

        return action, log_prob

    def evaluate_action(
        self,
        features: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a given action.

        Returns:
            log_prob: (batch,)
            entropy: (batch,)
            value: (batch, 1)
        """
        action_logits, value = self.forward(features, goal)

        log_probs = []
        entropies = []

        for i, logits in enumerate(action_logits):
            dist = torch.distributions.Categorical(logits=logits)
            a = action[:, i].long()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

        log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        entropy = torch.stack(entropies, dim=1).sum(dim=1)

        return log_prob, entropy, value


class FeudalRLAgent:
    """
    Complete Feudal RL agent with manager and worker.
    """

    def __init__(
        self,
        observation_space,
        grid_width: int = 20,
        grid_height: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Feature extractor (shared)
        self.feature_extractor = SpatialFeatureExtractor(
            observation_space,
            features_dim=512
        ).to(device)

        # Manager network
        self.manager = ManagerNetwork(
            feature_dim=512,
            grid_width=grid_width,
            grid_height=grid_height,
            num_goal_types=4
        ).to(device)

        # Worker network
        self.worker = WorkerNetwork(
            feature_dim=512,
            goal_embedding_dim=64,
            action_space_dims=[10, 8, grid_width, grid_height, grid_width, grid_height]
        ).to(device)

        # Current goal (maintained across steps)
        self.current_goal = None
        self.goal_step_counter = 0
        self.manager_horizon = 10  # Update goal every N steps

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Select action using manager-worker hierarchy.

        Returns:
            action: Primitive action array
            goal: Current goal (for logging/debugging)
        """
        # Convert observation to tensor (grid, units, and global_features)
        obs_tensor = {
            k: torch.as_tensor(v).unsqueeze(0).float().to(self.device)
            for k, v in observation.items()
            if k in ['grid', 'units', 'global_features']
        }

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(obs_tensor)

            # Update goal if needed
            if self.current_goal is None or self.goal_step_counter >= self.manager_horizon:
                if deterministic:
                    # Use mode of distribution
                    goal_x_logits, goal_y_logits, goal_type_logits, _ = self.manager(features)
                    goal_x = goal_x_logits.argmax(dim=1)
                    goal_y = goal_y_logits.argmax(dim=1)
                    goal_type = goal_type_logits.argmax(dim=1)
                    self.current_goal = torch.stack([goal_x, goal_y, goal_type], dim=1)
                else:
                    self.current_goal, _ = self.manager.sample_goal(features)

                self.goal_step_counter = 0

            # Worker selects action conditioned on goal
            if deterministic:
                action_logits, _ = self.worker(features, self.current_goal)
                action = torch.stack([logits.argmax(dim=1) for logits in action_logits], dim=1)
            else:
                action, _ = self.worker.sample_action(features, self.current_goal)

            self.goal_step_counter += 1

        return action.cpu().numpy()[0], self.current_goal.cpu().numpy()[0]

    def reset_goal(self):
        """Reset current goal (call at episode start)."""
        self.current_goal = None
        self.goal_step_counter = 0


def compute_intrinsic_reward(
    state: Dict[str, np.ndarray],
    goal: np.ndarray,
    next_state: Dict[str, np.ndarray]
) -> float:
    """
    Compute intrinsic reward for worker based on goal achievement.

    Args:
        state: Current state observation
        goal: (3,) array [goal_x, goal_y, goal_type]
        next_state: Next state observation

    Returns:
        Intrinsic reward
    """
    goal_x, goal_y, _ = int(goal[0]), int(goal[1]), int(goal[2])

    # Goal types: 0=attack, 1=defend, 2=capture, 3=expand

    # Distance-based reward (encourage moving toward goal location)
    units = next_state['units']  # (H, W, 3)

    # Find player's units (assuming player 1)
    player_units = (units[:, :, 1] == 1)

    if player_units.any():
        # Get positions of player units
        unit_positions = np.argwhere(player_units)

        # Compute distances to goal
        distances = np.abs(unit_positions[:, 0] - goal_y) + np.abs(unit_positions[:, 1] - goal_x)
        min_distance = distances.min()

        # Reward based on proximity (closer = better)
        distance_reward = -min_distance * 0.1

        # Bonus if unit reached goal
        if (goal_y, goal_x) in map(tuple, unit_positions):
            distance_reward += 5.0
    else:
        distance_reward = -10.0

    return distance_reward
