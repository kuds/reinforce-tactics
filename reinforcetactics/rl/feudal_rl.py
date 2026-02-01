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


def _compute_gae(rewards, values, dones, last_value, gamma, gae_lambda,
                  segment_lengths=None):
    """
    Compute Generalized Advantage Estimation.

    For the manager, segment_lengths adjusts the discount to gamma^k
    where k is the number of worker steps in each manager segment.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        discount = (gamma ** segment_lengths[t]) if segment_lengths is not None else gamma
        delta = rewards[t] + discount * next_val * non_terminal - values[t]
        last_gae = delta + discount * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class FeudalRolloutBuffer:
    """Rollout buffer for feudal RL with separate manager and worker storage."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all stored data."""
        # Worker storage (one per env step)
        self.w_obs_grid = []
        self.w_obs_units = []
        self.w_obs_global = []
        self.w_actions = []
        self.w_log_probs = []
        self.w_values = []
        self.w_goals = []
        self.w_rewards = []
        self.w_dones = []

        # Manager storage (one per goal-setting event)
        self.m_obs_grid = []
        self.m_obs_units = []
        self.m_obs_global = []
        self.m_goals = []
        self.m_log_probs = []
        self.m_values = []
        self.m_rewards = []
        self.m_dones = []
        self.m_segment_lengths = []

    def add_worker_step(self, obs, action, log_prob, value, goal,
                        extrinsic_reward, intrinsic_reward, done,
                        worker_reward_alpha):
        """Add a single worker step to the buffer."""
        self.w_obs_grid.append(obs['grid'])
        self.w_obs_units.append(obs['units'])
        self.w_obs_global.append(obs['global_features'])
        self.w_actions.append(action)
        self.w_log_probs.append(log_prob)
        self.w_values.append(value)
        self.w_goals.append(goal)
        self.w_rewards.append(intrinsic_reward + worker_reward_alpha * extrinsic_reward)
        self.w_dones.append(done)

    def add_manager_step(self, obs, goal, log_prob, value):
        """Record a goal-setting event (reward/done filled later)."""
        self.m_obs_grid.append(obs['grid'])
        self.m_obs_units.append(obs['units'])
        self.m_obs_global.append(obs['global_features'])
        self.m_goals.append(goal)
        self.m_log_probs.append(log_prob)
        self.m_values.append(value)

    def end_manager_segment(self, cumulative_reward, done, segment_length):
        """Finalize a manager goal segment with its accumulated reward."""
        self.m_rewards.append(cumulative_reward)
        self.m_dones.append(done)
        self.m_segment_lengths.append(segment_length)

    def finalize(self):
        """Convert all lists to numpy arrays."""
        self.w_obs_grid = np.stack(self.w_obs_grid)
        self.w_obs_units = np.stack(self.w_obs_units)
        self.w_obs_global = np.stack(self.w_obs_global)
        self.w_actions = np.array(self.w_actions, dtype=np.int64)
        self.w_log_probs = np.array(self.w_log_probs, dtype=np.float32)
        self.w_values = np.array(self.w_values, dtype=np.float32)
        self.w_goals = np.array(self.w_goals, dtype=np.float32)
        self.w_rewards = np.array(self.w_rewards, dtype=np.float32)
        self.w_dones = np.array(self.w_dones, dtype=np.float32)

        self.m_obs_grid = np.stack(self.m_obs_grid)
        self.m_obs_units = np.stack(self.m_obs_units)
        self.m_obs_global = np.stack(self.m_obs_global)
        self.m_goals = np.array(self.m_goals, dtype=np.float32)
        self.m_log_probs = np.array(self.m_log_probs, dtype=np.float32)
        self.m_values = np.array(self.m_values, dtype=np.float32)
        self.m_rewards = np.array(self.m_rewards, dtype=np.float32)
        self.m_dones = np.array(self.m_dones, dtype=np.float32)
        self.m_segment_lengths = np.array(self.m_segment_lengths, dtype=np.int64)

    def compute_advantages(self, last_w_value, last_m_value, gamma, gae_lambda):
        """Compute GAE advantages for both worker and manager."""
        self.w_advantages, self.w_returns = _compute_gae(
            self.w_rewards, self.w_values, self.w_dones,
            last_w_value, gamma, gae_lambda
        )
        self.m_advantages, self.m_returns = _compute_gae(
            self.m_rewards, self.m_values, self.m_dones,
            last_m_value, gamma, gae_lambda,
            segment_lengths=self.m_segment_lengths
        )


class FeudalRLAgent:
    """
    Complete Feudal RL agent with manager and worker.
    Supports both inference (select_action) and training (collect_rollout + update).
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

    # ------------------------------------------------------------------
    # Training methods
    # ------------------------------------------------------------------

    def setup_training(self, learning_rate: float = 3e-4,
                       manager_lr_scale: float = 1.0,
                       worker_lr_scale: float = 1.0):
        """Initialize optimizer with per-component learning rates."""
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_extractor.parameters(), 'lr': learning_rate},
            {'params': self.manager.parameters(), 'lr': learning_rate * manager_lr_scale},
            {'params': self.worker.parameters(), 'lr': learning_rate * worker_lr_scale},
        ])
        self.feature_extractor.train()
        self.manager.train()
        self.worker.train()
        self._last_obs = None

    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert a single observation dict to batched tensor dict on device."""
        return {
            k: torch.as_tensor(v).unsqueeze(0).float().to(self.device)
            for k, v in obs.items()
            if k in ['grid', 'units', 'global_features']
        }

    def _batch_obs_to_tensor(self, grid, units, global_feat):
        """Convert pre-stacked numpy arrays to tensor dict on device."""
        return {
            'grid': torch.as_tensor(grid).float().to(self.device),
            'units': torch.as_tensor(units).float().to(self.device),
            'global_features': torch.as_tensor(global_feat).float().to(self.device),
        }

    def collect_rollout(self, env, n_steps: int, gamma: float, gae_lambda: float,
                        worker_reward_alpha: float = 0.5) -> FeudalRolloutBuffer:
        """
        Collect n_steps of experience using the feudal hierarchy.

        Args:
            env: Gymnasium environment
            n_steps: Number of environment steps to collect
            gamma: Discount factor
            gae_lambda: GAE lambda
            worker_reward_alpha: Weight of extrinsic reward in worker reward

        Returns:
            Filled FeudalRolloutBuffer with computed advantages
        """
        buf = FeudalRolloutBuffer()
        obs = self._last_obs
        manager_reward_accum = 0.0
        manager_step_count = 0

        self.feature_extractor.eval()
        self.manager.eval()
        self.worker.eval()

        for _ in range(n_steps):
            obs_tensor = self._obs_to_tensor(obs)

            with torch.no_grad():
                features = self.feature_extractor(obs_tensor)

                # Check if manager needs to set a new goal
                need_new_goal = (self.current_goal is None
                                 or self.goal_step_counter >= self.manager_horizon)

                if need_new_goal:
                    # Close previous manager segment if one exists
                    if self.current_goal is not None and manager_step_count > 0:
                        buf.end_manager_segment(manager_reward_accum,
                                                done=False,
                                                segment_length=manager_step_count)
                        manager_reward_accum = 0.0
                        manager_step_count = 0

                    # Sample new goal
                    goal, m_log_prob = self.manager.sample_goal(features)
                    _, _, m_value = self.manager.evaluate_goal(features, goal)
                    buf.add_manager_step(obs, goal.cpu().numpy()[0],
                                         m_log_prob.item(), m_value.item())
                    self.current_goal = goal
                    self.goal_step_counter = 0

                # Worker selects action conditioned on goal
                action, w_log_prob = self.worker.sample_action(features, self.current_goal)
                _, _, w_value = self.worker.evaluate_action(
                    features, self.current_goal, action)

            # Step environment
            action_np = action.cpu().numpy()[0]
            next_obs, ext_reward, terminated, truncated, _info = env.step(action_np)
            done = terminated or truncated

            # Compute intrinsic reward
            goal_np = self.current_goal.cpu().numpy()[0]
            int_reward = compute_intrinsic_reward(obs, goal_np, next_obs)

            # Store worker transition
            buf.add_worker_step(obs, action_np, w_log_prob.item(), w_value.item(),
                                goal_np, ext_reward, int_reward, done,
                                worker_reward_alpha)

            manager_reward_accum += ext_reward
            manager_step_count += 1
            self.goal_step_counter += 1

            if done:
                buf.end_manager_segment(manager_reward_accum, done=True,
                                        segment_length=manager_step_count)
                manager_reward_accum = 0.0
                manager_step_count = 0
                obs, _ = env.reset()
                self.reset_goal()
            else:
                obs = next_obs

        # Close any pending manager segment
        if manager_step_count > 0:
            buf.end_manager_segment(manager_reward_accum, done=False,
                                    segment_length=manager_step_count)

        # Bootstrap last values for GAE
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs)
            features = self.feature_extractor(obs_tensor)
            # Need a goal for worker value bootstrap
            if self.current_goal is None:
                self.current_goal, _ = self.manager.sample_goal(features)
            _, last_w_value = self.worker(features, self.current_goal)
            _, _, _, last_m_value = self.manager(features)

        self._last_obs = obs

        buf.finalize()
        buf.compute_advantages(last_w_value.item(), last_m_value.item(),
                               gamma, gae_lambda)

        self.feature_extractor.train()
        self.manager.train()
        self.worker.train()

        return buf

    def update(self, buf: FeudalRolloutBuffer, n_epochs: int, batch_size: int,
               clip_range: float, ent_coef: float, vf_coef: float,
               max_grad_norm: float) -> Dict[str, float]:
        """
        Run PPO update for both manager and worker.

        Returns dict of loss metrics.
        """
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        all_params = (list(self.feature_extractor.parameters())
                      + list(self.manager.parameters())
                      + list(self.worker.parameters()))

        n_worker = len(buf.w_rewards)
        n_manager = len(buf.m_rewards)
        w_batch_size = min(batch_size, n_worker)
        m_batch_size = min(batch_size, n_manager)

        # Convert to tensors
        w_actions_t = torch.as_tensor(buf.w_actions).to(self.device)
        w_old_lp = torch.as_tensor(buf.w_log_probs).to(self.device)
        w_adv = torch.as_tensor(buf.w_advantages).to(self.device)
        w_ret = torch.as_tensor(buf.w_returns).to(self.device)
        w_goals_t = torch.as_tensor(buf.w_goals).float().to(self.device)

        m_goals_t = torch.as_tensor(buf.m_goals).float().to(self.device)
        m_old_lp = torch.as_tensor(buf.m_log_probs).to(self.device)
        m_adv = torch.as_tensor(buf.m_advantages).to(self.device)
        m_ret = torch.as_tensor(buf.m_returns).to(self.device)

        # Normalize advantages
        w_adv = (w_adv - w_adv.mean()) / (w_adv.std() + 1e-8)
        if n_manager > 1:
            m_adv = (m_adv - m_adv.mean()) / (m_adv.std() + 1e-8)

        metrics = {}

        for _epoch in range(n_epochs):
            # --- Worker update ---
            w_indices = np.random.permutation(n_worker)
            for start in range(0, n_worker, w_batch_size):
                idx = w_indices[start:start + w_batch_size]
                b_obs = self._batch_obs_to_tensor(
                    buf.w_obs_grid[idx], buf.w_obs_units[idx], buf.w_obs_global[idx])
                b_actions = w_actions_t[idx]
                b_old_lp = w_old_lp[idx]
                b_adv = w_adv[idx]
                b_ret = w_ret[idx]
                b_goals = w_goals_t[idx]

                features = self.feature_extractor(b_obs)
                new_lp, entropy, values = self.worker.evaluate_action(
                    features, b_goals, b_actions)

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), b_ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                self.optimizer.step()

            metrics['worker_policy_loss'] = policy_loss.item()
            metrics['worker_value_loss'] = value_loss.item()
            metrics['worker_entropy'] = -entropy_loss.item()

            # --- Manager update ---
            m_indices = np.random.permutation(n_manager)
            for start in range(0, n_manager, m_batch_size):
                idx = m_indices[start:start + m_batch_size]
                b_obs = self._batch_obs_to_tensor(
                    buf.m_obs_grid[idx], buf.m_obs_units[idx], buf.m_obs_global[idx])
                b_goals = m_goals_t[idx]
                b_old_lp = m_old_lp[idx]
                b_adv = m_adv[idx]
                b_ret = m_ret[idx]

                features = self.feature_extractor(b_obs)
                new_lp, entropy, values = self.manager.evaluate_goal(
                    features, b_goals)

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), b_ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                self.optimizer.step()

            metrics['manager_policy_loss'] = policy_loss.item()
            metrics['manager_value_loss'] = value_loss.item()
            metrics['manager_entropy'] = -entropy_loss.item()

        return metrics

    def save_checkpoint(self, path):
        """Save all network weights and optimizer state."""
        from pathlib import Path as _Path  # pylint: disable=import-outside-toplevel
        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'manager': self.manager.state_dict(),
            'worker': self.worker.state_dict(),
            'optimizer': self.optimizer.state_dict()
                       if hasattr(self, 'optimizer') else None,
        }, path)

    def load_checkpoint(self, path):
        """Load network weights and optionally optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        self.manager.load_state_dict(checkpoint['manager'])
        self.worker.load_state_dict(checkpoint['worker'])
        if checkpoint.get('optimizer') and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent over n_episodes.

        Returns dict with mean_reward, std_reward, win_rate.
        """
        self.feature_extractor.eval()
        self.manager.eval()
        self.worker.eval()

        rewards = []
        wins = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            self.reset_goal()
            ep_reward = 0.0
            done = False

            while not done:
                action, _ = self.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated

            rewards.append(ep_reward)
            if info.get('winner') == 1:
                wins += 1

        self.feature_extractor.train()
        self.manager.train()
        self.worker.train()

        rewards_arr = np.array(rewards)
        return {
            'mean_reward': float(rewards_arr.mean()),
            'std_reward': float(rewards_arr.std()),
            'win_rate': wins / max(n_episodes, 1),
        }


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
