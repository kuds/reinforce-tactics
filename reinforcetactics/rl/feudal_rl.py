"""
Feudal Reinforcement Learning Architecture
Manager-Worker hierarchy for strategy games
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from reinforcetactics.rl.gym_env import StructuredActionMasks

# Keys used from the observation dict for feature extraction.
# Other keys (action_mask, visibility, etc.) are excluded.
_OBS_KEYS = ("grid", "units", "global_features")


def _apply_action_masks(
    action_logits: List[torch.Tensor],
    action_masks: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Set logits at masked-out (False) positions to -inf so Categorical
    sampling can never pick them. Each mask is bool, shape (B, dim_i) or
    (dim_i,); the latter is broadcast across the batch. A mask that is
    entirely False (no legal action in that dim) is left untouched —
    setting every logit to -inf would NaN the softmax. End-turn is always
    legal in dim 0, so an all-False mask in any other dim is a bug
    upstream rather than something to silently mask away here.
    """
    if len(action_masks) != len(action_logits):
        raise ValueError(f"Got {len(action_masks)} masks for {len(action_logits)} action heads")
    masked = []
    for logits, mask in zip(action_logits, action_masks):
        if mask is None:
            masked.append(logits)
            continue
        mask_t = mask.to(dtype=torch.bool, device=logits.device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0).expand_as(logits)
        # Skip masking if no legal action in this dim (avoids NaN softmax).
        if not mask_t.any():
            masked.append(logits)
            continue
        masked.append(logits.masked_fill(~mask_t, float("-inf")))
    return masked


class StructuredMaskProvider:
    """
    Adapter from a single-env ``StructuredActionMasks`` to per-stage batched
    masks for ``AutoregressiveActionHead.sample_with_provider``.

    The provider is what lets us mask exactly at each AR stage (atype -> src
    -> unit_type -> target) when the mask at later stages depends on values
    sampled at earlier stages. It assumes batch size 1, which is the shape
    the feudal rollout loop actually uses (single env stepping).
    """

    def __init__(
        self,
        masks: StructuredActionMasks,
        grid_height: int,
        grid_width: int,
        num_action_types: int = 10,
        num_unit_types: int = 8,
        device: str = "cpu",
    ):
        self.masks = masks
        self.H = grid_height
        self.W = grid_width
        self.A = num_action_types
        self.U = num_unit_types
        self.device = device

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        # numpy bool arrays may share memory; copy to be safe before sending to torch.
        return torch.as_tensor(arr.copy(), dtype=torch.bool, device=self.device).unsqueeze(0)

    def atype_mask(self) -> torch.Tensor:
        return self._to_tensor(self.masks.atype)  # (1, A)

    def src_mask(self, atype: torch.Tensor) -> torch.Tensor:
        a = int(atype.item())
        return self._to_tensor(self.masks.source[a].reshape(-1))  # (1, H*W)

    def unit_type_mask(self, atype: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
        # Only meaningful for create_unit (atype=0); for other atypes any
        # unit_type is acceptable since the env ignores the slot.
        sx_i, sy_i = int(sx.item()), int(sy.item())
        m = self.masks.unit_type.get((sx_i, sy_i))
        if m is None:
            m = np.ones(self.U, dtype=bool)
        return self._to_tensor(m)  # (1, U)

    def target_mask(self, atype: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
        a = int(atype.item())
        sx_i, sy_i = int(sx.item()), int(sy.item())
        m = self.masks.target.get((a, sx_i, sy_i))
        if m is None:
            m = np.ones((self.H, self.W), dtype=bool)
        return self._to_tensor(m.reshape(-1))  # (1, H*W)


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
        if "global_features" in observation_space.spaces:
            self.n_global = observation_space["global_features"].shape[0]
        else:
            self.n_global = 0

        # Compute CNN output shape by doing one forward pass
        with torch.no_grad():
            sample_obs = observation_space.sample()
            grid = torch.as_tensor(sample_obs["grid"]).float()
            units = torch.as_tensor(sample_obs["units"]).float()
            combined = torch.cat([grid, units], dim=-1).permute(2, 0, 1).unsqueeze(0)
            n_flatten = self.cnn(combined).flatten(1).shape[1]

        # Linear projection: CNN output + global features -> features_dim
        self.linear = nn.Sequential(nn.Linear(n_flatten + self.n_global, features_dim), nn.ReLU())

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from observations.

        Args:
            observations: Dict with 'grid', 'units', and optionally 'global_features'

        Returns:
            Feature tensor of shape (batch, features_dim)
        """
        grid = observations["grid"]  # (B, H, W, 3)
        units = observations["units"]  # (B, H, W, 3)

        # Combine and permute to (B, C, H, W)
        combined = torch.cat([grid, units], dim=-1)  # (B, H, W, 6)
        combined = combined.permute(0, 3, 1, 2)  # (B, 6, H, W)

        # CNN forward
        features = self.cnn(combined)  # (B, 64, H, W)
        features = features.flatten(1)  # (B, 64*H*W)

        # Concatenate global features (gold, turn, unit counts, current player)
        if self.n_global > 0 and "global_features" in observations:
            global_feat = observations["global_features"]  # (B, 6)
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
        num_goal_types: int = 4,  # attack, defend, capture, expand
    ):
        super().__init__()

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_goal_types = num_goal_types

        # Process features
        self.mlp = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())

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

    def sample_goal(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a goal from the policy.

        Returns:
            goal: (batch, 3) - [goal_x, goal_y, goal_type]
            log_prob: (batch,) - log probability of sampled goal
            value: (batch, 1) - value estimate
        """
        goal_x_logits, goal_y_logits, goal_type_logits, value = self.forward(features)

        # Sample from categorical distributions
        goal_x_dist = torch.distributions.Categorical(logits=goal_x_logits)
        goal_y_dist = torch.distributions.Categorical(logits=goal_y_logits)
        goal_type_dist = torch.distributions.Categorical(logits=goal_type_logits)

        goal_x = goal_x_dist.sample()
        goal_y = goal_y_dist.sample()
        goal_type = goal_type_dist.sample()

        # Compute log probability
        log_prob = goal_x_dist.log_prob(goal_x) + goal_y_dist.log_prob(goal_y) + goal_type_dist.log_prob(goal_type)

        goal = torch.stack([goal_x, goal_y, goal_type], dim=1)

        return goal, log_prob, value

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

        log_prob = goal_x_dist.log_prob(goal_x) + goal_y_dist.log_prob(goal_y) + goal_type_dist.log_prob(goal_type)

        entropy = goal_x_dist.entropy() + goal_y_dist.entropy() + goal_type_dist.entropy()

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
        action_space_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if action_space_dims is None:
            action_space_dims = [10, 8, 20, 20, 20, 20]  # 10 action types, 8 unit types

        self.action_space_dims = action_space_dims

        # Goal embedding
        self.goal_embedding = nn.Sequential(
            nn.Linear(3, goal_embedding_dim),  # goal is (x, y, type)
            nn.ReLU(),
            nn.Linear(goal_embedding_dim, goal_embedding_dim),
            nn.ReLU(),
        )

        # Combined processing
        combined_dim = feature_dim + goal_embedding_dim
        self.mlp = nn.Sequential(nn.Linear(combined_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())

        # Action heads (one for each dimension)
        self.action_heads = nn.ModuleList([nn.Linear(256, dim) for dim in action_space_dims])

        # Value head
        self.value_head = nn.Linear(256, 1)

    def forward(
        self,
        features: torch.Tensor,
        goal: torch.Tensor,
        action_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, feature_dim)
            goal: (batch, 3) - [goal_x, goal_y, goal_type]
            action_masks: Optional list of 6 boolean tensors aligned with
                ``action_space_dims``. Each tensor has shape (batch, dim_i)
                or (dim_i,) (broadcast). Disallowed actions get -inf logit
                so Categorical sampling never picks them. End-turn (a single
                always-legal action) is the safety net the env guarantees.

        Returns:
            action_logits: List of (batch, action_dim) tensors (post-mask)
            value: (batch, 1)
        """
        # Embed goal
        goal_emb = self.goal_embedding(goal.float())

        # Combine features and goal
        combined = torch.cat([features, goal_emb], dim=1)
        x = self.mlp(combined)

        # Compute action logits
        action_logits = [head(x) for head in self.action_heads]

        if action_masks is not None:
            action_logits = _apply_action_masks(action_logits, action_masks)

        # Compute value
        value = self.value_head(x)

        return action_logits, value

    def sample_action(
        self,
        features: torch.Tensor,
        goal: torch.Tensor,
        action_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action. ``action_masks`` is forwarded to ``forward``.

        Returns:
            action: (batch, len(action_space_dims))
            log_prob: (batch,)
            value: (batch, 1)
        """
        action_logits, value = self.forward(features, goal, action_masks=action_masks)

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

        return action, log_prob, value

    def evaluate_action(
        self,
        features: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        action_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a given action. When ``action_masks`` is provided it must
        match the masks used at sample time — otherwise the new log_prob is
        computed under a different distribution than the old one and PPO's
        importance-sampling ratio is biased.

        Returns:
            log_prob: (batch,)
            entropy: (batch,)
            value: (batch, 1)
        """
        action_logits, value = self.forward(features, goal, action_masks=action_masks)

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


class AutoregressiveActionHead(nn.Module):
    """
    AlphaStar-style autoregressive head over the env's 6-tuple action
    [action_type, unit_type, src_x, src_y, tgt_x, tgt_y].

    Joint distribution factorizes as
        p(atype) * p(src_xy | atype) * p(unit_type | atype, src) * p(tgt_xy | atype, src).

    Replaces the WorkerNetwork's product-of-marginals 6-head independent
    Categoricals. The independent product cannot represent dependencies
    between dimensions (e.g. "if action_type=move, source must be one of
    *my* units"), which is exactly what the existing per-dimension mask
    over-approximates.

    All sub-head masks are optional; when present they zero out illegal
    logits prior to softmax. Mask plumbing into the rollout buffer is a
    follow-up — at this stage the head is structurally autoregressive but
    can be used unmasked.

    Mask shapes (all bool):
        atype:      (B, A)
        src:        (B, H * W)
        unit_type:  (B, U)         — for the unit-type pick at the chosen src
        target:     (B, H * W)     — for the target pick at the chosen src
    """

    def __init__(
        self,
        feature_dim: int,
        grid_height: int,
        grid_width: int,
        num_action_types: int = 10,
        num_unit_types: int = 8,
        hidden_dim: int = 256,
        atype_emb_dim: int = 32,
        src_emb_dim: int = 32,
    ):
        super().__init__()
        self.H = grid_height
        self.W = grid_width
        self.A = num_action_types
        self.U = num_unit_types

        # Stage 1: action type
        self.atype_logits = nn.Linear(feature_dim, num_action_types)
        self.atype_emb = nn.Embedding(num_action_types, atype_emb_dim)

        # Stage 2: source position, conditioned on (features, atype)
        self.src_trunk = nn.Sequential(
            nn.Linear(feature_dim + atype_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_height * grid_width),
        )

        # Source-position embedding shared by stages 3 and 4.
        # Coordinates are normalized to [0, 1] before the linear projection
        # so that this layer remains valid if the grid size changes.
        self.src_pos_proj = nn.Linear(2, src_emb_dim)

        tail_in = feature_dim + atype_emb_dim + src_emb_dim
        # Stage 3: unit type (only meaningful for create_unit)
        self.ut_head = nn.Sequential(
            nn.Linear(tail_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_unit_types),
        )
        # Stage 4: target position
        self.tgt_head = nn.Sequential(
            nn.Linear(tail_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, grid_height * grid_width),
        )

    @staticmethod
    def _apply_mask(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        return logits.masked_fill(~mask, -1e9)

    def _src_pos_emb(self, sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
        x_norm = sx.float() / max(self.W - 1, 1)
        y_norm = sy.float() / max(self.H - 1, 1)
        return self.src_pos_proj(torch.stack([x_norm, y_norm], dim=-1))

    def _atype_dist(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return torch.distributions.Categorical(
            logits=self._apply_mask(self.atype_logits(features), mask)
        )

    def _src_dist(self, features: torch.Tensor, atype: torch.Tensor, mask: Optional[torch.Tensor] = None):
        ae = self.atype_emb(atype)
        logits = self.src_trunk(torch.cat([features, ae], dim=-1))
        return torch.distributions.Categorical(logits=self._apply_mask(logits, mask))

    def _tail_input(self, features: torch.Tensor, atype: torch.Tensor, sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
        ae = self.atype_emb(atype)
        se = self._src_pos_emb(sx, sy)
        return torch.cat([features, ae, se], dim=-1)

    def _ut_dist(self, features, atype, sx, sy, mask=None):
        return torch.distributions.Categorical(
            logits=self._apply_mask(self.ut_head(self._tail_input(features, atype, sx, sy)), mask)
        )

    def _tgt_dist(self, features, atype, sx, sy, mask=None):
        return torch.distributions.Categorical(
            logits=self._apply_mask(self.tgt_head(self._tail_input(features, atype, sx, sy)), mask)
        )

    def sample(
        self,
        features: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action autoregressively.

        Returns:
            action: (B, 6) long — [atype, unit_type, src_x, src_y, tgt_x, tgt_y]
            log_prob: (B,) float — joint log probability of the sampled action
        """
        masks = masks or {}

        atype_dist = self._atype_dist(features, masks.get("atype"))
        atype = atype_dist.probs.argmax(dim=-1) if deterministic else atype_dist.sample()
        lp_a = atype_dist.log_prob(atype)

        src_dist = self._src_dist(features, atype, masks.get("src"))
        src_idx = src_dist.probs.argmax(dim=-1) if deterministic else src_dist.sample()
        lp_s = src_dist.log_prob(src_idx)
        sy = (src_idx // self.W).long()
        sx = (src_idx % self.W).long()

        ut_dist = self._ut_dist(features, atype, sx, sy, masks.get("unit_type"))
        ut = ut_dist.probs.argmax(dim=-1) if deterministic else ut_dist.sample()
        lp_u = ut_dist.log_prob(ut)

        tgt_dist = self._tgt_dist(features, atype, sx, sy, masks.get("target"))
        tgt_idx = tgt_dist.probs.argmax(dim=-1) if deterministic else tgt_dist.sample()
        lp_t = tgt_dist.log_prob(tgt_idx)
        ty = (tgt_idx // self.W).long()
        tx = (tgt_idx % self.W).long()

        action = torch.stack([atype, ut, sx, sy, tx, ty], dim=1).long()
        log_prob = lp_a + lp_s + lp_u + lp_t
        return action, log_prob

    def sample_with_provider(
        self,
        features: torch.Tensor,
        provider: StructuredMaskProvider,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample autoregressively, fetching masks from a provider between stages.

        Used at rollout time: the source mask depends on the sampled atype,
        the unit_type mask depends on the sampled (sx, sy), etc., so they
        cannot be pre-computed. The conditional masks actually applied are
        returned alongside the action so they can be stored in the rollout
        buffer and replayed verbatim during the PPO update.

        Returns:
            action: (B, 6) long
            log_prob: (B,)
            conditional_masks: dict with keys ``atype``, ``src``, ``unit_type``,
                ``target`` — each the bool tensor passed to that stage's
                Categorical at sample time.
        """
        atype_mask = provider.atype_mask()
        atype_dist = self._atype_dist(features, atype_mask)
        atype = atype_dist.probs.argmax(dim=-1) if deterministic else atype_dist.sample()
        lp_a = atype_dist.log_prob(atype)

        src_mask = provider.src_mask(atype)
        src_dist = self._src_dist(features, atype, src_mask)
        src_idx = src_dist.probs.argmax(dim=-1) if deterministic else src_dist.sample()
        lp_s = src_dist.log_prob(src_idx)
        sy = (src_idx // self.W).long()
        sx = (src_idx % self.W).long()

        ut_mask = provider.unit_type_mask(atype, sx, sy)
        ut_dist = self._ut_dist(features, atype, sx, sy, ut_mask)
        ut = ut_dist.probs.argmax(dim=-1) if deterministic else ut_dist.sample()
        lp_u = ut_dist.log_prob(ut)

        tgt_mask = provider.target_mask(atype, sx, sy)
        tgt_dist = self._tgt_dist(features, atype, sx, sy, tgt_mask)
        tgt_idx = tgt_dist.probs.argmax(dim=-1) if deterministic else tgt_dist.sample()
        lp_t = tgt_dist.log_prob(tgt_idx)
        ty = (tgt_idx // self.W).long()
        tx = (tgt_idx % self.W).long()

        action = torch.stack([atype, ut, sx, sy, tx, ty], dim=1).long()
        log_prob = lp_a + lp_s + lp_u + lp_t
        conditional_masks = {
            "atype": atype_mask,
            "src": src_mask,
            "unit_type": ut_mask,
            "target": tgt_mask,
        }
        return action, log_prob, conditional_masks

    def evaluate(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint log_prob and entropy of `action` under the AR policy.

        Args:
            action: (B, 6) long — [atype, unit_type, src_x, src_y, tgt_x, tgt_y]

        Returns:
            log_prob: (B,)
            entropy: (B,)  — sum of per-stage entropies (chain-rule decomposition)
        """
        masks = masks or {}
        atype = action[:, 0].long()
        ut = action[:, 1].long()
        sx = action[:, 2].long()
        sy = action[:, 3].long()
        tx = action[:, 4].long()
        ty = action[:, 5].long()

        atype_dist = self._atype_dist(features, masks.get("atype"))
        src_dist = self._src_dist(features, atype, masks.get("src"))
        ut_dist = self._ut_dist(features, atype, sx, sy, masks.get("unit_type"))
        tgt_dist = self._tgt_dist(features, atype, sx, sy, masks.get("target"))

        src_idx = sy * self.W + sx
        tgt_idx = ty * self.W + tx

        log_prob = (
            atype_dist.log_prob(atype)
            + src_dist.log_prob(src_idx)
            + ut_dist.log_prob(ut)
            + tgt_dist.log_prob(tgt_idx)
        )
        entropy = atype_dist.entropy() + src_dist.entropy() + ut_dist.entropy() + tgt_dist.entropy()
        return log_prob, entropy


class AutoregressiveWorkerNetwork(nn.Module):
    """
    Drop-in replacement for ``WorkerNetwork`` with an AlphaStar-style
    autoregressive policy head.

    Same external API as ``WorkerNetwork`` (``forward``, ``sample_action``,
    ``evaluate_action``) so ``FeudalRLAgent`` can swap between the two via
    a constructor flag without touching its rollout/update code.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        goal_embedding_dim: int = 64,
        grid_width: int = 20,
        grid_height: int = 20,
        num_action_types: int = 10,
        num_unit_types: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Mirror WorkerNetwork's action_space_dims for diagnostic compatibility.
        self.action_space_dims = [
            num_action_types,
            num_unit_types,
            grid_width,
            grid_height,
            grid_width,
            grid_height,
        ]
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.goal_embedding = nn.Sequential(
            nn.Linear(3, goal_embedding_dim),
            nn.ReLU(),
            nn.Linear(goal_embedding_dim, goal_embedding_dim),
            nn.ReLU(),
        )

        combined_dim = feature_dim + goal_embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

        self.head = AutoregressiveActionHead(
            feature_dim=hidden_dim,
            grid_height=grid_height,
            grid_width=grid_width,
            num_action_types=num_action_types,
            num_unit_types=num_unit_types,
            hidden_dim=hidden_dim,
        )

        self.value_head = nn.Linear(hidden_dim, 1)

    def _shared(self, features: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        goal_emb = self.goal_embedding(goal.float())
        return self.trunk(torch.cat([features, goal_emb], dim=1))

    def forward(self, features: torch.Tensor, goal: torch.Tensor) -> Tuple[None, torch.Tensor]:
        """
        Mirrors ``WorkerNetwork.forward`` for interface compatibility.

        The autoregressive head cannot expose a flat list of independent
        per-dimension logits, so the first return value is None (the AR
        head's ``sample`` / ``evaluate`` methods are the supported API).
        Returns ``(None, value)``.
        """
        h = self._shared(features, goal)
        return None, self.value_head(h)

    def sample_action(
        self,
        features: torch.Tensor,
        goal: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._shared(features, goal)
        action, log_prob = self.head.sample(h, masks=masks, deterministic=deterministic)
        value = self.value_head(h)
        return action, log_prob, value

    def evaluate_action(
        self,
        features: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._shared(features, goal)
        log_prob, entropy = self.head.evaluate(h, action, masks=masks)
        value = self.value_head(h)
        return log_prob, entropy, value

    def sample_action_with_provider(
        self,
        features: torch.Tensor,
        goal: torch.Tensor,
        provider: StructuredMaskProvider,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h = self._shared(features, goal)
        action, log_prob, conditional_masks = self.head.sample_with_provider(
            h, provider, deterministic=deterministic
        )
        value = self.value_head(h)
        return action, log_prob, value, conditional_masks


def _compute_gae(rewards, values, dones, last_value, gamma, gae_lambda, segment_lengths=None):
    """
    Compute Generalized Advantage Estimation.

    For the manager, segment_lengths adjusts the discount to gamma^k
    where k is the number of worker steps in each manager segment.
    The GAE propagation term uses the *next* segment's length to correctly
    discount the accumulated advantage from future segments.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        discount = (gamma ** segment_lengths[t]) if segment_lengths is not None else gamma
        delta = rewards[t] + discount * next_val * non_terminal - values[t]
        # For GAE propagation, use the *next* segment's discount to properly
        # weight the future advantage. At the last step, use the current discount
        # as a fallback (the last_gae is 0.0 at initialization anyway).
        if segment_lengths is not None and t < n - 1:
            next_discount = gamma ** segment_lengths[t + 1]
        else:
            next_discount = discount
        last_gae = delta + next_discount * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class FeudalRolloutBuffer:
    """Rollout buffer for feudal RL with separate manager and worker storage.

    When ``store_masks`` is True (autoregressive worker mode), each worker
    step also stores the four conditional AR masks that were applied at
    sample time. The PPO update replays them through ``evaluate_action`` so
    new and old log-probs are computed under identical mask supports.
    """

    def __init__(self, store_masks: bool = False):
        self.store_masks = store_masks
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
        # Per-dim action masks for the legacy 6-head WorkerNetwork: list of
        # 6-tuples (each entry a np.bool array sized to that worker dim), or
        # an empty list if the env didn't provide masks. Stored so update()
        # can re-apply the same masking and keep PPO ratios well-defined.
        self.w_action_masks: List[Tuple[np.ndarray, ...]] = []
        # Conditional AR mask storage for AutoregressiveWorkerNetwork (only
        # populated when store_masks=True). One per worker step; each holds
        # the four conditional masks that were applied at sample time.
        self.w_mask_atype: List[np.ndarray] = []
        self.w_mask_src: List[np.ndarray] = []
        self.w_mask_unit_type: List[np.ndarray] = []
        self.w_mask_target: List[np.ndarray] = []

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

    def add_worker_step(
        self,
        obs,
        action,
        log_prob,
        value,
        goal,
        extrinsic_reward,
        intrinsic_reward,
        done,
        worker_reward_alpha,
        action_masks=None,
        masks: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Add a single worker step to the buffer.

        ``masks`` is required when ``store_masks`` is True and must contain the
        four conditional AR masks (``atype``, ``src``, ``unit_type``, ``target``)
        that were applied at sample time. Stored verbatim for replay during
        the PPO update.
        """
        self.w_obs_grid.append(obs["grid"])
        self.w_obs_units.append(obs["units"])
        self.w_obs_global.append(obs["global_features"])
        self.w_actions.append(action)
        self.w_log_probs.append(log_prob)
        self.w_values.append(value)
        self.w_goals.append(goal)
        self.w_rewards.append(intrinsic_reward + worker_reward_alpha * extrinsic_reward)
        self.w_dones.append(done)
        if action_masks is not None:
            self.w_action_masks.append(tuple(np.asarray(m, dtype=bool) for m in action_masks))
        if self.store_masks:
            if masks is None:
                raise ValueError("FeudalRolloutBuffer(store_masks=True) requires masks per worker step")
            self.w_mask_atype.append(masks["atype"])
            self.w_mask_src.append(masks["src"])
            self.w_mask_unit_type.append(masks["unit_type"])
            self.w_mask_target.append(masks["target"])

    def add_manager_step(self, obs, goal, log_prob, value):
        """Record a goal-setting event (reward/done filled later)."""
        self.m_obs_grid.append(obs["grid"])
        self.m_obs_units.append(obs["units"])
        self.m_obs_global.append(obs["global_features"])
        self.m_goals.append(goal)
        self.m_log_probs.append(log_prob)
        self.m_values.append(value)

    def end_manager_segment(self, cumulative_reward, done, segment_length):
        """Finalize a manager goal segment with its accumulated reward."""
        self.m_rewards.append(cumulative_reward)
        self.m_dones.append(done)
        self.m_segment_lengths.append(segment_length)

    @property
    def has_manager_data(self) -> bool:
        """Check whether the buffer contains any finalized manager segments."""
        return len(self.m_rewards) > 0

    @property
    def has_action_masks(self) -> bool:
        """True iff the rollout captured per-dim worker masks for every step.

        Works both before and after ``finalize()``. Pre-finalize the field is
        a list of per-step tuples (length matches w_rewards). Post-finalize
        it is a tuple of stacked arrays (length 6, matching the action dims).
        """
        if isinstance(self.w_action_masks, tuple):
            return len(self.w_action_masks) > 0
        return len(self.w_action_masks) == len(self.w_rewards) and len(self.w_action_masks) > 0

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
        # Pivot from list-of-tuples to tuple-of-arrays so each dim can be
        # batch-indexed: w_action_masks[dim_i] is shape (N, dim_size_i).
        # Skipped (left as []) when no masks were captured — update() then
        # falls back to the un-masked code path.
        if self.has_action_masks:
            n_dims = len(self.w_action_masks[0])
            self.w_action_masks = tuple(np.stack([step_masks[d] for step_masks in self.w_action_masks]) for d in range(n_dims))

        if self.store_masks and len(self.w_mask_atype) > 0:
            self.w_mask_atype = np.stack(self.w_mask_atype).astype(bool)
            self.w_mask_src = np.stack(self.w_mask_src).astype(bool)
            self.w_mask_unit_type = np.stack(self.w_mask_unit_type).astype(bool)
            self.w_mask_target = np.stack(self.w_mask_target).astype(bool)

        if self.has_manager_data:
            self.m_obs_grid = np.stack(self.m_obs_grid)
            self.m_obs_units = np.stack(self.m_obs_units)
            self.m_obs_global = np.stack(self.m_obs_global)
            self.m_goals = np.array(self.m_goals, dtype=np.float32)
            self.m_log_probs = np.array(self.m_log_probs, dtype=np.float32)
            self.m_values = np.array(self.m_values, dtype=np.float32)
            self.m_rewards = np.array(self.m_rewards, dtype=np.float32)
            self.m_dones = np.array(self.m_dones, dtype=np.float32)
            self.m_segment_lengths = np.array(self.m_segment_lengths, dtype=np.int64)
        else:
            # Empty arrays so downstream code can check len() == 0
            self.m_obs_grid = np.empty((0,), dtype=np.float32)
            self.m_obs_units = np.empty((0,), dtype=np.float32)
            self.m_obs_global = np.empty((0,), dtype=np.float32)
            self.m_goals = np.empty((0, 3), dtype=np.float32)
            self.m_log_probs = np.empty((0,), dtype=np.float32)
            self.m_values = np.empty((0,), dtype=np.float32)
            self.m_rewards = np.empty((0,), dtype=np.float32)
            self.m_dones = np.empty((0,), dtype=np.float32)
            self.m_segment_lengths = np.empty((0,), dtype=np.int64)

    def compute_advantages(self, last_w_value, last_m_value, gamma, gae_lambda):
        """Compute GAE advantages for both worker and manager."""
        self.w_advantages, self.w_returns = _compute_gae(
            self.w_rewards, self.w_values, self.w_dones, last_w_value, gamma, gae_lambda
        )
        if len(self.m_rewards) > 0:
            self.m_advantages, self.m_returns = _compute_gae(
                self.m_rewards,
                self.m_values,
                self.m_dones,
                last_m_value,
                gamma,
                gae_lambda,
                segment_lengths=self.m_segment_lengths,
            )
        else:
            self.m_advantages = np.empty((0,), dtype=np.float32)
            self.m_returns = np.empty((0,), dtype=np.float32)


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
        agent_player: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        autoregressive_worker: bool = False,
    ):
        self.device = device
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.agent_player = agent_player
        self.autoregressive_worker = autoregressive_worker

        # Feature extractor (shared)
        self.feature_extractor = SpatialFeatureExtractor(observation_space, features_dim=512).to(device)

        # Manager network
        self.manager = ManagerNetwork(feature_dim=512, grid_width=grid_width, grid_height=grid_height, num_goal_types=4).to(
            device
        )

        # Worker network. The autoregressive variant samples action dimensions
        # in dependency order p(atype) p(src|atype) p(ut|atype, src) p(tgt|atype, src),
        # which is the AlphaStar factorization and a prerequisite for replacing
        # the per-dimension over-approximation mask with a per-stage exact mask.
        if autoregressive_worker:
            self.worker = AutoregressiveWorkerNetwork(
                feature_dim=512,
                goal_embedding_dim=64,
                grid_width=grid_width,
                grid_height=grid_height,
            ).to(device)
        else:
            self.worker = WorkerNetwork(
                feature_dim=512,
                goal_embedding_dim=64,
                action_space_dims=[10, 8, grid_width, grid_height, grid_width, grid_height],
            ).to(device)

        # Current goal (maintained across steps)
        self.current_goal: Optional[torch.Tensor] = None
        self.goal_step_counter = 0
        self.manager_horizon = 10  # Update goal every N steps

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
        action_masks: Optional[Tuple[np.ndarray, ...]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Select action using manager-worker hierarchy.

        Args:
            action_masks: Optional 6-tuple of per-dimension bool numpy arrays
                from ``env.action_masks()`` (multi_discrete mode). Applied to
                the worker's logits so illegal actions cannot be sampled or
                argmax'd. Manager goals are not masked (goal space is
                unconstrained by env legality).

        Returns:
            action: Primitive action array
            goal: Current goal (for logging/debugging)
        """
        obs_tensor = self._obs_to_tensor(observation)
        worker_masks = self._masks_to_tensors(action_masks)

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
                    self.current_goal, _, _ = self.manager.sample_goal(features)

                self.goal_step_counter = 0

            # Worker selects action conditioned on goal
            assert self.current_goal is not None
            if self.autoregressive_worker:
                # AR worker has no env-mask plumbing at inference time; the
                # legacy per-dim ``worker_masks`` would not match its head
                # API, so they are dropped here.
                action, _, _ = self.worker.sample_action(
                    features, self.current_goal, deterministic=deterministic
                )
            elif deterministic:
                action_logits, _ = self.worker(features, self.current_goal, action_masks=worker_masks)
                action = torch.stack([logits.argmax(dim=1) for logits in action_logits], dim=1)
            else:
                action, _, _ = self.worker.sample_action(features, self.current_goal, action_masks=worker_masks)

            self.goal_step_counter += 1

        return action.cpu().numpy()[0], self.current_goal.cpu().numpy()[0]

    def _masks_to_tensors(self, action_masks: Optional[Tuple[np.ndarray, ...]]) -> Optional[List[torch.Tensor]]:
        """Convert a tuple of per-dim numpy bool masks (from env.action_masks())
        into a list of torch tensors on self.device, matching the worker's
        action_space_dims. Returns None if no masks supplied.
        """
        if action_masks is None:
            return None
        return [torch.as_tensor(m, dtype=torch.bool, device=self.device) for m in action_masks]

    def reset_goal(self):
        """Reset current goal (call at episode start)."""
        self.current_goal = None
        self.goal_step_counter = 0

    # ------------------------------------------------------------------
    # Training methods
    # ------------------------------------------------------------------

    def setup_training(self, learning_rate: float = 3e-4, manager_lr_scale: float = 1.0, worker_lr_scale: float = 1.0):
        """Initialize separate optimizers for worker and manager to prevent
        feature extractor drift from interleaved updates."""
        # Worker optimizer: updates feature extractor + worker
        self.worker_optimizer = torch.optim.Adam(
            [
                {"params": self.feature_extractor.parameters(), "lr": learning_rate},
                {"params": self.worker.parameters(), "lr": learning_rate * worker_lr_scale},
            ]
        )
        # Manager optimizer: updates feature extractor + manager
        self.manager_optimizer = torch.optim.Adam(
            [
                {"params": self.feature_extractor.parameters(), "lr": learning_rate},
                {"params": self.manager.parameters(), "lr": learning_rate * manager_lr_scale},
            ]
        )
        self.feature_extractor.train()
        self.manager.train()
        self.worker.train()
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert a single observation dict to batched tensor dict on device.

        Only includes keys needed by the feature extractor (grid, units,
        global_features), filtering out action_mask, visibility, etc.
        """
        return {k: torch.as_tensor(v).unsqueeze(0).float().to(self.device) for k, v in obs.items() if k in _OBS_KEYS}

    def _batch_obs_to_tensor(self, grid, units, global_feat):
        """Convert pre-stacked numpy arrays to tensor dict on device."""
        return {
            "grid": torch.as_tensor(grid).float().to(self.device),
            "units": torch.as_tensor(units).float().to(self.device),
            "global_features": torch.as_tensor(global_feat).float().to(self.device),
        }

    def collect_rollout(
        self, env, n_steps: int, gamma: float, gae_lambda: float, worker_reward_alpha: float = 0.5
    ) -> FeudalRolloutBuffer:
        """
        Collect n_steps of experience using the feudal hierarchy.

        When the env exposes ``action_masks()`` (multi_discrete StrategyGameEnv),
        per-dim masks are captured at each step, applied to the worker's
        logits, and stashed in the buffer so ``update()`` re-applies the
        same masking — keeping PPO's importance-sampling ratio well-defined.

        End reasons and reward breakdowns from ``info`` are accumulated on
        the returned buffer (``buf.end_reasons``, ``buf.reward_breakdown``)
        so the training loop can surface diagnostics that mirror the
        PPO notebook's eval cards.
        """
        # In autoregressive mode the buffer also stores per-step conditional
        # masks so the PPO update can re-evaluate log-probs under the same
        # mask supports that were applied at sample time.
        use_ar_masks = self.autoregressive_worker and hasattr(env, "structured_action_masks")
        buf = FeudalRolloutBuffer(store_masks=use_ar_masks)
        obs = self._last_obs
        manager_reward_accum = 0.0
        manager_step_count = 0
        # Track whether we have opened a manager segment *in this buffer*.
        # Prevents closing a segment from a previous rollout in a fresh buffer.
        manager_segment_open = False

        env_supports_masks = hasattr(env, "action_masks")
        end_reasons: List[str] = []
        reward_breakdown_sums: Dict[str, float] = {}

        self.feature_extractor.eval()
        self.manager.eval()
        self.worker.eval()

        for _ in range(n_steps):
            obs_tensor = self._obs_to_tensor(obs)
            step_masks_np = env.action_masks() if env_supports_masks else None
            worker_mask_tensors = self._masks_to_tensors(step_masks_np)

            with torch.no_grad():
                features = self.feature_extractor(obs_tensor)

                # Check if manager needs to set a new goal
                need_new_goal = self.current_goal is None or self.goal_step_counter >= self.manager_horizon

                if need_new_goal:
                    # Close previous manager segment if one was opened in this buffer
                    if manager_segment_open and manager_step_count > 0:
                        buf.end_manager_segment(manager_reward_accum, done=False, segment_length=manager_step_count)
                        manager_reward_accum = 0.0
                        manager_step_count = 0

                    # Sample new goal (single forward pass returns goal, log_prob, value)
                    goal, m_log_prob, m_value = self.manager.sample_goal(features)
                    buf.add_manager_step(obs, goal.cpu().numpy()[0], m_log_prob.item(), m_value.squeeze(-1).item())
                    self.current_goal = goal
                    self.goal_step_counter = 0
                    manager_segment_open = True

                # Worker selects action conditioned on goal.
                ar_step_masks_np: Optional[Dict[str, np.ndarray]] = None
                if use_ar_masks:
                    provider = StructuredMaskProvider(
                        env.structured_action_masks(),
                        grid_height=self.grid_height,
                        grid_width=self.grid_width,
                        device=self.device,
                    )
                    action, w_log_prob, w_value, cond_masks = self.worker.sample_action_with_provider(
                        features, self.current_goal, provider
                    )
                    ar_step_masks_np = {
                        k: v.cpu().numpy().squeeze(0) for k, v in cond_masks.items()
                    }
                else:
                    action, w_log_prob, w_value = self.worker.sample_action(
                        features, self.current_goal, action_masks=worker_mask_tensors
                    )

            # Step environment
            action_np = action.cpu().numpy()[0]
            next_obs, ext_reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Surface info diagnostics so the training loop can show them.
            for k, v in info.get("reward_breakdown", {}).items():
                reward_breakdown_sums[k] = reward_breakdown_sums.get(k, 0.0) + float(v)

            # Compute intrinsic reward
            assert self.current_goal is not None
            goal_np = self.current_goal.cpu().numpy()[0]
            int_reward = compute_intrinsic_reward(obs, goal_np, next_obs, agent_player=self.agent_player)

            # Store worker transition
            buf.add_worker_step(
                obs,
                action_np,
                w_log_prob.item(),
                w_value.squeeze(-1).item(),
                goal_np,
                ext_reward,
                int_reward,
                done,
                worker_reward_alpha,
                action_masks=step_masks_np,
                masks=ar_step_masks_np,
            )

            manager_reward_accum += ext_reward
            manager_step_count += 1
            self.goal_step_counter += 1

            if done:
                if manager_segment_open and manager_step_count > 0:
                    buf.end_manager_segment(manager_reward_accum, done=True, segment_length=manager_step_count)
                manager_reward_accum = 0.0
                manager_step_count = 0
                manager_segment_open = False
                reason = info.get("end_reason")
                if reason is not None:
                    end_reasons.append(reason)
                obs, _ = env.reset()
                self.reset_goal()
            else:
                obs = next_obs

        # Close any pending manager segment
        if manager_segment_open and manager_step_count > 0:
            buf.end_manager_segment(manager_reward_accum, done=False, segment_length=manager_step_count)

        # Bootstrap last values for GAE
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs)
            features = self.feature_extractor(obs_tensor)
            # Need a goal for worker value bootstrap
            if self.current_goal is None:
                self.current_goal, _, _ = self.manager.sample_goal(features)
            _, last_w_value = self.worker(features, self.current_goal)
            _, _, _, last_m_value = self.manager(features)

        self._last_obs = obs

        buf.finalize()
        buf.compute_advantages(last_w_value.item(), last_m_value.item(), gamma, gae_lambda)
        buf.end_reasons = end_reasons
        buf.reward_breakdown = reward_breakdown_sums

        self.feature_extractor.train()
        self.manager.train()
        self.worker.train()

        return buf

    def update(
        self,
        buf: FeudalRolloutBuffer,
        n_epochs: int,
        batch_size: int,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
    ) -> Dict[str, float]:
        """
        Run PPO update for both manager and worker.

        Uses separate optimizers to avoid feature extractor drift from
        interleaved worker/manager gradient steps.

        Returns dict of loss metrics.
        """
        import torch.nn.functional as F  # pylint: disable=import-outside-toplevel

        worker_params = list(self.feature_extractor.parameters()) + list(self.worker.parameters())
        manager_params = list(self.feature_extractor.parameters()) + list(self.manager.parameters())

        n_worker = len(buf.w_rewards)
        n_manager = len(buf.m_rewards)
        w_batch_size = min(batch_size, n_worker)
        m_batch_size = min(batch_size, max(n_manager, 1))

        # Convert to tensors
        w_actions_t = torch.as_tensor(buf.w_actions).to(self.device)
        w_old_lp = torch.as_tensor(buf.w_log_probs).to(self.device)
        w_adv = torch.as_tensor(buf.w_advantages).to(self.device)
        w_ret = torch.as_tensor(buf.w_returns).to(self.device)
        w_goals_t = torch.as_tensor(buf.w_goals).float().to(self.device)
        # Per-dim worker masks: tuple of bool tensors (N, dim_i). These are
        # only used inside the inner training loop where we slice by mini-
        # batch index. None when collect_rollout couldn't capture masks
        # (e.g. test envs without action_masks()).
        if buf.has_action_masks:
            w_masks_t = tuple(torch.as_tensor(m, dtype=torch.bool, device=self.device) for m in buf.w_action_masks)
        else:
            w_masks_t = None

        if n_manager > 0:
            m_goals_t = torch.as_tensor(buf.m_goals).float().to(self.device)
            m_old_lp = torch.as_tensor(buf.m_log_probs).to(self.device)
            m_adv = torch.as_tensor(buf.m_advantages).to(self.device)
            m_ret = torch.as_tensor(buf.m_returns).to(self.device)

        # Normalize advantages
        w_adv = (w_adv - w_adv.mean()) / (w_adv.std() + 1e-8)
        if n_manager > 1:
            m_adv = (m_adv - m_adv.mean()) / (m_adv.std() + 1e-8)

        # Accumulators for averaged metrics
        w_policy_loss_sum = 0.0
        w_value_loss_sum = 0.0
        w_entropy_sum = 0.0
        w_batch_count = 0
        m_policy_loss_sum = 0.0
        m_value_loss_sum = 0.0
        m_entropy_sum = 0.0
        m_batch_count = 0

        for _epoch in range(n_epochs):
            # --- Worker update ---
            w_indices = np.random.permutation(n_worker)
            for start in range(0, n_worker, w_batch_size):
                idx = w_indices[start : start + w_batch_size]
                b_obs = self._batch_obs_to_tensor(buf.w_obs_grid[idx], buf.w_obs_units[idx], buf.w_obs_global[idx])
                b_actions = w_actions_t[idx]
                b_old_lp = w_old_lp[idx]
                b_adv = w_adv[idx]
                b_ret = w_ret[idx]
                b_goals = w_goals_t[idx]

                features = self.feature_extractor(b_obs)
                if buf.store_masks:
                    # Autoregressive worker: replay the conditional masks
                    # actually applied at sample time so PPO ratios are
                    # computed under identical mask supports.
                    b_ar_masks = {
                        "atype": torch.as_tensor(buf.w_mask_atype[idx]).to(self.device),
                        "src": torch.as_tensor(buf.w_mask_src[idx]).to(self.device),
                        "unit_type": torch.as_tensor(buf.w_mask_unit_type[idx]).to(self.device),
                        "target": torch.as_tensor(buf.w_mask_target[idx]).to(self.device),
                    }
                    new_lp, entropy, values = self.worker.evaluate_action(
                        features, b_goals, b_actions, masks=b_ar_masks
                    )
                else:
                    b_masks = [m[idx] for m in w_masks_t] if w_masks_t is not None else None
                    new_lp, entropy, values = self.worker.evaluate_action(
                        features, b_goals, b_actions, action_masks=b_masks
                    )

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), b_ret)
                mean_entropy = entropy.mean()

                # ent_coef * (-entropy) subtracts entropy from loss, encouraging exploration
                loss = policy_loss + vf_coef * value_loss - ent_coef * mean_entropy

                self.worker_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(worker_params, max_grad_norm)
                self.worker_optimizer.step()

                w_policy_loss_sum += policy_loss.item()
                w_value_loss_sum += value_loss.item()
                w_entropy_sum += mean_entropy.item()
                w_batch_count += 1

            # --- Manager update ---
            if n_manager > 0:
                m_indices = np.random.permutation(n_manager)
                for start in range(0, n_manager, m_batch_size):
                    idx = m_indices[start : start + m_batch_size]
                    b_obs = self._batch_obs_to_tensor(buf.m_obs_grid[idx], buf.m_obs_units[idx], buf.m_obs_global[idx])
                    b_goals = m_goals_t[idx]
                    b_old_lp = m_old_lp[idx]
                    b_adv = m_adv[idx]
                    b_ret = m_ret[idx]

                    features = self.feature_extractor(b_obs)
                    new_lp, entropy, values = self.manager.evaluate_goal(features, b_goals)

                    ratio = torch.exp(new_lp - b_old_lp)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values.squeeze(-1), b_ret)
                    mean_entropy = entropy.mean()

                    loss = policy_loss + vf_coef * value_loss - ent_coef * mean_entropy

                    self.manager_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(manager_params, max_grad_norm)
                    self.manager_optimizer.step()

                    m_policy_loss_sum += policy_loss.item()
                    m_value_loss_sum += value_loss.item()
                    m_entropy_sum += mean_entropy.item()
                    m_batch_count += 1

        metrics = {
            "worker_policy_loss": w_policy_loss_sum / max(w_batch_count, 1),
            "worker_value_loss": w_value_loss_sum / max(w_batch_count, 1),
            "worker_entropy": w_entropy_sum / max(w_batch_count, 1),
            "manager_policy_loss": m_policy_loss_sum / max(m_batch_count, 1),
            "manager_value_loss": m_value_loss_sum / max(m_batch_count, 1),
            "manager_entropy": m_entropy_sum / max(m_batch_count, 1),
        }

        return metrics

    def save_checkpoint(self, path):
        """Save all network weights and optimizer state."""
        from pathlib import Path as _Path  # pylint: disable=import-outside-toplevel

        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "feature_extractor": self.feature_extractor.state_dict(),
                "manager": self.manager.state_dict(),
                "worker": self.worker.state_dict(),
                "worker_optimizer": self.worker_optimizer.state_dict() if hasattr(self, "worker_optimizer") else None,
                "manager_optimizer": self.manager_optimizer.state_dict() if hasattr(self, "manager_optimizer") else None,
            },
            path,
        )

    def load_checkpoint(self, path):
        """Load network weights and optionally optimizer state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.manager.load_state_dict(checkpoint["manager"])
        self.worker.load_state_dict(checkpoint["worker"])
        if checkpoint.get("worker_optimizer") and hasattr(self, "worker_optimizer"):
            self.worker_optimizer.load_state_dict(checkpoint["worker_optimizer"])
        if checkpoint.get("manager_optimizer") and hasattr(self, "manager_optimizer"):
            self.manager_optimizer.load_state_dict(checkpoint["manager_optimizer"])
        # Backward compatibility: load old single-optimizer checkpoints
        if checkpoint.get("optimizer") and not checkpoint.get("worker_optimizer"):
            if hasattr(self, "worker_optimizer"):
                # Can't reliably split old optimizer state; skip
                pass

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
            if info.get("winner") == self.agent_player:
                wins += 1

        self.feature_extractor.train()
        self.manager.train()
        self.worker.train()

        rewards_arr = np.array(rewards)
        return {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "win_rate": wins / max(n_episodes, 1),
        }


def compute_intrinsic_reward(
    state: Dict[str, np.ndarray],
    goal: np.ndarray,
    next_state: Dict[str, np.ndarray],
    agent_player: int = 1,
) -> float:
    """
    Compute intrinsic reward for worker based on goal achievement.

    The reward is goal-type-aware:
      0=attack:  reward proximity to enemy units near goal
      1=defend:  reward own unit presence at goal (friendly structure area)
      2=capture: reward proximity to goal (neutral/enemy structures)
      3=expand:  reward spreading units toward goal location

    Args:
        state: Current state observation
        goal: (3,) array [goal_x, goal_y, goal_type]
        next_state: Next state observation
        agent_player: Which player the agent controls (1 or 2)

    Returns:
        Intrinsic reward
    """
    goal_x, goal_y, goal_type = int(goal[0]), int(goal[1]), int(goal[2])

    units = next_state["units"]  # (H, W, 3)
    grid = next_state["grid"]  # (H, W, 3)

    # Find player's and opponent's units using agent_player
    player_units = units[:, :, 1] == agent_player
    opponent_units = units[:, :, 1] == (3 - agent_player)

    if not player_units.any():
        return -10.0

    player_positions = np.argwhere(player_units)

    # Base distance reward: proximity of closest own unit to goal
    distances = np.abs(player_positions[:, 0] - goal_y) + np.abs(player_positions[:, 1] - goal_x)
    min_distance = distances.min()
    distance_reward = -min_distance * 0.1

    # Bonus if unit reached goal
    unit_at_goal = (goal_y, goal_x) in map(tuple, player_positions)
    if unit_at_goal:
        distance_reward += 5.0

    # Goal-type-specific bonus
    if goal_type == 0:
        # Attack: bonus for proximity to enemy units near the goal
        if opponent_units.any():
            opp_positions = np.argwhere(opponent_units)
            opp_to_goal = np.abs(opp_positions[:, 0] - goal_y) + np.abs(opp_positions[:, 1] - goal_x)
            # Reward if enemy units are near the goal (meaning we picked a good attack target)
            nearby_enemies = (opp_to_goal <= 3).sum()
            distance_reward += nearby_enemies * 1.0
    elif goal_type == 1:
        # Defend: bonus for having a unit at the goal location (hold position)
        if unit_at_goal:
            distance_reward += 3.0
        # Extra bonus if goal is near a friendly structure
        goal_owner = grid[goal_y, goal_x, 1] if goal_y < grid.shape[0] and goal_x < grid.shape[1] else 0
        if goal_owner == agent_player:
            distance_reward += 2.0
    elif goal_type == 2:
        # Capture: bonus for reaching a structure tile at the goal
        if unit_at_goal:
            goal_terrain = grid[goal_y, goal_x, 0] if goal_y < grid.shape[0] and goal_x < grid.shape[1] else 0
            # Terrain types for capturable structures (building/tower/HQ typically > 0)
            if goal_terrain > 0:
                distance_reward += 4.0
    elif goal_type == 3:
        # Expand: reward spread — count units in a radius around the goal
        in_radius = distances <= 4
        distance_reward += in_radius.sum() * 0.5

    return distance_reward
