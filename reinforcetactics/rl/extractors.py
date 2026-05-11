"""
Shared SB3 features extractors for Reinforce Tactics observations.

The canonical extractor lives here so MaskablePPO (via
``policy_kwargs.features_extractor_class``) and the feudal trainer (which
instantiates the extractor directly) share one implementation. Without
this, the policy network ran through SB3's default ``CombinedExtractor``,
which is just ``nn.Flatten`` on each Dict key — spatial structure in
``grid`` / ``units`` was discarded and the policy had to relearn
adjacency from a flat vector.

Layout assumptions match ``reinforcetactics.rl.observation``:
    grid:            (B, H, W, C_grid) float32 in [0, 1]
    units:           (B, H, W, C_units) float32 in [0, 1]
    global_features: (B, GLOBAL_FEATURES_DIM) float32 in [0, 1)

The two spatial planes are channel-concatenated and permuted to NCHW
before the convolution stack, then either flattened or globally
average-pooled (see ``pool``) and concatenated with the (already
tanh-normalized) global features.
"""

from typing import Dict, Optional

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SpatialFeatureExtractor(BaseFeaturesExtractor):
    """CNN-based features extractor for ``(grid, units, global_features)``.

    Architecture:
        cat(grid, units) -> permute to NCHW -> 3x Conv2d(3x3, ReLU) ->
        pool ('flatten' | 'avg') -> concat global_features -> Linear+ReLU
        -> (B, features_dim)

    Args:
        observation_space: A Dict space exposing ``grid`` and ``units``
            (channel-last) and optionally ``global_features``.
        features_dim: Output width of the trailing Linear projection.
            512 matches the feudal trainer; 256 is a smaller-footprint
            default that suits a PPO policy with its own MLP head.
        pool: How to collapse the CNN's ``(B, 64, H, W)`` output before
            the linear projection.
              - ``"flatten"`` (default): preserves positional info by
                flattening to ``64 * H * W``. Required when downstream
                heads consume per-cell features (e.g. feudal's worker
                spatial action heads).
              - ``"avg"``: ``AdaptiveAvgPool2d(1)`` collapses to 64-dim
                irrespective of map size, making the parameter count
                independent of ``pad_to_size``. Preferred for vanilla
                PPO over a fixed (and possibly heavily padded) action
                space, where the action head doesn't need per-cell
                features in the extractor.
    """

    _SUPPORTED_POOLS = ("flatten", "avg")

    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        pool: str = "flatten",
    ) -> None:
        super().__init__(observation_space, features_dim)

        if pool not in self._SUPPORTED_POOLS:
            raise ValueError(f"pool must be one of {self._SUPPORTED_POOLS}, got {pool!r}")

        # Pull spatial channel counts from the observation space rather
        # than hardcoding them, so this extractor follows the canonical
        # encoding in ``rl.observation`` even if its channel layout
        # changes.
        grid_channels = observation_space["grid"].shape[-1]
        unit_channels = observation_space["units"].shape[-1]
        n_input_channels = grid_channels + unit_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.pool = pool
        self._avg_pool: Optional[nn.Module] = nn.AdaptiveAvgPool2d(1) if pool == "avg" else None

        # Determine global_features size if present in the observation space.
        if "global_features" in observation_space.spaces:
            self.n_global = observation_space["global_features"].shape[0]
        else:
            self.n_global = 0

        # Probe the CNN output shape with a single forward pass on a
        # sample so we don't have to hand-compute it for every (H, W).
        with torch.no_grad():
            sample_obs = observation_space.sample()
            grid = torch.as_tensor(sample_obs["grid"]).float()
            units = torch.as_tensor(sample_obs["units"]).float()
            combined = torch.cat([grid, units], dim=-1).permute(2, 0, 1).unsqueeze(0)
            cnn_out = self.cnn(combined)
            if self._avg_pool is not None:
                cnn_out = self._avg_pool(cnn_out)
            n_flatten = cnn_out.flatten(1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.n_global, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the CNN trunk and project to ``features_dim``."""
        grid = observations["grid"]  # (B, H, W, C_grid)
        units = observations["units"]  # (B, H, W, C_units)

        combined = torch.cat([grid, units], dim=-1)  # (B, H, W, C)
        combined = combined.permute(0, 3, 1, 2)  # (B, C, H, W)

        features = self.cnn(combined)  # (B, 64, H, W)
        if self._avg_pool is not None:
            features = self._avg_pool(features)  # (B, 64, 1, 1)
        features = features.flatten(1)

        if self.n_global > 0 and "global_features" in observations:
            global_feat = observations["global_features"]  # (B, GLOBAL_FEATURES_DIM)
            features = torch.cat([features, global_feat], dim=1)

        return self.linear(features)
