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
before the convolution stack, then pooled (see ``pool``) and concatenated
with the (already tanh-normalized) global features.
"""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F

from reinforcetactics.rl.observation import NUM_TILE_TYPES


class SpatialFeatureExtractor(BaseFeaturesExtractor):
    """CNN-based features extractor for ``(grid, units, global_features)``.

    Architecture:
        [optional coord-conv channels] -> cat(grid, units) -> permute to
        NCHW -> 3x Conv2d(3x3, ReLU) [+ optional 4th conv block] -> pool
        ('flatten' | 'avg' | 'masked_avg') -> concat global_features ->
        Linear+ReLU -> (B, features_dim)

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
                irrespective of map size. Map-size-invariant but
                contaminated by padded zero cells when the live map is
                much smaller than the padded shape (e.g. a 6x6 map in a
                (10, 12) padded grid is 30% live, 70% pad).
              - ``"masked_avg"``: averages CNN features only over live
                cells (those with a tile-type one-hot bit set in
                ``grid[..., :NUM_TILE_TYPES]``). Same 64-dim output as
                ``"avg"`` but undiluted by padding. Recommended for
                vanilla PPO over curricula that pad small maps up to a
                shared shape.
        coord_conv: Append two normalized coordinate channels (``y/(H-1)``
            and ``x/(W-1)``) to the spatial input. Cheap inductive bias
            for "near top / near left" reasoning; useful since HQs and
            spawn structures sit at fixed map-relative offsets.
        extra_conv: Add a 4th Conv2d(64, 64, 3x3) block. Extends the
            receptive field from ~7x7 to ~9x9, which matters on larger
            maps where the default RF covers only a fraction of the
            board.
    """

    _SUPPORTED_POOLS = ("flatten", "avg", "masked_avg")

    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        pool: str = "flatten",
        coord_conv: bool = False,
        extra_conv: bool = False,
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
        n_spatial_channels = grid_channels + unit_channels

        self.pool = pool
        self.coord_conv = bool(coord_conv)
        self.extra_conv = bool(extra_conv)

        # Coord-conv buffer is sized to the (padded) observation shape.
        # Padded cells get coord values like real ones; under masked_avg
        # those cells are excluded from the spatial average, and under
        # plain avg / flatten the conv can learn to discount them via
        # the all-zero tile-type one-hot signature on pad cells.
        if self.coord_conv:
            h, w = observation_space["grid"].shape[:2]
            coord_y = torch.linspace(0.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
            coord_x = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
            self.register_buffer("coord_planes", torch.cat([coord_y, coord_x], dim=1))
            n_input_channels = n_spatial_channels + 2
        else:
            n_input_channels = n_spatial_channels

        conv_layers = [
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ]
        if self.extra_conv:
            conv_layers.extend(
                [
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ]
            )
        self.cnn = nn.Sequential(*conv_layers)

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
            if self.coord_conv:
                combined = torch.cat([combined, self.get_buffer("coord_planes")], dim=1)
            cnn_out = self.cnn(combined)
            if pool in ("avg", "masked_avg"):
                # Both collapse the spatial dims to 1x1; only the
                # weighting differs, which doesn't change the post-pool
                # feature width.
                cnn_out = cnn_out.mean(dim=(2, 3), keepdim=True)
            n_flatten = cnn_out.flatten(1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.n_global, features_dim),
            nn.ReLU(),
        )

    def _live_cell_mask(self, grid: torch.Tensor) -> torch.Tensor:
        """Build a (B, 1, H, W) float mask marking real (non-padded) cells.

        Pad cells produced by ``build_observation`` are all-zero across
        every channel; in particular every real cell has exactly one of
        ``grid[..., :NUM_TILE_TYPES]`` set, while pad cells have zero
        across that slice. We use that signature to recover the live
        region without needing to know ``(H_live, W_live)`` from the
        caller — handy because the live shape can vary per stage in a
        curriculum even when the padded shape is fixed.
        """
        tile_sum = grid[..., :NUM_TILE_TYPES].sum(dim=-1)  # (B, H, W)
        return (tile_sum > 0).float().unsqueeze(1)  # (B, 1, H, W)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the CNN trunk and project to ``features_dim``."""
        grid = observations["grid"]  # (B, H, W, C_grid)
        units = observations["units"]  # (B, H, W, C_units)

        combined = torch.cat([grid, units], dim=-1)  # (B, H, W, C)
        combined = combined.permute(0, 3, 1, 2)  # (B, C, H, W)

        if self.coord_conv:
            coords = self.get_buffer("coord_planes").expand(combined.shape[0], -1, -1, -1)
            combined = torch.cat([combined, coords], dim=1)

        features = self.cnn(combined)  # (B, 64, H, W)

        if self.pool == "avg":
            features = F.adaptive_avg_pool2d(features, 1)  # (B, 64, 1, 1)
            features = features.flatten(1)
        elif self.pool == "masked_avg":
            live = self._live_cell_mask(grid)  # (B, 1, H, W)
            # Clamp denominator to 1 to defend against (impossible in
            # practice) all-pad observations rather than producing NaN.
            denom = live.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
            features = (features * live).sum(dim=(2, 3), keepdim=True) / denom
            features = features.flatten(1)
        else:  # "flatten"
            features = features.flatten(1)

        if self.n_global > 0 and "global_features" in observations:
            global_feat = observations["global_features"]  # (B, GLOBAL_FEATURES_DIM)
            features = torch.cat([features, global_feat], dim=1)

        return self.linear(features)
