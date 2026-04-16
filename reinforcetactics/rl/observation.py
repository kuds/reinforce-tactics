"""
Shared observation builder for RL agents.

Single source of truth for turning a GameState into the observation dict
consumed by the gym environment, MCTS, and ModelBot. Prior to consolidation,
MCTS and ModelBot always placed player 1's gold and unit count first in
``global_features``, violating the agent-relative contract whenever
``current_player`` was 2 (MCTS) or the bot played as player 2 (ModelBot).

Observation contract (2-player):
    grid:            (H, W, 3) float32 - terrain, owner, structure HP
    units:           (H, W, 3) float32 - unit type, owner, HP %
    global_features: (6,) float32
        [own_gold, opp_gold, turn, own_units, opp_units, current_player]
    action_mask:     (N,) float32 - supplied by the caller
    visibility:      (H, W) uint8 - present only when fog_of_war is True

Masks are NOT built here: callers pass in a pre-computed mask. Phase 2 will
consolidate action-mask building in a separate module.
"""

from typing import Any, Dict, Optional

import numpy as np


def build_observation(
    game_state: Any,
    perspective_player: int,
    action_mask: np.ndarray,
    fog_of_war: Optional[bool] = None,
) -> Dict[str, np.ndarray]:
    """Build an RL observation from ``perspective_player``'s viewpoint.

    Args:
        game_state: The GameState to observe.
        perspective_player: Player whose gold/unit count go first in
            ``global_features``. Under fog of war, only this player's
            visibility is applied to grid and units.
        action_mask: Precomputed legal-action mask. Ownership stays with the
            caller so this builder has no dependency on action-mask logic.
        fog_of_war: Override for the FOW flag. When ``None``, falls back to
            ``game_state.fog_of_war``.

    Returns:
        Observation dict with keys ``grid``, ``units``, ``global_features``,
        ``action_mask``, and (when FOW is enabled) ``visibility``.
    """
    if fog_of_war is None:
        fog_of_war = bool(getattr(game_state, "fog_of_war", False))

    opp = 3 - perspective_player

    if fog_of_war:
        state_arrays = game_state.to_numpy(for_player=perspective_player)
    else:
        state_arrays = game_state.to_numpy()

    own_gold = game_state.player_gold.get(perspective_player, 0)
    own_units = sum(1 for u in game_state.units if u.player == perspective_player)

    if fog_of_war:
        # Enemy gold is hidden; only visible enemy units are counted.
        opp_gold: float = 0
        opp_units = sum(
            1 for u in game_state.units if u.player == opp and game_state.is_position_visible(u.x, u.y, perspective_player)
        )
    else:
        opp_gold = game_state.player_gold.get(opp, 0)
        opp_units = sum(1 for u in game_state.units if u.player == opp)

    global_features = np.array(
        [
            own_gold,
            opp_gold,
            game_state.turn_number,
            own_units,
            opp_units,
            game_state.current_player,
        ],
        dtype=np.float32,
    )

    obs: Dict[str, np.ndarray] = {
        "grid": state_arrays["grid"].astype(np.float32),
        "units": state_arrays["units"].astype(np.float32),
        "global_features": global_features,
        "action_mask": action_mask,
    }

    if fog_of_war and "visibility" in state_arrays:
        obs["visibility"] = state_arrays["visibility"]

    return obs
