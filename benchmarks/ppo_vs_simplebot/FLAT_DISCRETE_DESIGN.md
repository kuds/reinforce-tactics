# Flat Discrete Action Space Design

## Problem Statement

The original `MultiDiscrete([10, 8, 6, 6, 6, 6])` action space uses **per-dimension
masking**, which creates a severe over-approximation problem. Each of the 6 dimensions
is masked independently, so the mask is the union of valid values across all legal
actions. This means:

- Total combinations: `10 × 8 × 6 × 6 × 6 × 6 = 103,680`
- Typical legal actions per step: ~20–50
- Mask-valid combinations: ~3,000–4,000 (union of per-dimension values)
- **~99% of sampled actions are game-invalid**

The agent wastes nearly all of its training signal on invalid-action penalties (-10 each),
making it impossible to learn useful gameplay behaviors.

## Solution: Flat `Discrete(N)` with Exact Masking

Replace the `MultiDiscrete` space with a single `Discrete(512)` space where each index
maps to a specific legal action. The mask is **exact**: only truly valid actions are
marked as valid.

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Max action count | 512 | Sufficient for 6×6 maps (~50-80 legal actions typical). Configurable via `max_flat_actions` parameter. |
| Action encoding | Same 6-tuple `[action_type, unit_type, from_x, from_y, to_x, to_y]` | Reuses existing `_execute_action()` logic unchanged |
| Action enumeration | Per-step from `get_legal_actions()` | Exact match with game rules, rebuilt every step |
| Deduplication | Tuple-based `seen` set | Prevents duplicate entries from different code paths |
| End turn | Always appended last | Guarantees agent always has at least one valid action |
| Backward compatibility | `action_space_type` parameter | Both `'multi_discrete'` and `'flat_discrete'` available |

### How It Works

```
Step N:
  1. Environment calls _build_flat_actions()
  2. Enumerates ALL legal actions from game_state.get_legal_actions()
  3. Converts each to [action_type, unit_type, from_x, from_y, to_x, to_y]
  4. Stores in self._current_actions (list of numpy arrays)
  5. Mask = [True] * len(_current_actions) + [False] * (512 - len)
  6. Policy samples action INDEX from masked Discrete(512)
  7. step() resolves: action = self._current_actions[index]
  8. Executes via existing _execute_action() path
```

### Action Mapping

The `_ACTION_KEY_MAP` translates `get_legal_actions()` output format to the 6-tuple:

| Game Action | action_type | from_fields | to_fields |
|-------------|-------------|-------------|-----------|
| create_unit | 0 | (building x,y) | (building x,y) |
| move | 1 | (from_x, from_y) | (to_x, to_y) |
| attack | 2 | (attacker.x, .y) | (target.x, .y) |
| seize | 3 | (unit.x, .y) | (tile.x, .y) |
| heal | 4 | (healer.x, .y) | (target.x, .y) |
| cure | 4 | (curer.x, .y) | (target.x, .y) |
| end_turn | 5 | (0, 0) | (0, 0) |
| paralyze | 6 | (paralyzer.x, .y) | (target.x, .y) |
| haste | 7 | (sorcerer.x, .y) | (target.x, .y) |
| defence_buff | 8 | (sorcerer.x, .y) | (target.x, .y) |
| attack_buff | 9 | (sorcerer.x, .y) | (target.x, .y) |

## Configuration

### Switching Between Modes

Both modes are available via the `action_space_type` parameter:

```python
from reinforcetactics.rl.masking import make_maskable_env

# Per-dimension masking (original, higher invalid rate)
env_multi = make_maskable_env(
    opponent="bot",
    action_space_type="multi_discrete"  # default
)

# Exact masking (recommended for training)
env_flat = make_maskable_env(
    opponent="bot",
    action_space_type="flat_discrete",
    max_flat_actions=512
)
```

Vectorized environments:

```python
from reinforcetactics.rl.masking import make_maskable_vec_env

vec_env = make_maskable_vec_env(
    n_envs=4,
    opponent="bot",
    action_space_type="flat_discrete",
    max_flat_actions=512
)
```

### Impact on Model/Policy

The `action_space_type` changes the **action space shape**, which means models trained
with one mode are **not compatible** with the other. Specifically:

- `multi_discrete`: Action space is `MultiDiscrete([10, 8, 6, 6, 6, 6])`. Policy outputs
  6 independent categorical distributions (one per dimension). MaskablePPO applies
  per-dimension masks.

- `flat_discrete`: Action space is `Discrete(512)`. Policy outputs a single categorical
  distribution over 512 actions. MaskablePPO applies a single exact mask.

**You cannot load a model trained with `multi_discrete` and evaluate it with
`flat_discrete`** (or vice versa). Each mode produces a different neural network
architecture. Choose one mode before training and use it consistently.

### Choosing `max_flat_actions`

The parameter should be larger than the maximum number of legal actions in any game state:

| Map Size | Max Units/Side | Typical Max Legal Actions | Recommended `max_flat_actions` |
|----------|---------------|--------------------------|-------------------------------|
| 6×6 | ~5 | ~80 | 256–512 |
| 10×10 | ~10 | ~300 | 512 |
| 14×10 | ~15 | ~600 | 1024 |
| 16×16 | ~20 | ~1200 | 1536–2048 |

If the actual legal actions exceed `max_flat_actions`, a warning is logged and the list
is truncated. Padding indices (beyond the action list) are always masked as invalid.

## Bug Fixes Included

### 1. Move destination validation in `get_legal_actions()` (game_state.py)

**Problem:** `get_legal_actions()` used pathfinding reachability (`is_destination=False`)
to enumerate move targets. This allows passing through friendly units during pathfinding
but also included friendly-occupied positions as valid destinations. When `move_unit()`
later validates with `is_destination=True`, these positions are rejected.

**Fix:** Added a secondary `can_move_to_position(..., is_destination=True)` check before
adding move actions to the legal actions list. This eliminates the ~16.6% invalid move
rate that affected both `multi_discrete` and `flat_discrete` modes.

### 2. Cure action field name mismatch (gym_env.py)

**Problem:** `_ACTION_KEY_MAP` mapped `'cure'` with `src_fields='healer'`, but
`get_legal_actions()` returns cure actions with key `'curer'`.

**Fix:** Updated mapping to `'cure': (4, 'curer', 'target')`.

### 3. Double end_turn in opponent handling (gym_env.py)

**Problem:** After the agent plays `end_turn`, the environment calls
`_opponent_turn()` which invokes `SimpleBot.take_turn()`. The bot internally
calls `game_state.end_turn()`. The environment then also called
`game_state.end_turn()`, double-toggling `current_player` and leaving it
pointing at the opponent instead of the agent.

**Fix:** The environment now conditionally calls `end_turn` only if
`current_player != agent_player` after the opponent's turn.

## Scalability Considerations

For maps larger than ~14×10 with 20+ units, the flat Discrete approach may hit
practical limits:

- **Action space size**: With 20 units each having 10-20 moves, 5 attacks, plus
  abilities, legal actions can reach 1000-1500. A `Discrete(2048)` space works but
  increases memory and training time.

- **Credit assignment**: The policy must learn to distinguish 2048 discrete actions,
  many of which are semantically similar (e.g., moving the same unit to adjacent cells).

### Alternative architectures for large maps

1. **Two-stage autoregressive**: Select unit first, then select action for that unit.
   Smaller action spaces with exact masking at both stages.

2. **Action embedding / pointer networks**: Encode each legal action as a feature vector,
   score them with a learned network. Scales naturally with action count (AlphaStar-style).

3. **Hybrid approach**: Use flat Discrete for small maps during initial development,
   switch to two-stage or embedding-based approaches for larger maps.

For the current 6×6 benchmark, `Discrete(512)` is more than sufficient and eliminates
the root cause of the 0% win rate.
