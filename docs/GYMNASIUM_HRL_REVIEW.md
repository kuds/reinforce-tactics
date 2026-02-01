# Gymnasium & HRL Code Review

**Date:** February 2026
**Branch:** claude/review-gymnasium-hrl-RzN1N
**Reviewer:** Code Review (Automated)

---

## Executive Summary

After thorough review of the RL/Gymnasium and HRL codebase, I identified **11 specific issues** (6 bugs, 5 design issues) plus documentation gaps. The most critical bugs are in `ModelBot` (3 bugs making it unusable with trained models) and `WorkerNetwork` (dimension mismatches). The HRL training loop is entirely unimplemented.

---

## 1. Gymnasium Integration - Bugs

### BUG-1 (HIGH): `ModelBot` action mask size mismatch

**File:** `reinforcetactics/game/model_bot.py:143-146`

The `ModelBot._get_observation()` creates an action mask of size `6 * W * H`:
```python
'action_mask': np.ones(
    6 * self.game_state.grid.width * self.game_state.grid.height,
    dtype=np.float32
)
```

But the Gymnasium environment (`gym_env.py:185`) defines the action space size as `10 * W * H`:
```python
def _get_action_space_size(self) -> int:
    return 10 * self.grid_width * self.grid_height
```

**Impact:** Any model trained in `StrategyGameEnv` will receive observations with mismatched `action_mask` shapes when deployed via `ModelBot`, causing crashes or silent incorrect behavior.

**Fix:** Change `model_bot.py:144` to use `10 * W * H`.

---

### BUG-2 (HIGH): `ModelBot` only supports 4 of 8 unit types

**File:** `reinforcetactics/game/model_bot.py:197-198`

```python
unit_codes = ['W', 'M', 'C', 'A']
if unit_type < 0 or unit_type >= len(unit_codes):
    return False
```

The environment supports 8 unit types (`W, M, C, A, K, R, S, B`), but `ModelBot._create_unit()` only maps the first 4. Any trained model that attempts to create a Knight, Rogue, Sorcerer, or Barbarian will silently fail.

**Fix:** Change to `unit_codes = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']`.

---

### BUG-3 (HIGH): `ModelBot` ignores action types 6-9

**File:** `reinforcetactics/game/model_bot.py:172-187`

Only action types 0-5 are handled (create, move, attack, seize, heal, end_turn). Action types 6 (paralyze), 7 (haste), 8 (defence_buff), and 9 (attack_buff) fall through to `"Unknown action type"` warning and return `False`.

**Impact:** A trained agent deployed via `ModelBot` loses access to all special abilities (paralyze, haste, buffs), which are key tactical options.

**Fix:** Add handlers for action types 6-9 with corresponding game state calls.

---

### BUG-4 (MEDIUM): `episode_stats['length']` is never incremented

**File:** `reinforcetactics/rl/gym_env.py:176`

The `episode_stats` dict initializes `'length': 0` but no code ever increments it. The `current_step` counter is incremented on line 714, but `episode_stats['length']` stays at 0 throughout the episode. This metric is dead and will always report 0 in episode info.

**Fix:** Add `self.episode_stats['length'] = self.current_step` in the `step()` method.

---

### BUG-5 (MEDIUM): `seize` action gives base reward before checking success

**File:** `reinforcetactics/rl/gym_env.py:564-571`

```python
result = self.game_state.seize(unit)
reward += 1.0                    # Always added, even if seize fails
if result['captured']:
    reward += 20.0
```

The base `1.0` reward is added before checking if `seize()` succeeded. If the unit isn't on a capturable tile or the seize fails for other reasons, the reward is still given. Additionally, if `seize()` doesn't return a dict with `'captured'` key, this will raise a `KeyError`.

**Fix:** Check the result status before adding any reward.

---

### BUG-6 (LOW): Stale docstrings reference 8 action types, code uses 10

**File:** `reinforcetactics/rl/gym_env.py:238-252, 490`

- `_get_action_mask()` docstring describes 8 action types (0-7) and says size is `8 * W * H`
- `get_action_mask_flat()` docstring says `(8 * W * H,)`
- Actual code implements 10 action types (0-9) with size `10 * W * H`

**Fix:** Update docstrings to reflect the current 10 action types.

---

## 2. HRL (Hierarchical RL) - Bugs & Issues

### BUG-7 (HIGH): `WorkerNetwork` default dimensions are wrong

**File:** `reinforcetactics/rl/feudal_rl.py:207`

```python
action_space_dims: list = [6, 3, 20, 20, 20, 20]
```

- First dim should be **10** (action types), not 6
- Second dim should be **8** (unit types), not 3

The `FeudalRLAgent` constructor (`feudal_rl.py:352-355`) also passes wrong values:
```python
action_space_dims=[6, 3, grid_width, grid_height, grid_width, grid_height]
```

**Impact:** The worker network output dimensions don't match the environment's actual action space, making the HRL agent incompatible with the environment.

**Fix:** Change defaults to `[10, 8, 20, 20, 20, 20]` and constructor to `[10, 8, grid_width, ...]`.

---

### ISSUE-8 (HIGH): Feudal RL training is completely unimplemented

**File:** `train/train_feudal_rl.py:184-202`

```python
def train_feudal_rl(args):
    # TODO: Implement custom Feudal RL training loop
    print("Feudal RL training not yet fully implemented")
    return train_flat_baseline(args)
```

The `FeudalRLAgent`, `ManagerNetwork`, and `WorkerNetwork` classes exist but have:
- No training loop
- No optimizer
- No loss computation
- No buffer/rollout storage

They work for inference only via `select_action()`.

---

### ISSUE-9 (MEDIUM): Hierarchical mode `step()` doesn't handle Dict actions

**File:** `reinforcetactics/rl/gym_env.py:717`

When `hierarchical=True`, the action space is `spaces.Dict({'goal': Discrete, 'primitive': MultiDiscrete})`. But `step()` always calls `self._encode_action(action)` which expects a flat numpy array of 6 elements:
```python
action_type = int(action[0])  # Crashes if action is a Dict
```

**Impact:** Hierarchical mode is unusable; passing a Dict action will crash.

**Fix:** Add Dict action handling in `step()` that extracts the `'primitive'` component.

---

### ISSUE-10 (MEDIUM): `SpatialFeatureExtractor` ignores `global_features`

**File:** `reinforcetactics/rl/feudal_rl.py:47-71`

The CNN only processes 'grid' and 'units' channels (6 total). It completely ignores:
- `global_features` (gold, turn number, unit counts, current player)
- `action_mask`

For a strategy game with economy and turn management, gold reserves and unit counts are critical signals the network should have access to.

**Suggestion:** Concatenate `global_features` after the CNN flatten step, before the final linear layer.

---

### ISSUE-11 (LOW): `compute_intrinsic_reward()` is dead code

**File:** `reinforcetactics/rl/feudal_rl.py:417-460`

This function is defined but never imported or called anywhere in the codebase. It would be used by the missing feudal training loop.

---

## 3. Gymnasium Integration - Improvement Suggestions

### Action masking is an over-approximation

The `action_masks()` method (`gym_env.py:340-483`) returns per-dimension masks (union of valid values per dimension). Since dimensions are interdependent (e.g., valid `to_x/to_y` depends on `action_type` AND `from_x/from_y`), the agent can still select invalid combinations that satisfy each individual dimension mask but not the joint constraint.

**Options:**
- Flatten the action space entirely for exact masking (increases action space but eliminates invalid combos)
- Accept the over-approximation but increase the invalid action penalty to discourage bad combos
- Use a factored action space with conditional masking

### Dense reward shaping runs every step (not just on turn boundaries)

`_calculate_reward()` at `gym_env.py:678-705` computes income diff, unit diff, and structure control diff on every single action, not just on `end_turn`. This means the same state-based signals are applied multiple times within a single turn. Consider computing these only on turn transitions.

### `'random'` opponent type does nothing

`gym_env.py:664-676` -- The random opponent fetches legal actions but discards them:
```python
elif self.opponent_type == 'random':
    legal_actions = self.game_state.get_legal_actions(player=2)
    if legal_actions and legal_actions.get('end_turn'):
        pass  # Just end turn for random opponent
```

A random opponent that actually executes random valid actions would be more useful for training diversity.

### PettingZoo is an unused dependency

`pyproject.toml` lists `pettingzoo` as a base dependency, but it's never imported anywhere in the codebase. This adds unnecessary installation overhead. Remove it or move to optional dependencies if multi-agent support is planned.

### Self-play weight swapping is expensive

`self_play.py:344-373` -- `_get_opponent_action()` copies the current model weights, loads opponent weights, gets an action, then restores original weights. For a 50-action opponent turn, this means 50 full model weight swaps. Consider maintaining a separate opponent policy network.

---

## 4. README & Documentation - Outstanding Work

### `docs-site/docs/implementation-status.md` is severely outdated

Items listed as "missing" that have been implemented:
- "Complete action masking" -- implemented in `rl/masking.py`
- "Map editor GUI" -- implemented in `ui/menus/map_editor/`
- "Tournament/ladder system" -- implemented in `reinforcetactics/tournament/`
- "Fog of war" -- implemented in `core/visibility.py` + `gym_env.py`
- "Action masking returns all 1s" -- no longer true (Known Issues #1)
- References non-existent `rl/action_space.py`
- Lists only SimpleBot; MediumBot and AdvancedBot now exist

### Main `README.md` installation issues

- Uses `pip install pygame` but `pyproject.toml` specifies `pygame-ce`
- Doesn't mention `pip install -e .` or `pip install .[gui]` (the v0.2.0 install method)
- References `maps/1v1/test_map.csv` which does not exist (maps use `beginner.csv`, `crossroads.csv`, etc.)

### `examples/README.md` pending examples

Lists as "Coming Soon":
- Headless training script
- Custom bot implementation tutorial
- Multi-player game setup
- Replay analysis tool

### `docs/IMPROVEMENT_PLAN.md` remaining items (post-tournament consolidation)

1. **Configuration System Unification** (HIGH) -- reward configs hardcoded, settings scattered
2. **LLM Bot Code Consolidation** (MEDIUM) -- three similar bot classes with duplicated logic
3. **Directory Structure Clarification** (MEDIUM) -- dual `/game/` directories
4. **Headless Mode Improvements** (MEDIUM) -- progress callbacks for remote training
5. **Code Quality** (VARIOUS) -- magic strings to enums, type hints, splitting large files
6. **Testing Improvements** (MEDIUM) -- integration tests, performance benchmarks

### `sys.path.insert()` anti-pattern in 10+ files

Found in: `main.py`, `cli/commands.py` (3 occurrences), `examples/llm_bot_demo.py`, `scripts/generate_map_previews.py`, `docker/tournament/run_tournament.py`, and multiple notebooks. With `pyproject.toml` supporting `pip install -e .`, these should be removed.

---

## Priority Summary

| # | Issue | Severity | File | Category |
|---|-------|----------|------|----------|
| 1 | ModelBot mask size `6*W*H` vs env `10*W*H` | HIGH BUG | model_bot.py:143 | Gymnasium |
| 2 | ModelBot only maps 4/8 unit types | HIGH BUG | model_bot.py:197 | Gymnasium |
| 3 | ModelBot ignores action types 6-9 | HIGH BUG | model_bot.py:172 | Gymnasium |
| 7 | WorkerNetwork dims `[6,3,...]` vs env `[10,8,...]` | HIGH BUG | feudal_rl.py:207,355 | HRL |
| 8 | Feudal RL training is unimplemented | HIGH | train_feudal_rl.py:197 | HRL |
| 4 | `episode_stats['length']` never incremented | MEDIUM BUG | gym_env.py:176 | Gymnasium |
| 5 | `seize` gives reward before checking success | MEDIUM BUG | gym_env.py:567 | Gymnasium |
| 9 | Hierarchical step() doesn't handle Dict actions | MEDIUM BUG | gym_env.py:717 | HRL |
| 10 | Feature extractor ignores global_features | MEDIUM | feudal_rl.py:47 | HRL |
| - | PettingZoo is unused dependency | MEDIUM | pyproject.toml:29 | Config |
| - | implementation-status.md severely outdated | MEDIUM | docs-site | Docs |
| - | README install instructions incorrect | MEDIUM | README.md | Docs |
| 6 | Stale docstrings (8 vs 10 action types) | LOW | gym_env.py:238 | Gymnasium |
| 11 | `compute_intrinsic_reward` is dead code | LOW | feudal_rl.py:417 | HRL |
