# Codebase Review — Reinforce Tactics v0.2.0

> **Date:** February 2026
> **Scope:** Full review covering bugs, RL improvements, UI updates, game balance, architecture, testing, training pipelines, and documentation.

---

## Executive Summary

Reinforce Tactics is a well-structured turn-based strategy game with an impressive feature set: 8 unit types with unique abilities, a Gymnasium RL environment, AlphaZero/MCTS, Feudal RL, self-play training, LLM bots, a tournament system, and a full Pygame UI with sprite animations. The codebase is modular and generally clean.

This review identifies **28 bugs** (6 critical, 10 high, 12 medium), **12 RL enhancement opportunities**, **11 UI/UX improvements**, **5 game balance concerns**, **9 architecture/testing gaps**, and **6 training pipeline issues**.

---

## 1. Bugs — Critical

### BUG-1: MCTS deepcopy uses stale unit/tile references (Critical)
**File:** `reinforcetactics/rl/mcts.py:174-176, 439-446`

When `_select_child` deepcopies a parent's game state and then calls `_execute_action_on_state`, the `action_data` dict still references Unit objects from the **parent** state, not the deepcopied child. Actions like `move`, `attack`, `heal`, `paralyze`, etc. all pass original object references to the new game state.

```python
# mcts.py line 174-175 — 'unit' is from the PARENT state
game_state.move_unit(action_data['unit'], ...)  # Wrong unit object!
```

**Impact:** Every non-end-turn MCTS action operates on wrong state, producing incorrect tree search results. AlphaZero is fundamentally broken.

**Fix:** After deepcopy, re-resolve unit references by position (e.g., find unit at `(unit.x, unit.y)` in the child state's unit list).

### BUG-2: MCTS PUCT sign flip has wrong guard condition
**File:** `reinforcetactics/rl/mcts.py:421-426`

```python
if child.player != node.player and child.game_state is not None:
    q = -q
```

For lazily-expanded children (game_state is None), `child.player == 0` (set on line 56 when game_state is None). Since `0 != node.player` is always true, but `game_state is not None` is false, the sign flip is skipped. This means unexpanded opponent nodes are treated as if they're good for the parent, biasing selection toward unexpanded opponent nodes.

### BUG-3: Self-play flipped observation doesn't recalculate action mask
**File:** `reinforcetactics/rl/self_play.py:447`

```python
if 'action_mask' in obs:
    flipped['action_mask'] = obs['action_mask'].copy()  # Wrong! Uses agent's mask for opponent
```

The opponent model receives the agent's action mask, not its own legal actions. The opponent will attempt illegal actions.

### BUG-4: Self-play `_get_obs_for_player` broken when agent is player 2
**File:** `reinforcetactics/rl/self_play.py:496`

The function flips the observation when `player == 2`, regardless of who the agent is. When `agent_player == 2`, calling `_get_obs_for_player(1)` returns the un-flipped observation (player 2's perspective), giving the opponent model the wrong view.

### BUG-5: Winner determination broken for 3+ players
**File:** `reinforcetactics/core/game_state.py:500-502, 518-521`

```python
self.winner = defeated_player + 1 if defeated_player < self.num_players else 1
```

When player 2 is eliminated in a 3-player game, player 3 is immediately declared the winner even though player 1 is still alive. Should only declare a winner when exactly one player remains.

### BUG-6: `train/train_eval.py` imports nonexistent modules (Broken)
**File:** `train/train_eval.py`

Imports `reinforcetactics.entities`, `reinforcetactics.leaderboard`, `reinforcetactics.model_manager`, `reinforcetactics.envs`, `reinforcetactics.curriculum` — none of which exist. The entire file is dead code that crashes immediately.

**Fix:** Delete `train/train_eval.py` or rewrite using current module structure.

---

## 2. Bugs — High

### BUG-7: All bots try to move to occupied positions
**File:** `reinforcetactics/game/bot.py:67-90`

`find_best_move_position` uses `is_destination=False` in the pathfinding lambda, which allows passing through friendly units. The returned positions include tiles occupied by friendly units. When the bot calls `move_unit()`, the move fails silently and the bot wastes its turn.

### BUG-8: All bots skip seizing structures at full health
**File:** `reinforcetactics/game/bot.py:184-185, 867-869, 1165-1167`

The early-seize check in SimpleBot, MediumBot, and AdvancedBot requires `tile.health < tile.max_health`. Bots standing on enemy/neutral structures at full health won't seize them via the fast path.

### BUG-9: MediumBot kill calculation ignores defence reduction
**File:** `reinforcetactics/game/bot.py:654-662`

`find_killable_targets` sums raw damage without applying `apply_defence_reduction()`. Bots think they can kill targets they actually can't, leading to wasted coordinated attacks.

### BUG-10: AdvancedBot uses wrong attribute name for haste check
**File:** `reinforcetactics/game/bot.py:1582, 1598`

```python
not getattr(a, 'hasted', False)  # Wrong! Should be 'is_hasted'
```

Always returns `False` because `hasted` doesn't exist. The bot tries to haste already-hasted units. The mechanic rejects it silently, and the Sorcerer wastes its turn.

### BUG-11: AdvancedBot Sorcerer abilities don't check return values
**File:** `reinforcetactics/game/bot.py:1578-1632`

```python
self.game_state.haste(unit, knight)
return True  # Returns True even if haste was on cooldown and failed!
```

Failed buff/haste attempts are treated as successes, preventing the bot from trying other actions.

### BUG-12: LLM bots completely missing Sorcerer abilities
**File:** `reinforcetactics/game/llm_bot.py:778-898, 930-949, 951-1006`

The `_format_legal_actions()`, `_format_prompt()`, and `_execute_actions()` methods have no handlers for HASTE, DEFENCE_BUFF, or ATTACK_BUFF. Sorcerer abilities are invisible to all LLM bots.

### BUG-13: LLM bot CREATE_UNIT template only shows 4 of 8 unit types
**File:** `reinforcetactics/game/llm_bot.py:935`

Template shows `"unit_type": "W|M|C|A"`, omitting K, R, S, B. The LLM may never create Knights, Rogues, Sorcerers, or Barbarians.

### BUG-14: End-turn fills entire flat mask slice, biasing policy
**File:** `reinforcetactics/rl/gym_env.py:368`

```python
flat_mask[5 * area: 6 * area] = 1.0  # Marks ALL W*H positions as valid for end_turn
```

On a 20×20 map, this creates 400 "end turn" entries in the mask vs. a handful for other actions. This heavily biases the policy toward ending turns early.

### BUG-15: AlphaZero evaluation creates opponent with wrong architecture
**File:** `reinforcetactics/rl/alphazero_trainer.py:463-466`

Opponent network uses default `num_res_blocks` and `channels` instead of the trainer's values. If trained with non-default architecture, `load_state_dict` will crash.

### BUG-16: Feudal RL intrinsic reward hardcodes player 1
**File:** `reinforcetactics/rl/feudal_rl.py:885`

```python
player_units = (units[:, :, 1] == 1)  # Always player 1!
```

When agent plays as player 2, intrinsic reward is computed using opponent's units, completely inverting the signal.

---

## 3. Bugs — Medium

### BUG-17: Counter-damage calculated twice in `attack_unit`
**File:** `reinforcetactics/game/mechanics.py:415-458`

Counter-damage is computed and applied (lines 415-435), then recomputed independently for the return dict (lines 437-458). The two may diverge due to Rogue evasion randomness. Displayed damage can differ from actual.

### BUG-18: Random opponent hardcoded to player 2
**File:** `reinforcetactics/rl/gym_env.py:687`

`_random_opponent_turn()` uses `player=2`, breaking self-play with swapped players.

### BUG-19: `make_curriculum_env` mutates shared config dict
**File:** `reinforcetactics/rl/masking.py:391`

`config.update(kwargs)` modifies the `difficulty_configs` dict entry in-place. Second call with same difficulty gets corrupted defaults.

### BUG-20: MediumBot temporarily mutates live game state
**File:** `reinforcetactics/game/bot.py:639-646`

Unit position is temporarily changed to compute attackable enemies. If an exception occurs, the position is permanently corrupted.

### BUG-21: Settings shallow copy corrupts class defaults
**File:** `reinforcetactics/utils/settings.py:71-89`

`DEFAULT_SETTINGS.copy()` is shallow — nested dicts are shared references. Calling `result[key].update(value)` in `_merge_with_defaults` permanently modifies `DEFAULT_SETTINGS` nested dicts for all future instances.

### BUG-22: Random map generation uses ocean base instead of grass
**File:** `reinforcetactics/utils/file_io.py:309`

`np.full((height, width), 'o', dtype=object)` creates an all-ocean map. The comment on line 68 says "Replace NaN with grass" but replaces with `'o'` (ocean). Neutral towers check `map_data[y, x] == 'p'` which is never true.

### BUG-23: Feudal RL goal_type is completely ignored
**File:** `reinforcetactics/rl/feudal_rl.py:877`

```python
goal_x, goal_y, _ = int(goal[0]), int(goal[1]), int(goal[2])
```

The goal type (attack/defend/capture/expand) is extracted and discarded. All goal types use the same distance metric, making the manager's goal_type output unlearnable.

### BUG-24: AlphaZero "epochs" are single batches
**File:** `reinforcetactics/rl/alphazero_trainer.py:391-439`

Each "epoch" samples one batch. With a 100K replay buffer and batch_size=256, each epoch sees 0.256% of data. Should iterate over the full buffer.

### BUG-25: AlphaZero checkpoint doesn't save architecture parameters
**File:** `reinforcetactics/rl/alphazero_trainer.py:554-582`

Only saves `grid_height`, `grid_width`, `num_simulations`. Architecture params like `num_res_blocks` and `channels` are lost, causing mismatches on reload.

### BUG-26: Mixed self-play training mode is non-functional
**File:** `train/train_self_play.py:444-457`

Bot environments are created and assigned to `_` (discarded). The `MixedTrainingCallback` sets a flag but never switches environments. Mixed mode is identical to pure self-play.

### BUG-27: `wandb.finish()` crashes when import failed
**File:** `train/train_feudal_rl.py:413`

Unguarded `wandb.finish()` raises `NameError` when the `wandb` import failed.

### BUG-28: `Renderer._draw_tile` fog overlay allocates new surface every frame
**File:** `reinforcetactics/ui/renderer.py:384-386`

Creates a new `pygame.Surface` per fog tile per frame. On a 20×20 map with significant fog, this is hundreds of allocations per frame. Should be cached.

---

## 4. RL Enhancements

### RL-1: Observation normalization missing
**File:** `reinforcetactics/rl/gym_env.py:229-276`

Raw game values (gold up to 10000, turn_number unbounded, mixed-scale grid channels) are passed without normalization. Neural networks train much better with normalized inputs.

### RL-2: Default to flat_discrete action space
The `multi_discrete` space has 12.8M combinations for a 20×20 map. `flat_discrete` eliminates invalid combinations entirely and should be the default.

### RL-3: Reward shaping may dominate terminal signal
Per-step shaping can accumulate to compete with the +/-1000 terminal reward. Log decomposed reward components during training to verify balance.

### RL-4: Self-play lacks ELO-based opponent selection
The `prioritized` strategy uses raw cumulative win rates. The existing `tournament/elo.py` could be integrated for more principled strength-based selection.

### RL-5: Curriculum learning not wired into training scripts
`make_curriculum_env()` defines easy/medium/hard presets but no training script uses them. A proper curriculum would significantly improve learning.

### RL-6: AlphaZero training lacks temperature annealing
The standard schedule (temperature=1.0 for first N moves, then →0) isn't implemented.

### RL-7: Feudal RL intrinsic reward is too simplistic
All goal types use the same distance metric. Goal-type-specific rewards (damage for attack, structure HP for capture, etc.) would make the manager meaningful.

### RL-8: No multi-agent RL support
PettingZoo is already a dependency but unused. A PettingZoo-compatible environment would enable 2v2 and 1v1v1 multi-agent training.

### RL-9: No temporal context in observations
The observation is a single snapshot. Frame stacking, previous action encoding, or LSTM/Transformer policies would provide temporal context.

### RL-10: `max_steps=200` may be too low
200 individual actions ≈ 16-33 full turns. Games on larger maps may never reach completion. Consider 500+ or proportional to map size.

### RL-11: No buff/debuff status in observation
The units array (type, owner, HP) lacks paralysis, haste, buff status, and cooldowns. The agent is blind to ability effects.

### RL-12: AlphaZero policy head has 51.2M parameters
`nn.Linear(32 * 20 * 20, 10 * 20 * 20)` = 12800×4000 matrix. Consider a factored approach (separate spatial and action-type predictions).

---

## 5. UI/UX Improvements

### UI-1: No visual feedback for buff/debuff status
No visual indicators on the board for paralyzed, hasted, or buffed units. Players must hover each unit to check.

### UI-2: No undo/cancel for ability actions
Special abilities (paralyze, haste, buffs) with cooldowns are irreversible. Add a confirmation step.

### UI-3: No action log / game log panel
No visible log of recent actions. Opponent (bot) actions are invisible unless you notice state changes.

### UI-4: No structure capture progress indicator
No health bar or progress overlay on structures being contested.

### UI-5: Map editor unsaved changes lost without warning
**File:** `reinforcetactics/ui/menus/map_editor/map_editor.py:103-106`

The QUIT handler has a `pass` with a TODO comment. Unsaved changes are silently lost.

### UI-6: Bot turns block the render loop
**File:** `game/input_handler.py:405-419`

`_process_bot_turns` runs synchronously, freezing the screen during AI computation. Especially problematic for LLM bots making API calls.

### UI-7: Turn transition is instantaneous
No visual banner, animation, or sound when turns switch. Easy to miss when opponent's turn completes.

### UI-8: `pygame.quit()` called in multiple places
**File:** `game/game_loop.py:304, 382, 435`

Both `start_new_game` and `load_saved_game` call `pygame.quit()`. Starting a new game after one ends requires reinitializing pygame.

### UI-9: Error feedback only via console `print()`
Map editor save results, player config validation errors, model validation errors, and file I/O errors are all printed to stdout. Users interacting with the GUI never see these messages.

### UI-10: Replay stepping backward replays from the beginning
**File:** `reinforcetactics/utils/replay_player.py:415-431`

Going back one step replays all prior steps from scratch, creating a new GameState and Renderer each time. O(N) per step-back, O(N²) for repeated backward stepping.

### UI-11: Single-target auto-execute removes player choice
**File:** `game/action_executor.py:93-123`

When there's exactly one valid target, the action executes immediately without entering target selection. No way to cancel.

---

## 6. Game Balance Concerns

### BAL-1: Barbarian mislabeled as "glass cannon"
At 400 gold, the Barbarian has the **highest HP (20)** and **highest movement (5)** in the game. With 2 defence (only 10% damage reduction vs Warrior's 30%), the Barbarian is actually a mobile tank, not fragile. Two Warriors (same cost) provide more total HP, more defence, and two actions per turn.

**Recommendation:** Either reduce HP to 14-16 to justify "glass cannon", increase cost to 500+, or give it a unique ability to justify the premium.

### BAL-2: Archer is underpowered at 250 gold
5 attack with 1 defence yields ~3.5 effective damage against a Warrior. Can't attack adjacent units, making it useless when closed on.

**Recommendation:** Increase attack to 7-8 or reduce cost to 175-200.

### BAL-3: Sorcerer kit is overloaded
Four abilities (ranged attack, haste, +35% defence buff, +35% attack buff) at 400 gold. A Sorcerer + Knight combo with haste + charge + attack buff can deal 20+ damage twice in one turn.

**Recommendation:** Make haste and buffs mutually exclusive per turn, or increase cooldowns.

### BAL-4: Paralyze enables near-permanent lockdown
3-turn paralysis with 2-turn cooldown = unit is paralyzed 60% of the time. Devastating against expensive units.

**Recommendation:** Reduce to 2 turns, or add post-paralysis immunity (1 turn).

### BAL-5: Structure regen is too fast
50% HP/turn means towers fully heal in 2 turns, making structure denial non-viable.

**Recommendation:** Reduce to 20-25% per turn.

---

## 7. Architecture & Code Quality

### ARCH-1: Duplicated unit type mappings across 4+ files
`unit_type_to_idx` mapping is defined independently in `gym_env.py`, `mcts.py`, `bot.py`, and `model_bot.py`. Should be a single constant in `constants.py`.

### ARCH-2: `torch.load` without `weights_only=True`
**File:** `reinforcetactics/rl/feudal_rl.py:813`

Security risk (arbitrary code execution via pickle) on PyTorch < 2.6.

### ARCH-3: Duplicated action dispatch logic
**File:** `game/action_executor.py` and `game/input_handler.py:266-286`

Same attack/paralyze/heal/cure/haste dispatch exists in both files. Any new action type must be added in both places.

### ARCH-4: `id(unit)` as sprite animator key is fragile
**File:** `reinforcetactics/ui/sprite_animator.py:267-268`

Python `id()` can be reused after garbage collection. Destroyed + newly created units at the same memory address get stale animation state.

### ARCH-5: No type checking or schema validation for map data
Map data is a pandas DataFrame with no schema validation. Malformed maps cause cryptic errors deep in initialization.

### ARCH-6: Game state mutation is not isolated
No command pattern or transactional boundary. Multiple callers (gym env, bots, input handler, action executor) all mutate state directly.

### ARCH-7: `GameState.get_legal_actions()` called redundantly
Called multiple times per step (masking, flat actions, opponent). Should cache and invalidate on state change.

### ARCH-8: LLM bot Cleric heal range only checks distance 1
**File:** `reinforcetactics/game/llm_bot.py:751-774`

`_compute_move_then_actions` for Cleric only checks adjacent positions, missing heal opportunities at distance 2.

### ARCH-9: `WorkerNetwork` mutable default argument
**File:** `reinforcetactics/rl/feudal_rl.py:220`

```python
action_space_dims: list = [10, 8, 20, 20, 20, 20]  # Mutable default!
```

Shared across all calls. Should use `None` with a conditional.

---

## 8. Testing Gaps

### TEST-1: No tests for Feudal RL (HIGH)
`feudal_rl.py` has complex Manager-Worker hierarchy, rollout buffer, GAE, and PPO update with zero test coverage.

### TEST-2: No direct MCTS tests (HIGH)
MCTS is only tested indirectly via `test_alphazero.py`. Missing: node expansion, backpropagation values, PUCT selection, temperature correctness.

### TEST-3: No tests for `masking.py` utilities (HIGH)
`make_maskable_env`, `make_maskable_vec_env`, `make_curriculum_env`, `validate_action_mask` are untested. `verify_mask.py` uses `unittest` (inconsistent) and covers only 2 narrow scenarios.

### TEST-4: No training smoke tests
No test runs even 100 steps of MaskablePPO to verify the full training pipeline works.

### TEST-5: No integration tests (full game)
No end-to-end test running a full game from start to finish (create units → move → attack → capture HQ → verify winner).

### TEST-6: Missing unit test fixtures for 4 of 8 unit types
`conftest.py` has fixtures for W, M, C, A but not K, R, S, B. Knight charge, Rogue flank/evade, Sorcerer buffs, and Barbarian mechanics lack shared fixtures.

### TEST-7: No multi-player (>2) environment tests
`test_gym_env.py` covers 2-player flat-discrete well but has no 3+ player tests.

### TEST-8: No flat-discrete benchmark results
The proposed fix (flat_discrete mode) to the 0% win-rate problem has **never been empirically validated** in the repo.

---

## 9. Documentation Gaps

### DOC-1: Zero RL training documentation
The docs site has no training-related content despite RL being the project's core purpose. Missing: "Training Your First Agent", "Understanding the Environment", "Action Masking Explained", "Reward Engineering Guide".

### DOC-2: Implementation status page is stale
Lists "MCTS, minimax" as future work — MCTS is already fully implemented. Claims "feature-complete" despite 0% PPO win rate and Phase 2 at 0%.

### DOC-3: Flat-discrete action space not documented
Implementation status page only documents the MultiDiscrete action space despite flat-discrete being the recommended mode.

### DOC-4: Game mechanics docs don't mention Haste exception
States "attacking or moving ends a unit's turn" without noting the Haste ability contradiction.

### DOC-5: Benchmark "expected results" have no source
Claims win rates from "notebook reference" — no such notebook exists.

---

## 10. Training Pipeline Issues

### TRAIN-1: `train/train_eval.py` is completely broken
Imports from nonexistent modules. Should be deleted.

### TRAIN-2: Mixed self-play mode is non-functional
Bot environments created and discarded. Mixed mode behaves identically to pure self-play.

### TRAIN-3: No graceful interrupt handling
None of the training scripts handle `KeyboardInterrupt`. Long training runs lose progress if interrupted.

### TRAIN-4: AlphaZero training uses `sys.path` hack
Unnecessary `sys.path.insert` that the other training scripts don't use.

### TRAIN-5: Feudal RL hardcodes `max_steps=500`
Not exposed as CLI argument. Causes degenerate episodes on small maps.

### TRAIN-6: CI does not install the package properly
Uses `pip install -r requirements.txt` instead of `pip install -e ".[dev]"`.

---

## 11. Quick Wins (Low-effort, High-impact)

| # | Category | Description | Effort |
|---|----------|-------------|--------|
| 1 | Bug | Fix MCTS stale unit references (BUG-1) | ~1 hr |
| 2 | Bug | Fix end-turn mask filling entire slice (BUG-14) | ~10 min |
| 3 | Bug | Fix bot `find_best_move_position` using `is_destination=True` (BUG-7) | ~10 min |
| 4 | Bug | Fix AdvancedBot `'hasted'` → `'is_hasted'` (BUG-10) | ~5 min |
| 5 | Bug | Fix all bots' seize health check (BUG-8) | ~10 min |
| 6 | Bug | Delete broken `train/train_eval.py` (BUG-6) | ~2 min |
| 7 | Bug | Fix random map ocean→grass base (BUG-22) | ~5 min |
| 8 | Bug | Add Sorcerer abilities to LLM bot (BUG-12, BUG-13) | ~1 hr |
| 9 | RL | Default to `flat_discrete` action space | ~10 min |
| 10 | RL | Fix end-turn mask to single entry | ~10 min |
| 11 | Code | Extract `UNIT_TYPE_TO_IDX` constant | ~15 min |
| 12 | Code | Add `weights_only=True` to `torch.load` | ~5 min |
| 13 | Code | Fix Settings shallow copy | ~15 min |
| 14 | Train | Add `KeyboardInterrupt` handler to training scripts | ~15 min |
| 15 | CI | Change to `pip install -e ".[dev]"` | ~5 min |

---

## 12. Prioritized Recommendation Summary

### Tier 1 — Must Fix (Bugs affecting correctness of core systems)
1. **BUG-1:** MCTS stale references — AlphaZero tree search completely broken
2. **BUG-2:** MCTS PUCT sign flip guard — biases search toward opponent nodes
3. **BUG-3:** Self-play flipped observation wrong mask — opponent plays illegally
4. **BUG-5:** Winner determination for 3+ players — wrong winner declared
5. **BUG-7:** Bot pathfinding returns occupied positions — bots waste turns
6. **BUG-14:** End-turn mask fills entire slice — policy heavily biased to end turn early
7. **BUG-6:** `train/train_eval.py` broken imports — dead code confuses users

### Tier 2 — Should Fix (Significant impact on training and gameplay)
8. **BUG-9:** MediumBot ignores defence in kill calculation
9. **BUG-10+11:** AdvancedBot Sorcerer attribute name + return value bugs
10. **BUG-12+13:** LLM bots missing Sorcerer abilities + 4 unit types
11. **BUG-16:** Feudal RL hardcodes player 1
12. **RL-1:** Observation normalization
13. **RL-2:** Default flat_discrete action space
14. **RL-11:** Add buff/debuff to observations
15. **UI-6:** Async bot turns (unblock render loop)
16. **DOC-1:** Create RL training documentation

### Tier 3 — Nice to Have (Polish, depth, and quality of life)
17. **RL-5:** Curriculum learning integration
18. **RL-7:** Goal-type-specific feudal intrinsic rewards
19. **BAL-1–5:** Game balance tuning
20. **UI-1:** Visual status indicators for buffs/debuffs
21. **UI-3:** Action log panel
22. **TEST-1–4:** Feudal RL tests, MCTS tests, masking tests, training smoke tests
23. **ARCH-1:** Extract duplicated constants
24. **TRAIN-2:** Fix mixed self-play mode
