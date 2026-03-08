# Codebase Review: Reinforce Tactics (advancedbot)

## Critical Bugs

### 1. `max_turns` is never enforced ✅ FIXED
- **File:** `reinforcetactics/core/game_state.py:95, 779-842`
- `max_turns` is stored and serialized but never checked in `end_turn()`. Games run indefinitely. Also not restored from save data in `from_dict()` (line 1259).
- **Fix:** `max_turns` enforcement added in `end_turn()`.

### 2. Broken cache interleaving ✅ FIXED
- **File:** `reinforcetactics/core/game_state.py:293-300, 868-1013`
- A single `_cache_valid` flag is shared between `get_unit_count()` and `get_legal_actions()`. When one sets it `True`, the other returns stale results.
- **Fix:** Separate cache flags for `unit_count` and `legal_actions`.

### 3. Dead attacker state mutation ✅ FIXED
- **File:** `reinforcetactics/core/game_state.py:502-522`
- After attacker is killed and removed from `self.units`, code unconditionally sets `attacker.can_move = False`. Should be guarded with `if result['attacker_alive']`.
- **Fix:** `can_move`/`can_attack` set only when `result['attacker_alive']` is True.

### 4. Missing `moving_unit` in legal actions BFS ✅ FIXED
- **File:** `reinforcetactics/core/game_state.py:906`
- `get_legal_actions()` omits `moving_unit=unit`, so the unit blocks itself during pathfinding.
- **Fix:** `moving_unit=unit` now passed to `get_legal_actions()`.

### 5. Action mask misaligned with action space
- **File:** `reinforcetactics/rl/gym_env.py:181-185, 234-340`
- MultiDiscrete action space is combinatorial but mask is flat union-based. Agent routinely selects invalid combos and gets penalized.
- **Note:** This is an acknowledged limitation of per-dimension masking with MultiDiscrete spaces. A proper fix requires the auto-regressive action head planned for Phase 3.2.

### 6. Random opponent is a no-op ✅ FIXED
- **File:** `reinforcetactics/rl/gym_env.py:669-681`
- The `'random'` opponent fetches legal actions then does `pass`.
- **Fix:** `_random_opponent_turn()` now fully implemented with random action execution.

### 7. Opponent turn never called for 'random'/'self' ✅ FIXED
- **File:** `reinforcetactics/rl/gym_env.py:86, 611`
- `self.opponent` stays `None` for these modes, so `_opponent_turn()` is never invoked.
- **Fix:** `_opponent_turn()` dispatches on `opponent_type` string, not `opponent` object.

### 8. ClaudeBot crashes when max_tokens is None ✅ FIXED
- **File:** `reinforcetactics/game/llm_bot.py:1370-1380`
- Anthropic API requires `max_tokens`. When None, the parameter is omitted causing a validation error.
- **Fix:** `max_tokens` defaults to 4096 when None to satisfy the Anthropic API requirement.

## High-Severity Design Issues

### 9. Non-potential-based reward shaping dominates terminal signal ✅ FIXED
- **File:** `reinforcetactics/rl/gym_env.py:691-708`
- State-based rewards applied every step. With structure_control=5.0 and a 2-structure lead over 500 steps, shaping yields 5,000 reward vs 1,000 win reward.
- **Fix:** Reward shaping converted to potential-based (Phi(s') - Phi(s)) so only changes in advantage are rewarded, not maintaining a lead.

### 10. Hardcoded rewards bypass reward_config ✅ FIXED
- **File:** `reinforcetactics/rl/gym_env.py:534-661`
- 12+ hardcoded reward values not controlled by `reward_config`.
- **Fix:** All reward values now read from `reward_config` dict with sensible defaults.

### 11. Bare exception catch silences bugs ✅ FIXED
- **File:** `reinforcetactics/rl/gym_env.py:663-665`
- All exceptions in `_execute_action` converted to "invalid action" penalty. Real bugs become invisible.
- **Fix:** `TypeError` and `AttributeError` now re-raised as programming errors. Game-logic exceptions still caught gracefully.

### 12. Unbounded recursion in bot act methods ✅ FIXED
- **File:** `reinforcetactics/game/bot.py` (20+ call sites)
- `act_with_unit` calls itself recursively on haste. If can_move never becomes False, stack overflow.
- **Fix:** `MAX_RECURSION_DEPTH=10` guard added to prevent haste stack overflow.

### 13. Mountain vision bonus is dead code ✅ FIXED
- **File:** `reinforcetactics/core/visibility.py:302-326`
- `calculate_vision_radius()` implements mountain +1 range but is never called.
- **Fix:** `PlayerVisibility.update()` now uses `calculate_vision_radius()` for both units and structures, enabling terrain-based vision bonuses.

### 14. Model bot action mask is all ones ✅ FIXED
- **File:** `reinforcetactics/game/model_bot.py:149-152`
- Returns `np.ones(...)`, giving the model no guidance on legal actions.
- **Fix:** Real masks computed from legal actions.

### 15. Self-play weight swap has no finally guard ✅ FIXED
- **File:** `reinforcetactics/rl/self_play.py:346-373`
- Two `load_state_dict` calls per opponent action with no exception safety.
- **Fix:** `try/finally` guards ensure original params are always restored.

### 16. swap_players broken in self-play ✅ FIXED
- **File:** `reinforcetactics/rl/self_play.py:453, 629`
- Opponent turn always assumes player 2 regardless of swap state.
- **Fix:** `agent_player` attribute added to `StrategyGameEnv`. `_execute_action`, `_get_obs`, `_compute_potential`, and `step()` all use `agent_player` instead of hardcoded player 1. `SelfPlayEnv` propagates `agent_player` to base env on reset.

### 17. Feudal RL _last_obs is None on first call
- **File:** `reinforcetactics/rl/feudal_rl.py:596`

### 18. Tournament race condition
- **File:** `reinforcetactics/tournament/runner.py:251`
- `completed_count` incremented outside lock on error path.

## Architectural Concerns

### 19. GameState is a God Object (~50KB)
- Handles state, CRUD, combat, abilities, turns, income, legal actions, fog-of-war, coordinates, recording, serialization, player config.

### 20. Massive duplication in bot.py
- `manhattan_distance`, `find_best_move_position`, `__init__` all duplicated across SimpleBot/MediumBot.
- Haste check pattern repeated 17 times.

### 21. Ability methods are boilerplate copies
- **File:** `reinforcetactics/core/game_state.py:526-633`
- Six 7-line methods with identical structure.

### 22-23. No UnitType enum; TileType enum inconsistently used
- Unit types are raw strings throughout. TileType exists but most code uses `'m'`, `'f'`, `'h'` literals.

## Performance Concerns

### 24. O(n) unit lookup by position
- **File:** `reinforcetactics/core/game_state.py:302-307`
- Linear scan inside loops. Should use a position index dict.

### 25. Python loops for numpy masking
- **File:** `reinforcetactics/core/game_state.py:1077-1085`
- Should be vectorized numpy operations.

### 26-27. Repeated BFS and grid scans per turn
- `get_reachable_positions` and `find_our_hq` called redundantly.

## Security Concerns

### 28. No path sanitization in file I/O
- `save_game`, `save_replay`, `save_map` accept arbitrary paths with `mkdir(parents=True)`.

### 29. API keys in world-readable plaintext
- **File:** `reinforcetactics/utils/settings.py`

### 30. Shallow copy corrupts DEFAULT_SETTINGS
- **File:** `reinforcetactics/utils/settings.py:74-85`

## Top 5 Recommendations

1. **Fix RL action masking architecture** - Switch to auto-regressive action head (Phase 3.2) for proper per-sub-action masking.
2. **Decompose GameState** - Extract legal actions, combat, abilities, serialization into focused classes.
3. ~~**Fix caching and max_turns bugs**~~ ✅ Done — Separate cache flags, turn limit enforced.
4. ~~**Consolidate bot hierarchy**~~ ✅ Partial — Recursion guard added; full BotBase refactor remains.
5. **Add UnitType enum, use TileType consistently** - Prevents typo/refactoring bugs.

## Resolution Summary

| Category | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| Critical Bugs (#1-8) | 8 | 7 | 1 (#5 — architectural, planned for Phase 3.2) |
| High-Severity (#9-18) | 10 | 8 | 2 (#17, #18) |
| Architectural (#19-23) | 5 | 0 | 5 |
| Performance (#24-27) | 4 | 0 | 4 |
| Security (#28-30) | 3 | 0 | 3 |
