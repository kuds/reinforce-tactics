# Codebase Review: Reinforce Tactics (advancedbot)

## Critical Bugs

### 1. `max_turns` is never enforced
- **File:** `reinforcetactics/core/game_state.py:95, 779-842`
- `max_turns` is stored and serialized but never checked in `end_turn()`. Games run indefinitely. Also not restored from save data in `from_dict()` (line 1259).

### 2. Broken cache interleaving
- **File:** `reinforcetactics/core/game_state.py:293-300, 868-1013`
- A single `_cache_valid` flag is shared between `get_unit_count()` and `get_legal_actions()`. When one sets it `True`, the other returns stale results.

### 3. Dead attacker state mutation
- **File:** `reinforcetactics/core/game_state.py:502-522`
- After attacker is killed and removed from `self.units`, code unconditionally sets `attacker.can_move = False`. Should be guarded with `if result['attacker_alive']`.

### 4. Missing `moving_unit` in legal actions BFS
- **File:** `reinforcetactics/core/game_state.py:906`
- `get_legal_actions()` omits `moving_unit=unit`, so the unit blocks itself during pathfinding.

### 5. Action mask misaligned with action space
- **File:** `reinforcetactics/rl/gym_env.py:181-185, 234-340`
- MultiDiscrete action space is combinatorial but mask is flat union-based. Agent routinely selects invalid combos and gets penalized.

### 6. Random opponent is a no-op
- **File:** `reinforcetactics/rl/gym_env.py:669-681`
- The `'random'` opponent fetches legal actions then does `pass`.

### 7. Opponent turn never called for 'random'/'self'
- **File:** `reinforcetactics/rl/gym_env.py:86, 611`
- `self.opponent` stays `None` for these modes, so `_opponent_turn()` is never invoked.

### 8. ClaudeBot crashes when max_tokens is None
- **File:** `reinforcetactics/game/llm_bot.py:1370-1380`
- Anthropic API requires `max_tokens`. When None, the parameter is omitted causing a validation error.

## High-Severity Design Issues

### 9. Non-potential-based reward shaping dominates terminal signal
- **File:** `reinforcetactics/rl/gym_env.py:691-708`
- State-based rewards applied every step. With structure_control=5.0 and a 2-structure lead over 500 steps, shaping yields 5,000 reward vs 1,000 win reward.

### 10. Hardcoded rewards bypass reward_config
- **File:** `reinforcetactics/rl/gym_env.py:534-661`
- 12+ hardcoded reward values not controlled by `reward_config`.

### 11. Bare exception catch silences bugs
- **File:** `reinforcetactics/rl/gym_env.py:663-665`
- All exceptions in `_execute_action` converted to "invalid action" penalty. Real bugs become invisible.

### 12. Unbounded recursion in bot act methods
- **File:** `reinforcetactics/game/bot.py` (20+ call sites)
- `act_with_unit` calls itself recursively on haste. If can_move never becomes False, stack overflow.

### 13. Mountain vision bonus is dead code
- **File:** `reinforcetactics/core/visibility.py:302-326`
- `calculate_vision_radius()` implements mountain +1 range but is never called.

### 14. Model bot action mask is all ones
- **File:** `reinforcetactics/game/model_bot.py:149-152`
- Returns `np.ones(...)`, giving the model no guidance on legal actions.

### 15. Self-play weight swap has no finally guard
- **File:** `reinforcetactics/rl/self_play.py:346-373`
- Two `load_state_dict` calls per opponent action with no exception safety.

### 16. swap_players broken in self-play
- **File:** `reinforcetactics/rl/self_play.py:453, 629`
- Opponent turn always assumes player 2 regardless of swap state.

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

1. **Fix RL action masking architecture** - Switch to flat action space with proper masking, fix dead opponents, consolidate rewards.
2. **Decompose GameState** - Extract legal actions, combat, abilities, serialization into focused classes.
3. **Fix caching and max_turns bugs** - Separate cache flags, implement turn limit.
4. **Consolidate bot hierarchy** - Create BotBase, replace recursion with bounded loops.
5. **Add UnitType enum, use TileType consistently** - Prevents typo/refactoring bugs.
