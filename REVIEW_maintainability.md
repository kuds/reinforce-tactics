# Maintainability & Code Quality Review

**Date:** 2026-03-06
**Scope:** Full repository — architecture, code quality, consolidation opportunities, and quality-of-life improvements.

---

## Executive Summary

Reinforce Tactics is a well-architected project with clean separation between core game logic, RL environments, UI, and the tournament system. Recent PRs have already addressed linting (ruff/mypy), some code consolidation, and a 50% coverage floor. This review identifies the **next tier** of improvements: systemic duplication, data integrity bugs, and structural patterns that will become increasingly painful as the codebase grows.

**Findings by severity:**

| Severity | Count | Summary |
|----------|-------|---------|
| Critical | 2 | Data corruption bug in Settings; broken action mask in ModelBot |
| High | 8 | Duplicated core logic (observations, action dispatch, masks); logic inconsistencies between RL environments |
| Medium | 12 | Magic numbers, missing enums, competing path systems, print vs logging |
| Low | 10 | Dead code, minor inconsistencies, style issues |

---

## Critical Issues

### 1. Settings shallow copy corrupts defaults at runtime

**File:** `reinforcetactics/utils/settings.py:54,57,180`

`DEFAULT_SETTINGS.copy()` is a shallow copy of a nested dict. Any runtime mutation to a nested value (e.g., `settings["graphics"]["sprites_path"] = "new"`) permanently corrupts the class-level `DEFAULT_SETTINGS` for all future instances and `reset_to_defaults()` calls.

**Fix:** Use `copy.deepcopy(self.DEFAULT_SETTINGS)` in all three locations.

### 2. ModelBot action mask is always all-zeros, then overridden to all-ones

**File:** `reinforcetactics/game/model_bot.py:184-245`

`_compute_action_mask` accesses legal actions with `.get("targets", [])` and `.get("positions", [])` (dict-style), but `get_legal_actions()` returns actions with direct key access (e.g., `action["target"].x`). The mask loops iterate over empty lists, producing all-zeros. The fallback at lines 244-245 then sets `mask[:] = 1.0`, making every action appear valid — completely defeating action masking.

**Fix:** Align the data access pattern with `get_legal_actions()` return format, matching `gym_env.py`'s mask-building code.

---

## High Priority Issues

### 3. Observation building duplicated 4 times with divergent player-perspective logic

**Files:**
- `reinforcetactics/rl/gym_env.py:219-277` (correct — respects `agent_player`)
- `reinforcetactics/rl/mcts.py:281-339` (bug — always puts player 1 gold first)
- `reinforcetactics/game/model_bot.py` (bug — same as MCTS)
- `reinforcetactics/rl/feudal_rl.py` (independent reimplementation)

**Recommendation:** Extract a shared `build_observation(game_state, perspective_player, grid)` function in a new `reinforcetactics/rl/observation.py` module. All four consumers should call this single implementation.

### 4. Action dispatch chain duplicated 4 times and already diverging

**Files:**
- `gym_env.py:510-662` (152 lines, 10-branch if/elif)
- `self_play.py:455-543` (88 lines, copy of the above)
- `mcts.py:250-278` (condensed version)
- `model_bot.py:255-304` (another variant)

All four implement the same "decode flat action → call game_state method" logic with slight variations. They are already diverging (e.g., paralyze allows M+S in gym_env but only M in self_play).

**Recommendation:** Extract a shared `execute_action(game_state, action_type, params)` dispatcher. Each consumer maps their input format to the shared dispatcher's interface.

### 5. Action mask building duplicated 3 times with inconsistent implementations

**Files:**
- `gym_env.py:295-385`
- `mcts.py:304-337`
- `model_bot.py:161-253` (broken, see Critical #2)

**Recommendation:** Extract a shared `build_action_mask(game_state, player, grid_dims)` function that all three consumers use.

### 6. Paralyze unit-type check inconsistency

- `gym_env.py:607` — allows Mage (`M`) **and** Sorcerer (`S`)
- `self_play.py:514` — allows only Mage (`M`)
- `model_bot.py:439` — allows only Mage (`M`)

One side has a logic bug. This should be defined once and referenced everywhere.

### 7. 8 near-identical "find units in range" methods in GameMechanics

**File:** `reinforcetactics/game/mechanics.py`

`get_adjacent_enemies`, `get_adjacent_allies`, `get_adjacent_paralyzed_allies`, `get_healable_allies`, `get_curable_allies`, `get_hasteable_allies`, `get_defence_buffable_allies`, `get_attack_buffable_allies` — all follow the identical pattern: iterate units, check player/health, compute Manhattan distance, apply a filter predicate.

**Recommendation:** Replace with a single `get_units_in_range(center, units, min_range, max_range, predicate)` function. This eliminates ~150 lines of duplication.

### 8. Counter-damage calculated twice in `attack_unit()`

**File:** `reinforcetactics/game/mechanics.py:412-428 vs 431-447`

The actual counter-damage application and the response-dict counter-damage are computed independently with the same formula. If a bug fix is applied to one block but not the other, they will silently diverge.

**Fix:** Compute counter-damage once, store in a variable, use for both application and response.

### 9. AlphaZero checkpoint missing architecture parameters

**File:** `reinforcetactics/rl/alphazero_trainer.py:487-491, 556-575`

`_save_checkpoint` omits `num_res_blocks` and `channels` from the config dict. `_evaluation_phase` creates an opponent network with default architecture parameters. If training uses non-default values, `load_state_dict` will fail with a shape mismatch.

**Fix:** Include `num_res_blocks` and `channels` in saved config; use them when loading.

### 10. `generate_random_map` base fill bug

**File:** `reinforcetactics/utils/file_io.py:301,347`

Line 301: `np.full((height, width), "o")` fills with ocean, but comment says "grass". Since tower placement (line 347) checks for grass tiles (`"p"`), towers are never placed on randomly generated maps.

**Fix:** Change `"o"` to `"p"` at line 301, or update the tower placement logic.

---

## Medium Priority Issues

### 11. No `UnitType` enum — raw single-character strings everywhere

Unit types (`"W"`, `"M"`, `"C"`, `"A"`, `"K"`, `"R"`, `"S"`, `"B"`) are scattered as raw string comparisons across 15+ files. `TileType` exists for tiles but there is no equivalent for units.

**Recommendation:** Create a `UnitType` enum in `constants.py` mirroring `TileType`. Reference it everywhere unit types are checked.

### 12. Inconsistent use of `TileType` enum

`constants.py` defines `TileType`. `game_state.py` uses it for some comparisons (`TileType.TOWER.value`). But `tile.py`, `mechanics.py`, and `grid.py` exclusively use raw strings (`"m"`, `"f"`, `"h"`). The codebase should pick one approach.

### 13. Duplicated tile-type encoding maps

- `grid.py:60` defines `tile_type_encoding` locally (missing ocean `"o"`)
- `game_state.py:1051` defines `unit_type_encoding` locally

These should be shared constants in `constants.py`, and ocean should be included.

### 14. Two competing path systems

`FileIO` uses hardcoded relative paths (`"saves"`, `"replays"`, `"maps/..."`) while `Settings` manages configurable paths. Neither references the other. Games saved via `FileIO` will ignore user-configured paths from `Settings`.

### 15. `print()` vs `logging` split

The tournament system properly uses `logging`. Everything else (`FileIO`, `Settings`, `Language`, `Tile`) uses `print()` with emoji. This prevents output control and breaks testability.

**Recommendation:** Adopt `logging` throughout. Replace all `print()` in library code with appropriate log levels.

### 16. Duplicated map-padding logic

`FileIO._pad_map` + `FileIO.add_water_border` and `ReplayPlayer._pad_map_for_replay` implement the same algorithm independently.

### 17. `UNIT_COLORS` dict is redundant

**File:** `reinforcetactics/constants.py:162-171`

Every entry in `UNIT_COLORS` duplicates `UNIT_DATA[key]["color"]`. This is dead data that can diverge silently.

### 18. Magic numbers across reward shaping and game rules

| Location | Value | Meaning |
|----------|-------|---------|
| `mechanics.py:236` | `0.9` | Max defence damage reduction cap |
| `game_state.py:717-724` | `1`, `2` | Tower/Building heal amounts |
| `feudal_rl.py:457` | `10` | Manager horizon |
| `feudal_rl.py:857-863` | `0.1`, `5.0`, `-10.0` | Intrinsic reward weights |
| `self_play.py:414-417` | `50`, `5` | Max actions, max consecutive invalid |
| `gym_env.py:86` | `20, 20` | Default map size |
| `llm_bot.py:632` | `150`, `100`, `50` | Building income values |

**Recommendation:** Define named constants in `constants.py` for gameplay values, and make RL hyperparameters constructor parameters with documented defaults.

### 19. LLM bot creates fresh API client on every call

**File:** `reinforcetactics/game/llm_bot.py:1248` and subclass equivalents

Each `_call_llm` invocation constructs a new API client, wasting connection setup time and preventing HTTP connection pooling.

**Fix:** Create the client once in `__init__` and reuse it.

### 20. Renderer allocates surfaces in hot loops

**File:** `reinforcetactics/ui/renderer.py:383-385, 674, 692-693, 710-711`

Fog overlays, movement overlays, target overlays, and attack range overlays all create new `pygame.Surface` per tile per frame. The `_get_overlay` caching pattern exists (line 531) but is not applied consistently.

### 21. Hardcoded random opponent player

**File:** `reinforcetactics/rl/gym_env.py:678`

`_random_opponent_turn` hardcodes `player=2`. If the agent is player 2, the random opponent should be player 1. Should use `3 - self.agent_player`.

### 22. `_merge_with_defaults` only handles one nesting level

**File:** `reinforcetactics/utils/settings.py:63-72`

If a loaded settings file has a sub-dict missing some keys, those defaults are lost after merge. Should use recursive merge.

---

## Low Priority / Quality of Life

### 23. `GameMechanics` is a static-only class
All methods are `@staticmethod`. A plain module of functions would be simpler. Similarly, `FileIO` is stateless — all `@staticmethod`.

### 24. `get_legal_actions()` is 127 lines
**File:** `game_state.py:885-1011`. Should be decomposed into per-action-type helper methods.

### 25. Visibility system does 3 full grid scans per update
**File:** `visibility.py:127-192`. Could use `grid.get_capturable_tiles(player)` instead.

### 26. `TILE_COLORS` has duplicate keys
**File:** `constants.py:46-66`. Both `TileType.GRASS.value` and `"p"` are used as keys with the same value.

### 27. Language system stores 770 lines of Python dict literals
**File:** `language.py:7-773`. Adding translations requires editing Python source. Should use external JSON/TOML files.

### 28. Incomplete translation coverage
Spanish and Chinese are missing `map_editor.*` keys. No warning is emitted.

### 29. CSV export doesn't escape values
**File:** `tournament/results.py:389-407`. Should use the `csv` module.

### 30. Dead code
- `file_io.py:556-584` — `export_replay_video` is a stub
- `masking.py:236-240` — unused imports suppressed with noqa
- `feudal_rl.py:838` — `goal_type` computed but never used
- `gym_env.py:47` — `ALL_UNIT_TYPES = ALL_UNIT_TYPES` redundant shadowing

### 31. Inconsistent bot API key testing
**File:** `tournament/bots.py:451-506`. OpenAI makes a real API call; Anthropic only instantiates client; Google calls list_models. Rigor varies by provider.

### 32. Menu base class `_populate_option_rects` / `_draw_content` duplication
**File:** `ui/menus/base.py:183-323`. Rect geometry is computed twice — once for hit-testing, once for rendering.

---

## Recommended Action Plan

### Phase 1 — Fix bugs and data corruption (1-2 days)
1. Fix `Settings` shallow copy → `deepcopy` (Critical #1)
2. Fix `ModelBot` action mask data access (Critical #2)
3. Fix `generate_random_map` ocean/grass bug (High #10)
4. Fix paralyze unit-type inconsistency (High #6)
5. Fix counter-damage double-calculation (High #8)
6. Fix random opponent hardcoded player (Medium #21)
7. Fix AlphaZero checkpoint architecture params (High #9)

### Phase 2 — Consolidate duplicated RL logic (3-5 days)
1. Extract shared `build_observation()` function
2. Extract shared `execute_action()` dispatcher
3. Extract shared `build_action_mask()` function
4. Create `UnitType` enum and use it everywhere
5. Unify tile-type encoding maps into `constants.py`

### Phase 3 — Code quality improvements (2-3 days)
1. Extract `get_units_in_range()` generic helper in mechanics
2. Adopt `logging` throughout (replace `print()`)
3. Unify path management (Settings ↔ FileIO)
4. Cache renderer overlay surfaces consistently
5. Decompose long methods (`get_legal_actions`, `_execute_action`, `_serialize_game_state`)

### Phase 4 — Quality of life (ongoing)
1. Externalize language translations to JSON
2. Add `UnitType` and `TileType` enum usage consistency
3. Remove dead code and stubs
4. Reuse LLM API clients across calls
5. Harden `_merge_with_defaults` to recursive merge
