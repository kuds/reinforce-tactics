# Fog of War Implementation Proposal

**Author:** Claude
**Date:** 2026-01-25
**Status:** Draft

## Executive Summary

This proposal outlines the design and implementation plan for adding a **Fog of War (FOW)** system to Reinforce Tactics. Currently, the game operates with complete information—both players see all units, structures, and resources at all times. Introducing FOW will add strategic depth, create more realistic gameplay scenarios, and provide a more challenging environment for reinforcement learning research.

---

## 1. Motivation

### 1.1 Current State

Reinforce Tactics currently exposes all game state information to both players:
- All unit positions, types, and health values
- All structure ownership and health
- All player gold amounts
- Complete terrain visibility

This "perfect information" model simplifies gameplay but limits strategic depth and doesn't reflect realistic tactical scenarios.

### 1.2 Benefits of Fog of War

| Benefit | Description |
|---------|-------------|
| **Strategic Depth** | Players must scout, predict, and react to incomplete information |
| **Realistic Tactics** | Ambushes, flanking maneuvers, and surprise attacks become viable strategies |
| **RL Research Value** | Partial observability (POMDP) presents a more challenging learning problem |
| **Gameplay Variety** | Different unit compositions become valuable (scouts vs brute force) |
| **Replayability** | Each game feels different due to information uncertainty |

---

## 2. Design Overview

### 2.1 Core Concept

Each player maintains their own **visibility map** that determines what they can see. Areas outside visibility are either:
- **Unexplored**: Never seen (terrain unknown)
- **Shrouded**: Previously explored but not currently visible (terrain known, units/ownership hidden)
- **Visible**: Currently within sight range (full information)

### 2.2 Visibility Sources

| Source | Base Range | Notes |
|--------|------------|-------|
| **Units** | 3 tiles | All units provide visibility |
| **Headquarters** | 4 tiles | Starting base provides larger vision |
| **Buildings** | 3 tiles | Owned secondary structures |
| **Towers** | 5 tiles | Elevated position provides best vision |

### 2.3 Visibility Model Options

#### Option A: Simple Radius (Recommended for v1)
- Circular visibility based on Chebyshev distance (king's movement)
- No terrain blocking
- Easiest to implement and understand

#### Option B: True Line-of-Sight
- Ray-casting from source to target
- Mountains block vision, forests partially obscure
- More realistic but computationally expensive

#### Option C: Hybrid
- Radius-based with terrain modifiers
- Mountains provide +1 vision when occupied
- Forests reduce incoming vision by 1 tile

**Recommendation:** Start with **Option A** for initial implementation, with architecture supporting future Option C upgrade.

---

## 3. Technical Design

### 3.1 Data Structures

#### 3.1.1 Visibility Map

```python
# New class in core/visibility.py
class VisibilityMap:
    """Tracks visibility state for a single player."""

    def __init__(self, width: int, height: int):
        # 0 = unexplored, 1 = shrouded, 2 = visible
        self.state = np.zeros((height, width), dtype=np.uint8)
        self.last_seen_units = {}  # (x, y) -> UnitSnapshot
        self.last_seen_structures = {}  # (x, y) -> StructureSnapshot

    def update(self, game_state: GameState, player: int) -> None:
        """Recalculate visibility based on current unit/structure positions."""
        pass

    def is_visible(self, x: int, y: int) -> bool:
        return self.state[y, x] == 2

    def is_explored(self, x: int, y: int) -> bool:
        return self.state[y, x] >= 1
```

#### 3.1.2 Unit Snapshot (for memory of last-seen enemies)

```python
@dataclass
class UnitSnapshot:
    """Snapshot of unit information when last visible."""
    unit_type: str
    owner: int
    health: int
    position: Tuple[int, int]
    turn_seen: int
```

### 3.2 GameState Modifications

#### 3.2.1 New Attributes

```python
class GameState:
    def __init__(self, ..., fog_of_war: bool = False):
        self.fog_of_war = fog_of_war
        if fog_of_war:
            self.visibility_maps = {
                1: VisibilityMap(self.grid.width, self.grid.height),
                2: VisibilityMap(self.grid.width, self.grid.height),
            }
```

#### 3.2.2 New Methods

```python
class GameState:
    def get_visible_units(self, player: int) -> List[Unit]:
        """Return units visible to the specified player."""
        if not self.fog_of_war:
            return self.units

        visible = []
        vis_map = self.visibility_maps[player]
        for unit in self.units:
            if unit.owner == player or vis_map.is_visible(unit.x, unit.y):
                visible.append(unit)
        return visible

    def get_visible_tiles(self, player: int) -> List[Tuple[int, int, Tile]]:
        """Return tiles with full information for the specified player."""
        pass

    def get_obs_for_player(self, player: int) -> Dict:
        """Generate observation dictionary with FOW applied."""
        pass
```

### 3.3 Visibility Calculation

```python
# In core/visibility.py

VISIBILITY_RANGES = {
    'unit_default': 3,
    'headquarters': 4,
    'building': 3,
    'tower': 5,
}

def calculate_visibility(game_state: GameState, player: int) -> np.ndarray:
    """Calculate visibility mask for a player.

    Returns:
        2D numpy array where True = visible
    """
    height, width = game_state.grid.height, game_state.grid.width
    visible = np.zeros((height, width), dtype=bool)

    # Vision from units
    for unit in game_state.units:
        if unit.owner == player:
            _add_vision_radius(visible, unit.x, unit.y,
                             VISIBILITY_RANGES['unit_default'])

    # Vision from structures
    for y in range(height):
        for x in range(width):
            tile = game_state.grid.get_tile(x, y)
            if tile.owner == player and tile.type in ('h', 'b', 't'):
                range_key = {'h': 'headquarters', 'b': 'building', 't': 'tower'}[tile.type]
                _add_vision_radius(visible, x, y, VISIBILITY_RANGES[range_key])

    return visible

def _add_vision_radius(visible: np.ndarray, cx: int, cy: int, radius: int):
    """Add circular vision around a point."""
    height, width = visible.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                if max(abs(dx), abs(dy)) <= radius:  # Chebyshev distance
                    visible[ny, nx] = True
```

### 3.4 Observation Space Modifications

#### 3.4.1 Current Observation (for reference)

```python
# gym_env.py current implementation
obs = {
    'grid': (H, W, 3),        # terrain, owner, structure_hp
    'units': (H, W, 3),       # unit_type, owner, health
    'global_features': (6,),  # gold_p1, gold_p2, turn, etc.
    'action_mask': (H*W*10,)  # valid actions
}
```

#### 3.4.2 FOW Observation

```python
# Modified for fog of war
obs = {
    'grid': (H, W, 4),        # terrain, owner, structure_hp, visibility_state
    'units': (H, W, 3),       # unit_type, owner, health (hidden units = 0)
    'memory': (H, W, 4),      # last_seen: unit_type, owner, health, turns_ago
    'global_features': (5,),  # own_gold, turn, player_id (enemy gold hidden)
    'action_mask': (H*W*10,)  # valid actions (only for visible targets)
}
```

### 3.5 Action Masking Changes

With FOW enabled, players cannot:
- Attack units they cannot see
- Target abilities (paralyze, heal) on non-visible tiles
- Seize structures they haven't discovered

```python
def get_legal_actions(self, player: int = None) -> List[Dict]:
    """Get legal actions, filtered by visibility if FOW enabled."""
    actions = self._get_all_possible_actions(player)

    if self.fog_of_war and player is not None:
        vis_map = self.visibility_maps[player]
        actions = [a for a in actions if self._is_action_valid_with_fow(a, vis_map)]

    return actions

def _is_action_valid_with_fow(self, action: Dict, vis_map: VisibilityMap) -> bool:
    """Check if action target is visible."""
    if action['type'] in ('attack', 'paralyze', 'heal'):
        return vis_map.is_visible(action['to_x'], action['to_y'])
    if action['type'] == 'seize':
        return vis_map.is_explored(action['to_x'], action['to_y'])
    return True  # Movement and creation always allowed
```

---

## 4. Implementation Plan

### Phase 1: Core Visibility System (Foundation)

**Files to create:**
- `reinforcetactics/core/visibility.py` - VisibilityMap class and calculations

**Files to modify:**
- `reinforcetactics/core/game_state.py` - Add FOW flag and visibility integration

**Deliverables:**
- VisibilityMap class with update/query methods
- Basic visibility calculation (radius-based)
- GameState integration with optional FOW flag

### Phase 2: Observation & Action Filtering

**Files to modify:**
- `reinforcetactics/rl/gym_env.py` - Update observation generation
- `reinforcetactics/rl/masking.py` - FOW-aware action masking

**Deliverables:**
- Modified observation space with visibility layer
- Memory system for last-seen units
- Action filtering based on visibility

### Phase 3: AI & Bot Updates

**Files to modify:**
- `reinforcetactics/game/bot.py` - Update rule-based bots for FOW
- `reinforcetactics/game/llm_bot.py` - Update LLM state representation

**Deliverables:**
- Bots operate correctly with partial information
- Exploration behavior for AI agents
- LLM prompts include visibility context

### Phase 4: UI & Replay System

**Files to modify:**
- `reinforcetactics/ui/game_renderer.py` - Render FOW overlay
- `reinforcetactics/utils/replay.py` - Support FOW in replays

**Deliverables:**
- Visual fog rendering (darkened/hidden areas)
- Replay viewer with FOW toggle (player POV vs omniscient)
- Minimap with exploration state

### Phase 5: Testing & Balancing

**New files:**
- `tests/test_visibility.py` - Visibility calculation tests
- `tests/test_fow_integration.py` - Full game tests with FOW

**Deliverables:**
- Comprehensive test coverage
- Balance adjustments to visibility ranges
- Performance benchmarks

---

## 5. Configuration Options

### 5.1 Game Settings

```python
# New settings in utils/settings.py
FOW_SETTINGS = {
    'enabled': False,  # Master toggle
    'mode': 'simple',  # 'simple', 'los', 'hybrid'
    'remember_terrain': True,  # Keep terrain after exploring
    'remember_structures': True,  # Remember structure locations
    'remember_units': False,  # Show last-known unit positions
    'hide_enemy_gold': True,  # Hide opponent's gold amount
    'unit_vision': {
        'warrior': 3,
        'mage': 3,
        'cleric': 3,
        'archer': 4,  # Scouts further
        'knight': 3,
        'rogue': 4,   # Good vision
        'sorcerer': 3,
        'barbarian': 2,  # Limited vision
    },
    'structure_vision': {
        'headquarters': 4,
        'building': 3,
        'tower': 5,
    },
}
```

### 5.2 Environment Options

```python
# gym_env.py initialization
env = ReinforceTacticsEnv(
    map_name="standard",
    fog_of_war=True,
    fow_settings=custom_settings,  # Optional overrides
)
```

---

## 6. Special Considerations

### 6.1 Unit-Specific Behaviors

| Unit | FOW Interaction |
|------|-----------------|
| **Archer** | Extended vision range (4 tiles) - natural scout |
| **Rogue** | Extended vision (4 tiles) + can see into forests |
| **Mage** | Paralyze requires vision of target |
| **Cleric** | Heal requires vision of ally (prevents healing unseen units) |
| **Knight** | Charge damage bonus calculated on visible path only |

### 6.2 Structure Interactions

- **Capturing**: Must have explored the tile (know structure exists)
- **Tower Vision**: Provides largest vision radius as strategic asset
- **Neutral Structures**: Visible once in range, ownership updates when visible

### 6.3 Edge Cases

1. **Unit dies out of vision**: Remove from memory after N turns
2. **Structure changes hands out of vision**: Update when re-observed
3. **Unit moves into vision mid-turn**: Becomes visible immediately
4. **Simultaneous discovery**: Both players see each other at same time

---

## 7. RL Training Considerations

### 7.1 POMDP Complexity

FOW transforms the environment from an MDP to a POMDP (Partially Observable MDP), requiring:
- Recurrent networks (LSTM/GRU) to maintain belief state
- Attention mechanisms for memory integration
- Potentially larger networks to handle uncertainty

### 7.2 Curriculum Learning Approach

```
Stage 1: No FOW (baseline training)
    ↓
Stage 2: FOW with full memory (all seen info remembered)
    ↓
Stage 3: FOW with decaying memory (info fades over time)
    ↓
Stage 4: Full FOW (realistic partial observability)
```

### 7.3 Reward Shaping

Consider additional reward signals for FOW games:
- Small reward for exploring new tiles
- Reward for discovering enemy positions
- Penalty for losing units in unexplored territory

---

## 8. Backward Compatibility

### 8.1 Default Behavior

- FOW is **disabled by default** to maintain backward compatibility
- Existing saved games, replays, and trained models work unchanged
- All APIs remain functional with `fog_of_war=False`

### 8.2 API Compatibility

```python
# Existing code continues to work
game = GameState(map_data)  # No FOW
game.get_legal_actions()    # Returns all actions

# New FOW mode is opt-in
game = GameState(map_data, fog_of_war=True)
game.get_legal_actions(player=1)  # Returns visible actions only
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# test_visibility.py
def test_unit_vision_radius():
    """Units should see within their vision range."""

def test_tower_extended_vision():
    """Towers should provide 5-tile vision radius."""

def test_visibility_update_on_move():
    """Visibility should update when units move."""

def test_hidden_enemy_not_in_obs():
    """Hidden enemies should not appear in observation."""
```

### 9.2 Integration Tests

```python
# test_fow_integration.py
def test_cannot_attack_hidden_unit():
    """Attack action should be invalid for non-visible targets."""

def test_gym_env_fow_observation():
    """Gym environment should return FOW-filtered observations."""

def test_game_playable_with_fow():
    """Complete game should be playable with FOW enabled."""
```

### 9.3 Performance Tests

- Visibility calculation should complete in < 1ms for standard maps
- Memory overhead should be < 10% increase per game state
- No impact on non-FOW games

---

## 10. Future Enhancements

### 10.1 Advanced Visibility

- **Line-of-sight**: True raycasting with terrain blocking
- **Stealth units**: Rogues invisible until adjacent
- **Detection abilities**: Reveal hidden areas temporarily

### 10.2 Information Warfare

- **Decoys**: Fake unit projections
- **Intel gathering**: Ability to see enemy gold/production
- **Communication**: Signal allies about enemy positions

### 10.3 Map Features

- **Watchtowers**: Neutral structures providing vision when captured
- **Fog generators**: Tiles that block all vision
- **Reveal zones**: Areas where all units are visible

---

## 11. Open Questions

1. **Memory persistence**: How long should last-seen unit info persist?
2. **Partial visibility**: Should units at edge of vision show reduced info (type but not health)?
3. **Replay default**: Should replays default to omniscient view or player POV?
4. **Tournament mode**: Should competitive tournaments use FOW?
5. **Performance**: Is per-unit vision calculation acceptable, or should we optimize?

---

## 12. Conclusion

Implementing fog of war will significantly enhance Reinforce Tactics by:
- Adding strategic depth through information asymmetry
- Creating a more challenging RL environment (POMDP)
- Enabling new tactical possibilities (ambushes, scouting, deception)
- Aligning gameplay with realistic tactical scenarios

The phased implementation approach minimizes risk while maintaining backward compatibility. Starting with simple radius-based visibility provides immediate value while the architecture supports future enhancements.

---

## Appendix A: File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `core/visibility.py` | **New** | VisibilityMap class and calculations |
| `core/game_state.py` | Modify | Add FOW flag, visibility integration |
| `rl/gym_env.py` | Modify | FOW observation space |
| `rl/masking.py` | Modify | FOW action filtering |
| `game/bot.py` | Modify | Partial information handling |
| `game/llm_bot.py` | Modify | FOW state representation |
| `ui/game_renderer.py` | Modify | FOW visual rendering |
| `utils/replay.py` | Modify | FOW replay support |
| `utils/settings.py` | Modify | FOW configuration |
| `tests/test_visibility.py` | **New** | Visibility tests |
| `tests/test_fow_integration.py` | **New** | Integration tests |

## Appendix B: Estimated Complexity

| Phase | Complexity | Notes |
|-------|------------|-------|
| Phase 1 | Medium | Core system, most architectural decisions |
| Phase 2 | Medium | Observation space changes, action filtering |
| Phase 3 | Low-Medium | Bot updates are incremental |
| Phase 4 | Medium | UI rendering requires careful design |
| Phase 5 | Low | Testing and refinement |
