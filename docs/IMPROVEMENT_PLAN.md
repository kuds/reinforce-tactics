# Reinforce Tactics - Codebase Improvement Plan

**Created:** January 2026
**Status:** Review & Discussion

This document outlines recommended improvements for the Reinforce Tactics codebase, organized by priority and impact area.

---

## Executive Summary

After reviewing the codebase (~22,000+ lines of Python across 102 files), I've identified several key areas for improvement:

| Category | High Priority | Medium Priority | Low Priority |
|----------|---------------|-----------------|--------------|
| Code Consolidation | 3 items | 2 items | 2 items |
| Configuration Management | 1 item | 1 item | - |
| UI/UX Improvements | 1 item | 2 items | 1 item |
| Headless Mode | - | 2 items | 1 item |
| Code Quality | 1 item | 2 items | 2 items |

---

## 1. Tournament System Consolidation (HIGH PRIORITY)

### Current State
There are **three separate tournament implementations**:
1. `scripts/tournament.py` (1,112 lines) - Standalone CLI script
2. `docker/tournament/run_tournament.py` (1,595 lines) - Docker-based runner
3. `notebooks/bot_tournament.ipynb` & `llm_bot_tournament.ipynb` - Jupyter notebooks

### Problems
- **Significant code duplication**: EloRatingSystem class is duplicated in both Python files
- **Different feature sets**: Docker version has GCS upload, resume capability; script version has different bot discovery
- **Inconsistent interfaces**: Different configuration methods (JSON vs CLI args)
- **Maintenance burden**: Bug fixes must be applied in multiple places

### Recommended Solution

#### 1.1 Create a Unified Tournament Library
Create `reinforcetactics/tournament/` package with shared components:

```
reinforcetactics/tournament/
├── __init__.py
├── core.py           # TournamentRunner base class
├── elo.py            # EloRatingSystem (single source of truth)
├── bots.py           # BotDescriptor, TournamentBot, bot discovery
├── schedule.py       # Game scheduling, round-robin generation
├── results.py        # Results saving, CSV/JSON export
├── config.py         # Configuration parsing & validation
└── integrations/
    ├── gcs.py        # GCSUploader
    └── resume.py     # Resume functionality
```

**Benefits:**
- Single EloRatingSystem implementation
- Shared bot discovery and creation logic
- Consistent results format across all runners
- Easier testing and maintenance

#### 1.2 Refactor Existing Runners to Use Library
- `scripts/tournament.py` → thin CLI wrapper around library
- `docker/tournament/run_tournament.py` → thin Docker wrapper with GCS integration
- Notebooks → import library directly

**Estimated effort:** 2-3 days
**Impact:** High (reduces ~1,000 lines of duplicated code)

---

## 2. Configuration System Unification (HIGH PRIORITY)

### Current State
Configuration is scattered across multiple systems:
- `reinforcetactics/utils/settings.py` - Game settings (JSON file)
- `docker/tournament/config.json` - Tournament config (separate schema)
- `cli/commands.py` - Training args (argparse only)
- `reinforcetactics/rl/gym_env.py` - Reward config (hardcoded dict)

### Problems
- Settings not accessible to notebooks
- Training hyperparameters not persisted
- Tournament config separate from main settings
- Reward configuration buried in code

### Recommended Solution

#### 2.1 Extend Settings System
```python
# reinforcetactics/utils/settings.py
DEFAULT_SETTINGS = {
    # Existing settings...
    'training': {
        'default_algorithm': 'MaskablePPO',
        'default_timesteps': 100000,
        'reward_weights': {
            'win': 1000.0,
            'loss': -1000.0,
            'income_diff': 0.1,
            'unit_diff': 1.0,
            'structure_control': 5.0,
            'invalid_action': -10.0,
            'turn_penalty': -0.1
        }
    },
    'tournament': {
        'default_games_per_side': 2,
        'default_max_turns': 500,
        'map_pool_mode': 'cycle',
        'log_conversations': False
    }
}
```

#### 2.2 Add Configuration Layering
```
Base settings (settings.json)
    └── CLI overrides (argparse)
        └── Environment overrides (gym constructor)
            └── Tournament config (docker/config.json)
```

**Estimated effort:** 1-2 days
**Impact:** High (unified configuration, better reproducibility)

---

## 3. LLM Bot Code Consolidation (MEDIUM PRIORITY)

### Current State
`reinforcetactics/game/llm_bot.py` (1,611 lines) contains three similar bot classes:
- `OpenAIBot`
- `ClaudeBot`
- `GeminiBot`

### Problems
- Similar structure with provider-specific API calls
- Duplicated response parsing logic
- Action execution logic repeated

### Recommended Solution

#### 3.1 Extract Common Logic to Base Class
The abstract `LLMBot` base class already exists but can be extended:

```python
class LLMBot(ABC):
    """Base class already handles most shared logic."""

    # Add these shared methods:
    def _parse_actions_from_response(self, response_text: str) -> List[Dict]:
        """Unified action parsing logic."""
        pass

    def _validate_and_execute_actions(self, actions: List[Dict]) -> None:
        """Unified action validation and execution."""
        pass

    @abstractmethod
    def _call_api(self, messages: List[Dict]) -> str:
        """Provider-specific API call - subclasses implement this."""
        pass
```

#### 3.2 Consider Provider Pattern
```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages, **kwargs) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def complete(self, messages, **kwargs):
        # OpenAI-specific implementation
        pass

# Single LLMBot class uses provider
class LLMBot:
    def __init__(self, provider: LLMProvider, ...):
        self.provider = provider
```

**Estimated effort:** 1-2 days
**Impact:** Medium (reduces duplication, easier to add new providers)

---

## 4. Directory Structure Clarification (MEDIUM PRIORITY)

### Current State
Two `/game/` directories exist:
- `/home/user/reinforce-tactics/game/` - UI-facing code (game_loop.py, input_handler.py)
- `/home/user/reinforce-tactics/reinforcetactics/game/` - Core game logic (bot.py, llm_bot.py)

### Problems
- Confusing for new contributors
- Import statements are inconsistent
- `sys.path.insert()` anti-pattern in some files

### Recommended Solution

#### 4.1 Rename Top-Level `/game/` Directory
Rename to better reflect its purpose:
```
/game/ → /ui_session/
# OR
/game/ → Move contents into reinforcetactics/ui/session/
```

#### 4.2 Fix Path Manipulation Anti-Pattern
Replace:
```python
# Bad
sys.path.insert(0, str(Path(__file__).parent.parent))
```

With proper package installation:
```bash
pip install -e .  # Install package in editable mode
```

Update `setup.py` or `pyproject.toml` to include all subpackages.

**Estimated effort:** 0.5-1 day
**Impact:** Medium (clearer structure, proper imports)

---

## 5. Headless Mode Improvements (MEDIUM PRIORITY)

### Current State
Headless mode works well for training:
- `render_mode=None` skips pygame initialization
- GameState is completely independent of rendering
- Used effectively in `cli/commands.py` and training scripts

### Improvements

#### 5.1 Add Headless Tournament Mode
Currently tournaments require pygame even when not displaying:
```python
# Add explicit headless flag to tournament runners
class TournamentRunner:
    def __init__(self, ..., headless: bool = True):
        if not headless:
            pygame.init()  # Only if needed for visualization
```

#### 5.2 Add Progress Callbacks for Headless Training
```python
# In gym_env.py or self_play.py
class ProgressCallback:
    def on_episode_end(self, episode: int, reward: float):
        pass

    def on_training_update(self, timesteps: int, metrics: dict):
        pass
```

This enables:
- Remote monitoring
- Integration with experiment tracking (W&B, MLflow)
- Better notebook integration

#### 5.3 Environment Variable for Complete Headless
```python
# reinforcetactics/__init__.py
import os
if os.environ.get('REINFORCE_HEADLESS', '').lower() == 'true':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
```

**Estimated effort:** 1 day
**Impact:** Medium (better CI/CD, remote training support)

---

## 6. UI Code Organization (LOW PRIORITY)

### Current State
30 menu files across 5 subdirectories:
```
reinforcetactics/ui/menus/
├── base.py (394 lines)
├── main_menu.py
├── game_setup/ (3 files)
├── in_game/ (5 files)
├── map_editor/ (5 files)
├── save_load/ (4 files)
└── settings/ (5 files)
```

### Observations
- **Well organized**: Hierarchical structure is logical
- **Good base class**: `Menu` base class provides consistent behavior
- **Minor improvements possible**:

#### 6.1 Extract Common UI Components
Create `reinforcetactics/ui/components/`:
```
components/
├── button.py        # Reusable button component
├── text_input.py    # Text input field
├── list_view.py     # Scrollable list
└── dialog.py        # Modal dialog base
```

#### 6.2 Add UI Theme System
```python
# reinforcetactics/ui/theme.py
class Theme:
    BACKGROUND = (30, 30, 40)
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)
    BUTTON_NORMAL = (60, 60, 80)
    BUTTON_HOVER = (80, 80, 100)
    # etc.

# Allows easy theming/dark mode support
```

**Estimated effort:** 1-2 days
**Impact:** Low (nice to have, improves maintainability)

---

## 7. Code Quality Improvements (VARIOUS)

### 7.1 Replace Magic Strings with Enums (LOW PRIORITY)
```python
# Current
unit_type = 'W'  # Magic string

# Proposed
class UnitType(Enum):
    WARRIOR = 'W'
    MAGE = 'M'
    CLERIC = 'C'
    ARCHER = 'A'
    KNIGHT = 'K'
    RIFLEMAN = 'R'
    SNIPER = 'S'
    BOMBER = 'B'
```

### 7.2 Add Type Hints to Core Modules (MEDIUM PRIORITY)
Files lacking comprehensive type hints:
- `game_state.py` (partial)
- `bot.py` (partial)
- `mechanics.py` (partial)

### 7.3 Add Docstrings to Public APIs (LOW PRIORITY)
Generate API documentation with Sphinx or MkDocs.

### 7.4 Consider Splitting Large Files (MEDIUM PRIORITY)
- `bot.py` (1,622 lines) → Consider splitting into:
  - `bot_base.py` - Base classes and mixins
  - `simple_bot.py` - SimpleBot
  - `medium_bot.py` - MediumBot
  - `advanced_bot.py` - AdvancedBot

---

## 8. Testing Improvements (MEDIUM PRIORITY)

### Current State
- Good test coverage: 9,250 lines of tests
- Tests cover mechanics, gym_env, llm_bot, etc.

### Improvements

#### 8.1 Add Integration Tests for Tournament System
```python
# tests/test_tournament_integration.py
def test_full_tournament_run():
    """Test complete tournament with minimal bots."""
    pass

def test_tournament_resume():
    """Test tournament resume functionality."""
    pass
```

#### 8.2 Add Performance Benchmarks
```python
# tests/benchmarks/
def benchmark_game_simulation():
    """Measure games per second in headless mode."""
    pass

def benchmark_observation_generation():
    """Measure observation computation time."""
    pass
```

---

## Implementation Roadmap

### Phase 1: Core Consolidation (1-2 weeks)
1. Create tournament library package
2. Unify configuration system
3. Fix path manipulation issues

### Phase 2: Code Quality (1 week)
1. Extract LLM bot common logic
2. Add type hints to core modules
3. Split large files

### Phase 3: Enhancements (1 week)
1. Headless mode improvements
2. UI component extraction
3. Add integration tests

---

## Summary of Estimated Effort

| Improvement | Effort | Priority | Impact |
|-------------|--------|----------|--------|
| Tournament consolidation | 2-3 days | High | High |
| Configuration unification | 1-2 days | High | High |
| LLM bot consolidation | 1-2 days | Medium | Medium |
| Directory restructure | 0.5-1 day | Medium | Medium |
| Headless improvements | 1 day | Medium | Medium |
| UI organization | 1-2 days | Low | Low |
| Code quality (enums, types) | 2-3 days | Low-Medium | Medium |
| Testing improvements | 1-2 days | Medium | Medium |

**Total estimated effort:** 2-3 weeks for complete implementation

---

## Questions for Discussion

1. **Tournament consolidation**: Should we maintain backward compatibility with existing tournament configs, or is a clean break acceptable?

2. **Configuration system**: Should training hyperparameters be persisted to settings.json, or kept separate for reproducibility?

3. **Directory structure**: Preference between renaming `/game/` to `/ui_session/` vs. moving into `reinforcetactics/ui/`?

4. **LLM providers**: Interest in supporting additional LLM providers (e.g., Mistral, Llama via Ollama)?

5. **Priority order**: Which improvements would you like tackled first?
