# Test Plan for Reinforce Tactics

This document outlines the testing strategy, current coverage status, and plans for improvement.

## Table of Contents

1. [Test Infrastructure](#test-infrastructure)
2. [Test Categories & Markers](#test-categories--markers)
3. [Current Coverage Status](#current-coverage-status)
4. [Long-Running Test Handling](#long-running-test-handling)
5. [Coverage Improvement Plan](#coverage-improvement-plan)
6. [Test Execution Guide](#test-execution-guide)

---

## Test Infrastructure

### Framework & Tools

- **Test Framework**: pytest
- **Python Versions**: 3.10, 3.11, 3.12
- **CI/CD**: GitHub Actions
- **Code Quality**: flake8 (linting), pylint (code quality >= 9.75)

### Test Directory Structure

```
tests/
├── conftest.py           # Shared fixtures and pytest configuration
├── TEST_PLAN.md          # This document
├── pytest.ini            # Pytest configuration (markers, timeout settings)
│
├── # Core Tests
├── test_unit.py          # Unit class tests
├── test_tile.py          # Tile class tests
├── test_game_state.py    # GameState tests
│
├── # Game Logic Tests
├── test_mechanics.py     # Combat, movement, abilities
├── test_medium_bot.py    # MediumBot AI
├── test_advanced_bot.py  # AdvancedBot AI
├── test_llm_bot.py       # LLM bot integration
├── test_tournament.py    # Tournament system & ModelBot
│
├── # RL Tests
├── test_gym_env.py       # Gymnasium environment
│
├── # UI Tests
├── test_menus.py         # Menu system
├── test_input_handler.py # Input handling
├── test_language_menu.py # Localization
├── test_fonts.py         # Font rendering
├── test_map_preview.py   # Map preview component
├── test_map_editor.py    # Map editor
│
├── # Utility Tests
├── test_save_replay.py   # Save/load, replay system
└── verify_mask.py        # Action masking verification
```

---

## Test Categories & Markers

### Pytest Markers

Tests are categorized using pytest markers to allow selective execution:

| Marker | Description | Typical Duration |
|--------|-------------|------------------|
| `@pytest.mark.unit` | Fast unit tests | < 100ms |
| `@pytest.mark.integration` | Integration tests | 100ms - 2s |
| `@pytest.mark.slow` | Long-running tests | > 2s |
| `@pytest.mark.gpu` | Tests requiring GPU | Variable |
| `@pytest.mark.external` | Tests requiring external services (API keys) | Variable |
| `@pytest.mark.ui` | Tests requiring pygame display | < 500ms |

### Running Tests by Category

```bash
# Run only fast unit tests (default for quick feedback)
pytest -m "unit"

# Run all tests except slow ones
pytest -m "not slow"

# Run integration tests
pytest -m "integration"

# Run slow tests only (for CI nightly builds)
pytest -m "slow"

# Run all tests (full suite)
pytest

# Skip external API tests (no API keys)
pytest -m "not external"
```

---

## Current Coverage Status

### Fully Tested Modules

| Module | Test File | Status | Notes |
|--------|-----------|--------|-------|
| `core/unit.py` | `test_unit.py` | ✅ Complete | All unit types, serialization |
| `core/tile.py` | `test_tile.py` | ✅ Complete | All tile types, ownership |
| `core/game_state.py` | `test_game_state.py` | ✅ Complete | Win conditions, actions |
| `game/mechanics.py` | `test_mechanics.py` | ✅ Complete | Combat, movement, abilities |
| `game/bot.py` | `test_medium_bot.py`, `test_advanced_bot.py` | ✅ Complete | All bot difficulties |
| `game/llm_bot.py` | `test_llm_bot.py` | ✅ Complete | All LLM providers |
| `rl/gym_env.py` | `test_gym_env.py` | ✅ Complete | Spaces, steps, rewards |
| `ui/menus/` | `test_menus.py` | ✅ Complete | All menu classes |
| `utils/replay_player.py` | `test_save_replay.py` | ✅ Complete | Record, playback, export |

### Partially Tested Modules

| Module | Test File | Coverage | Missing |
|--------|-----------|----------|---------|
| `game/model_bot.py` | `test_tournament.py` | ~60% | Full model loading, action selection |
| `utils/file_io.py` | Various | ~50% | Map validation, error handling |

### Untested Modules (Coverage Gaps)

| Module | Size | Priority | Reason |
|--------|------|----------|--------|
| `game/llm_prompts.py` | 15KB | Medium | Prompt templates (mostly config) |
| `rl/feudal_rl.py` | 14KB | High | Complex RL components |
| `utils/experiment_tracker.py` | 5KB | Low | External service integration |
| `utils/settings.py` | 5.5KB | Medium | Settings persistence |
| `ui/renderer.py` | - | Low | Requires display mocking |
| `ui/menus/credits_menu.py` | - | Low | Simple static content |

---

## Long-Running Test Handling

### Identification of Slow Tests

Tests are considered "slow" if they:
- Take more than 2 seconds to complete
- Perform multiple game simulations
- Train RL models (even briefly)
- Make external API calls (LLM tests)
- Involve video/replay rendering

### Slow Tests List

The following test classes/functions should be marked as slow:

1. **RL Environment Tests** (`test_gym_env.py`)
   - `TestFullGameSimulation` - Full game episodes
   - `TestEpisodeStatistics` - Multiple episode runs
   - `TestRewardCalculation` (multi-step scenarios)

2. **Bot Tests**
   - `TestAdvancedBotMapAnalysis` - Large map analysis
   - `TestModelBotWithMockModel` - Model inference simulation

3. **Tournament Tests** (`test_tournament.py`)
   - `TestTournamentRound` - Full match simulation
   - `TestBotRankings` - Multiple games

4. **Replay Tests** (`test_save_replay.py`)
   - `TestReplayVideoExport` - Video encoding
   - `TestReplayPadding` - Multi-frame processing

5. **LLM Bot Tests** (`test_llm_bot.py`)
   - Any test that actually calls APIs (when API keys present)

### Timeout Configuration

```ini
# pytest.ini
[pytest]
timeout = 30           # Default timeout per test (30 seconds)
timeout_method = signal
```

Individual slow tests can override:
```python
@pytest.mark.timeout(120)  # 2 minute timeout for this specific test
def test_full_game_simulation():
    ...
```

---

## Coverage Improvement Plan

### Phase 1: High Priority (Add Missing Test Coverage)

#### 1. `rl/feudal_rl.py` Tests
```python
# tests/test_feudal_rl.py
class TestFeudalRL:
    """Tests for the Feudal RL hierarchical action space."""

    def test_hierarchical_action_space_creation(self):
        """Test hierarchical action space is properly defined."""
        pass

    def test_high_level_policy(self):
        """Test high-level strategic decisions."""
        pass

    def test_low_level_policy(self):
        """Test low-level tactical execution."""
        pass

    def test_goal_embedding(self):
        """Test goal representation between levels."""
        pass
```

#### 2. `utils/settings.py` Tests
```python
# tests/test_settings.py
class TestSettings:
    """Tests for settings persistence."""

    def test_load_default_settings(self):
        """Test loading settings with defaults."""
        pass

    def test_save_settings(self):
        """Test saving settings to file."""
        pass

    def test_api_key_storage(self):
        """Test secure API key storage."""
        pass

    def test_invalid_settings_file(self):
        """Test handling of corrupted settings."""
        pass
```

#### 3. Enhanced `game/model_bot.py` Tests
```python
# Add to test_tournament.py or create test_model_bot.py
class TestModelBotEnhanced:
    """Enhanced ModelBot tests."""

    @pytest.mark.slow
    def test_modelbot_action_masking(self):
        """Test that ModelBot respects action masks."""
        pass

    @pytest.mark.slow
    def test_modelbot_multiple_turns(self):
        """Test ModelBot over multiple game turns."""
        pass

    def test_modelbot_invalid_model_path(self):
        """Test graceful handling of invalid model paths."""
        pass
```

### Phase 2: Medium Priority

#### 4. `game/llm_prompts.py` Tests
```python
# tests/test_llm_prompts.py
class TestLLMPrompts:
    """Tests for LLM prompt generation."""

    def test_system_prompt_generation(self):
        """Test system prompt contains required sections."""
        pass

    def test_game_state_prompt_formatting(self):
        """Test game state is formatted correctly in prompt."""
        pass

    def test_action_format_instructions(self):
        """Test action format instructions are clear."""
        pass
```

#### 5. `utils/experiment_tracker.py` Tests
```python
# tests/test_experiment_tracker.py
class TestExperimentTracker:
    """Tests for experiment tracking."""

    def test_tracker_without_external_services(self):
        """Test tracker works without wandb/tensorboard."""
        pass

    def test_log_metrics(self):
        """Test metric logging."""
        pass

    def test_config_persistence(self):
        """Test config is saved to disk."""
        pass
```

### Phase 3: Low Priority

- `ui/renderer.py` - Requires display mocking, low ROI
- `ui/menus/credits_menu.py` - Static content, minimal logic

---

## Test Execution Guide

### Local Development

```bash
# Quick test (unit tests only, < 30 seconds)
pytest -m "unit" -q

# Standard test (exclude slow tests, ~2 minutes)
pytest -m "not slow"

# Full test suite (all tests, ~5-10 minutes)
pytest

# With coverage report
pytest --cov=reinforcetactics --cov-report=html
```

### CI/CD Pipeline

#### Fast Pipeline (PRs, every push)
```yaml
# Run on every PR - fast feedback
pytest -m "not slow" --timeout=60
```

#### Full Pipeline (nightly, releases)
```yaml
# Nightly full test suite
pytest --timeout=120

# Or explicitly include slow tests
pytest -m "slow or not slow" --timeout=120
```

### Pre-commit Hook (optional)
```bash
# Quick sanity check before commit
pytest -m "unit" -x --timeout=30 -q
```

---

## Best Practices

### Writing New Tests

1. **Mark tests appropriately**:
   ```python
   @pytest.mark.unit
   def test_simple_function():
       ...

   @pytest.mark.slow
   @pytest.mark.integration
   def test_full_game_simulation():
       ...
   ```

2. **Use fixtures from conftest.py** for common objects

3. **Set appropriate timeouts** for tests that might hang:
   ```python
   @pytest.mark.timeout(60)
   def test_potentially_slow():
       ...
   ```

4. **Mock external services** to avoid flaky tests:
   ```python
   @pytest.mark.external
   def test_with_real_api():
       # Only runs when explicitly requested
       ...
   ```

### Test Naming Convention

- `test_<function_name>_<scenario>` for unit tests
- `test_<class>_<behavior>` for class tests
- `test_<feature>_integration` for integration tests

---

## Maintenance

- **Review quarterly**: Check for new untested modules
- **Update markers**: Ensure slow tests are properly marked
- **Monitor CI times**: If tests slow down, investigate and optimize
- **Track coverage**: Aim for >80% on core modules
