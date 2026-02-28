# Submitting Reinforce Tactics to Kaggle Environments

This document explains how to submit Reinforce Tactics as a custom
environment to the [kaggle-environments](https://github.com/Kaggle/kaggle-environments)
repository.

## Prerequisites

- Fork the [Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments) repo
- Python >= 3.8 with `kaggle-environments` installed:
  ```bash
  pip install kaggle-environments
  ```

## File Structure

Copy the following files into the kaggle-environments repo:

```
kaggle_environments/envs/reinforce_tactics/
    __init__.py                  # Package init (empty or re-exports)
    reinforce_tactics.json       # Environment specification
    reinforce_tactics.py         # Interpreter, renderer, specification, agents
    reinforce_tactics_engine/    # Vendored game engine (self-contained)
        __init__.py              # Exports GameState, UNIT_DATA
        constants.py             # Game constants and unit data
        core/
            __init__.py
            tile.py              # Tile class
            unit.py              # Unit class
            grid.py              # TileGrid class
            visibility.py        # Fog of war visibility
            game_state.py        # Core GameState logic
        game/
            __init__.py
            mechanics.py         # Combat, healing, buffs mechanics
    agents/
        __init__.py
        random_agent.py          # Minimal baseline agent
        simple_bot_agent.py      # Strategic baseline agent
```

The `reinforce_tactics_engine/` directory is a vendored copy of the core
game engine with all imports rewritten as relative imports. This follows
the same pattern as [Lux AI Season 3](https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs/lux_ai_s3),
which vendors its engine as the `luxai_s3/` sub-package. No external
`pip install` is required — the environment is fully self-contained.

## Step-by-Step PR Process

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USER/kaggle-environments.git
cd kaggle-environments
git checkout -b add-reinforce-tactics
```

### 2. Create the Environment Directory

```bash
mkdir -p kaggle_environments/envs/reinforce_tactics/agents
```

### 3. Copy Files

Copy the entire `reinforce_tactics/` directory (including the
`reinforce_tactics_engine/` sub-package and `agents/`) into the
kaggle-environments envs directory.

The game engine is vendored as a self-contained sub-package with
relative imports, so no `pip install` of the main `reinforcetactics`
package is needed. Map generation is inlined in the interpreter to
avoid pygame or other UI dependencies.

### 4. Register the Environment

The kaggle-environments package auto-discovers environments in
`kaggle_environments/envs/`. Each subdirectory that contains a Python
module exporting `specification`, `interpreter`, `renderer`, and
optionally `agents` is automatically registered.

Verify registration:

```python
from kaggle_environments import make
env = make("reinforce_tactics")
print(env.specification)
```

### 5. Add Tests

Create `kaggle_environments/envs/reinforce_tactics/test_reinforce_tactics.py`
(or adapt `tests/test_kaggle_env.py` from this repository). The test
file should verify:

- Specification loads correctly
- Game initialises with default config
- Actions execute properly (create, move, attack, seize, heal, etc.)
- Win conditions (HQ capture, unit elimination)
- Draw condition (max turns)
- Built-in agents run without errors

### 6. Submit the PR

```bash
git add kaggle_environments/envs/reinforce_tactics/
git commit -m "Add Reinforce Tactics environment"
git push origin add-reinforce-tactics
```

Then open a PR against `Kaggle/kaggle-environments` with:

**Title:** Add Reinforce Tactics environment

**Description:**
```
## Summary
- Adds Reinforce Tactics, a turn-based tactical strategy game
- 2-player game with 8 unit types, structures, gold economy
- Supports fog of war, configurable unit types, and custom maps
- Includes random and aggressive built-in agents

## Environment Details
- Turn-based: players alternate submitting action lists
- Action format: list of command dicts (like Lux AI)
- Win conditions: capture enemy HQ or eliminate all enemy units
- Configurable: map size, unit types, fog of war, starting gold

## Files
- reinforce_tactics.json: Environment specification
- reinforce_tactics.py: Interpreter, renderer, built-in agents
- reinforce_tactics_engine/: Vendored game engine (self-contained)
- agents/: Standalone baseline agents for competition submissions

## Testing
- 72 unit tests covering specification, serialisation, actions, win
  conditions, renderer, and full game simulations
```

## Testing Locally

### Quick test with built-in agents

```python
from kaggle_environments import make

env = make(
    "reinforce_tactics",
    configuration={"mapWidth": 20, "mapHeight": 20, "mapSeed": 42}
)

# Run with two random agents
result = env.run(["random", "random"])

# Print final state
print(f"Steps: {len(env.steps)}")
for i, agent in enumerate(result[-1]):
    print(f"Agent {i}: status={agent.status}, reward={agent.reward}")
```

### Test with custom agents

```python
def my_agent(observation, configuration):
    actions = []
    player = observation.player + 1
    gold = observation.gold[observation.player]
    structures = observation.structures
    units = observation.units

    # Your strategy here...

    actions.append({"type": "end_turn"})
    return actions

env = make("reinforce_tactics")
result = env.run([my_agent, "random"])
```

### Render in Jupyter notebook

```python
env = make("reinforce_tactics")
env.run(["random", "aggressive"])
env.render(mode="ansi")  # ASCII output
```

## Action Reference

Each turn, an agent returns a list of action dicts:

| Action | Fields | Description |
|--------|--------|-------------|
| `create_unit` | `unit_type`, `x`, `y` | Create a unit at a building |
| `move` | `from_x`, `from_y`, `to_x`, `to_y` | Move a unit |
| `attack` | `from_x`, `from_y`, `to_x`, `to_y` | Attack an enemy unit |
| `seize` | `x`, `y` | Seize the structure at position |
| `heal` | `from_x`, `from_y`, `to_x`, `to_y` | Cleric heals an ally |
| `cure` | `from_x`, `from_y`, `to_x`, `to_y` | Cleric cures paralysis |
| `paralyze` | `from_x`, `from_y`, `to_x`, `to_y` | Mage paralyzes an enemy |
| `haste` | `from_x`, `from_y`, `to_x`, `to_y` | Sorcerer hastes an ally |
| `defence_buff` | `from_x`, `from_y`, `to_x`, `to_y` | Sorcerer buffs defence |
| `attack_buff` | `from_x`, `from_y`, `to_x`, `to_y` | Sorcerer buffs attack |
| `end_turn` | *(none)* | End the current turn |

## Unit Types

| Code | Name | Cost | HP | ATK | DEF | Move | Special |
|------|------|------|----|-----|-----|------|---------|
| W | Warrior | 200 | 15 | 10 | 6 | 3 | - |
| M | Mage | 300 | 10 | 8/12 | 4 | 2 | Paralyze (range 2) |
| C | Cleric | 200 | 8 | 2 | 4 | 2 | Heal, Cure (range 2) |
| A | Archer | 250 | 15 | 5 | 1 | 3 | Ranged 2-3 (4 on mountain) |
| K | Knight | 350 | 18 | 8 | 5 | 4 | Charge (+50% after 3+ tiles) |
| R | Rogue | 350 | 12 | 9 | 3 | 4 | Flank (+50%), Evade (15%) |
| S | Sorcerer | 400 | 10 | 6/8 | 3 | 2 | Haste, Buffs (range 2) |
| B | Barbarian | 400 | 20 | 10 | 2 | 5 | High mobility |
