---
sidebar_position: 5
id: tournament-system
title: Tournament System
---

# Tournament System

This document describes the tournament system for Reinforce Tactics, which allows running round-robin tournaments between different bot types.

:::tip
Looking for tournament results? Check out the [Bot Tournaments](./tournaments.md) page!
:::

## Overview

The tournament system automatically discovers and runs competitions between:
- **SimpleBot**: Built-in rule-based bot (always included)
- **LLM Bots**: OpenAI, Claude, and Gemini bots (if API keys configured)
- **Model Bots**: Trained Stable-Baselines3 models (from `models/` directory)

## Quick Start

Run a tournament with default settings:

```bash
python3 scripts/tournament.py
```

This will:
- Use the `maps/1v1/6x6_beginner.csv` map
- Discover all available bots
- Run 4 games per matchup (2 per side)
- Save results to `tournament_results/`

## Command-Line Options

```bash
python3 scripts/tournament.py [OPTIONS]
```

### Options

- `--map PATH`: Path to map file (default: `maps/1v1/6x6_beginner.csv`)
- `--models-dir PATH`: Directory containing trained models (default: `models/`)
- `--output-dir PATH`: Directory for results and replays (default: `tournament_results/`)
- `--games-per-side INT`: Number of games per side in each matchup (default: 2)
- `--test`: Test mode - adds duplicate SimpleBots for testing

### Examples

Run a tournament on a different map:
```bash
python3 scripts/tournament.py --map maps/1v1/10x10_easy.csv
```

Run more games per matchup:
```bash
python3 scripts/tournament.py --games-per-side 5
```

Save results to a custom directory:
```bash
python3 scripts/tournament.py --output-dir my_tournament
```

Test the tournament system:
```bash
python3 scripts/tournament.py --test --games-per-side 1
```

## Bot Discovery

### SimpleBot
Always included. No configuration needed.

### LLM Bots
Automatically included if:
1. API key is configured in `settings.json`
2. Required package is installed (`openai`, `anthropic`, or `google-generativeai`)
3. API connection test passes

Configure API keys in `settings.json`:
```json
{
  "llm_api_keys": {
    "openai": "sk-...",
    "anthropic": "sk-ant-...",
    "google": "AIza..."
  }
}
```

### Model Bots
Automatically discovered from the `models/` directory:
1. Place trained `.zip` model files in `models/`
2. Models must be Stable-Baselines3 compatible (PPO, A2C, or DQN)
3. Models must be trained on the Reinforce Tactics environment

Example model file: `models/ppo_best_model.zip`

## Tournament Format

### Round-Robin Structure
Every bot plays against every other bot exactly once.

### Matchup Structure
Each matchup consists of `2 × games-per-side` games:
- `games-per-side` games with Bot A as Player 1
- `games-per-side` games with Bot B as Player 1

This accounts for first-move advantage.

Example with `--games-per-side 2`:
- Game 1: Bot A (P1) vs Bot B (P2)
- Game 2: Bot A (P1) vs Bot B (P2)
- Game 3: Bot B (P1) vs Bot A (P2)
- Game 4: Bot B (P1) vs Bot A (P2)

### Game Execution
- All games run in **headless mode** (no rendering) for speed
- Maximum 500 turns per game (prevents infinite games)
- Games end when:
  - One player wins (captures enemy HQ or eliminates all units/buildings)
  - Turn limit reached (counts as draw)

## Output Files

The tournament generates the following outputs:

### `tournament_results/tournament_results.json`
Complete tournament data in JSON format:
```json
{
  "timestamp": "2025-12-10T22:09:52.145889",
  "map": "maps/1v1/6x6_beginner.csv",
  "games_per_side": 2,
  "rankings": [
    {
      "bot": "SimpleBot",
      "wins": 5,
      "losses": 1,
      "draws": 2,
      "total_games": 8,
      "win_rate": 0.625
    }
  ],
  "matchups": [...]
}
```

### `tournament_results/tournament_results.csv`
Simple CSV format for spreadsheet import:
```csv
Bot,Wins,Losses,Draws,Total Games,Win Rate
SimpleBot,5,1,2,8,0.625
OpenAIBot,3,3,2,8,0.375
```

### `tournament_results/replays/`
Replay files for every game:
- Format: `matchup{N}_game{M}_{BotA}_vs_{BotB}.json`
- Example: `matchup001_game01_SimpleBot_vs_OpenAIBot.json`
- Can be played back using the game's replay system

## ModelBot Integration

The `ModelBot` class allows trained Stable-Baselines3 models to participate in tournaments.

### Creating Compatible Models

Train a model using the Reinforcement Learning environment:

```python
from stable_baselines3 import PPO
from reinforcetactics.rl.gym_env import StrategyGameEnv

# Create environment
env = StrategyGameEnv(
    map_file='maps/1v1/6x6_beginner.csv',
    opponent='bot',
    render_mode=None
)

# Train model
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save('models/my_trained_bot')
```

The saved model will be automatically discovered and used in tournaments.

### Action Translation

ModelBot automatically translates between:
- Model actions (MultiDiscrete format)
- Game actions (create_unit, move, attack, seize, heal)

Action format: `[action_type, unit_type, from_x, from_y, to_x, to_y]`

## Troubleshooting

### "Need at least 2 bots for a tournament"
- Only SimpleBot was found
- Add LLM API keys or train some models
- Or use `--test` flag to add a duplicate SimpleBot

### LLM bot not discovered
- Check API key in `settings.json`
- Install required package: `pip install openai` (or `anthropic`, `google-generativeai`)
- Verify API key is valid and has credits

### Model bot not discovered
- Ensure `.zip` file is in `models/` directory
- Verify model is Stable-Baselines3 compatible
- Check that `stable-baselines3` is installed: `pip install stable-baselines3`

### Games ending in draws
- Map may be too large or defensive positions too strong
- Try a smaller map or increase turn limit in code
- Check bot logic is aggressive enough

## Testing

Run the test suite:
```bash
python3 -m pytest tests/test_tournament.py -v
```

Quick tournament test:
```bash
python3 scripts/tournament.py --test --games-per-side 1 --output-dir /tmp/test
```

## Architecture

### Key Components

1. **BotDescriptor**: Describes a bot and knows how to instantiate it
2. **TournamentRunner**: Manages tournament execution
3. **ModelBot**: Wrapper for Stable-Baselines3 models
4. **Bot discovery**: Automatic detection of available bots
5. **Results tracking**: Win/loss/draw statistics

### Code Structure

```
scripts/
  tournament.py          # Main tournament script
reinforcetactics/
  game/
    bot.py              # SimpleBot implementation
    llm_bot.py          # LLM bot implementations
    model_bot.py        # ModelBot for trained models
    __init__.py         # Exports all bot types
tests/
  test_tournament.py    # Tournament system tests
```

## Future Enhancements

Possible improvements:
- Swiss-system tournament format
- ELO rating calculation
- Parallel game execution
- Real-time progress visualization
- Tournament brackets for elimination format
- Head-to-head statistics
- Performance profiling per bot
