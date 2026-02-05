---
sidebar_position: 5
id: tournament-system
title: Tournament System
---

# Tournament System

This document describes the tournament system for Reinforce Tactics, which allows running round-robin tournaments between different bot types.

:::tip
Looking for tournament results? Check out the [Bot Tournaments](./tournaments) page!
:::

## Overview

The tournament system automatically discovers and runs competitions between:
- **SimpleBot**: Built-in basic rule-based bot (always included)
- **MediumBot**: Built-in improved rule-based bot with advanced strategies (always included)
- **AdvancedBot**: Built-in sophisticated bot extending MediumBot with map analysis, enhanced unit composition, mountain positioning, ranged combat prioritization, and special ability usage (always included)
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

- `--map PATH`: Path to single map file (for backward compatibility)
- `--maps PATH [PATH ...]`: List of map file paths to use in evaluation
- `--map-dir PATH`: Directory to load all maps from (alternative to listing individual maps)
- `--map-pool-mode {cycle,random,all}`: How to select maps: `cycle` (default), `random`, or `all`
- `--models-dir PATH`: Directory containing trained models (default: `models/`)
- `--output-dir PATH`: Directory for results and replays (default: `tournament_results/`)
- `--games-per-side INT`: Number of games per side in each matchup (default: 2)
- `--max-turns INT`: Maximum turns per game (default: 500)
- `--test`: Test mode - adds duplicate SimpleBots for testing
- `--log-conversations`: Enable LLM conversation logging to JSON files
- `--conversation-log-dir PATH`: Directory for conversation logs (default: `output_dir/llm_conversations/`)
- `--concurrent INT`: Number of concurrent games (default: 1, sequential)
- `--no-llm`: Skip LLM bot discovery
- `--no-models`: Skip trained model bot discovery

### Examples

Run a tournament on a specific map:
```bash
python3 scripts/tournament.py --map maps/1v1/beginner.csv
```

Run a tournament across multiple maps:
```bash
python3 scripts/tournament.py --maps maps/1v1/beginner.csv maps/1v1/funnel_point.csv
```

Run a tournament on all maps in a directory:
```bash
python3 scripts/tournament.py --map-dir maps/1v1/ --map-pool-mode all
```

Run more games per matchup:
```bash
python3 scripts/tournament.py --games-per-side 5
```

Save results to a custom directory:
```bash
python3 scripts/tournament.py --output-dir my_tournament
```

Run with concurrent games (no LLM bots):
```bash
python3 scripts/tournament.py --no-llm --concurrent 4
```

Test the tournament system:
```bash
python3 scripts/tournament.py --test --games-per-side 1
```

## Bot Discovery

### SimpleBot, MediumBot & AdvancedBot
All three built-in bots are always included. No configuration needed.

- **SimpleBot**: Basic strategy with single-unit purchases and simple targeting
- **MediumBot**: Advanced strategy with coordinated attacks and maximized unit production
- **AdvancedBot**: Extends MediumBot with map analysis, optimized unit composition (Warriors 25%, Archers 20%, Mages 15%, Knights 10%, Rogues 10%, Barbarians 8%, Clerics 7%, Sorcerers 5%), mountain positioning for archers, ranged combat prioritization, and special ability usage (Mage Paralyze, Cleric Heal)

### LLM Bots
Automatically included if:
1. API key is configured in `settings.json`
2. Required package is installed (`openai`, `anthropic`, or `google-generativeai`)
3. API connection test passes

#### Supported Models

**OpenAI (Default: gpt-5-mini-2025-08-07)**
- GPT-5: `gpt-5-mini-2025-08-07` (recommended for cost-effectiveness)
- GPT-4o family: `gpt-4o`, `gpt-4o-mini`
- O-series: `o1`, `o1-mini`, `o3-mini`

**Anthropic Claude (Default: claude-haiku-4-5-20251001)**
- Claude 4.5: `claude-haiku-4-5-20251001` (recommended), `claude-sonnet-4-5-20250929`
- Claude 4: `claude-sonnet-4-20250514`
- Claude 3.5: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`

**Google Gemini (Default: gemini-2.5-flash)**
- Gemini 2.5: `gemini-2.5-flash` (recommended)
- Gemini 2.0: `gemini-2.0-flash`
- Gemini 1.5: `gemini-1.5-pro`, `gemini-1.5-flash`

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

You can also specify custom models by setting environment variables or modifying bot initialization code.

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
  - One player wins (captures enemy HQ or eliminates all enemy units)
  - Turn limit reached (counts as draw)

### ELO Rating System
The tournament tracks ELO ratings for all bots:
- **Starting rating**: 1500 for all bots
- **K-factor**: 32 (standard chess rating adjustment)
- Ratings are updated after each game based on expected vs actual outcomes
- Final rankings include ELO rating and change from initial rating

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
      "win_rate": 0.625,
      "elo": 1564,
      "elo_change": 64
    }
  ],
  "matchups": [...],
  "elo_history": {...}
}
```

### `tournament_results/tournament_results.csv`
Simple CSV format for spreadsheet import:
```csv
Bot,Wins,Losses,Draws,Total Games,Win Rate,Elo,Elo Change
SimpleBot,5,1,2,8,0.625,1564,+64
OpenAIBot,3,3,2,8,0.375,1436,-64
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
3. **TournamentConfig**: Unified configuration for tournament settings
4. **ELO Rating**: Rating system with configurable K-factor
5. **TournamentSchedule**: Round-robin scheduling with resume support
6. **ModelBot**: Wrapper for Stable-Baselines3 models
7. **Bot discovery**: Automatic detection of available bots (built-in, LLM, model)
8. **Results tracking**: Win/loss/draw statistics with CSV/JSON export

### Code Structure

```
scripts/
  tournament.py              # Main tournament CLI script
reinforcetactics/
  game/
    bot.py                   # SimpleBot, MediumBot, AdvancedBot
    llm_bot.py               # LLM bot implementations (OpenAI, Claude, Gemini)
    model_bot.py             # ModelBot for trained models
  tournament/
    bots.py                  # Bot descriptors and discovery
    runner.py                # Tournament execution engine
    config.py                # Tournament configuration
    schedule.py              # Round-robin scheduling with resume support
    results.py               # Results tracking and export
    elo.py                   # ELO rating system
tests/
  test_tournament.py         # Tournament system tests
  test_tournament_library.py # Tournament library tests
```

## Docker Tournament Runner

For more advanced tournament features, see the Docker-based tournament runner in `docker/tournament/`:

```bash
cd docker/tournament
docker-compose up --build
```

The Docker tournament runner includes:
- **ELO rating system**: Tracks bot skill ratings throughout the tournament
- **Concurrent game execution**: Run multiple games in parallel (configurable 1-32)
- **Resume capability**: Continue interrupted tournaments from where they left off
- **Google Cloud Storage**: Upload results to GCS for cloud deployments
- **Multi-map tournaments**: Play across multiple maps with per-map configuration
- **LLM API rate limiting**: Configurable delay between API calls

See `docker/tournament/README.md` for detailed configuration options.

## Future Enhancements

Possible improvements:
- Swiss-system tournament format
- Real-time progress visualization
- Tournament brackets for elimination format
- Head-to-head statistics
- Performance profiling per bot
