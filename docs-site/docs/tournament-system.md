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
- **MasterBot**: Master-tier bot extending AdvancedBot with threat maps, HP-ascending focus fire, HQ-snipe priority, and Haste follow-through
- **MixedBot**: Curriculum-only bot that samples one of two inner bots per episode (used for bootstrap training, not for standings)
- **LLM Bots**: OpenAI, Claude, and Gemini bots (if API keys configured)
- **Model Bots**: Trained Stable-Baselines3 models (from `models/` directory)

## Quick Start

Run a tournament with default settings:

```bash
python3 scripts/tournament.py
```

This will:
- Use the `maps/1v1/beginner.csv` map
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

### SimpleBot, MediumBot, AdvancedBot & MasterBot
The built-in scripted bots are always available. No configuration needed.

- **SimpleBot**: Basic strategy with single-unit purchases and simple targeting
- **MediumBot**: Advanced strategy with coordinated attacks and maximized unit production
- **AdvancedBot**: Extends MediumBot with map analysis, optimized unit composition (Warriors 25%, Archers 20%, Mages 15%, Knights 10%, Rogues 10%, Barbarians 8%, Clerics 7%, Sorcerers 5%), mountain positioning for archers, ranged combat prioritization, and special ability usage (Mage Paralyze, Cleric Heal)
- **MasterBot**: Extends AdvancedBot with a per-turn threat map (used for retreat tiles and Knight charge landings), HP-ascending focus fire, HQ-snipe priority in the conquer phase, and Haste follow-through (a hasted unit that seizes/attacks/charges still gets its second action)

### Stochastic Tiebreak

All scripted bots accept an optional `rng` (a `random.Random` instance). When provided, the bot uses it to tiebreak at every sort / max / best-tracking site so two runs of the same scenario don't produce byte-identical games. Without an `rng`, the bots remain deterministic. This is plumbed through `MixedBot` to its inner bot so the curriculum-bridge bot's stochastic tiebreaking actually engages.

### Curriculum Bridge (MixedBot)

`MixedBot` is not a standalone strategy — it is a training-time curriculum bridge. On construction it samples one of two inner bots with probability `p_hard` and delegates `take_turn()` to that instance for the lifetime of the episode (the environment reconstructs the opponent on every `reset()`, so the choice effectively resamples per episode). Configure it via `opponent_kwargs` in `configs/ppo/bootstrap.yaml`, e.g. `{easy: simple, hard: medium, p_hard: 0.5}` for the simple→medium bridge.

### LLM Bots
Automatically included if:
1. API key is configured in `settings.json`
2. Required package is installed (`openai`, `anthropic`, or `google-genai`)
3. API connection test passes

#### Supported Models

**OpenAI (Default: gpt-5-mini-2025-08-07)**
- GPT-5.2: `gpt-5.2`
- GPT-5: `gpt-5-2025-08-07`, `gpt-5-mini-2025-08-07` (recommended for cost-effectiveness), `gpt-5-nano-2025-08-07`
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, the `-pro` models and the o-series only accept the default temperature, so OpenAIBot ignores `temperature` for them

**Anthropic Claude (Default: claude-haiku-4-5-20251001)**
- Claude 5.x: `claude-fable-5-1`, `claude-fable-5`, `claude-opus-5-5`, `claude-opus-5`, `claude-sonnet-5`
- Claude 4.6–4.8: `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`
- Claude 4.5: `claude-haiku-4-5-20251001` (recommended), `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20250929`
- Opus 4.7, Opus 4.8 and the 5.x models reject sampling parameters, so ClaudeBot ignores `temperature` for them
- Don't use `claude-opus-4-1-20250805` (retired 2026-08-05) or the deprecated `claude-sonnet-4-20250514` / `claude-opus-4-20250514`

**Google Gemini (Default: gemini-2.5-flash)**
- Gemini 3 (preview): `gemini-3-pro-preview`, `gemini-3-flash-preview`
- Gemini 2.5: `gemini-2.5-flash` (recommended), `gemini-2.5-pro`, `gemini-2.5-flash-lite`

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

#### When an LLM bot can't play

Rate limits, overloads, timeouts and connection errors are retried with jittered backoff (honouring `Retry-After`): at least 3 attempts, and more while the request has taken less than 60 seconds in all; a turn whose retries run out is passed. An LLM bot raises `LLMBotError` instead of passing turns when the failure can't fix itself (missing SDK, rejected API key, unknown model, malformed request) or after 3 turns in a row in which the API gave no reply at all. The tournament then ends that game as an error: it is reported with an `error` message in the results JSON, counted under `errors` in the standings, and left out of wins, losses, draws and Elo.

A reply that arrives but holds no usable actions (an empty reply, a refusal or safety block, prose, or JSON cut off at `max_tokens`) is the model's own play, not an infrastructure failure: the turn is passed, the game goes on and is scored normally, and the reply is counted as `llm_empty_reply` or `llm_unparseable_reply` in the replay's `game_info.capabilities_p1` / `capabilities_p2`. Actions that aren't legal at the moment they would run are skipped and counted as `llm_illegal_action` (and `llm_illegal_<type>`).

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
  "map": "maps/1v1/beginner.csv",
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
env = StrategyGameEnv(map_file="maps/1v1/beginner.csv", opponent="bot", render_mode=None)

# Train model
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("models/my_trained_bot")
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
- Install required package: `pip install openai` (or `anthropic`, `google-genai`)
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
    bot.py                   # SimpleBot, MediumBot, AdvancedBot, MasterBot, MixedBot
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
