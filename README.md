# Reinforce Tactics - 2D Turn-Based Strategy Game

[![GitHub Stars](https://img.shields.io/github/stars/kuds/reinforce-tactics)](https://github.com/kuds/reinforce-tactics/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/kuds/reinforce-tactics)](https://github.com/kuds/reinforce-tactics/network/members)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/kuds/reinforce-tactics)](https://github.com/kuds/reinforce-tactics/commits/main)
[![GitHub Issues](https://img.shields.io/github/issues/kuds/reinforce-tactics)](https://github.com/kuds/reinforce-tactics/issues)
[![GitHub License](https://img.shields.io/github/license/kuds/reinforce-tactics)](https://github.com/kuds/reinforce-tactics/blob/main/LICENSE)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/kuds/reinforce-tactics/deploy-docusaurus.yml)](https://github.com/kuds/reinforce-tactics/actions/workflows/deploy-docusaurus.yml)
[![Documentation](https://img.shields.io/badge/docs-reinforcetactics.com-blue)](https://reinforcetactics.com)

![](images/reinforce_tactics_logo.svg)

A modular turn-based strategy game built with Pygame and Gymnasium for reinforcement learning.

## Features

- **Turn-based tactical gameplay** with multiple unit types
- **4 unit types**: Warrior, Mage, Cleric, Archer (each with unique abilities)
- **Combat system** with attacks, counter-attacks, paralysis, and healing
- **Economic system** with income from controlled structures
- **Structure capture**: Towers, Buildings, and Headquarters
- **Save/Load system** for continuing games
- **Replay system** for watching past games with video export capability
- **AI opponent** (SimpleBot) for single-player games
- **Gymnasium integration** for reinforcement learning
- **Headless mode** for fast training without rendering

## Installation

### Basic Installation (GUI Mode)

```bash
pip install pygame pandas numpy
```

### Full Installation (with RL)

```bash
pip install pygame pandas numpy gymnasium stable-baselines3[extra]
```

### Optional (for replay video recording)

```bash
pip install opencv-python
```

## Project Structure

```
reinforce-tactics/
├── main.py                    # Main entry point (train, evaluate, play, stats modes)
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker support
├── reinforcetactics/          # Main package
│   ├── core/                  # Core game logic
│   ├── game/                  # Game mechanics & bots
│   ├── ui/                    # Pygame UI components
│   ├── rl/                    # Reinforcement learning
│   └── utils/                 # Utilities (file_io, settings, language, replay)
├── train/                     # Training scripts (including feudal RL)
├── eval/                      # Evaluation scripts
├── examples/                  # Example scripts (LLM bot demo)
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks
├── maps/                      # Map files
├── scripts/                   # Utility scripts
└── docs-site/                 # Docusaurus documentation site
```

## Quick Start

### Play the Game (GUI)

```bash
python main.py
```

### Train an RL Agent

```bash
# Train against bot opponent
python main.py --mode train --algorithm ppo --timesteps 1000000 --opponent bot

# Train with self-play
python main.py --mode train --algorithm ppo --timesteps 1000000 --opponent self

# Train with A2C algorithm
python main.py --mode train --algorithm a2c --timesteps 500000 --opponent bot

# Train with DQN algorithm
python main.py --mode train --algorithm dqn --timesteps 1000000 --opponent bot
```

### Test a Trained Agent

```bash
python main.py --mode evaluate --model models/ppo_model.zip --episodes 10
```

### Use as Gymnasium Environment

```python
from reinforcetactics.rl.gym_env import StrategyGameEnv

# Create environment
env = StrategyGameEnv(
    map_file='maps/1v1/test_map.csv',  # or None for random
    opponent='bot',  # 'bot', 'random', or 'self'
    render_mode=None  # None, 'human', or 'rgb_array'
)

# Standard Gym API
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Headless Mode (Fast Training)

```python
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils.file_io import FileIO

# Load map
map_data = FileIO.load_map('maps/1v1/test_map.csv')

# Create game without rendering
game = GameState(map_data)
bot = SimpleBot(game, player=2)

# Game loop
while not game.game_over:
    # Your agent's actions here
    # ...
    
    game.end_turn()
    
    # Bot plays
    bot.take_turn()
    game.end_turn()
```

## Replay System with Video Export

The replay system allows you to watch and record past games. Replays are automatically saved after each game and can be exported to video files.

### Watching Replays

From the main menu, select "Watch Replay" to view saved replays. The replay player includes:

- **Play/Pause** (▶/⏸): Control replay playback (Space key or click)
- **Restart** (⟲): Restart from the beginning (R key or click)
- **Speed Control** (+/-): Change playback speed (0.25x to 4x)
- **Progress Bar**: Click to seek to any point in the replay
- **Video Recording** (⏺ Rec/⏹ Save): Record and save replay as video (V key or click)

### Recording Replay to Video

To export a replay as a video file:

1. Open a replay from the main menu
2. Press **V** or click the **⏺ Rec** button to start recording
3. The replay will capture frames as it plays
4. Press **V** again or click **⏹ Save** to stop and save the video
5. Video will be saved to the `videos/` directory as MP4

**Requirements**: Install `opencv-python` for video export functionality:
```bash
pip install opencv-python
```

**Video Settings**:
- Format: MP4 (H.264)
- Frame Rate: 30 FPS
- Resolution: Matches game window size

### Programmatic Replay Control

```python
from reinforcetactics.utils.replay_player import ReplayPlayer
from reinforcetactics.utils.file_io import FileIO
import pandas as pd

# Load replay
replay_data = FileIO.load_replay('replays/game_20231201_120000.json')
map_data = pd.DataFrame(replay_data['game_info']['initial_map'])

# Create replay player
player = ReplayPlayer(replay_data, map_data)

# Start recording
player.start_recording()

# Run replay (captures frames automatically)
player.run()

# Or manually control:
player.start_recording()
# ... render frames with player.draw() ...
player.stop_recording()
video_path = player.save_video()  # Returns path to saved video
```

## LLM Bots (AI Opponents)

Reinforce Tactics supports LLM-powered bots that can play the game using GPT, Claude, or Gemini models. These bots understand game rules and make strategic decisions in natural language.

### Supported Providers and Models

#### OpenAI (GPT)
Default model: `gpt-4o-mini`

**GPT-4o Family** (latest, most capable):
- `gpt-4o` - Latest flagship model
- `gpt-4o-mini` - Excellent cost/performance balance (default)
- `gpt-4o-2024-11-20`, `gpt-4o-2024-08-06`, `gpt-4o-2024-05-13` - Dated snapshots
- `gpt-4o-mini-2024-07-18` - Mini model snapshot

**GPT-4 Turbo** (high performance):
- `gpt-4-turbo` - Latest turbo model
- `gpt-4-turbo-2024-04-09`, `gpt-4-turbo-preview` - Turbo variants
- `gpt-4-0125-preview`, `gpt-4-1106-preview` - Preview releases

**GPT-4** (stable, proven):
- `gpt-4` - Original GPT-4
- `gpt-4-0613` - Stable snapshot

**GPT-3.5 Turbo** (fast and economical):
- `gpt-3.5-turbo` - Latest 3.5 model
- `gpt-3.5-turbo-0125`, `gpt-3.5-turbo-1106` - Dated snapshots

**O1 Reasoning Models** (advanced reasoning):
- `o1` - Latest reasoning model
- `o1-2024-12-17` - Dated snapshot
- `o1-mini` - Smaller reasoning model
- `o1-mini-2024-09-12` - Mini snapshot
- `o1-preview`, `o1-preview-2024-09-12` - Preview versions

**O3 Models** (if available):
- `o3-mini`, `o3-mini-2025-01-31`

#### Anthropic (Claude)
Default model: `claude-3-5-haiku-20241022`

**Claude 4** (latest generation):
- `claude-sonnet-4-20250514` - Latest Claude 4 Sonnet

**Claude 3.5** (high performance):
- `claude-3-5-sonnet-20241022` - Latest 3.5 Sonnet
- `claude-3-5-sonnet-20240620` - Earlier 3.5 Sonnet
- `claude-3-5-haiku-20241022` - Latest 3.5 Haiku (default, fast and economical)

**Claude 3** (proven models):
- `claude-3-opus-20240229` - Most capable Claude 3
- `claude-3-sonnet-20240229` - Balanced Claude 3
- `claude-3-haiku-20240307` - Fast Claude 3

#### Google (Gemini)
Default model: `gemini-2.0-flash`

**Gemini 2.0** (latest generation):
- `gemini-2.0-flash` - Latest Flash model (default, fast and high-quality)
- `gemini-2.0-flash-exp` - Experimental Flash variant
- `gemini-2.0-flash-lite` - Lightweight Flash variant
- `gemini-2.0-flash-thinking-exp` - Enhanced reasoning variant

**Gemini 1.5** (stable and capable):
- `gemini-1.5-pro` - High-capability model with 2M token context
- `gemini-1.5-pro-latest` - Latest Pro snapshot
- `gemini-1.5-flash` - Fast and efficient
- `gemini-1.5-flash-latest` - Latest Flash snapshot
- `gemini-1.5-flash-8b` - 8B parameter Flash variant

**Gemini 1.0** (legacy):
- `gemini-1.0-pro` - Original Gemini Pro
- `gemini-pro` - Alias for 1.0 Pro

### Installation

Install the LLM provider packages you need:

```bash
# For OpenAI GPT models
pip install openai>=1.0.0

# For Anthropic Claude models
pip install anthropic>=0.18.0

# For Google Gemini models
pip install google-generativeai>=0.4.0
```

### Configuration

Set your API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY='your-api-key-here'

# Anthropic
export ANTHROPIC_API_KEY='your-api-key-here'

# Google
export GOOGLE_API_KEY='your-api-key-here'
```

Or create a config file at `~/.reinforce-tactics/config.json`:

```json
{
  "openai_api_key": "your-openai-key",
  "anthropic_api_key": "your-anthropic-key",
  "google_api_key": "your-google-key"
}
```

### Usage Examples

#### Play Against an LLM Bot

```python
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.llm_bot import OpenAIBot, ClaudeBot, GeminiBot
from reinforcetactics.utils.file_io import FileIO

# Load a map
map_data = FileIO.load_map('maps/1v1/test_map.csv')
game = GameState(map_data, num_players=2)

# Create an LLM bot (automatically uses API key from environment)
bot = OpenAIBot(game, player=2)  # Or ClaudeBot, GeminiBot

# Game loop
while not game.game_over:
    # Human player makes moves
    # ... your moves here ...
    
    game.end_turn()
    
    # Bot's turn
    bot.take_turn()
    game.end_turn()
```

#### Custom Model Selection

```python
# OpenAI models
bot = OpenAIBot(game, player=2, model='gpt-4o')          # Latest flagship
bot = OpenAIBot(game, player=2, model='gpt-4o-mini')     # Default, cost-effective
bot = OpenAIBot(game, player=2, model='gpt-4-turbo')     # High performance
bot = OpenAIBot(game, player=2, model='o1-mini')         # Reasoning model

# Claude models
bot = ClaudeBot(game, player=2, model='claude-sonnet-4-20250514')        # Latest Claude 4
bot = ClaudeBot(game, player=2, model='claude-3-5-haiku-20241022')       # Default, fast
bot = ClaudeBot(game, player=2, model='claude-3-5-sonnet-20241022')      # High performance
bot = ClaudeBot(game, player=2, model='claude-3-opus-20240229')          # Most capable

# Gemini models
bot = GeminiBot(game, player=2, model='gemini-2.0-flash')                # Default, latest
bot = GeminiBot(game, player=2, model='gemini-2.0-flash-thinking-exp')  # Enhanced reasoning
bot = GeminiBot(game, player=2, model='gemini-1.5-pro')                  # Long context
bot = GeminiBot(game, player=2, model='gemini-1.5-flash')                # Fast and efficient
```

#### Manual API Key

```python
# Pass API key directly (not recommended for production)
bot = OpenAIBot(game, player=2, api_key='sk-...')
```

#### Using LLM Bots in GUI Mode

Configure player settings in the game menu to select an LLM bot type. The bot will automatically be instantiated with the appropriate API key from your environment.

### Cost and Performance Considerations

**API Costs by Model Tier:**

*Budget Tier (Best for frequent gameplay):*
- OpenAI GPT-4o-mini: ~$0.15/1M input tokens, ~$0.60/1M output tokens
- OpenAI GPT-3.5-turbo: ~$0.50/1M input tokens, ~$1.50/1M output tokens
- Claude 3 Haiku / 3.5 Haiku: ~$0.25/1M input tokens, ~$1.25/1M output tokens
- Gemini Flash models: Free tier available, then ~$0.075/1M input tokens

*Standard Tier (Balanced performance):*
- OpenAI GPT-4o: ~$2.50/1M input tokens, ~$10/1M output tokens
- OpenAI GPT-4 Turbo: ~$10/1M input tokens, ~$30/1M output tokens
- Claude 3 Sonnet / 3.5 Sonnet: ~$3/1M input tokens, ~$15/1M output tokens
- Gemini 1.5 Pro: ~$1.25/1M input tokens, ~$5/1M output tokens

*Premium Tier (Best strategic play):*
- OpenAI O1 series: ~$15/1M input tokens, ~$60/1M output tokens
- Claude 3 Opus / Sonnet 4: ~$15/1M input tokens, ~$75/1M output tokens

**Performance:**
- Average response time: 1-3 seconds per turn
- Tokens per turn: ~1000-2000 input, ~200-500 output
- Cost per game (20-30 turns):
  - Budget: $0.001-0.01
  - Standard: $0.01-0.05
  - Premium: $0.05-0.20

**Recommendations:**
- **For practice/testing**: `gpt-4o-mini`, `claude-3-5-haiku-20241022`, or `gemini-2.0-flash`
- **For competitive play**: `gpt-4o`, `claude-3-5-sonnet-20241022`, or `gemini-1.5-pro`
- **For maximum performance**: `o1`, `claude-sonnet-4-20250514`, or `gemini-2.0-flash-thinking-exp`
- Always monitor API usage to avoid unexpected costs
- Consider using budget models first to understand game mechanics

### How It Works

1. **Game State Serialization**: The bot converts the current game state into a JSON format including units, buildings, gold, and legal actions.

2. **Strategic Reasoning**: The LLM receives game rules and the current state, then decides on actions using strategic thinking.

3. **Action Execution**: The bot's response is parsed and validated, then actions are executed sequentially.

4. **Error Handling**: Invalid actions are skipped with warnings. API failures trigger exponential backoff retry logic.

### Troubleshooting

**"API key not provided" Error:**
- Ensure your environment variable is set correctly
- Check the variable name matches: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`

**Import Errors:**
- Install the required package: `pip install openai` (or `anthropic`, `google-generativeai`)

**Rate Limiting:**
- The bot automatically retries with exponential backoff
- Consider upgrading your API tier if you hit limits frequently

**Bot Makes Invalid Moves:**
- This is expected occasionally due to LLM unpredictability
- Invalid moves are automatically skipped with no game impact
- Check logs for details on which actions were rejected

## Game Rules

![](images/rt_demo.gif)

### Units

- **Warrior (W)**: 200 gold, 3 movement, 15 HP, 10 ATK
  - Melee attacker with high health

- **Mage (M)**: 250 gold, 2 movement, 10 HP, 8/12 ATK
  - Ranged attacker (stronger at distance)
  - Can paralyze enemies for 3 turns

- **Cleric (C)**: 200 gold, 2 movement, 8 HP, 2 ATK
  - Support unit that can heal allies (+5 HP)
  - Can cure paralyzed allies

- **Archer (A)**: 250 gold, 3 movement, 15 HP, 5 ATK
  - Indirect ranged unit (1-2 spaces, or 1-3 on mountains)
  - Cannot attack at distance 0 (adjacent)
  - Melee units cannot counter-attack Archers
  - Other Archers and Mages can counter if in range

### Combat

- Units attack adjacent enemies
- Defending units counter-attack with 90% damage
- Paralyzed units cannot move, attack, or counter-attack
- Units can't move or attack the turn they're created

### Structures

- **Headquarters (HQ)**: 50 HP, generates $150 income/turn
  - Capturing enemy HQ wins the game!
  
- **Buildings**: 40 HP, generates $100 income/turn
  - Used to create new units
  
- **Towers**: 30 HP, generates $50 income/turn
  - Control key strategic points

- Structures are captured by standing on them and using "Seize"
- Unit's current HP is reduced from structure HP each turn
- Structures regenerate 50% HP/turn if abandoned

### Economy

- Starting gold: $1000
- Income generated at start of each turn
- Use gold to create units at your buildings

## Map Format

Maps are CSV files with tile codes:

```
p,p,p,b_1,h_1,b_1,p,p,p
p,p,p,p,p,p,p,p,p
p,t,p,p,p,p,p,t,p
p,p,p,p,p,p,p,p,p
p,p,p,b_2,h_2,b_2,p,p,p
```

Tile codes:
- `p` - Grass (walkable)
- `w` - Water (not walkable)
- `m` - Mountain (not walkable)
- `f` - Forest (walkable)
- `r` - Road (walkable)
- `t` - Tower (neutral or `t_1`, `t_2` for owned)
- `b` - Building (requires owner, e.g., `b_1`)
- `h` - Headquarters (requires owner, e.g., `h_1`)

## RL Environment Details

### Observation Space

Multi-dict space with:
- `grid`: (H, W, 3) array - tile info (type, owner, structure HP)
- `units`: (H, W, 3) array - unit info (type, owner, HP)
- `gold`: (2,) array - gold for both players
- `current_player`: Discrete(2) - current player index
- `action_mask`: MultiBinary - legal action mask

### Action Space

Multi-discrete: `[action_type, unit_type, from_x, from_y, to_x, to_y]`

Action types:
0. Create unit
1. Move
2. Attack
3. Paralyze
4. Heal
5. Cure
6. Seize
7. End turn

### Reward Configuration

```python
reward_config = {
    'win': 1000.0,           # Win game
    'loss': -1000.0,         # Lose game
    'income_diff': 0.0,      # Gold advantage per turn
    'unit_diff': 0.0,        # Unit count advantage
    'structure_control': 0.0,# Structure control bonus
    'invalid_action': -10.0   # Invalid action penalty
}
```

Start with sparse rewards (only win/loss), then add dense rewards if learning is slow.

## Development

### Running Tests

```bash
# Run the full test suite
python -m pytest tests/

# Quick game test
python -c "from reinforcetactics.rl.gym_env import StrategyGameEnv; env = StrategyGameEnv(); env.reset(); print('OK')"
```

### Creating Custom Maps

1. Create CSV file in `maps/1v1/`
2. Use tile codes (see Map Format above)
3. Ensure each player has at least 1 HQ
4. Minimum size is 20x20 (auto-padded if smaller)

### Extending the Game

- Add new unit types in `constants.py` and `core/unit.py`
- Modify combat rules in `game/mechanics.py`
- Create new AI opponents in `game/bot.py`
- Add custom reward functions in `rl/gym_env.py`

## Troubleshooting

**"No maps found"**: Create `maps/1v1/` directory and add CSV map files

**Pygame window not appearing**: Check if running in headless environment

**RL training slow**: Use headless mode (`render_mode=None`)

**Invalid actions during training**: Action masking not fully implemented yet, agent will learn to avoid invalid actions through penalties

## Documentation Site

This project includes a comprehensive Docusaurus-based documentation site located in the `docs-site/` directory.

### Running the Documentation Site Locally

```bash
# Navigate to the docs site directory
cd docs-site

# Install dependencies (first time only)
npm install

# Start the development server
npm start
```

The site will be available at `http://localhost:3000`.

### Building for Production

```bash
cd docs-site
npm run build
```

The static files will be generated in the `docs-site/build/` directory.

### Documentation Structure

The documentation includes:
- **Getting Started**: Overview and quick start guide
- **Implementation Status**: Current state of features and development roadmap
- **Project Timeline**: Research timeline and task list

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Areas for improvement:
- Better action masking for RL
- More sophisticated AI opponents
- Additional unit types and abilities
- Multiplayer support (3-4 players)
- Map editor GUI
- Tournament/ladder system

## Citing This Project

If you use Reinforce Tactics in your research, please cite it as follows:

### BibTeX

```bibtex
@software{reinforce_tactics,
  author = {Michael Kudlaty},
  title = {Reinforce Tactics: A Turn-Based Strategy Game for Reinforcement Learning},
  year = {2025},
  url = {https://github.com/kuds/reinforce-tactics},
  note = {A modular turn-based strategy game built with Pygame and Gymnasium for reinforcement learning research}
}
```

### Plain Text

Michael Kudlaty. (2025). Reinforce Tactics: A Turn-Based Strategy Game for Reinforcement Learning. GitHub. https://github.com/kuds/reinforce-tactics

## Credits

Built with:
- Pygame for rendering
- Gymnasium for RL interface
- Stable-Baselines3 for RL algorithms
- NumPy and Pandas for data handling

## To Do List
- [x] Game play save and load
- [x] Implement Replay Functionality
- [ ] Implement Basic AI (Deterministics, no NN)
  - [x] Easy
  - [ ] Normal
  - [ ] Hard
- [X] Implement Range Units
- [X] Implement Routing Functionality for Units
- [X] Buildings Healing Units
- [ ] Implement Terriority Defenese
- [X] Language Support
  - [x] Korean
  - [x] Spanish
  - [x] French
  - [x] Chinese
- [x] Implement Saving playback to video
- [x] Implement headless mode
- [ ] Implement game play stats
- [ ] Unit Artwork
- [ ] Terriority Artwork
- [x] Support Gymnasium framework
- [ ] Support Ray rllib
- [ ] Support PettingZoo
- [x] Support Docker
- [x] Website
