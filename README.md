# Reinforce Tactics - 2D Turn-Based Strategy Game

![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/:user/:repo/total)

![](images/reinforce_tactics_logo.svg)

A modular turn-based strategy game built with Pygame and Gymnasium for reinforcement learning.

## Features

- **Turn-based tactical gameplay** with multiple unit types
- **3 unit types**: Warrior, Mage, Cleric (each with unique abilities)
- **Combat system** with attacks, counter-attacks, paralysis, and healing
- **Economic system** with income from controlled structures
- **Structure capture**: Towers, Buildings, and Headquarters
- **Save/Load system** for continuing games
- **Replay system** for watching past games
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
strategy_game/
├── constants.py              # Game constants
├── main.py                   # GUI entry point
├── train_rl_agent.py        # RL training script
├── core/                     # Core game logic (no rendering)
│   ├── tile.py
│   ├── unit.py
│   ├── grid.py
│   └── game_state.py
├── game/                     # Game mechanics
│   ├── mechanics.py
│   └── bot.py
├── ui/                       # Pygame UI
│   ├── renderer.py
│   └── menus.py
├── rl/                       # Reinforcement learning
│   ├── gym_env.py
│   └── action_space.py
├── utils/                    # Utilities
│   └── file_io.py
└── maps/                     # Map files
    └── 1v1/
```

## Quick Start

### Play the Game (GUI)

```bash
python main.py
```

### Train an RL Agent

```bash
# Train against bot opponent
python train_rl_agent.py train --opponent bot --total-timesteps 1000000

# Train with self-play
python train_rl_agent.py train --opponent self --total-timesteps 1000000

# Custom rewards (dense rewards for faster learning)
python train_rl_agent.py train --reward-income 0.01 --reward-units 10 --reward-structures 5
```

### Test a Trained Agent

```bash
python train_rl_agent.py test --model-path ./models/PPO_final.zip --n-episodes 5
```

### Use as Gymnasium Environment

```python
from rl.gym_env import StrategyGameEnv

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
from core.game_state import GameState
from game.bot import SimpleBot
from utils.file_io import FileIO

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

```python
# Test environment
python train_rl_agent.py train --check-env --total-timesteps 1000

# Quick game test
python -c "from rl import StrategyGameEnv; env = StrategyGameEnv(); env.reset(); print('OK')"
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
- [ ] Implement Range Units
- [ ] Implement Routing Functionality for Units
- [ ] Buildings Healing Units
- [ ] Implement Terriority Defenese
- [ ] Language Support
  - [ ] Korean
  - [ ] Spanish
  - [ ] French
  - [ ] Chinese
- [x] Implement Saving playback to video
- [ ] Implement headless mode
- [ ] Implement game play stats
- [ ] Unit Artwork
- [ ] Terriority Artwork
- [ ] Support Gymanisum framework
- [ ] Support Ray rllib
- [ ] Support PettingZoo
- [ ] Support Docker
- [ ] Website
