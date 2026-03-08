---
sidebar_position: 1
id: intro
title: Welcome to Reinforce Tactics
slug: /
---

# Welcome to Reinforce Tactics

![Reinforce Tactics Logo](/img/logo.svg)

**Reinforce Tactics** is a modular turn-based strategy game built specifically for reinforcement learning research and experimentation. This project combines classic tactical gameplay with modern RL capabilities, providing a rich environment for developing and testing reinforcement learning algorithms.

## 🎮 What is Reinforce Tactics?

Reinforce Tactics is a 2D turn-based strategy game featuring:

- **Turn-based tactical gameplay** with multiple unit types
- **8 unit types**: Warrior, Mage, Cleric, Archer, Knight, Rogue, Sorcerer, and Barbarian (each with unique abilities)
- **Combat system** with attacks, counter-attacks, paralysis, and healing
- **Economic system** with income from controlled structures
- **Structure capture**: Towers, Buildings, and Headquarters
- **Save/Load system** for continuing games
- **Replay system** for watching past games
- **AI opponents**: SimpleBot, MediumBot, AdvancedBot, and LLM-powered bots (GPT, Claude, Gemini)
- **Full Gymnasium integration** for RL training
- **Headless mode** for fast training without rendering
- **Multiple training algorithms**: PPO, A2C, DQN via Stable-Baselines3, AlphaZero with MCTS, and Feudal RL
- **Action masking**: MaskablePPO and legal-action masking across all bot types
- **Self-play**: Train agents against copies of themselves
- **Fog of War**: Line-of-sight visibility with terrain bonuses
- **Map Editor**: In-game editor for creating and modifying maps
- **Multi-player modes**: 1v1, 1v1v1 (free-for-all), and 2v2 (team) maps
- **Tournament system**: Round-robin tournaments with ELO ratings and Docker support
- **Sprite animations**: Per-team palette swapping and movement path transitions
- **Multi-language**: English, Korean, Spanish, French, Chinese
- **Docker support** for easy deployment

## 🤖 Why Reinforce Tactics?

This project is designed to be:

- **Research-Friendly**: Modular architecture makes it easy to extend and customize
- **RL-Ready**: Standard Gymnasium interface for seamless integration with RL libraries
- **Educational**: Clear code structure perfect for learning RL and game development
- **Performant**: Headless mode enables fast training without rendering overhead

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kuds/reinforce-tactics.git
cd reinforce-tactics

# Basic Installation (core + RL dependencies)
pip install .

# With GUI support
pip install ".[gui]"

# With LLM bot support
pip install ".[llm]"

# Full Installation (all extras)
pip install ".[all]"
```

### Play the Game

```bash
python main.py --mode play
```

### Train an RL Agent

```bash
python main.py --mode train --algorithm ppo --timesteps 1000000 --opponent bot
```

### Use as Gymnasium Environment

```python
from reinforcetactics.rl.gym_env import StrategyGameEnv

# Create environment
env = StrategyGameEnv(
    map_file='maps/1v1/beginner.csv',
    opponent='bot',
    render_mode=None
)

# Standard Gym API
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## 📚 Documentation Structure

This documentation is organized into several sections:

- **Getting Started** (this page): Overview and quick start guide
- **Game Mechanics**: Units, combat system, structures, and terrain
- **Bot Tournaments**: Official tournament results and analysis
- **Maps**: Available maps with previews and descriptions
- **Tournament System**: Technical guide for running tournaments
- **Implementation Status**: Current state of the project and completed features

## 🔗 Useful Links

- [GitHub Repository](https://github.com/kuds/reinforce-tactics)
- [Main README](https://github.com/kuds/reinforce-tactics#readme)
- [Issues](https://github.com/kuds/reinforce-tactics/issues)
- [License (Apache 2.0)](https://github.com/kuds/reinforce-tactics/blob/main/LICENSE)

## 💡 Contributing

Contributions are welcome! Whether you're interested in:
- Improving RL algorithms
- Adding new unit types or game mechanics
- Enhancing documentation
- Fixing bugs

Feel free to open issues or submit pull requests on GitHub.

## 📜 License

This project is licensed under the Apache License 2.0 - feel free to use and modify!
