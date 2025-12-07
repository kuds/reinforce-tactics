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
- **3 unit types**: Warrior, Mage, Cleric (each with unique abilities)
- **Combat system** with attacks, counter-attacks, paralysis, and healing
- **Economic system** with income from controlled structures
- **Structure capture**: Towers, Buildings, and Headquarters
- **AI opponents** for training and evaluation
- **Full Gymnasium integration** for RL training

## 🤖 Why Reinforce Tactics?

This project is designed to be:

- **Research-Friendly**: Modular architecture makes it easy to extend and customize
- **RL-Ready**: Standard Gymnasium interface for seamless integration with RL libraries
- **Educational**: Clear code structure perfect for learning RL and game development
- **Performant**: Headless mode enables fast training without rendering overhead

## 🚀 Quick Start

### Installation

```bash
# Basic Installation (GUI Mode)
pip install pygame pandas numpy

# Full Installation (with RL)
pip install pygame pandas numpy gymnasium stable-baselines3[extra]
```

### Play the Game

```bash
python main.py
```

### Train an RL Agent

```bash
python train_rl_agent.py train --opponent bot --total-timesteps 1000000
```

### Use as Gymnasium Environment

```python
from rl.gym_env import StrategyGameEnv

# Create environment
env = StrategyGameEnv(
    map_file='maps/1v1/test_map.csv',
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
- **Implementation Status**: Current state of the project and completed features
- **Project Timeline**: Research timeline and development roadmap

## 🔗 Useful Links

- [GitHub Repository](https://github.com/kuds/reinforce-tactics)
- [Main README](https://github.com/kuds/reinforce-tactics#readme)
- [Issues](https://github.com/kuds/reinforce-tactics/issues)
- [License (MIT)](https://github.com/kuds/reinforce-tactics/blob/main/LICENSE)

## 💡 Contributing

Contributions are welcome! Whether you're interested in:
- Improving RL algorithms
- Adding new unit types or game mechanics
- Enhancing documentation
- Fixing bugs

Feel free to open issues or submit pull requests on GitHub.

## 📜 License

This project is licensed under the MIT License - feel free to use and modify!
