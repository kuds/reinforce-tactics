---
sidebar_position: 4
id: tournaments
title: Bot Tournaments
---

# Bot Tournaments

Welcome to the Reinforce Tactics Bot Tournament page! This page tracks official tournament results between different bot types, showcasing the performance of rule-based bots, LLM-powered bots, and trained RL models.

## What are Tournaments?

Tournaments in Reinforce Tactics pit different bot types against each other in a round-robin format. Each matchup consists of multiple games where bots alternate playing as Player 1 and Player 2 to account for first-move advantage.

### Tournament Format

- **Round-Robin**: Every bot plays against every other bot
- **Fair Play**: Equal games with each bot as Player 1 and Player 2
- **Map**: Typically played on the 6x6 beginner map for quick matches
- **Results**: Win/Loss/Draw records with win rate statistics

## Bot Types

### SimpleBot
A rule-based AI that follows a simple strategy:
- Purchases the most expensive units first
- Prioritizes capturing buildings and towers
- Attacks nearby enemy units
- Always included in tournaments

### LLM Bots
AI-powered by Large Language Models:
- **OpenAIBot**: Powered by GPT models
- **ClaudeBot**: Powered by Anthropic's Claude
- **GeminiBot**: Powered by Google's Gemini
- Uses natural language reasoning to make strategic decisions

### Model Bots
Trained using Reinforcement Learning:
- Uses Stable-Baselines3 (PPO, A2C, or DQN)
- Trained through self-play and opponent challenges
- Learns optimal strategies through experience

## Running Your Own Tournament

Want to run a tournament? It's easy! See the [Tournament System](./tournament-system.md) guide for detailed instructions.

Quick start:
```bash
python3 scripts/tournament.py
```

## Official Tournament Results

Below are the results from official tournaments run on the Reinforce Tactics platform.

### Tournament #1 - [Date TBD]

**Configuration:**
- Map: `maps/1v1/6x6_beginner.csv`
- Games per matchup: 4 (2 per side)
- Max turns per game: 500

**Results:**

| Rank | Bot Name | Wins | Losses | Draws | Total Games | Win Rate |
|------|----------|------|--------|-------|-------------|----------|
| 1    | TBD      | -    | -      | -     | -           | -        |
| 2    | TBD      | -    | -      | -     | -           | -        |
| 3    | TBD      | -    | -      | -     | -           | -        |
| 4    | TBD      | -    | -      | -     | -           | -        |

**Analysis:**
_Tournament analysis will be added after completion._

**Notable Matches:**
_Interesting matchups and key moments will be highlighted here._

---

### Tournament #2 - [Date TBD]

**Configuration:**
- Map: `maps/1v1/10x10_easy.csv`
- Games per matchup: 4 (2 per side)
- Max turns per game: 500

**Results:**

| Rank | Bot Name | Wins | Losses | Draws | Total Games | Win Rate |
|------|----------|------|--------|-------|-------------|----------|
| 1    | TBD      | -    | -      | -     | -           | -        |
| 2    | TBD      | -    | -      | -     | -           | -        |
| 3    | TBD      | -    | -      | -     | -           | -        |
| 4    | TBD      | -    | -      | -     | -           | -        |

**Analysis:**
_Tournament analysis will be added after completion._

---

### Tournament #3 - [Date TBD]

**Configuration:**
- Map: TBD
- Games per matchup: TBD
- Max turns per game: 500

**Results:**

| Rank | Bot Name | Wins | Losses | Draws | Total Games | Win Rate |
|------|----------|------|--------|-------|-------------|----------|
| 1    | TBD      | -    | -      | -     | -           | -        |
| 2    | TBD      | -    | -      | -     | -           | -        |
| 3    | TBD      | -    | -      | -     | -           | -        |
| 4    | TBD      | -    | -      | -     | -           | -        |

**Analysis:**
_Tournament analysis will be added after completion._

---

## Historical Statistics

### Overall Performance (All Tournaments)

| Bot Type    | Total Wins | Total Losses | Total Draws | Overall Win Rate |
|-------------|------------|--------------|-------------|------------------|
| SimpleBot   | TBD        | TBD          | TBD         | TBD              |
| OpenAIBot   | TBD        | TBD          | TBD         | TBD              |
| ClaudeBot   | TBD        | TBD          | TBD         | TBD              |
| GeminiBot   | TBD        | TBD          | TBD         | TBD              |
| Model Bots  | TBD        | TBD          | TBD         | TBD              |

### Head-to-Head Records

Coming soon! This section will show direct matchup statistics between bot types.

## How to Contribute Results

Have you run a tournament? Share your results with the community!

1. Run a tournament using `scripts/tournament.py`
2. Save the results (CSV/JSON files)
3. Submit a pull request with your results
4. Include replay files for verification

## Insights and Analysis

### Strategy Patterns

As tournaments are completed, we'll analyze:
- Opening strategies and unit compositions
- Economic vs. military balance
- Successful tactical patterns
- Common mistakes and pitfalls

### Bot Strengths and Weaknesses

Each bot type has different characteristics:
- **SimpleBot**: Predictable but consistent
- **LLM Bots**: Creative but sometimes unpredictable
- **Model Bots**: Optimized but may overfit to training conditions

### Map-Specific Performance

Different maps favor different strategies:
- Small maps (6x6): Fast, aggressive play
- Medium maps (10x10): Balanced economic/military
- Large maps (24x24+): Long-term strategy and positioning

## Tournament Archive

All tournament replays are saved and can be watched using the game's replay system:

```bash
# Load a replay file
python3 main.py --mode play --replay path/to/replay.json
```

Replay files include:
- Complete action history
- Game metadata (bots, map, duration)
- Can be analyzed programmatically

## Future Tournaments

Upcoming tournament ideas:
- **Map Variety**: Tournaments on different map sizes and layouts
- **Specialized Competitions**: Economy-focused, combat-focused, speed-run
- **Evolution**: Tournament of evolved/improved model versions
- **Human vs. Bot**: Special exhibitions with human players

## Resources

- [Tournament System Guide](./tournament-system.md) - Technical documentation
- [Implementation Status](./implementation-status.md) - Current features
- [Getting Started](./intro.md) - Learn how to play

---

_Last updated: December 2025_

_Tournament results will be updated as official competitions are completed._
