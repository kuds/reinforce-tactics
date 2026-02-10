# Reinforce Tactics -- Turn-Based Tactical Strategy for LLM PvP

Hi everyone! I'd like to share **Reinforce Tactics**, a turn-based tactical strategy game designed to benchmark LLM reasoning in PvP scenarios.

**Task link:** [TODO: insert Kaggle task link here]

## What is it?

Reinforce Tactics is a grid-based strategy game (think Fire Emblem / Advance Wars) where two players compete by recruiting units, capturing structures, and battling for control of the map. The game ends when a player captures the enemy HQ or eliminates all opposing units.

- GitHub: https://github.com/kuds/reinforce-tactics
- Docs: https://reinforcetactics.com

## Why it's a strong LLM benchmark

**Deep decision space.** Each turn, an LLM must decide: which units to recruit, where to move them, who to attack, and when to use special abilities -- all within a shared economy of limited gold. There's no single dominant strategy; every decision involves trade-offs.

**8 asymmetric unit types with interacting abilities.** Warriors, Mages, Clerics, Archers, Knights, Rogues, Sorcerers, and Barbarians each have fundamentally different mechanics:
- Mages can **paralyze** enemies (disabling them for 3 turns)
- Knights deal **+50% charge damage** when moving 3+ tiles before attacking
- Rogues get **+50% flank damage** when the target is adjacent to an ally
- Sorcerers can cast **haste** (granting an extra action), or **buff** allies' attack/defense
- Clerics **heal** and **cure** paralysis
- Archers gain **extended range on mountains**; Rogues gain **extra evasion in forests**

Winning requires understanding how these abilities combine -- a Sorcerer buffing a charging Knight creates a devastating combo, but leaves the Sorcerer exposed. LLMs need to reason about synergies, cooldowns, and positioning simultaneously.

**Multi-dimensional strategic reasoning.** The game tests several cognitive capabilities at once:
- **Spatial reasoning**: Movement paths, attack ranges, terrain bonuses, flanking positions
- **Resource management**: Balancing gold income from captured structures vs. spending on units
- **Temporal planning**: Cooldown tracking (paralyze, haste, buffs), multi-turn positioning
- **Opponent modeling**: Predicting enemy moves, defending key structures, controlling map space
- **Risk assessment**: Counter-attack damage, evasion chances, unit vulnerability when paralyzed

**Economic layer adds depth.** Structures (HQ, Buildings, Towers) generate gold each turn. Controlling more structures means more income, which means more units. But over-extending to capture structures can leave your HQ vulnerable. LLMs must balance aggression with defense.

## Built for LLM evaluation

- **Multi-provider support**: GPT, Claude, and Gemini bots out of the box
- **Tournament system**: Run round-robin tournaments between LLMs with Docker, parallel game execution, and full conversation logging
- **Structured I/O**: Game state is sent as JSON; LLMs respond with structured action lists
- **Reasoning mode**: Optional chain-of-thought reasoning to analyze *how* models think about tactics
- **Two-phase planning**: Optional plan-then-execute mode to study LLM planning vs. execution quality
- **Multiple maps**: Varied map layouts test adaptability across different tactical scenarios
- **Fog of War**: Optional partial observability for even harder reasoning challenges
- **Gymnasium integration**: Standard RL environment API for comparing LLMs against trained RL agents

## What makes it unique for Game Arena

Unlike simpler PvP games (tic-tac-toe, connect-four, poker), Reinforce Tactics creates a rich tactical environment where LLMs must reason about:
1. **Combinatorial unit interactions** (8 unit types x terrain bonuses x status effects x cooldowns)
2. **Economy and tempo** (when to invest in units vs. when to push for a win)
3. **Spatial positioning across a 2D grid** (not just abstract choices, but physical map control)
4. **Long-horizon planning** (games last 20-50+ turns with compounding consequences)

Early tournament results show meaningful differentiation between models -- stronger reasoners consistently make better use of unit synergies, terrain advantages, and economic timing.

Looking forward to feedback and seeing how different LLMs perform in the arena!
