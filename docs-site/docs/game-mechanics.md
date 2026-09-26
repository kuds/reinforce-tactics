---
sidebar_position: 2
id: game-mechanics
title: Game Mechanics
---

# Game Mechanics

This page provides detailed information about the game mechanics, units, structures, and combat system in Reinforce Tactics.

## Units

Reinforce Tactics features **8 distinct unit types**, each with unique abilities and tactical roles:

### Unit Statistics

| Unit | Code | Cost | Health | Movement | Attack | Defence | Special Abilities |
|------|------|------|--------|----------|--------|---------|-------------------|
| **Warrior** | W | 200 | 15 | 3 | 10 | 6 | Melee only (range 1) |
| **Mage** | M | 300 | 10 | 2 | 8 (adjacent) / 12 (range) | 4 | Can attack at range 1-2, Can PARALYZE enemies: they lose their next 2 turns (2-turn cooldown) |
| **Cleric** | C | 200 | 10 | 3 | 2 | 4 | Can HEAL allies (+5 HP) at range 1-2, Can CURE paralyzed units at range 1-2 |
| **Archer** | A | 250 | 15 | 3 | 5 | 1 | Ranged attack (2-3 spaces), **+1 range on mountains (2-4)**, Cannot attack adjacent (distance 1), Melee units cannot counter-attack Archers |
| **Knight** | K | 350 | 18 | 4 | 8 | 5 | CHARGE: +50% damage if moved 3+ tiles before attacking |
| **Rogue** | R | 350 | 12 | 4 | 9 | 3 | FLANK: +50% damage if target adjacent to ally, EVADE: 15% dodge (30% in forest) |
| **Sorcerer** | S | 350 | 12 | 2 | 6 (adjacent) / 8 (range) | 3 | Can attack at range 1-2, HASTE: Grant ally extra action, ATTACK/DEFENCE BUFF: +50% damage/reduction for 3 turns (2-turn cooldown) |
| **Barbarian** | B | 400 | 20 | 5 | 10 | 2 | High HP glass cannon with excellent mobility |

### Unit Details

#### Warrior (W)
- **Role**: Frontline melee fighter
- **Cost**: $200
- **Stats**: 15 HP, 3 Movement, 10 Attack, 6 Defence
- **Abilities**: Standard melee attacks at range 1
- **Best for**: Defending positions and engaging in close combat

#### Mage (M)
- **Role**: Ranged attacker with crowd control
- **Cost**: $300
- **Stats**: 10 HP, 2 Movement, 8/12 Attack, 4 Defence
- **Abilities**:
  - Attacks at distance 1 (adjacent): 8 damage
  - Attacks at distance 2 (range): 12 damage
  - Can PARALYZE enemies: the target loses its next 2 turns (2-turn cooldown)
- **Best for**: Disabling key enemy units and dealing ranged damage

#### Cleric (C)
- **Role**: Support and healing
- **Cost**: $200
- **Stats**: 10 HP, 3 Movement, 2 Attack, 4 Defence
- **Abilities**:
  - Can HEAL allies for 5 HP per action (range 1-2)
  - Can CURE paralyzed units (range 1-2)
  - Weak combat capabilities (2 attack)
- **Best for**: Keeping your army healthy and removing status effects

#### Knight (K)
- **Role**: Heavy cavalry
- **Cost**: $350
- **Stats**: 18 HP, 4 Movement, 8 Attack, 5 Defence
- **Abilities**:
  - CHARGE: +50% damage if moved 3+ tiles before attacking
- **Best for**: Rapid flanking maneuvers and high-impact charges

#### Rogue (R)
- **Role**: Fast melee assassin
- **Cost**: $350
- **Stats**: 12 HP, 4 Movement, 9 Attack, 3 Defence
- **Abilities**:
  - FLANK: +50% damage if target is adjacent to another friendly unit
  - EVADE: 15% chance to dodge counter-attacks (30% in forest)
- **Best for**: Coordinated attacks with allies and hit-and-run tactics

#### Sorcerer (S)
- **Role**: Support caster with buffs
- **Cost**: $350
- **Stats**: 12 HP, 2 Movement, 6/8 Attack, 3 Defence
- **Abilities**:
  - Attacks at distance 1 (adjacent): 6 damage
  - Attacks at distance 2 (range): 8 damage
  - HASTE: Grant one of your own units an extra full action this turn (2-turn cooldown)
  - DEFENCE BUFF: Give ally -50% damage taken for 3 turns (2-turn cooldown)
  - ATTACK BUFF: Give ally +50% damage dealt for 3 turns (2-turn cooldown)
- **Best for**: Amplifying ally effectiveness and providing tactical flexibility

#### Barbarian (B)
- **Role**: Glass cannon
- **Cost**: $400
- **Stats**: 20 HP, 5 Movement, 10 Attack, 2 Defence
- **Abilities**:
  - High HP and mobility for rapid strikes
- **Best for**: Fast aggressive plays and overwhelming enemies

## Structures

Structures provide income and serve as strategic objectives. They can be captured by enemy units.

### Structure Statistics

| Structure | Code | Max Health | Income/Turn |
|-----------|------|------------|-------------|
| **Headquarters (HQ)** | h | 50 HP | $150 |
| **Building** | b | 40 HP | $100 |
| **Tower** | t | 30 HP | $50 |

### Structure Mechanics

#### Capturing Structures
- Units can capture enemy structures by standing on them and using the "Seize" action
- Each turn the unit seizes, the structure takes damage equal to the unit's current HP
- Once the structure's HP reaches 0, it is captured and becomes owned by the capturing player
- Capturing an enemy HQ wins the game immediately in a game with two sides (1v1, 2v2); in a free-for-all it eliminates the HQ's owner once it has no HQ left
- You cannot seize a teammate's structure

#### Win Conditions
- **Capture Enemy HQ**: With two sides, capturing an enemy headquarters wins the game for your side immediately
- **Eliminate All Units**: If a player loses all their units, they are eliminated regardless of how many structures they own
- **Free-for-all** (three or more sides, e.g. 1v1v1): losing your last HQ, your last unit or resigning eliminates you, and the game goes on until one side is left

#### Structure Regeneration
- Abandoned structures (not owned by any player) regenerate **50% of their max HP per turn**
- This regeneration rate is defined by `STRUCTURE_REGEN_RATE = 0.5`
- Regeneration stops once a structure is captured by a player

#### Unit Creation
- Units can be created at owned **Buildings** only
- The creating structure must not have a unit standing on it
- Costs are deducted from the player's gold reserves

#### Income Generation
- At the start of each player's turn, they receive income from all owned structures:
  - Headquarters: $150 per turn
  - Buildings: $100 per turn
  - Towers: $50 per turn
- Players start with **$250** in gold (`STARTING_GOLD = 250`)

## Combat System

### Basic Combat
- Units can attack enemy units within their attack range
- Damage calculation involves both the attacker's attack stat and the defender's defence stat
- When a melee unit attacks another melee unit, the defender can counter-attack

### Counter-Attack Mechanics
- Counter-attacks occur when a melee unit (range 1) is attacked by another melee unit
- Counter-attacks deal **80% of normal damage** (`COUNTER_ATTACK_MULTIPLIER = 0.8`)
- Archers are special: melee units **cannot counter-attack Archers** even when attacked
- Units cannot counter-attack if they are paralyzed

### Status Effects

#### Paralysis
- Mages can paralyze enemy units
- Paralyzed units cannot move or attack
- The target loses its **next 2 turns** (`PARALYZE_DURATION = 2`) and is free again when its third turn starts
- A paralyzed unit cannot counter-attack, from the moment it is paralyzed until its first free turn starts. This includes the paralyzer's turn just before that. It cannot be paralyzed again in that window either.
- Clerics can cure paralysis with their CURE ability (the cured unit can act at once if it is its turn)
- Status durations always count the affected unit's own turns. Internally, the per-unit counter counts that unit's turn starts until the status ends, and it ticks as its owner's turn starts. A paralysis is cast during the victim's opponent's turn, so it is stored as `PARALYZE_DURATION + 1`. This is the rule the game has always played by; before September 2026 the constant was 3, the stored value.

#### Healing
- Clerics can heal friendly units
- Each heal action restores **5 HP** (`HEAL_AMOUNT = 5`)
- Units cannot be healed above their maximum HP

#### Haste
- Sorcerers can grant Haste to their own units within range 1-2. A paralyzed unit or a teammate's unit can't be hasted: neither could act on your turn.
- A hasted unit gets **one extra full action** this turn. When it spends its action by attacking, using an ability or seizing (or choosing Wait in the GUI), it is refreshed and may move and act once more. A unit that has already acted when it is hasted is refreshed at once.
- The engine applies this rule itself, so it plays the same for human players, scripted and LLM bots, RL agents and MCTS. The RL action space has no Wait, so there a hasted unit's first action ends only when it acts.
- After using Haste, the Sorcerer has a **2 turn cooldown** (`HASTE_COOLDOWN = 2`)

#### Defence Buff
- Sorcerers can grant Defence Buff to friendly units within range 1-2
- Buffed units take **50% less damage** for 3 turns (`SORCERER_DEFENCE_BUFF_AMOUNT = 0.50`)
- After using Defence Buff, the Sorcerer has a **2 turn cooldown** (`SORCERER_BUFF_COOLDOWN = 2`)

#### Attack Buff
- Sorcerers can grant Attack Buff to friendly units within range 1-2
- Buffed units deal **50% more damage** for 3 turns (`SORCERER_ATTACK_BUFF_AMOUNT = 0.50`)
- After using Attack Buff, the Sorcerer has a **2 turn cooldown** (`SORCERER_BUFF_COOLDOWN = 2`)

### Combat Abilities

#### Charge (Knight)
- Knights deal **+50% damage** when attacking after moving 3 or more tiles (`CHARGE_BONUS = 0.5`)
- The Knight must move at least **3 tiles** (`CHARGE_MIN_DISTANCE = 3`) before attacking to trigger Charge
- Charge bonus is applied automatically when the condition is met

#### Flank (Rogue)
- Rogues deal **+50% damage** when attacking an enemy that is adjacent to one of your other units (`FLANK_BONUS = 0.5`)
- Positioning is key: coordinate your units to enable flanking attacks

#### Evade (Rogue)
- Rogues have a **15% chance** to completely dodge counter-attacks (`ROGUE_EVADE_CHANCE = 0.15`)
- **Forest bonus**: When standing in forest terrain, evade chance increases by 15% to **30%** total (`ROGUE_FOREST_EVADE_BONUS = 0.15`)

## Terrain Bonuses

### Mountain Bonus for Archers
- When an Archer is positioned on a mountain tile, their attack range is extended
- Normal range: 2-3 spaces
- Mountain range: 2-4 spaces (+1 maximum range)
- This bonus is automatically applied when checking attack range and calculating attackable enemies

### Forest Bonus for Rogues
- When a Rogue is positioned on a forest tile, their evade chance is increased
- Normal evade: 15% dodge chance
- Forest evade: 30% dodge chance (+15% bonus)
- This bonus is automatically applied when calculating counter-attack outcomes

## Game Rules

### Victory Conditions
- **Capture enemy HQ**: With two sides (1v1, 2v2), capturing an enemy Headquarters wins the game for your side
- **Eliminate all enemy units**: A player whose last unit dies is eliminated; a side wins when every other side is eliminated
- **Resign**: A player who resigns is eliminated
- **Free-for-all** (three or more sides): capturing a player's last HQ eliminates that player, and the last side standing wins

### Teams
- A map declares teams on each player's HQ tile code as `h_<player>_<team>`, e.g. `h_3_1` is player 3's HQ on team 1. Other structures may repeat the suffix but must agree. Code can also pass `GameState(teams={player: team})`; a conflict with the map raises `ValueError`.
- Without any declaration every player is its own side (free-for-all), so 1v1 and 1v1v1 maps need no change.
- Teammates are allies for every rule. They never attack, paralyze or seize each other's units and structures, and they move through each other's units. A teammate standing next to a target counts for a Rogue's flank. Clerics can heal and cure a teammate's units, and Sorcerers can buff them; a buff lasts 3 of that unit's own turns. Haste targets only your own units.
- Income, gold, unit creation, auto-heal and vision stay per player.
- The bundled 2v2 map plays players 1 and 3 (top-left, bottom-right) against players 2 and 4 (top-right, bottom-left), so turns alternate between the teams. The 2v2 mode gives those teams to a map that declares none, such as a random map.

### Elimination
- An eliminated player's units are removed and its structures turn neutral. Structure HP is kept, so a unit that was seizing one keeps its progress. A neutral HQ is then an ordinary structure: capturing it eliminates nobody.
- Eliminated players are skipped in the turn order, so they get no turns, income or new units. A player eliminated during its own turn (resigning, or losing its last unit to a counter-attack) has nothing left to do but end the turn; the GUI ends it for a player who resigns.
- In games with more than two seats every elimination is recorded as an `eliminate` action. Replays and saves reproduce eliminations, and saves record the set as `eliminated_players`.
- When the last opposing side is eliminated, the game ends and the board is left as it stands.

### Turn Structure
1. **Start of turn**: Status effects and cooldowns tick, units become ready (a unit still paralyzed stays disabled), income is collected, units on owned structures auto-heal, and fog of war updates for the player. Player 1's very first turn (turn 0) skips this step and is played on starting gold, while Player 2 collects income before its first move. The engine override `begin_first_turn: true` runs the step for Player 1's first turn too, for balance sweeps; it is recorded in saves and replays.
2. **Action Phase**: Move units, attack enemies, create new units, capture structures
3. **End Turn**: Pass control to the next player still in the game

A move can be cancelled until the unit acts: the GUI's Cancel Move, ESC or right-click. The unit returns to where its action started, and the move leaves no trace in the action log. Under fog of war, what the move revealed is hidden again, and only the latest action can be taken back, so moving and cancelling cannot be used to scout.

### Movement Rules
- Units can move up to their movement range in Manhattan distance (sum of horizontal and vertical movement)
- Units cannot move through water or ocean tiles
- Units cannot move through other units (friendly or enemy)
- After moving, units can still attack if they haven't attacked yet

### Attack Rules
- Units can attack before or after moving (but only once per turn)
- Each unit type has specific attack range requirements
- Attacking or moving ends that unit's turn (it cannot move after attacking, and vice versa)

## Strategic Tips

### Unit Composition
- **Balance your army**: Mix melee units (Warriors, Knights, Barbarians) with ranged units (Mages, Archers, Sorcerers) and support (Clerics)
- **Protect support units**: Keep Clerics, Sorcerers, and Archers behind your frontline
- **Use terrain**: Position Archers on mountains for extended range, and Rogues in forests for increased evasion

### Economy Management
- **Capture structures early**: More income means more units
- **Protect your structures**: Losing income puts you at a disadvantage
- **Consider unit costs**: Warriors and Clerics ($200) are cost-effective; Knights ($350) and Barbarians ($400) are expensive but powerful

### Combat Tactics
- **Use Mages to disable**: Paralyze key enemy units before engaging
- **Heal efficiently**: Keep your units healthy with Clerics to maximize their effectiveness
- **Archer positioning**: Keep Archers at range 2-3 to avoid counter-attacks
- **Counter-attack advantage**: Let enemies attack into your defensive positions when possible
- **Knight charges**: Position Knights far from enemies to maximize charge damage (+50%)
- **Rogue flanking**: Coordinate your units to enable Rogue flanking attacks (+50% damage)
- **Sorcerer buffs**: Use Haste on high-value units for devastating double actions

### Advanced Tactics
- **Mountain control**: Fight for mountain tiles to give your Archers extended range
- **Forest control**: Position Rogues in forests for 30% evasion chance
- **Structure denial**: Capture or destroy enemy structures to starve their economy
- **Unit synergy**: Combine paralysis (Mage) with healing (Cleric) to maintain board control
- **Buff combos**: Use Sorcerer's Attack Buff on Knights before a charge for massive damage
- **Barbarian raids**: Use Barbarians' high movement (5) to quickly capture distant structures or hunt down ranged units
- **Haste combos**: Use Sorcerer's Haste to let units move, attack, and move again for repositioning
