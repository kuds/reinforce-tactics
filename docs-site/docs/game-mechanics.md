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
| **Mage** | M | 300 | 10 | 2 | 8 (adjacent) / 12 (range) | 4 | Can attack at range 1-2, Can PARALYZE enemies for 3 turns (2-turn cooldown) |
| **Cleric** | C | 200 | 8 | 2 | 2 | 4 | Can HEAL allies (+5 HP) at range 1-2, Can CURE paralyzed units at range 1-2 |
| **Archer** | A | 250 | 15 | 3 | 5 | 1 | Ranged attack (2-3 spaces), **+1 range on mountains (2-4)**, Cannot attack adjacent (distance 1), Melee units cannot counter-attack Archers |
| **Knight** | K | 350 | 18 | 4 | 8 | 5 | CHARGE: +50% damage if moved 3+ tiles before attacking |
| **Rogue** | R | 350 | 12 | 4 | 9 | 3 | FLANK: +50% damage if target adjacent to ally, EVADE: 15% dodge (30% in forest) |
| **Sorcerer** | S | 400 | 10 | 2 | 6 (adjacent) / 8 (range) | 3 | Can attack at range 1-2, HASTE: Grant ally extra action, ATTACK/DEFENCE BUFF: +35% damage/reduction for 3 turns |
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
  - Can PARALYZE enemies for 3 turns (2-turn cooldown)
- **Best for**: Disabling key enemy units and dealing ranged damage

#### Cleric (C)
- **Role**: Support and healing
- **Cost**: $200
- **Stats**: 8 HP, 2 Movement, 2 Attack, 4 Defence
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
- **Cost**: $400
- **Stats**: 10 HP, 2 Movement, 6/8 Attack, 3 Defence
- **Abilities**:
  - Attacks at distance 1 (adjacent): 6 damage
  - Attacks at distance 2 (range): 8 damage
  - HASTE: Grant an ally an extra action this turn (3-turn cooldown)
  - DEFENCE BUFF: Give ally -35% damage taken for 3 turns (3-turn cooldown)
  - ATTACK BUFF: Give ally +35% damage dealt for 3 turns (3-turn cooldown)
- **Best for**: Amplifying ally effectiveness and providing tactical flexibility

#### Barbarian (B)
- **Role**: Glass cannon
- **Cost**: $400
- **Stats**: 20 HP, 5 Movement, 10 Attack, 2 Defence
- **Abilities**:
  - High HP and mobility for rapid strikes
- **Best for**: Fast aggressive plays and overwhelming enemies

#### Archer (A)
- **Role**: Long-range attacker
- **Cost**: $250
- **Stats**: 15 HP, 3 Movement, 5 Attack, 1 Defence
- **Abilities**:
  - Ranged attacks at distance 2-3 (cannot attack adjacent enemies at distance 1)
  - **Mountain bonus**: +1 attack range when on mountains (range becomes 2-4)
  - Melee units cannot counter-attack Archers
- **Best for**: Harassing enemies from safe distance, especially effective when positioned on mountains

#### Knight (K)
- **Role**: Mobile heavy cavalry
- **Cost**: $350
- **Stats**: 18 HP, 4 Movement, 8 Attack, 5 Defence
- **Abilities**:
  - **CHARGE**: Deals +50% damage if the Knight moved 3 or more tiles before attacking
  - Good balance of mobility, durability, and damage
- **Best for**: Flanking maneuvers and devastating charge attacks on key targets

#### Rogue (R)
- **Role**: Agile assassin
- **Cost**: $300
- **Stats**: 12 HP, 4 Movement, 9 Attack, 3 Defence
- **Abilities**:
  - **FLANK**: Deals +50% damage when attacking an enemy that is adjacent to one of your other units
  - **EVADE**: 25% chance to completely dodge counter-attacks
  - **Forest bonus**: Evade chance increases to 35% when standing in forest terrain
- **Best for**: Coordinated attacks with other units to maximize flank damage, hit-and-run tactics

#### Sorcerer (S)
- **Role**: Support caster with buffs
- **Cost**: $300
- **Stats**: 10 HP, 2 Movement, 6/8 Attack (adjacent/range), 3 Defence
- **Abilities**:
  - Attacks at distance 1 (adjacent): 6 damage
  - Attacks at distance 2 (range): 8 damage
  - **HASTE**: Grant an ally an extra action this turn (range 1-2, 3 turn cooldown)
  - **DEFENCE BUFF**: Target ally takes 50% less damage for 3 turns (range 1-2, 3 turn cooldown)
  - **ATTACK BUFF**: Target ally deals 50% more damage for 3 turns (range 1-2, 3 turn cooldown)
- **Best for**: Empowering key units with buffs, enabling powerful combos with Haste

#### Barbarian (B)
- **Role**: High-speed berserker
- **Cost**: $400
- **Stats**: 20 HP, 5 Movement, 10 Attack, 2 Defence
- **Abilities**:
  - Highest movement speed in the game (5 tiles)
  - Highest base health (20 HP)
  - Strong attack power (10 damage)
  - Low defence makes them vulnerable to focused fire
- **Best for**: Rapid strikes, chasing down ranged units, and overwhelming enemies with speed

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
- Capturing the enemy's HQ wins the game immediately

#### Win Conditions
- **Capture Enemy HQ**: Capturing the enemy's headquarters wins the game immediately
- **Eliminate All Units**: If a player loses all their units, they lose the game regardless of how many structures they own

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
- Paralysis lasts **3 turns** (`PARALYZE_DURATION = 3`)
- Clerics can cure paralysis with their CURE ability

#### Healing
- Clerics can heal friendly units
- Each heal action restores **5 HP** (`HEAL_AMOUNT = 5`)
- Units cannot be healed above their maximum HP

#### Haste
- Sorcerers can grant Haste to friendly units within range 1-2
- Hasted units can take an additional action this turn
- After using Haste, the Sorcerer has a **3 turn cooldown** (`HASTE_COOLDOWN = 3`)

#### Defence Buff
- Sorcerers can grant Defence Buff to friendly units within range 1-2
- Buffed units take **50% less damage** for 3 turns (`SORCERER_DEFENCE_BUFF_AMOUNT = 0.50`)
- After using Defence Buff, the Sorcerer has a **3 turn cooldown** (`SORCERER_BUFF_COOLDOWN = 3`)

#### Attack Buff
- Sorcerers can grant Attack Buff to friendly units within range 1-2
- Buffed units deal **50% more damage** for 3 turns (`SORCERER_ATTACK_BUFF_AMOUNT = 0.50`)
- After using Attack Buff, the Sorcerer has a **3 turn cooldown** (`SORCERER_BUFF_COOLDOWN = 3`)

### Combat Abilities

#### Charge (Knight)
- Knights deal **+50% damage** when attacking after moving 3 or more tiles (`CHARGE_BONUS = 0.5`)
- The Knight must move at least **3 tiles** (`CHARGE_MIN_DISTANCE = 3`) before attacking to trigger Charge
- Charge bonus is applied automatically when the condition is met

#### Flank (Rogue)
- Rogues deal **+50% damage** when attacking an enemy that is adjacent to one of your other units (`FLANK_BONUS = 0.5`)
- Positioning is key: coordinate your units to enable flanking attacks

#### Evade (Rogue)
- Rogues have a **25% chance** to completely dodge counter-attacks (`ROGUE_EVADE_CHANCE = 0.25`)
- **Forest bonus**: When standing in forest terrain, evade chance increases by 10% to **35%** total (`ROGUE_FOREST_EVADE_BONUS = 0.10`)

## Terrain Bonuses

### Mountain Bonus for Archers
- When an Archer is positioned on a mountain tile, their attack range is extended
- Normal range: 2-3 spaces
- Mountain range: 2-4 spaces (+1 maximum range)
- This bonus is automatically applied when checking attack range and calculating attackable enemies

### Forest Bonus for Rogues
- When a Rogue is positioned on a forest tile, their evade chance is increased
- Normal evade: 25% dodge chance
- Forest evade: 35% dodge chance (+10% bonus)
- This bonus is automatically applied when calculating counter-attack outcomes

## Game Rules

### Victory Conditions
- **Capture enemy HQ**: Win by capturing the opponent's Headquarters
- **Eliminate all enemy units**: Win if the opponent has no units remaining and cannot create new ones

### Turn Structure
1. **Income Phase**: Receive income from owned structures
2. **Action Phase**: Move units, attack enemies, create new units, capture structures
3. **End Turn**: Pass control to the next player

### Movement Rules
- Units can move up to their movement range in Manhattan distance (sum of horizontal and vertical movement)
- Units cannot move through water, ocean, or mountain tiles
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
- **Forest control**: Position Rogues in forests for 35% evasion chance
- **Structure denial**: Capture or destroy enemy structures to starve their economy
- **Unit synergy**: Combine paralysis (Mage) with healing (Cleric) to maintain board control
- **Buff combos**: Use Sorcerer's Attack Buff on Knights before a charge for massive damage
- **Barbarian raids**: Use Barbarians' high movement (5) to quickly capture distant structures or hunt down ranged units
- **Haste combos**: Use Sorcerer's Haste to let units move, attack, and move again for repositioning
