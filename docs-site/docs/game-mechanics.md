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
| **Cleric** | C | 200 | 10 | 3 | 2 | 4 | Can HEAL allies (+7 HP) at range 1-3, Can CURE paralyzed units at range 1-3 |
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
  - Can PARALYZE enemies for 3 turns (2-turn cooldown)
- **Best for**: Disabling key enemy units and dealing ranged damage

#### Cleric (C)
- **Role**: Support and healing
- **Cost**: $200
- **Stats**: 10 HP, 3 Movement, 2 Attack, 4 Defence
- **Abilities**:
  - Can HEAL allies for 7 HP per action (range 1-3)
  - Can CURE paralyzed units (range 1-3)
  - Weak combat capabilities (2 attack)
- **Best for**: Keeping your army healthy and removing status effects

#### Archer (A)
- **Role**: Indirect ranged attacker
- **Cost**: $250
- **Stats**: 15 HP, 3 Movement, 5 Attack, 1 Defence
- **Abilities**:
  - Attacks at distance 2-3 only (2-4 when standing on a mountain); cannot attack adjacent enemies
  - Only Archers, Mages and Sorcerers can counter-attack an Archer
- **Best for**: Wearing down melee units from safety, especially from mountains

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
  - HASTE: Grant an ally an extra action this turn (2-turn cooldown)
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
- Capturing the enemy's HQ wins the game immediately

#### Win Conditions
- **Capture Enemy HQ**: Capturing the enemy's headquarters wins the game immediately
- **Eliminate All Units**: If a player loses all their units, they lose the game regardless of how many structures they own

#### Structure Regeneration
- A partly seized structure recovers when the seizing stops:
  - If the seizing unit moves off it, its HP is restored to full at the end of that player's turn
  - If the seizing unit is killed on it, it regenerates **50% of its max HP** (`STRUCTURE_REGEN_RATE = 0.5`) at the end of every turn until it is back to full; a unit stepping onto it stops the regeneration
- A captured structure starts at full HP under its new owner

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
- Damage calculation involves both the attacker's attack stat and the defender's defence stat (each point of defence reduces damage by 5%, at most 90%)
- A defender that survives and can reach its attacker counter-attacks

### Counter-Attack Mechanics
- A surviving defender counter-attacks if the attacker is within its own attack range (a Warrior hit by a Mage from 2 tiles away cannot)
- Counter-attacks deal **80% of normal damage** (`COUNTER_ATTACK_MULTIPLIER = 0.8`)
- Archers are special: only Archers, Mages and Sorcerers can counter-attack an Archer
- Units cannot counter-attack if they are paralyzed

### Status Effects

#### Paralysis
- Mages can paralyze enemy units
- Paralyzed units cannot move or attack
- Paralysis lasts **3 turns** (`PARALYZE_DURATION = 3`)
- Clerics can cure paralysis with their CURE ability

#### Healing
- Clerics can heal friendly units within range 1-3 (`CLERIC_HEAL_RANGE = 3`)
- Each heal action restores **7 HP** (`HEAL_AMOUNT = 7`)
- Units cannot be healed above their maximum HP

#### Haste
- Sorcerers can grant Haste to friendly units within range 1-2
- Hasted units can take an additional action this turn (can move, act, then move and act again)
- After using Haste, the Sorcerer has a **2 turn cooldown** (`HASTE_COOLDOWN = 2`)

#### Defence Buff
- Sorcerers can grant Defence Buff to friendly units within range 0-2 (including themselves)
- Buffed units take **50% less damage** for 3 turns (`SORCERER_DEFENCE_BUFF_AMOUNT = 0.50`)
- After using Defence Buff, the Sorcerer has a **2 turn cooldown** (`SORCERER_BUFF_COOLDOWN = 2`)

#### Attack Buff
- Sorcerers can grant Attack Buff to friendly units within range 0-2 (including themselves)
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

### Movement Cost
- By default every walkable tile (grass, road, forest, mountain and structures) costs 1 movement, so roads are not faster and forests and mountains are not slower
- Terrain movement costs can be switched on as an optional rule (below)

## Optional Rules

These rules are set per game through `engine_overrides` (in a training config's `env.engine_overrides`, or `GameState(..., engine_overrides=...)`). All of them are **off by default**, so the default game is exactly as described above. A game's overrides are stored in its saves and in a training run's `config.json`.

| Key | Default | Effect when set |
|-----|---------|-----------------|
| `terrain_move_cost` | every tile costs 1 | Movement points to enter each tile type, e.g. `{"r": 0.5, "f": 2, "m": 2}`. A unit may enter a tile while its path's total cost stays within its movement, so roads at 0.5 double the distance travelled along them. Only walkable tile codes (`p`, `r`, `f`, `m`, `b`, `h`, `t`) and positive costs are accepted |
| `charge_distance` | `"displacement"` | `"path"`: the Knight's Charge counts the tiles along the path of the move (going around a lake counts) instead of the straight-line distance from where the Knight started |
| `forest_concealment` | `false` | Under fog of war, a unit standing in forest is seen only by an enemy with a unit on or orthogonally next to its tile; the forest tile itself still counts as explored |
| `hq_always_visible` | `false` | Under fog of war, every HQ counts as explored for every player, so its position and owner are always known; units on it still need normal vision |

## Game Rules

### Victory Conditions
- **Capture enemy HQ**: Win by capturing the opponent's Headquarters
- **Eliminate all enemy units**: Win if the opponent has no units remaining and cannot create new ones

### Turn Structure
1. **Income Phase**: Receive income from owned structures
2. **Action Phase**: Move units, attack enemies, create new units, capture structures
3. **End Turn**: Pass control to the next player

### Movement Rules
- Units move up to their movement range in orthogonal steps (each walkable tile costs 1 movement by default; see Optional Rules)
- Units cannot move through water or ocean tiles
- Units can move through friendly units but not through enemies, and must end their move on an empty tile
- After moving, units can still attack if they haven't attacked yet

### Attack Rules
- Units can attack without moving, or move first and then attack (once per turn)
- Each unit type has specific attack range requirements
- Attacking, using an ability or seizing ends that unit's turn: it cannot move afterwards

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
