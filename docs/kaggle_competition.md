# Reinforce Tactics

*A two-player, turn-based tactical strategy game for the [Kaggle Environments](https://github.com/Kaggle/kaggle-environments) framework.*

---

## Overview

Command an army on a grid battlefield. Recruit units, march them across forests
and mountains, trade blows, and storm the enemy headquarters. You won't play by
hand — you'll write an **agent**: a Python function that reads the board each
turn and returns a list of orders. The agent that out-thinks its opponent over
many matches climbs the leaderboard.

Two armies start from opposite corners with a little gold and no units. From
there it's a race to build an economy, win the fights that matter, and **either
capture the enemy Headquarters or wipe out every enemy unit** before the turn
limit. Simple win conditions, deep tactics: terrain, eight distinct unit types,
HP-scaled combat, paralysis, buffs, and structure sieges all interact.

> **At a glance**
>
> | | |
> |---|---|
> | **Players** | 2 (your agent vs. one opponent) |
> | **Format** | Turn-based; players alternate submitting a *list* of orders |
> | **Goal** | Capture the enemy HQ **or** eliminate every enemy unit |
> | **Episode length** | 200 turns by default, then a draw |
> | **Score** | `+1` win · `0` draw · `-1` loss, fed into a skill-rating leaderboard |
> | **Limits (default)** | 5 s per turn · 1200 s per episode (enforced by the Kaggle harness; host-configurable) |
> | **Engine** | Self-contained, no pygame/UI deps — vendored like Lux AI S3 |

### What you'll build

Your submission is a single function. It's handed the current `observation` and
the match `configuration`, and it returns the orders to issue this turn:

```python
def my_agent(observation, configuration):
    actions = []
    # ... read the board, decide your moves ...
    actions.append({"type": "end_turn"})   # always finish with end_turn
    return actions
```

### Quick start

```python
from kaggle_environments import make

env = make("reinforce_tactics", configuration={"mapName": "beginner"})
env.run([my_agent, "simple_bot"])     # play a match against the strategic baseline
print(env.render(mode="ansi"))         # ASCII view of the board
```

The full game systems are in **Description** below; how submissions are ranked
is in **Evaluation**.

---

## Description

### 1. Turn flow & orders

Players act in strict alternation. On your turn you submit a **list of orders**,
all resolved in the order you give them, then control passes to your opponent. A
unit can normally **move once and act once per turn** (Haste is the exception).

Each order is a dict with a `"type"` key. The legal order types are:

| Order | Fields | Effect |
|---|---|---|
| `create_unit` | `unit_type`, `x`, `y` | Recruit a unit on one of **your Buildings** (costs gold) |
| `move` | `from_x`, `from_y`, `to_x`, `to_y` | Move a unit within its movement range |
| `attack` | `from_x`, `from_y`, `to_x`, `to_y` | Strike an enemy in range |
| `seize` | `x`, `y` | Chip away at / capture the structure your unit stands on |
| `heal` | `from_x`, `from_y`, `to_x`, `to_y` | Cleric restores an ally's HP |
| `cure` | `from_x`, `from_y`, `to_x`, `to_y` | Cleric removes paralysis from an ally |
| `paralyze` | `from_x`, `from_y`, `to_x`, `to_y` | Mage stuns an enemy |
| `haste` | `from_x`, `from_y`, `to_x`, `to_y` | Sorcerer lets an ally act again |
| `defence_buff` / `attack_buff` | `from_x`, `from_y`, `to_x`, `to_y` | Sorcerer buffs an ally |
| `end_turn` | *(none)* | Pass control to the opponent |

**Illegal orders are forgiving.** A well-formed but illegal order (unaffordable,
occupied tile, out of range) is simply **skipped as a no-op** — one bad order in
your list does **not** forfeit the game. Only a *malformed* order (not a dict)
counts as a broken agent and loses. Coordinates are `(x, y)` with `x` = column,
`y` = row, origin top-left.

### 2. Economy

Each player starts with **250 gold** and **no units**. At the start of every
turn you collect **income** from each structure you own:

| Structure | Income / turn |
|---|---|
| Headquarters (`h`) | **150** |
| Building (`b`) | **100** |
| Tower (`t`) | **50** |

Gold is spent recruiting units on your **Buildings** (not the HQ). The recruit
tile must be empty, and you may hold at most **50 units** at once. Your turn
income is therefore

$$\text{income} = 150\,n_h + 100\,n_b + 50\,n_t$$

where $n_h, n_b, n_t$ are the headquarters, buildings, and towers you currently
control. **Map control is economy** — every neutral tower you seize is a
permanent +50/turn and one fewer for the enemy.

### 3. The map & terrain

Maps are 2-D grids of single-character tile codes. Terrain changes how units
move, see, and fight:

| Tile | Code | Effect |
|---|---|---|
| Grass | `p` | Open ground, normal movement |
| Road | `r` | Normal movement |
| Forest | `f` | **+15% evasion** for Rogues (stacks to 30%) and **blocks ranged line-of-sight** |
| Mountain | `m` | Walkable; occupant gains **+1 vision**, and Archers gain **+1 attack range** |
| Water | `w` | **Impassable** |
| Ocean | `o` | Impassable border (small maps are padded to 20×20 with ocean) |
| Building | `b` | Capturable; gives income and recruits units |
| Tower | `t` | Capturable; gives income and a defensive perch |
| Headquarters | `h` | Capturable; **lose it and you lose the game** |

Structure tiles carry an owner suffix in the raw map (`h_1` = Player 1's HQ,
`b_2` = Player 2's building, bare `t` = neutral tower).

You can play on one of **19 built-in 1v1 maps** (e.g. `beginner`, `crossroads`,
`tower_rush`, `mountain_snipers`) or on a **randomly generated** map by setting
`mapName=""` with a `mapSeed` for reproducibility.

### 4. The roster

Eight unit types, each with a distinct role. **These are the competition
values** — see §9 for what differs from the raw engine defaults.

| Code | Unit | Cost | HP | ATK | DEF | Move | Range | Signature ability |
|---|---|---:|---:|---|---:|---:|---|---|
| `W` | Warrior | 300 | 15 | 10 | 6 | 3 | 1 | Durable, cost-efficient frontline |
| `M` | Mage | 300 | 10 | 8 / **12** | 4 | 2 | 1–2 | **Paralyze** (range 1–2, 3 turns) |
| `C` | Cleric | 200 | 10 | 2 | 4 | 3 | 1 | **Heal +7** & **Cure** (range 1–3) |
| `A` | Archer | 250 | 15 | 5 | 1 | 3 | 2–3 | Ranged; **rarely countered** |
| `K` | Knight | 350 | 18 | 8 | 5 | 4 | 1 | **Charge**: ×1.5 dmg after moving ≥3 |
| `R` | Rogue | 350 | 12 | 9 | 3 | 4 | 1 | **Flank**: ×1.5 dmg; 15%/30% evasion |
| `S` | Sorcerer | 350 | 12 | 6 / **8** | 3 | 2 | 1–2 | **Haste** + Attack/Defence buffs |
| `B` | Barbarian | 400 | 20 | 10 | 2 | 5 | 1 | Fast, high-HP shock trooper |

Notes on ranged units:

- **Mage / Sorcerer** hit at distance 1 *or* 2; their **distance-2 strike is
  stronger** (Mage 12 vs 8, Sorcerer 8 vs 6) and dodges the melee counter.
- **Archer** *cannot* hit adjacent (distance 1) — it attacks at distance
  **2–3** (2–**4** from a mountain) and is almost never counter-attacked.

### 5. Combat

When unit $a$ attacks target $t$, damage is computed in four stages. Let
$\text{ATK}_a$ be the attacker's base attack for that distance,
$\text{hp}_a/\text{HP}_a$ its current HP fraction, and $\text{DEF}_t$ the
target's defence.

**Stage 1 — HP-scaled base.** The competition uses the **`hp_scaled`** damage
model: a unit's output scales with how healthy it is.

$$B \;=\; \text{ATK}_a \cdot \frac{\text{hp}_a}{\text{HP}_a}$$

**Stage 2 — ability multipliers.** Each active bonus multiplies damage by
**1.5** (Knight *Charge* after moving ≥3 tiles; Rogue *Flank* when the target is
adjacent to another of your units; Sorcerer *Attack Buff*):

$$B' \;=\; B \cdot m_\text{charge}\cdot m_\text{flank}\cdot m_\text{atkbuff}, \qquad m_\bullet\in\{1,\,1.5\}$$

**Stage 3 — defence mitigation.** Each defence point removes 5% damage, capped
at 90%:

$$\rho \;=\; \min\!\big(0.05\cdot \text{DEF}_t,\; 0.9\big), \qquad D \;=\; \max\!\big(1,\; \lfloor B'(1-\rho)\rfloor\big)$$

**Stage 4 — defensive buff.** If the target has a Sorcerer *Defence Buff*, halve
again:

$$D_\text{final} \;=\; \max\!\big(1,\; \lfloor 0.5\,D\rfloor\big) \quad\text{(else } D_\text{final}=D)$$

#### Counter-attacks

If the target survives, isn't paralyzed, and is allowed to retaliate, it hits
back at **80%** strength, again HP-scaled and run through the *attacker's*
defence:

$$C \;=\; \max\!\Big(1,\;\Big\lfloor\, \text{ATK}_t\cdot \tfrac{\text{hp}_t}{\text{HP}_t}\cdot 0.8 \cdot m_\text{atkbuff}\,\Big\rfloor \cdot (1-\rho_a)\Big)$$

Counters are **denied** when:

- the **attacker is an Archer** and the target is not an Archer/Mage/Sorcerer
  (archers strike with impunity);
- the **target is paralyzed**;
- the attacker is a **Rogue** and a random evade roll $u < p_\text{evade}$
  succeeds, with $p_\text{evade}=0.15$ (**0.30 in forest**).

> **Why HP-scaling matters.** The counter is computed *after* the target absorbs
> your hit, so a freshly-wounded defender strikes back for less. Wounded units
> both *deal* and *survive* less — so **focus-firing one enemy to death is
> decisive**, and the even-trade stalemates of flat damage disappear. This is
> the single most important strategic consequence of the `hp_scaled` model.

### 6. Seizing structures

To capture a structure, stand a unit on it and issue `seize`. Each seize
subtracts the **unit's current HP** from the structure's HP:

$$\text{HP}^{\text{struct}} \;\leftarrow\; \text{HP}^{\text{struct}} - \text{hp}_u$$

When it reaches **0**, the structure flips to your control — and **seizing an
enemy HQ ends the game**. Structure HP pools:

| Structure | Max HP | Seizes for a full-HP Warrior (15 HP) |
|---|---:|---|
| Tower | 30 | 2 |
| Building | 40 | 3 |
| Headquarters | 50 | 4 |

Because capture scales with the seizer's HP, a **healthy unit captures faster**,
and stacking the assault matters. If a partly-damaged structure's occupant is
**killed**, the now-undefended structure **regenerates 50% of its max HP per
turn** until a unit stands on it again — so an interrupted siege can heal back
fast. Plan to capture with bodies to spare.

### 7. Abilities & status effects

| Ability | Caster | Range | Effect | Cooldown |
|---|---|---|---|---|
| **Paralyze** | Mage | 1–2 | Target can't move, act, or counter for **3 turns** | 2 turns |
| **Cure** | Cleric | 1–3 | Removes paralysis from an ally | — |
| **Heal** | Cleric | 1–3 | Restores **+7 HP** to an ally | — |
| **Haste** | Sorcerer | 1–2 | Ally may **move and act again** this turn | 2 turns |
| **Attack Buff** | Sorcerer | 1–2 | Ally deals **+50%** for 3 turns | 2 turns |
| **Defence Buff** | Sorcerer | 1–2 | Ally takes **−50%** for 3 turns | 2 turns |

Paralyze is a tempo weapon (a stunned unit can't even counter); Haste enables
double-moves for reach or burst; buffs swing a key fight. Healing/curing feeds
directly back into the HP-scaling loop — keeping a unit topped up keeps it
hitting at full power.

### 8. Fog of war (optional)

When `fogOfWar` is enabled, the **`units`** list in your observation is filtered
to only the enemies your units can currently see; the **board and structures
stay fully visible**. Mountains grant +1 vision and forests block ranged
line-of-sight, so scouting and positioning carry information value. Fog is
**off** by default.

### 9. Configuration & competition balance

The environment is created with `make("reinforce_tactics", configuration={...})`:

| Parameter | Default | Description |
|---|---|---|
| `episodeSteps` | 200 | Max turns before a draw |
| `mapName` | `"beginner"` | Built-in map name, or `""` for random generation |
| `mapWidth` / `mapHeight` | 20 / 20 | Random-map size (10–40), used only when `mapName=""` |
| `mapSeed` | -1 | Random-map seed (-1 = random) |
| `enabledUnits` | `W,M,C,A,K,R,S,B` | Which unit types may be recruited |
| `fogOfWar` | `false` | Hide enemy units outside vision |
| `startingGold` | 250 | Gold each player begins with |

> **⚠️ Competition balance.** The competition environment applies two balance
> changes on top of the base engine, and your agent should assume them:
>
> - **Warrior cost is 300** (raised from the engine default of 200).
> - **Combat uses the `hp_scaled` damage model** (wounded units hit softer — see
>   §5).
>
> All other unit stats in §4 are engine defaults.

### 10. The agent interface

Your agent is called once per turn as `agent(observation, configuration)` and
returns the order list. The `observation` exposes:

| Field | Shared? | Contents |
|---|---|---|
| `board` | yes | 2-D array of terrain codes (`p,w,m,f,r,b,h,t,o`) |
| `structures` | yes | List of `{x, y, type, owner, hp, maxHp}` for every capturable tile |
| `units` | **no** | List of `{type, owner, x, y, hp, maxHp, canMove, canAttack, paralyzedTurns, isHasted, distanceMoved, defenceBuffTurns, attackBuffTurns}` |
| `gold` | yes | `[player1_gold, player2_gold]` |
| `player` | — | Your index, `0` or `1` (engine players are 1-indexed, so `owner = player + 1`) |
| `turnNumber`, `mapWidth`, `mapHeight` | yes | Game clock and board dimensions |

#### A minimal agent

```python
def my_agent(observation, configuration):
    actions = []
    me = observation.player + 1                 # engine is 1-indexed
    gold = observation.gold[observation.player]
    occupied = {(u["x"], u["y"]) for u in observation.units}

    # Recruit a Warrior on each empty building we own
    for s in observation.structures:
        if s["type"] == "b" and s["owner"] == me and (s["x"], s["y"]) not in occupied:
            if gold >= 300:
                actions.append({"type": "create_unit", "unit_type": "W",
                                "x": s["x"], "y": s["y"]})
                gold -= 300

    # ... add your move / attack / seize logic here ...

    actions.append({"type": "end_turn"})
    return actions
```

### 11. Strategy primer

- **Grab towers early** — economy compounds; +50/turn now is a unit later.
- **Focus fire.** Under HP-scaling, two units killing one beats two units
  half-hurting two.
- **Respect terrain.** Forests hide Rogues and break archer lines; mountains
  extend vision and archer range.
- **Archers and Mages punch without reprisal** — screen them with
  Warriors/Knights and let them whittle the enemy down.
- **Don't start a siege you can't finish** — a killed seizer hands the structure
  a 50%/turn heal.
- **Buffs and Haste win the decisive fight**, not the whole game — spend them
  when a trade actually swings the match.

---

## Evaluation

### Match outcome → score

Every episode is a 1v1 match. At the end, each agent receives one of three
rewards:

| Outcome | Condition | Reward |
|---|---|---|
| **Win** | You **seize the enemy Headquarters**, or **eliminate all enemy units** | `+1` |
| **Loss** | The opponent does either of the above to you | `-1` |
| **Draw** | Neither happens within the turn limit (`episodeSteps`, default 200) | `0` |

The result is strictly zero-sum: one agent's `+1` is the other's `-1`, and a
draw is `0` for both.

### Leaderboard & ranking

As in other Kaggle **Simulation** competitions, your agent is continuously
matched against other submissions, and the per-episode results feed a
**skill-rating leaderboard**. Ratings rise after wins and fall after losses, so
the objective is a policy that **beats a wide field of opponents across many
maps**, not one that overfits to a single matchup. Because maps and (optionally)
the opening can vary between episodes, a robust agent that wins consistently
outranks a brittle one that only wins specific setups.

### Constraints

Time limits are enforced by the **Kaggle Environments harness**, not the game
engine, and are set by the competition host — the values below are the
environment defaults:

- **Per-turn time limit:** 5 s (`actTimeout`). Each agent also has an **overage
  bank** (`remainingOverageTime`, 60 s by default) it can draw from when a turn
  runs over budget; exhaust it and a slow turn flips your `status` to
  `TIMEOUT`, which is scored as a loss. (Local in-process `env.run` calls are
  timed but not hard-interrupted — enforcement bites in the sandboxed
  competition runner.)
- **Per-episode time limit:** 1200 s (`runTimeout`) of total wall-clock.
- **Robustness:** a *malformed* action (not a dict) forfeits the match, so guard
  your output. Merely *illegal* orders are safely ignored as no-ops, so you
  never need to perfectly validate every move yourself.

### Baselines

Two reference agents ship with the environment so you can benchmark before
submitting:

- **`random`** — ends its turn immediately; a sanity-check floor.
- **`simple_bot`** — a strategic baseline that recruits, advances, attacks, and
  seizes. Beating it consistently is a reasonable first milestone.

```python
from kaggle_environments import make, evaluate

# One match, rendered
env = make("reinforce_tactics", configuration={"mapName": "beginner"})
env.run([my_agent, "simple_bot"])
print(env.render(mode="ansi"))

# Win rate over several maps
rewards = evaluate(
    "reinforce_tactics",
    [my_agent, "simple_bot"],
    num_episodes=10,
    configuration={"mapName": ""},   # random maps
)
wins = sum(1 for r in rewards if r[0] == 1)
print(f"{wins}/{len(rewards)} wins as Player 0")
```

---

**Resources:** [Source on GitHub](https://github.com/kuds/reinforce-tactics) ·
[reinforcetactics.com](https://reinforcetactics.com) · License: Apache 2.0

> The unit stats and formulas above reflect the **competition overrides**
> the Kaggle adapter actually applies (`Warrior cost 300`, `hp_scaled` combat),
> drawn from the vendored engine (`reinforcetactics/kaggle/`) — the
> interpreter, `reinforce_tactics.json` spec, combat/seize in `mechanics.py`,
> and income/recruit rules in `game_state.py`.
