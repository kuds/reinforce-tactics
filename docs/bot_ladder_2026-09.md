# Scripted-bot ladder on the bootstrap curriculum maps — 2026-09

The rule-bot legality fixes (review `rulebots-1`, `-6`, `-8`, `-14`, `-19`)
made every scripted bot send only legal actions, and moved the ladder: on
the small curriculum maps MediumBot now beats AdvancedBot and MasterBot. The
review asked for the curriculum's stage order to be re-checked before the
next run (`docs/REVIEW_full_2026-09-26.md` §4.4 and §8 week 4; the numbers
under `rulebots-1` in `docs/REVIEW_full_2026-09-26_findings.md`). This is
that check for `configs/ppo/bootstrap.yaml`, measured with a new tool,
`scripts/eval/bot_ladder.py`.

This note recommends; it changes no curriculum. The only edits to
`configs/ppo/bootstrap.yaml` are comments pointing here: one at each
flagged stage, and short notes at the three stages that add no step, at
`intermediate_medium` and above the stage list. Reordering stages, and
re-baselining the thresholds and budgets that go with them, belong to the
validation-run package.

## Summary

- **5 of the 33 stages are flagged**: their opponent is weaker than the
  previous stage's opponent on the same map. Each loses the direct
  head-to-head with the previous opponent, significantly at 95%, and a
  rerun on fresh seeds confirmed all five (see "Confirmation on fresh
  seeds" below).

  | Stage | Its opponent | Previous stage's opponent | Head-to-head (this opponent's W-D-L) | Fresh seeds (100 games) |
  |---|---|---|---|---|
  | `starter_simple` | simple | random_20 | 2-7-41 | 5-19-76 |
  | `beginner_simple` | simple | random_20 | 7-17-26 | 4-40-56 |
  | `beginner_mixed_med_adv_50` | mix(medium/advanced,0.5) | medium | 15-0-35 | 35-0-65 |
  | `beginner_advanced` | advanced | mix(medium/advanced,0.5) | 17-0-33 | 24-0-76 |
  | `corner_points_medium` | medium | simple | 3-16-31 | 9-26-65 |

  `corner_points_medium` is flagged on the head-to-head alone. MediumBot
  rates *above* SimpleBot there (+183 against +139) because it beats the
  random bots, which SimpleBot only draws. But SimpleBot beats MediumBot,
  AdvancedBot and MasterBot head to head on corner_points. The other four
  are flagged by both the rating and the head-to-head.
- **3 stages add no measurable difficulty.** Their opponent's rating ties
  the previous one's, and every head-to-head game is a draw:
  `skirmish_random_15`, `skirmish_random_20` and `corner_points_random_20`.
  On skirmish and corner_points, RandomBot with `max_actions` 10, 15 or
  20 plays alike against every bot in the pool (see "Caveats" below).
- **The tier names no longer describe strength.** MediumBot beats
  AdvancedBot and MasterBot head to head on every curriculum map. Against
  AdvancedBot: 50-0-0 on starter, 46-0-4 on beginner, 46-0-4 on
  intermediate, 36-0-14 on skirmish and 34-0-16 on corner_points. Against
  MasterBot: 49-0-1, 42-0-8, 45-0-5, 31-0-19 and 30-0-20. AdvancedBot and
  MasterBot are roughly even with each other. On corner_points at 200
  turns, SimpleBot beats all three (31-16-3, 32-2-16 and 35-5-10). So the
  top of every map's ladder is MediumBot, or SimpleBot on corner_points.
  An AdvancedBot or MasterBot stage placed after MediumBot is always a
  step down.
- **Random opponents draw.** Scripted bots rarely finish a game against a
  RandomBot within the stage's `max_turns`. For example, SimpleBot against
  random_10 goes 7-43-0 on beginner, 2-48-0 on skirmish and 7-43-0 on
  corner_points. On the two smallest maps, RandomBot with
  `max_actions: 20` *beats* SimpleBot: 41-7-2 on starter and 26-17-7 on
  beginner.

### Recommended stage order

This is a recommendation only. It is based on the head-to-heads between
the opponents of adjacent stages and on the per-map ratings. In each
chain below, the number after an arrow is the next opponent's score
against the one before it (win 1, draw ½; above 0.5 means it is the
stronger bot).

| Map | Current order | Recommended order | Evidence |
|---|---|---|---|
| starter@20 | random → simple → medium | **simple → random → medium** (or drop `starter_simple`) | simple → random_20 **0.89**, random_20 → medium **0.91**. Fresh seeds: 0.85 and 0.87. |
| beginner@75 | balanced_random → random_10 → random_15 → random_20 → simple → mixed_50 → medium → mixed_med_adv_50 → advanced | balanced_random → random_10 → **simple** → random_15 → random_20 → mixed_50 → **advanced → mixed_med_adv_50 → medium** | random_10 → simple 0.57, simple → random_15 0.47 (the two tie), random_15 → random_20 0.52. random_20 → mixed_50 0.60, mixed_50 → advanced 0.60, advanced → mixed_med_adv_50 0.66, mixed_med_adv_50 → medium **0.70**. The ratings rise at every step from mixed_50 on (+153, +312, +388, +535). |
| intermediate@60 | balanced_random → random_20 → mixed(random/simple) → simple → medium | keep the order, and **add a mix(simple/medium, 0.5) bridge before `intermediate_medium`** | No flags. The single large step is simple → medium (0.91, 45-1-4). mix(simple/medium) splits it: simple → mix **0.72**, mix → medium **0.75** (rated +163, between −61 and +521). random_20 → mixed → simple are small steps: 0.50 and 0.47. |
| skirmish@120 | balanced_random → random_10 → random_15 → random_20 → mixed_25 → mixed_50 → simple → medium | keep the order; **`skirmish_random_15` and `skirmish_random_20` can go** | random_10, 15 and 20 are the same opponent here: ratings −165, −165 and −165, all draws against each other. MediumBot is the ceiling: AdvancedBot and MasterBot rate higher but lose to it head to head (0.28 and 0.38). |
| corner_points@200 | balanced_random → random_10 → random_15 → random_20 → mixed_25 → mixed_50 → simple → medium | … → mixed_50 → **medium → simple**; `corner_points_random_20` can go, and probably `_15` (6 rating points above `_10`, all draws against it) | mixed_50 → medium **0.67**, medium → simple **0.78**. Fresh seeds: 0.62 and 0.78. SimpleBot is the hardest bot to beat on this map: the others beat it 7% of the time, against 23% for MediumBot. |

Whole-curriculum notes:

- **Re-baseline any stage you move.** Thresholds and budgets were set for
  the old positions. For example, `beginner_medium` would become the
  block's final stage and should take the 5M budget `beginner_advanced`
  has now. That is the validation-run package's job.
- **Stop at the top of each map's ladder.** A MasterBot or AdvancedBot
  stage after MediumBot would be flagged on every map. If the curriculum
  needs a harder final opponent than MediumBot (SimpleBot on
  corner_points), it has to come from outside the scripted tiers:
  self-play or a frozen checkpoint.
- **Weigh the dropped random stages against their other purpose.** The
  10 → 15 → 20 ramps exist to teach the agent to *convert* games that a
  longer-surviving opponent would otherwise run to a draw (the comment at
  `beginner_random_15`). Bot against bot, the ramp adds no difficulty on
  skirmish and corner_points. Whether it still teaches something is a
  question for a training run.

## How it was measured

```
python scripts/eval/bot_ladder.py --config configs/ppo/bootstrap.yaml --seeds 25 --workers 4 --out-dir <dir>
```

- **Boards.** Every map a stage uses, at that stage's `max_turns`:
  starter@20, beginner@75, intermediate@60, skirmish@120 and
  corner_points@200. The runs use the config's `env.fog_of_war` (off),
  `env.enabled_units` (all) and `env.engine_overrides` (none). The games
  are bot against bot, so the env's `max_steps` does not apply.
- **Bots.** 15 in all. That is every opponent the curriculum's stages use,
  with its `opponent_kwargs`, plus the plain tiers `simple`, `medium`,
  `advanced`, `master`, `random`, `balanced_random` and `noop`, with
  duplicates removed. `random` is `random_20` (`max_actions` defaults to
  20). `mix(a/b,p)` is MixedBot with `easy: a`, `hard: b`, `p_hard: p`.
- **Games.** Every pair of bots on every board, over 25 seeds (0–24). Each
  seed plays one game per seat assignment: 50 games per pairing and 26,250
  games in all. The run took 63 minutes on 4 worker processes.
  `GameState(seed=k)` controls the engine's randomness. Each bot gets
  `random.Random("ladder:<k>:<bot>:vs:<other bot>")`. That seed is the same
  in both seats of a pair, and independent across pairings, so each
  MixedBot pairing draws its own 25 coin flips. A game's result depends
  only on the map, `max_turns`, the two bots, the seats and `k`. It does
  not depend on the opponent list, its order or the worker count, and the
  test suite checks this.
- **Code.** The engine and bots are those at `98d66ac`. This branch adds
  only the tool, its tests, this note and comments in `bootstrap.yaml`.
- **Ordering.** The rating is a Bradley–Terry fit on the Elo scale,
  centred on 0 per board. A draw counts as half a win for each side, and
  a prior of one virtual win and one virtual loss against an average bot
  keeps the ratings finite. "Mean score" is a bot's average score against
  the others. "Beaten" is how often the other bots beat it, counting wins
  only and averaged over the bots. Beaten is the number closest to a
  win-rate gate, because a draw does not promote.
- **Curriculum check.** Each stage's opponent is compared with the
  previous stage's opponent on the same map, both rated on the stage's
  board. A stage is **flagged** when its opponent rates lower, by at least
  1 Elo (a smaller gap counts as *tied*), *or* when it scores
  significantly below 0.5 against the previous opponent head to head.
  Significance uses the 95% Wilson interval, with a draw counting as half.
  The head-to-head is the better guide. An agent that has just promoted
  against the previous opponent at 65–90% meets the new one next. If the
  previous opponent beats the new one, that stage is no step up, whatever
  the pooled rating says.
- **Seats.** Player 1 moves first. Over every game on a board, seat 1
  scores 46% on starter, 50% on beginner, 51% on intermediate, 52% on
  skirmish and 51% on corner_points. The curriculum's opponent always sits
  in seat 2, because the env's agent is player 1 and bootstrap does not
  change that. The by-seat tables show that view: read the *column* for
  the opponent in seat 2.

The raw per-game records, the CSV and the full markdown report are not
committed. The command above plays the same games again. `--from-json`
re-rates a saved run without replaying it.

### Confirmation on fresh seeds

The flagged transitions were replayed with seeds 1000–1049, 100 games per
pairing, with a smaller bot pool per map:

```
python scripts/eval/bot_ladder.py --config configs/ppo/bootstrap.yaml \
    --maps maps/1v1/starter.csv maps/1v1/beginner.csv --seed-base 1000 --seeds 50 \
    --opponents random_10 random_15 random_20 simple "mixed:easy=simple,hard=medium,p_hard=0.5" \
        advanced "mixed:easy=medium,hard=advanced,p_hard=0.5" medium master
python scripts/eval/bot_ladder.py --config configs/ppo/bootstrap.yaml \
    --maps maps/1v1/corner_points.csv --seed-base 1000 --seeds 50 \
    --opponents "mixed:easy=balanced_random,hard=simple,p_hard=0.5" simple medium advanced master \
        "mixed:easy=simple,hard=medium,p_hard=0.5"
```

All five flags reproduce with the head-to-head significant (the table in
the Summary). One step in the recommended beginner order is weaker on the
fresh seeds: mixed_50 → advanced is 0.52 (52-0-48), against 0.60 on the
main run. AdvancedBot is a small step above mixed_50, not a large one. The
other recommended steps hold: random_20 → mixed_50 0.55,
advanced → mixed_med_adv_50 0.76, mixed_med_adv_50 → medium 0.65. On
corner_points: mixed_50 → medium 0.62, medium → simple 0.78.

The main run also agrees with the review's figures, which used other seeds
and 75 turns everywhere. There, AdvancedBot scored 5/0/45 against
MediumBot on beginner; here it scores 4-0-46. MasterBot scored 6/0/44
against MediumBot on beginner; here 8-0-42.

## Caveats

- **Bot against bot is a proxy.** The agent is a different player. A bot
  that the scripted tiers cannot finish in time, such as RandomBot, is not
  necessarily hard for a policy that has learned to finish games. The
  head-to-head between adjacent stage opponents is the closest proxy to
  "the agent just beat the previous one", but it is still a proxy.
- **The ratings depend on the pool.** 8 of the 15 bots play randomly all
  or part of the time. Everywhere but starter they mostly draw each other,
  which compresses their ratings. The results are also not transitive: on
  skirmish, AdvancedBot and MasterBot rate above MediumBot but lose to it,
  and on corner_points MediumBot rates above SimpleBot but loses to it.
  Read the ordering tables together with the crosstables.
- **The random-bot stages are separated only through third parties.**
  Everywhere but starter, random_10, 15 and 20 draw each other in 48–50 of
  50 games. Their order comes from how the other bots fare against them.
  On beginner that order is clear (−178, −77, +56). On skirmish and
  corner_points it is flat, because the `max_actions` cap rarely binds
  there. RandomBot stops once it has no legal action left. In a sample of
  6 games per map against SimpleBot, it took more than 10 actions in 5%
  of its turns on skirmish and 7% on corner_points, against 35% on
  beginner.
- **MixedBot's mix is sampled.** Each pairing gets 25 coin flips, one per
  seed, shared by both seats. The realised share of the hard bot therefore
  has a standard error of about 0.1.
- **The engine's randomness is small.** The only engine-side random event
  is the Rogue's evade roll. The spread between seeds comes almost
  entirely from the bots' own rngs.

## Curriculum check

Each stage's opponent against the previous stage's on the same map, both rated on this stage's board. Beaten as in the ordering tables. H2H: this stage's opponent's W-D-L against the previous one and its score with the 95% Wilson interval (*higher* / *lower* when it excludes 0.5). Flagged when the opponent rates weaker or its H2H is lower.

| Stage | Board | Opponent | Rating | Beaten | Previous stage (opponent) | Rating | Beaten | H2H | Score [95% CI] | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| starter_random | starter@20 | random_20 | +130 | 0.31 | - | n/a | n/a | - | - | first on map |
| starter_simple | starter@20 | simple | -77 | 0.46 | starter_random (random_20) | +130 | 0.31 | 2-7-41 | 0.11 [0.05, 0.23] | **FLAG: weaker** (rating weaker; H2H lower) |
| starter_medium | starter@20 | medium | +426 | 0.08 | starter_simple (simple) | -77 | 0.46 | 38-9-3 | 0.85 [0.73, 0.92] | rating harder; H2H higher |
| beginner_balanced_random | beginner@75 | balanced_random | -340 | 0.70 | - | n/a | n/a | - | - | first on map |
| beginner_random_10 | beginner@75 | random_10 | -178 | 0.35 | beginner_balanced_random (balanced_random) | -340 | 0.70 | 0-50-0 | 0.50 [0.37, 0.63] | rating harder; H2H within noise |
| beginner_random_15 | beginner@75 | random_15 | -77 | 0.29 | beginner_random_10 (random_10) | -178 | 0.35 | 0-50-0 | 0.50 [0.37, 0.63] | rating harder; H2H within noise |
| beginner_random_20 | beginner@75 | random_20 | +56 | 0.29 | beginner_random_15 (random_15) | -77 | 0.29 | 2-48-0 | 0.52 [0.39, 0.65] | rating harder; H2H within noise |
| beginner_simple | beginner@75 | simple | -87 | 0.41 | beginner_random_20 (random_20) | +56 | 0.29 | 7-17-26 | 0.31 [0.20, 0.45] | **FLAG: weaker** (rating weaker; H2H lower) |
| beginner_mixed_50 | beginner@75 | mix(simple/medium,0.5) | +153 | 0.24 | beginner_simple (simple) | -87 | 0.41 | 25-24-1 | 0.74 [0.60, 0.84] | rating harder; H2H higher |
| beginner_medium | beginner@75 | medium | +535 | 0.07 | beginner_mixed_50 (mix(simple/medium,0.5)) | +153 | 0.24 | 39-0-11 | 0.78 [0.65, 0.87] | rating harder; H2H higher |
| beginner_mixed_med_adv_50 | beginner@75 | mix(medium/advanced,0.5) | +388 | 0.15 | beginner_medium (medium) | +535 | 0.07 | 15-0-35 | 0.30 [0.19, 0.44] | **FLAG: weaker** (rating weaker; H2H lower) |
| beginner_advanced | beginner@75 | advanced | +312 | 0.21 | beginner_mixed_med_adv_50 (mix(medium/advanced,0.5)) | +388 | 0.15 | 17-0-33 | 0.34 [0.22, 0.48] | **FLAG: weaker** (rating weaker; H2H lower) |
| intermediate_balanced_random | intermediate@60 | balanced_random | -250 | 0.53 | - | n/a | n/a | - | - | first on map |
| intermediate_random_20 | intermediate@60 | random_20 | -118 | 0.30 | intermediate_balanced_random (balanced_random) | -250 | 0.53 | 15-35-0 | 0.65 [0.51, 0.77] | rating harder; H2H higher |
| intermediate_mixed_random_simple | intermediate@60 | mix(random_20/simple,0.5) | -99 | 0.32 | intermediate_random_20 (random_20) | -118 | 0.30 | 0-50-0 | 0.50 [0.37, 0.63] | rating harder; H2H within noise |
| intermediate_simple | intermediate@60 | simple | -61 | 0.31 | intermediate_mixed_random_simple (mix(random_20/simple,0.5)) | -99 | 0.32 | 2-43-5 | 0.47 [0.34, 0.61] | rating harder; H2H within noise |
| intermediate_medium | intermediate@60 | medium | +521 | 0.06 | intermediate_simple (simple) | -61 | 0.31 | 45-1-4 | 0.91 [0.80, 0.96] | rating harder; H2H higher |
| skirmish_balanced_random | skirmish@120 | balanced_random | -213 | 0.51 | - | n/a | n/a | - | - | first on map |
| skirmish_random_10 | skirmish@120 | random_10 | -165 | 0.34 | skirmish_balanced_random (balanced_random) | -213 | 0.51 | 0-50-0 | 0.50 [0.37, 0.63] | rating harder; H2H within noise |
| skirmish_random_15 | skirmish@120 | random_15 | -165 | 0.34 | skirmish_random_10 (random_10) | -165 | 0.34 | 0-50-0 | 0.50 [0.37, 0.63] | rating tied; H2H within noise |
| skirmish_random_20 | skirmish@120 | random_20 | -165 | 0.34 | skirmish_random_15 (random_15) | -165 | 0.34 | 0-50-0 | 0.50 [0.37, 0.63] | rating tied; H2H within noise |
| skirmish_mixed_25 | skirmish@120 | mix(random_20/simple,0.25) | -122 | 0.32 | skirmish_random_20 (random_20) | -165 | 0.34 | 1-49-0 | 0.51 [0.38, 0.64] | rating harder; H2H within noise |
| skirmish_mixed_50 | skirmish@120 | mix(random_20/simple,0.5) | -94 | 0.32 | skirmish_mixed_25 (mix(random_20/simple,0.25)) | -122 | 0.32 | 5-43-2 | 0.53 [0.39, 0.66] | rating harder; H2H within noise |
| skirmish_simple | skirmish@120 | simple | +8 | 0.27 | skirmish_mixed_50 (mix(random_20/simple,0.5)) | -94 | 0.32 | 7-42-1 | 0.56 [0.42, 0.69] | rating harder; H2H within noise |
| skirmish_medium | skirmish@120 | medium | +322 | 0.18 | skirmish_simple (simple) | +8 | 0.27 | 33-1-16 | 0.67 [0.53, 0.78] | rating harder; H2H higher |
| corner_points_balanced_random | corner_points@200 | balanced_random | -187 | 0.50 | - | n/a | n/a | - | - | first on map |
| corner_points_random_10 | corner_points@200 | random_10 | -131 | 0.34 | corner_points_balanced_random (balanced_random) | -187 | 0.50 | 0-50-0 | 0.50 [0.37, 0.63] | rating harder; H2H within noise |
| corner_points_random_15 | corner_points@200 | random_15 | -125 | 0.32 | corner_points_random_10 (random_10) | -131 | 0.34 | 0-50-0 | 0.50 [0.37, 0.63] | rating harder; H2H within noise |
| corner_points_random_20 | corner_points@200 | random_20 | -125 | 0.33 | corner_points_random_15 (random_15) | -125 | 0.32 | 0-50-0 | 0.50 [0.37, 0.63] | rating tied; H2H within noise |
| corner_points_mixed_25 | corner_points@200 | mix(balanced_random/simple,0.25) | -90 | 0.36 | corner_points_random_20 (random_20) | -125 | 0.33 | 0-50-0 | 0.50 [0.37, 0.63] | rating harder; H2H within noise |
| corner_points_mixed_50 | corner_points@200 | mix(balanced_random/simple,0.5) | -6 | 0.25 | corner_points_mixed_25 (mix(balanced_random/simple,0.25)) | -90 | 0.36 | 17-25-8 | 0.59 [0.45, 0.72] | rating harder; H2H within noise |
| corner_points_simple | corner_points@200 | simple | +139 | 0.07 | corner_points_mixed_50 (mix(balanced_random/simple,0.5)) | -6 | 0.25 | 24-23-3 | 0.71 [0.57, 0.82] | rating harder; H2H higher |
| corner_points_medium | corner_points@200 | medium | +183 | 0.23 | corner_points_simple (simple) | +139 | 0.07 | 3-16-31 | 0.22 [0.13, 0.35] | **FLAG: loses H2H** (rating harder; H2H lower) |

## Results by map

The ordering (weakest first), the crosstable over both seats, and the
by-seat crosstable for each board. Each cell is the row bot's W-D-L
against the column bot. For the curriculum's view of a stage opponent in
seat 2, read that opponent's column in the by-seat table. The per-pairing
tables with end reasons and mean game length are in the tool's output.

### maps/1v1/starter.csv, max_turns 20

Seat 1 (the player who moves first) scores 46.4% over every game on this board (2020-827-2403 W-D-L).

Strength ordering, weakest first. Rating: Bradley-Terry, Elo scale, centred on 0 (± one standard error).
Mean score: the bot's score against each other bot (win 1, draw 0.5), averaged.
Beaten: how often each other bot beat it (wins only), averaged; what a win-rate gate sees.

| # | Bot | Rating | Mean score | Beaten | W-D-L |
|---|---|---|---|---|---|
| 1 | noop | -529 ± 28 | 0.07 | 0.86 | 0-96-604 |
| 2 | balanced_random | -403 ± 23 | 0.13 | 0.78 | 31-124-545 |
| 3 | mix(balanced_random/simple,0.25) | -302 ± 20 | 0.20 | 0.69 | 58-162-480 |
| 4 | mix(balanced_random/simple,0.5) | -226 ± 19 | 0.26 | 0.62 | 93-174-433 |
| 5 | simple | -77 ± 16 | 0.40 | 0.46 | 176-201-323 |
| 6 | mix(random_20/simple,0.5) | +31 ± 16 | 0.51 | 0.37 | 277-163-260 |
| 7 | random_10 | +51 ± 15 | 0.54 | 0.33 | 278-193-229 |
| 8 | mix(random_20/simple,0.25) | +103 ± 15 | 0.59 | 0.31 | 351-131-218 |
| 9 | mix(simple/medium,0.5) | +117 ± 16 | 0.61 | 0.31 | 373-109-218 |
| 10 | random_20 | +130 ± 16 | 0.62 | 0.31 | 393-89-218 |
| 11 | master | +142 ± 16 | 0.64 | 0.34 | 435-24-241 |
| 12 | random_15 | +148 ± 16 | 0.64 | 0.28 | 401-101-198 |
| 13 | advanced | +149 ± 16 | 0.65 | 0.34 | 445-15-240 |
| 14 | mix(medium/advanced,0.5) | +242 ± 17 | 0.74 | 0.23 | 505-33-162 |
| 15 | medium | +426 ± 22 | 0.89 | 0.08 | 607-39-54 |

Crosstable: the row bot's W-D-L against the column bot over both seats (columns numbered as the rows).

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-29-21 | 0-27-23 | 0-19-31 | 0-0-50 | 0-0-50 | 0-18-32 | 0-1-49 | 0-0-50 | 0-0-50 | 0-0-50 | 0-2-48 | 0-0-50 | 0-0-50 | 0-0-50 |
| 2 | balanced_random | 21-29-0 | · | 7-29-14 | 1-24-25 | 1-14-35 | 0-9-41 | 0-7-43 | 0-2-48 | 1-8-41 | 0-1-49 | 0-0-50 | 0-1-49 | 0-0-50 | 0-0-50 | 0-0-50 |
| 3 | mix(balanced_random/simple,0.25) | 23-27-0 | 14-29-7 | · | 10-29-11 | 0-29-21 | 0-10-40 | 0-14-36 | 1-7-42 | 0-9-41 | 0-2-48 | 1-0-49 | 1-3-46 | 2-0-48 | 4-0-46 | 2-3-45 |
| 4 | mix(balanced_random/simple,0.5) | 31-19-0 | 25-24-1 | 11-29-10 | · | 1-27-22 | 1-18-31 | 1-13-36 | 1-15-34 | 3-15-32 | 1-1-48 | 4-1-45 | 1-4-45 | 5-0-45 | 5-4-41 | 3-4-43 |
| 5 | simple | 50-0-0 | 35-14-1 | 21-29-0 | 22-27-1 | · | 4-21-25 | 0-21-29 | 7-18-25 | 5-25-20 | 2-7-41 | 10-4-36 | 1-11-38 | 10-2-38 | 6-13-31 | 3-9-38 |
| 6 | mix(random_20/simple,0.5) | 50-0-0 | 41-9-0 | 40-10-0 | 31-18-1 | 25-21-4 | · | 9-21-20 | 8-23-19 | 9-15-26 | 9-16-25 | 17-5-28 | 12-9-29 | 11-2-37 | 11-2-37 | 4-12-34 |
| 7 | random_10 | 32-18-0 | 43-7-0 | 36-14-0 | 36-13-1 | 29-21-0 | 20-21-9 | · | 11-22-17 | 17-10-23 | 6-22-22 | 17-5-28 | 3-26-21 | 15-7-28 | 6-5-39 | 7-2-41 |
| 8 | mix(random_20/simple,0.25) | 49-1-0 | 48-2-0 | 42-7-1 | 34-15-1 | 25-18-7 | 19-23-8 | 17-22-11 | · | 28-9-13 | 20-13-17 | 17-3-30 | 15-11-24 | 21-2-27 | 12-3-35 | 4-2-44 |
| 9 | mix(simple/medium,0.5) | 50-0-0 | 41-8-1 | 41-9-0 | 32-15-3 | 20-25-5 | 26-15-9 | 23-10-17 | 13-9-28 | · | 21-4-25 | 32-1-17 | 18-9-23 | 24-0-26 | 19-2-29 | 13-2-35 |
| 10 | random_20 | 50-0-0 | 49-1-0 | 48-2-0 | 48-1-1 | 41-7-2 | 25-16-9 | 22-22-6 | 17-13-20 | 25-4-21 | · | 22-1-27 | 13-17-20 | 17-1-32 | 13-1-36 | 3-3-44 |
| 11 | master | 50-0-0 | 50-0-0 | 49-0-1 | 45-1-4 | 36-4-10 | 28-5-17 | 28-5-17 | 30-3-17 | 17-1-32 | 27-1-22 | · | 26-3-21 | 30-0-20 | 18-1-31 | 1-0-49 |
| 12 | random_15 | 48-2-0 | 49-1-0 | 46-3-1 | 45-4-1 | 38-11-1 | 29-9-12 | 21-26-3 | 24-11-15 | 23-9-18 | 20-17-13 | 21-3-26 | · | 19-1-30 | 14-2-34 | 4-2-44 |
| 13 | advanced | 50-0-0 | 50-0-0 | 48-0-2 | 45-0-5 | 38-2-10 | 37-2-11 | 28-7-15 | 27-2-21 | 26-0-24 | 32-1-17 | 20-0-30 | 30-1-19 | · | 14-0-36 | 0-0-50 |
| 14 | mix(medium/advanced,0.5) | 50-0-0 | 50-0-0 | 46-0-4 | 41-4-5 | 31-13-6 | 37-2-11 | 39-5-6 | 35-3-12 | 29-2-19 | 36-1-13 | 31-1-18 | 34-2-14 | 36-0-14 | · | 10-0-40 |
| 15 | medium | 50-0-0 | 50-0-0 | 45-3-2 | 43-4-3 | 38-9-3 | 34-12-4 | 41-2-7 | 44-2-4 | 35-2-13 | 44-3-3 | 49-0-1 | 44-2-4 | 50-0-0 | 40-0-10 | · |

By seat: the row bot's W-D-L in seat 1 (moving first) against the column bot in seat 2; the column bot's own seat-1 games are in its row. The diagonal is the mirror match, when played.

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-15-10 | 0-13-12 | 0-9-16 | 0-0-25 | 0-0-25 | 0-10-15 | 0-1-24 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 2 | balanced_random | 11-14-0 | · | 4-15-6 | 0-13-12 | 1-6-18 | 0-3-22 | 0-2-23 | 0-1-24 | 1-3-21 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 3 | mix(balanced_random/simple,0.25) | 11-14-0 | 8-14-3 | · | 6-13-6 | 0-14-11 | 0-6-19 | 0-5-20 | 1-1-23 | 0-4-21 | 0-1-24 | 0-0-25 | 1-0-24 | 0-0-25 | 1-0-24 | 1-2-22 |
| 4 | mix(balanced_random/simple,0.5) | 15-10-0 | 13-11-1 | 5-16-4 | · | 0-13-12 | 0-8-17 | 0-5-20 | 0-7-18 | 3-5-17 | 1-1-23 | 0-0-25 | 0-2-23 | 1-0-24 | 1-2-22 | 1-2-22 |
| 5 | simple | 25-0-0 | 17-8-0 | 10-15-0 | 10-14-1 | · | 3-8-14 | 0-9-16 | 3-9-13 | 0-15-10 | 0-3-22 | 1-1-23 | 0-5-20 | 3-1-21 | 1-5-19 | 1-5-19 |
| 6 | mix(random_20/simple,0.5) | 25-0-0 | 19-6-0 | 21-4-0 | 14-10-1 | 11-13-1 | · | 2-11-12 | 2-10-13 | 3-9-13 | 2-8-15 | 5-2-18 | 2-5-18 | 6-2-17 | 4-0-21 | 2-3-20 |
| 7 | random_10 | 17-8-0 | 20-5-0 | 16-9-0 | 16-8-1 | 13-12-0 | 8-10-7 | · | 3-11-11 | 6-7-12 | 4-8-13 | 8-3-14 | 1-10-14 | 8-0-17 | 4-2-19 | 2-0-23 |
| 8 | mix(random_20/simple,0.25) | 25-0-0 | 24-1-0 | 19-6-0 | 16-8-1 | 12-9-4 | 6-13-6 | 6-11-8 | · | 11-6-8 | 7-9-9 | 6-1-18 | 7-3-15 | 10-1-14 | 6-0-19 | 1-0-24 |
| 9 | mix(simple/medium,0.5) | 25-0-0 | 20-5-0 | 20-5-0 | 15-10-0 | 10-10-5 | 13-6-6 | 11-3-11 | 5-3-17 | · | 9-3-13 | 13-1-11 | 8-4-13 | 11-0-14 | 11-1-13 | 11-1-13 |
| 10 | random_20 | 25-0-0 | 24-1-0 | 24-1-0 | 25-0-0 | 19-4-2 | 10-8-7 | 9-14-2 | 8-4-13 | 12-1-12 | · | 9-0-16 | 3-10-12 | 6-1-18 | 3-0-22 | 1-1-23 |
| 11 | master | 25-0-0 | 25-0-0 | 24-0-1 | 20-1-4 | 13-3-9 | 10-3-12 | 14-2-9 | 12-2-11 | 6-0-19 | 11-1-13 | · | 9-2-14 | 9-0-16 | 5-1-19 | 0-0-25 |
| 12 | random_15 | 23-2-0 | 24-1-0 | 22-3-0 | 22-2-1 | 18-6-1 | 11-4-10 | 7-16-2 | 9-8-8 | 10-5-10 | 8-7-10 | 7-1-17 | · | 7-0-18 | 5-1-19 | 0-1-24 |
| 13 | advanced | 25-0-0 | 25-0-0 | 23-0-2 | 21-0-4 | 17-1-7 | 20-0-5 | 11-7-7 | 13-1-11 | 12-0-13 | 14-0-11 | 4-0-21 | 12-1-12 | · | 6-0-19 | 0-0-25 |
| 14 | mix(medium/advanced,0.5) | 25-0-0 | 25-0-0 | 22-0-3 | 19-2-4 | 12-8-5 | 16-2-7 | 20-3-2 | 16-3-6 | 16-1-8 | 14-1-10 | 12-0-13 | 15-1-9 | 17-0-8 | · | 10-0-15 |
| 15 | medium | 25-0-0 | 25-0-0 | 23-1-1 | 21-2-2 | 19-4-2 | 14-9-2 | 18-2-5 | 20-2-3 | 22-1-2 | 21-2-2 | 24-0-1 | 20-1-4 | 25-0-0 | 25-0-0 | · |

### maps/1v1/beginner.csv, max_turns 75

Seat 1 (the player who moves first) scores 49.8% over every game on this board (1969-1291-1990 W-D-L).

Strength ordering, weakest first. Rating: Bradley-Terry, Elo scale, centred on 0 (± one standard error).
Mean score: the bot's score against each other bot (win 1, draw 0.5), averaged.
Beaten: how often each other bot beat it (wins only), averaged; what a win-rate gate sees.

| # | Bot | Rating | Mean score | Beaten | W-D-L |
|---|---|---|---|---|---|
| 1 | noop | -596 ± 31 | 0.05 | 0.90 | 0-69-631 |
| 2 | balanced_random | -340 ± 20 | 0.19 | 0.70 | 54-154-492 |
| 3 | mix(balanced_random/simple,0.25) | -260 ± 18 | 0.25 | 0.61 | 78-195-427 |
| 4 | mix(balanced_random/simple,0.5) | -208 ± 17 | 0.30 | 0.56 | 110-196-394 |
| 5 | random_10 | -178 ± 17 | 0.33 | 0.35 | 1-454-245 |
| 6 | simple | -87 ± 17 | 0.41 | 0.41 | 166-249-285 |
| 7 | random_15 | -77 ± 17 | 0.42 | 0.29 | 96-403-201 |
| 8 | mix(random_20/simple,0.5) | -29 ± 16 | 0.47 | 0.37 | 220-224-256 |
| 9 | mix(random_20/simple,0.25) | +24 ± 17 | 0.53 | 0.30 | 251-237-212 |
| 10 | random_20 | +56 ± 17 | 0.56 | 0.29 | 284-216-200 |
| 11 | mix(simple/medium,0.5) | +153 ± 17 | 0.65 | 0.24 | 383-148-169 |
| 12 | master | +305 ± 19 | 0.78 | 0.21 | 542-11-147 |
| 13 | advanced | +312 ± 19 | 0.79 | 0.21 | 546-10-144 |
| 14 | mix(medium/advanced,0.5) | +388 ± 21 | 0.84 | 0.15 | 586-6-108 |
| 15 | medium | +535 ± 26 | 0.92 | 0.07 | 642-10-48 |

Crosstable: the row bot's W-D-L against the column bot over both seats (columns numbered as the rows).

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-4-46 | 0-5-45 | 0-0-50 | 0-49-1 | 0-0-50 | 0-11-39 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 2 | balanced_random | 46-4-0 | · | 6-28-16 | 2-12-36 | 0-50-0 | 0-8-42 | 0-37-13 | 0-6-44 | 0-4-46 | 0-2-48 | 0-3-47 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 3 | mix(balanced_random/simple,0.25) | 45-5-0 | 16-28-6 | · | 12-22-16 | 1-49-0 | 0-27-23 | 1-35-14 | 1-9-40 | 0-7-43 | 2-4-44 | 0-9-41 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 4 | mix(balanced_random/simple,0.5) | 50-0-0 | 36-12-2 | 16-22-12 | · | 4-46-0 | 0-26-24 | 1-36-13 | 1-16-33 | 1-16-33 | 1-9-40 | 0-13-37 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 5 | random_10 | 1-49-0 | 0-50-0 | 0-49-1 | 0-46-4 | · | 0-43-7 | 0-50-0 | 0-42-8 | 0-44-6 | 0-48-2 | 0-25-25 | 0-1-49 | 0-2-48 | 0-1-49 | 0-4-46 |
| 6 | simple | 50-0-0 | 42-8-0 | 23-27-0 | 24-26-0 | 7-43-0 | · | 4-45-1 | 3-30-17 | 5-28-17 | 7-17-26 | 1-24-25 | 0-0-50 | 0-0-50 | 0-0-50 | 0-1-49 |
| 7 | random_15 | 39-11-0 | 13-37-0 | 14-35-1 | 13-36-1 | 0-50-0 | 1-45-4 | · | 5-40-5 | 2-46-2 | 0-48-2 | 1-31-18 | 3-9-38 | 3-8-39 | 2-2-46 | 0-5-45 |
| 8 | mix(random_20/simple,0.5) | 50-0-0 | 44-6-0 | 40-9-1 | 33-16-1 | 8-42-0 | 17-30-3 | 5-40-5 | · | 5-35-10 | 2-32-16 | 6-13-31 | 5-0-45 | 2-0-48 | 2-1-47 | 1-0-49 |
| 9 | mix(random_20/simple,0.25) | 50-0-0 | 46-4-0 | 43-7-0 | 33-16-1 | 6-44-0 | 17-28-5 | 2-46-2 | 10-35-5 | · | 2-41-7 | 22-15-13 | 3-0-47 | 6-0-44 | 7-1-42 | 4-0-46 |
| 10 | random_20 | 50-0-0 | 48-2-0 | 44-4-2 | 40-9-1 | 2-48-0 | 26-17-7 | 2-48-0 | 16-32-2 | 7-41-2 | · | 13-14-23 | 10-1-39 | 13-0-37 | 8-0-42 | 5-0-45 |
| 11 | mix(simple/medium,0.5) | 50-0-0 | 47-3-0 | 41-9-0 | 37-13-0 | 25-25-0 | 25-24-1 | 18-31-1 | 31-13-6 | 13-15-22 | 23-14-13 | · | 26-0-24 | 20-0-30 | 16-1-33 | 11-0-39 |
| 12 | master | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 49-1-0 | 50-0-0 | 38-9-3 | 45-0-5 | 47-0-3 | 39-1-10 | 24-0-26 | · | 21-0-29 | 21-0-29 | 8-0-42 |
| 13 | advanced | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 48-2-0 | 50-0-0 | 39-8-3 | 48-0-2 | 44-0-6 | 37-0-13 | 30-0-20 | 29-0-21 | · | 17-0-33 | 4-0-46 |
| 14 | mix(medium/advanced,0.5) | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 49-1-0 | 50-0-0 | 46-2-2 | 47-1-2 | 42-1-7 | 42-0-8 | 33-1-16 | 29-0-21 | 33-0-17 | · | 15-0-35 |
| 15 | medium | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 46-4-0 | 49-1-0 | 45-5-0 | 49-0-1 | 46-0-4 | 45-0-5 | 39-0-11 | 42-0-8 | 46-0-4 | 35-0-15 | · |

By seat: the row bot's W-D-L in seat 1 (moving first) against the column bot in seat 2; the column bot's own seat-1 games are in its row. The diagonal is the mirror match, when played.

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-0-25 | 0-3-22 | 0-0-25 | 0-25-0 | 0-0-25 | 0-5-20 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 2 | balanced_random | 21-4-0 | · | 3-15-7 | 2-5-18 | 0-25-0 | 0-3-22 | 0-19-6 | 0-4-21 | 0-2-23 | 0-0-25 | 0-2-23 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 3 | mix(balanced_random/simple,0.25) | 23-2-0 | 9-13-3 | · | 6-9-10 | 0-25-0 | 0-14-11 | 0-18-7 | 0-5-20 | 0-4-21 | 1-2-22 | 0-5-20 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 4 | mix(balanced_random/simple,0.5) | 25-0-0 | 18-7-0 | 6-13-6 | · | 2-23-0 | 0-12-13 | 1-17-7 | 0-8-17 | 0-9-16 | 1-3-21 | 0-7-18 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 5 | random_10 | 1-24-0 | 0-25-0 | 0-24-1 | 0-23-2 | · | 0-23-2 | 0-25-0 | 0-19-6 | 0-23-2 | 0-23-2 | 0-11-14 | 0-1-24 | 0-1-24 | 0-0-25 | 0-2-23 |
| 6 | simple | 25-0-0 | 20-5-0 | 12-13-0 | 11-14-0 | 5-20-0 | · | 3-22-0 | 1-16-8 | 3-13-9 | 2-9-14 | 1-12-12 | 0-0-25 | 0-0-25 | 0-0-25 | 0-1-24 |
| 7 | random_15 | 19-6-0 | 7-18-0 | 7-17-1 | 6-19-0 | 0-25-0 | 1-23-1 | · | 2-19-4 | 0-24-1 | 0-23-2 | 0-16-9 | 3-2-20 | 1-3-21 | 1-1-23 | 0-2-23 |
| 8 | mix(random_20/simple,0.5) | 25-0-0 | 23-2-0 | 20-4-1 | 16-8-1 | 2-23-0 | 9-14-2 | 1-21-3 | · | 2-16-7 | 1-15-9 | 4-7-14 | 1-0-24 | 0-0-25 | 1-0-24 | 0-0-25 |
| 9 | mix(random_20/simple,0.25) | 25-0-0 | 23-2-0 | 22-3-0 | 17-7-1 | 4-21-0 | 8-15-2 | 1-22-2 | 3-19-3 | · | 2-21-2 | 10-9-6 | 1-0-24 | 3-0-22 | 2-0-23 | 2-0-23 |
| 10 | random_20 | 25-0-0 | 23-2-0 | 22-2-1 | 19-6-0 | 0-25-0 | 12-8-5 | 0-25-0 | 7-17-1 | 5-20-0 | · | 6-8-11 | 2-1-22 | 7-0-18 | 4-0-21 | 3-0-22 |
| 11 | mix(simple/medium,0.5) | 25-0-0 | 24-1-0 | 21-4-0 | 19-6-0 | 11-14-0 | 13-12-0 | 9-15-1 | 17-6-2 | 7-6-12 | 12-6-7 | · | 14-0-11 | 10-0-15 | 11-0-14 | 11-0-14 |
| 12 | master | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 18-7-0 | 21-0-4 | 23-0-2 | 17-0-8 | 13-0-12 | · | 4-0-21 | 7-0-18 | 5-0-20 |
| 13 | advanced | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 24-1-0 | 25-0-0 | 18-5-2 | 23-0-2 | 22-0-3 | 19-0-6 | 15-0-10 | 8-0-17 | · | 6-0-19 | 4-0-21 |
| 14 | mix(medium/advanced,0.5) | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 24-1-0 | 25-0-0 | 23-1-1 | 23-1-1 | 19-1-5 | 21-0-4 | 19-1-5 | 11-0-14 | 14-0-11 | · | 15-0-10 |
| 15 | medium | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 23-2-0 | 25-0-0 | 22-3-0 | 24-0-1 | 23-0-2 | 23-0-2 | 25-0-0 | 22-0-3 | 25-0-0 | 25-0-0 | · |

### maps/1v1/intermediate.csv, max_turns 60

Seat 1 (the player who moves first) scores 50.6% over every game on this board (1709-1890-1651 W-D-L).

Strength ordering, weakest first. Rating: Bradley-Terry, Elo scale, centred on 0 (± one standard error).
Mean score: the bot's score against each other bot (win 1, draw 0.5), averaged.
Beaten: how often each other bot beat it (wins only), averaged; what a win-rate gate sees.

| # | Bot | Rating | Mean score | Beaten | W-D-L |
|---|---|---|---|---|---|
| 1 | noop | -398 ± 20 | 0.13 | 0.74 | 0-181-519 |
| 2 | balanced_random | -250 ± 17 | 0.26 | 0.53 | 34-292-374 |
| 3 | random_10 | -180 ± 16 | 0.33 | 0.34 | 1-459-240 |
| 4 | mix(balanced_random/simple,0.25) | -180 ± 16 | 0.33 | 0.45 | 75-311-314 |
| 5 | random_15 | -168 ± 16 | 0.34 | 0.32 | 4-471-225 |
| 6 | mix(balanced_random/simple,0.5) | -144 ± 16 | 0.37 | 0.41 | 104-309-287 |
| 7 | random_20 | -118 ± 16 | 0.40 | 0.30 | 68-420-212 |
| 8 | mix(random_20/simple,0.25) | -108 ± 16 | 0.41 | 0.31 | 86-400-214 |
| 9 | mix(random_20/simple,0.5) | -99 ± 16 | 0.42 | 0.32 | 109-368-223 |
| 10 | simple | -61 ± 16 | 0.46 | 0.31 | 162-321-217 |
| 11 | mix(simple/medium,0.5) | +163 ± 18 | 0.68 | 0.18 | 382-190-128 |
| 12 | advanced | +313 ± 20 | 0.80 | 0.19 | 552-15-133 |
| 13 | master | +323 ± 20 | 0.81 | 0.19 | 561-7-132 |
| 14 | mix(medium/advanced,0.5) | +388 ± 22 | 0.85 | 0.14 | 589-11-100 |
| 15 | medium | +521 ± 26 | 0.92 | 0.06 | 633-25-42 |

Crosstable: the row bot's W-D-L against the column bot over both seats (columns numbered as the rows).

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-18-32 | 0-49-1 | 0-7-43 | 0-49-1 | 0-5-45 | 0-24-26 | 0-20-30 | 0-9-41 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 2 | balanced_random | 32-18-0 | · | 1-49-0 | 1-38-11 | 0-49-1 | 0-24-26 | 0-35-15 | 0-35-15 | 0-27-23 | 0-11-39 | 0-6-44 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 3 | random_10 | 1-49-0 | 0-49-1 | · | 0-49-1 | 0-50-0 | 0-45-5 | 0-50-0 | 0-48-2 | 0-48-2 | 0-44-6 | 0-25-25 | 0-0-50 | 0-0-50 | 0-1-49 | 0-1-49 |
| 4 | mix(balanced_random/simple,0.25) | 43-7-0 | 11-38-1 | 1-49-0 | · | 3-46-1 | 8-28-14 | 1-39-10 | 1-35-14 | 3-30-17 | 1-32-17 | 0-7-43 | 0-0-50 | 0-0-50 | 2-0-48 | 1-0-49 |
| 5 | random_15 | 1-49-0 | 1-49-0 | 0-50-0 | 1-46-3 | · | 1-46-3 | 0-50-0 | 0-48-2 | 0-48-2 | 0-42-8 | 0-31-19 | 0-3-47 | 0-2-48 | 0-2-48 | 0-5-45 |
| 6 | mix(balanced_random/simple,0.5) | 45-5-0 | 26-24-0 | 5-45-0 | 14-28-8 | 3-46-1 | · | 1-38-11 | 2-37-11 | 5-36-9 | 0-31-19 | 0-14-36 | 2-0-48 | 0-0-50 | 1-0-49 | 0-5-45 |
| 7 | random_20 | 26-24-0 | 15-35-0 | 0-50-0 | 10-39-1 | 0-50-0 | 11-38-1 | · | 1-47-2 | 0-50-0 | 0-48-2 | 0-28-22 | 0-6-44 | 1-4-45 | 2-1-47 | 2-0-48 |
| 8 | mix(random_20/simple,0.25) | 30-20-0 | 15-35-0 | 2-48-0 | 14-35-1 | 2-48-0 | 11-37-2 | 2-47-1 | · | 2-46-2 | 1-43-6 | 2-30-18 | 1-3-46 | 1-0-49 | 3-2-45 | 0-6-44 |
| 9 | mix(random_20/simple,0.5) | 41-9-0 | 23-27-0 | 2-48-0 | 17-30-3 | 2-48-0 | 9-36-5 | 0-50-0 | 2-46-2 | · | 5-43-2 | 1-21-28 | 2-3-45 | 1-1-48 | 3-2-45 | 1-4-45 |
| 10 | simple | 50-0-0 | 39-11-0 | 6-44-0 | 17-32-1 | 8-42-0 | 19-31-0 | 2-48-0 | 6-43-1 | 2-43-5 | · | 2-24-24 | 1-0-49 | 2-0-48 | 4-2-44 | 4-1-45 |
| 11 | mix(simple/medium,0.5) | 50-0-0 | 44-6-0 | 25-25-0 | 43-7-0 | 19-31-0 | 36-14-0 | 22-28-0 | 18-30-2 | 28-21-1 | 24-24-2 | · | 19-0-31 | 26-0-24 | 17-1-32 | 11-3-36 |
| 12 | advanced | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 47-3-0 | 48-0-2 | 44-6-0 | 46-3-1 | 45-3-2 | 49-0-1 | 31-0-19 | · | 25-0-25 | 13-0-37 | 4-0-46 |
| 13 | master | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 48-2-0 | 50-0-0 | 45-4-1 | 49-0-1 | 48-1-1 | 48-0-2 | 24-0-26 | 25-0-25 | · | 19-0-31 | 5-0-45 |
| 14 | mix(medium/advanced,0.5) | 50-0-0 | 50-0-0 | 49-1-0 | 48-0-2 | 48-2-0 | 49-0-1 | 47-1-2 | 45-2-3 | 45-2-3 | 44-2-4 | 32-1-17 | 37-0-13 | 31-0-19 | · | 14-0-36 |
| 15 | medium | 50-0-0 | 50-0-0 | 49-1-0 | 49-0-1 | 45-5-0 | 45-5-0 | 48-0-2 | 44-6-0 | 45-4-1 | 45-1-4 | 36-3-11 | 46-0-4 | 45-0-5 | 36-0-14 | · |

By seat: the row bot's W-D-L in seat 1 (moving first) against the column bot in seat 2; the column bot's own seat-1 games are in its row. The diagonal is the mirror match, when played.

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-9-16 | 0-25-0 | 0-4-21 | 0-24-1 | 0-2-23 | 0-13-12 | 0-10-15 | 0-5-20 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 2 | balanced_random | 16-9-0 | · | 0-25-0 | 0-20-5 | 0-25-0 | 0-16-9 | 0-19-6 | 0-19-6 | 0-14-11 | 0-6-19 | 0-2-23 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 3 | random_10 | 1-24-0 | 0-24-1 | · | 0-24-1 | 0-25-0 | 0-22-3 | 0-25-0 | 0-23-2 | 0-23-2 | 0-20-5 | 0-12-13 | 0-0-25 | 0-0-25 | 0-1-24 | 0-0-25 |
| 4 | mix(balanced_random/simple,0.25) | 22-3-0 | 6-18-1 | 0-25-0 | · | 1-23-1 | 3-15-7 | 1-16-8 | 0-18-7 | 1-14-10 | 0-18-7 | 0-4-21 | 0-0-25 | 0-0-25 | 2-0-23 | 1-0-24 |
| 5 | random_15 | 0-25-0 | 1-24-0 | 0-25-0 | 0-23-2 | · | 1-22-2 | 0-25-0 | 0-24-1 | 0-23-2 | 0-22-3 | 0-16-9 | 0-1-24 | 0-0-25 | 0-1-24 | 0-2-23 |
| 6 | mix(balanced_random/simple,0.5) | 22-3-0 | 17-8-0 | 2-23-0 | 7-13-5 | 1-24-0 | · | 1-17-7 | 1-17-7 | 1-19-5 | 0-14-11 | 0-7-18 | 2-0-23 | 0-0-25 | 0-0-25 | 0-4-21 |
| 7 | random_20 | 14-11-0 | 9-16-0 | 0-25-0 | 2-23-0 | 0-25-0 | 4-21-0 | · | 0-24-1 | 0-25-0 | 0-24-1 | 0-17-8 | 0-2-23 | 0-1-24 | 1-1-23 | 1-0-24 |
| 8 | mix(random_20/simple,0.25) | 15-10-0 | 9-16-0 | 0-25-0 | 7-17-1 | 1-24-0 | 4-20-1 | 1-23-1 | · | 0-24-1 | 0-20-5 | 0-15-10 | 1-2-22 | 0-0-25 | 1-2-22 | 0-2-23 |
| 9 | mix(random_20/simple,0.5) | 21-4-0 | 12-13-0 | 0-25-0 | 7-16-2 | 0-25-0 | 4-17-4 | 0-25-0 | 1-22-2 | · | 0-23-2 | 0-12-13 | 2-1-22 | 1-0-24 | 1-0-24 | 1-3-21 |
| 10 | simple | 25-0-0 | 20-5-0 | 1-24-0 | 10-14-1 | 5-20-0 | 8-17-0 | 1-24-0 | 1-23-1 | 0-20-5 | · | 1-12-12 | 1-0-24 | 2-0-23 | 4-2-19 | 1-1-23 |
| 11 | mix(simple/medium,0.5) | 25-0-0 | 21-4-0 | 12-13-0 | 22-3-0 | 10-15-0 | 18-7-0 | 14-11-0 | 8-15-2 | 15-9-1 | 12-12-1 | · | 11-0-14 | 14-0-11 | 12-1-12 | 11-2-12 |
| 12 | advanced | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 23-2-0 | 25-0-0 | 21-4-0 | 24-1-0 | 23-2-0 | 25-0-0 | 17-0-8 | · | 10-0-15 | 5-0-20 | 4-0-21 |
| 13 | master | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 23-2-0 | 25-0-0 | 21-3-1 | 24-0-1 | 24-1-0 | 25-0-0 | 13-0-12 | 10-0-15 | · | 9-0-16 | 3-0-22 |
| 14 | mix(medium/advanced,0.5) | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 24-1-0 | 24-0-1 | 24-0-1 | 23-0-2 | 21-2-2 | 25-0-0 | 20-0-5 | 17-0-8 | 15-0-10 | · | 14-0-11 |
| 15 | medium | 25-0-0 | 25-0-0 | 24-1-0 | 25-0-0 | 22-3-0 | 24-1-0 | 24-0-1 | 21-4-0 | 24-1-0 | 22-0-3 | 24-1-0 | 25-0-0 | 23-0-2 | 25-0-0 | · |

### maps/1v1/skirmish.csv, max_turns 120

Seat 1 (the player who moves first) scores 52.3% over every game on this board (1814-1863-1573 W-D-L).

Strength ordering, weakest first. Rating: Bradley-Terry, Elo scale, centred on 0 (± one standard error).
Mean score: the bot's score against each other bot (win 1, draw 0.5), averaged.
Beaten: how often each other bot beat it (wins only), averaged; what a win-rate gate sees.

| # | Bot | Rating | Mean score | Beaten | W-D-L |
|---|---|---|---|---|---|
| 1 | noop | -362 ± 19 | 0.14 | 0.71 | 0-201-499 |
| 2 | balanced_random | -213 ± 16 | 0.28 | 0.51 | 52-288-360 |
| 3 | random_15 | -165 ± 16 | 0.33 | 0.34 | 1-462-237 |
| 4 | random_10 | -165 ± 16 | 0.33 | 0.34 | 1-463-236 |
| 5 | random_20 | -165 ± 16 | 0.33 | 0.34 | 5-455-240 |
| 6 | mix(balanced_random/simple,0.25) | -135 ± 16 | 0.37 | 0.42 | 105-302-293 |
| 7 | mix(random_20/simple,0.25) | -122 ± 16 | 0.38 | 0.32 | 58-416-226 |
| 8 | mix(balanced_random/simple,0.5) | -103 ± 16 | 0.40 | 0.38 | 130-302-268 |
| 9 | mix(random_20/simple,0.5) | -94 ± 15 | 0.41 | 0.32 | 101-375-224 |
| 10 | simple | +8 ± 16 | 0.53 | 0.27 | 227-283-190 |
| 11 | mix(simple/medium,0.5) | +133 ± 17 | 0.66 | 0.23 | 376-166-158 |
| 12 | medium | +322 ± 20 | 0.82 | 0.18 | 570-5-125 |
| 13 | mix(medium/advanced,0.5) | +336 ± 20 | 0.83 | 0.17 | 576-7-117 |
| 14 | advanced | +358 ± 20 | 0.84 | 0.16 | 590-1-109 |
| 15 | master | +367 ± 21 | 0.85 | 0.15 | 595-0-105 |

Crosstable: the row bot's W-D-L against the column bot over both seats (columns numbered as the rows).

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-0-50 | 0-49-1 | 0-49-1 | 0-48-2 | 0-0-50 | 0-35-15 | 0-0-50 | 0-20-30 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 2 | balanced_random | 50-0-0 | · | 0-50-0 | 0-50-0 | 0-49-1 | 1-33-16 | 0-43-7 | 1-26-23 | 0-32-18 | 0-3-47 | 0-2-48 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 3 | random_15 | 1-49-0 | 0-50-0 | · | 0-50-0 | 0-50-0 | 0-48-2 | 0-49-1 | 0-49-1 | 0-49-1 | 0-40-10 | 0-28-22 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 4 | random_10 | 1-49-0 | 0-50-0 | 0-50-0 | · | 0-50-0 | 0-48-2 | 0-47-3 | 0-48-2 | 0-48-2 | 0-48-2 | 0-25-25 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 5 | random_20 | 2-48-0 | 1-49-0 | 0-50-0 | 0-50-0 | · | 2-48-0 | 0-49-1 | 0-49-1 | 0-45-5 | 0-44-6 | 0-23-27 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 6 | mix(balanced_random/simple,0.25) | 50-0-0 | 16-33-1 | 2-48-0 | 2-48-0 | 0-48-2 | · | 2-37-11 | 13-20-17 | 2-39-9 | 5-22-23 | 4-7-39 | 5-0-45 | 3-0-47 | 0-0-50 | 1-0-49 |
| 7 | mix(random_20/simple,0.25) | 15-35-0 | 7-43-0 | 1-49-0 | 3-47-0 | 1-49-0 | 11-37-2 | · | 6-39-5 | 2-43-5 | 0-42-8 | 4-32-14 | 3-0-47 | 4-0-46 | 0-0-50 | 1-0-49 |
| 8 | mix(balanced_random/simple,0.5) | 50-0-0 | 23-26-1 | 1-49-0 | 2-48-0 | 1-49-0 | 17-20-13 | 5-39-6 | · | 4-37-9 | 3-20-27 | 6-11-33 | 8-0-42 | 6-2-42 | 2-1-47 | 2-0-48 |
| 9 | mix(random_20/simple,0.5) | 30-20-0 | 18-32-0 | 1-49-0 | 2-48-0 | 5-45-0 | 9-39-2 | 5-43-2 | 9-37-4 | · | 1-42-7 | 7-17-26 | 9-3-38 | 3-0-47 | 1-0-49 | 1-0-49 |
| 10 | simple | 50-0-0 | 47-3-0 | 10-40-0 | 2-48-0 | 6-44-0 | 23-22-5 | 8-42-0 | 27-20-3 | 7-42-1 | · | 16-18-16 | 16-1-33 | 13-3-34 | 1-0-49 | 1-0-49 |
| 11 | mix(simple/medium,0.5) | 50-0-0 | 48-2-0 | 22-28-0 | 25-25-0 | 27-23-0 | 39-7-4 | 14-32-4 | 33-11-6 | 26-17-7 | 16-18-16 | · | 26-1-23 | 19-2-29 | 14-0-36 | 17-0-33 |
| 12 | medium | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 45-0-5 | 47-0-3 | 42-0-8 | 38-3-9 | 33-1-16 | 23-1-26 | · | 25-0-25 | 36-0-14 | 31-0-19 |
| 13 | mix(medium/advanced,0.5) | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 47-0-3 | 46-0-4 | 42-2-6 | 47-0-3 | 34-3-13 | 29-2-19 | 25-0-25 | · | 29-0-21 | 27-0-23 |
| 14 | advanced | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 47-1-2 | 49-0-1 | 49-0-1 | 36-0-14 | 14-0-36 | 21-0-29 | · | 24-0-26 |
| 15 | master | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 49-0-1 | 49-0-1 | 48-0-2 | 49-0-1 | 49-0-1 | 33-0-17 | 19-0-31 | 23-0-27 | 26-0-24 | · |

By seat: the row bot's W-D-L in seat 1 (moving first) against the column bot in seat 2; the column bot's own seat-1 games are in its row. The diagonal is the mirror match, when played.

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-0-25 | 0-25-0 | 0-24-1 | 0-24-1 | 0-0-25 | 0-17-8 | 0-0-25 | 0-10-15 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 2 | balanced_random | 25-0-0 | · | 0-25-0 | 0-25-0 | 0-25-0 | 0-17-8 | 0-22-3 | 1-15-9 | 0-14-11 | 0-2-23 | 0-2-23 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 3 | random_15 | 1-24-0 | 0-25-0 | · | 0-25-0 | 0-25-0 | 0-23-2 | 0-25-0 | 0-24-1 | 0-24-1 | 0-20-5 | 0-15-10 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 4 | random_10 | 0-25-0 | 0-25-0 | 0-25-0 | · | 0-25-0 | 0-23-2 | 0-25-0 | 0-25-0 | 0-24-1 | 0-23-2 | 0-13-12 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 5 | random_20 | 1-24-0 | 1-24-0 | 0-25-0 | 0-25-0 | · | 2-23-0 | 0-24-1 | 0-24-1 | 0-23-2 | 0-22-3 | 0-12-13 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 6 | mix(balanced_random/simple,0.25) | 25-0-0 | 8-16-1 | 0-25-0 | 0-25-0 | 0-25-0 | · | 2-18-5 | 6-11-8 | 1-20-4 | 3-11-11 | 2-3-20 | 2-0-23 | 2-0-23 | 0-0-25 | 1-0-24 |
| 7 | mix(random_20/simple,0.25) | 7-18-0 | 4-21-0 | 1-24-0 | 3-22-0 | 0-25-0 | 6-19-0 | · | 4-19-2 | 1-20-4 | 0-20-5 | 1-16-8 | 2-0-23 | 2-0-23 | 0-0-25 | 1-0-24 |
| 8 | mix(balanced_random/simple,0.5) | 25-0-0 | 14-11-0 | 0-25-0 | 2-23-0 | 0-25-0 | 9-9-7 | 3-20-2 | · | 2-18-5 | 2-11-12 | 2-7-16 | 3-0-22 | 5-0-20 | 2-1-22 | 2-0-23 |
| 9 | mix(random_20/simple,0.5) | 15-10-0 | 7-18-0 | 0-25-0 | 1-24-0 | 3-22-0 | 5-19-1 | 1-23-1 | 4-19-2 | · | 1-21-3 | 4-7-14 | 4-2-19 | 2-0-23 | 1-0-24 | 1-0-24 |
| 10 | simple | 25-0-0 | 24-1-0 | 5-20-0 | 0-25-0 | 3-22-0 | 12-11-2 | 3-22-0 | 15-9-1 | 4-21-0 | · | 10-5-10 | 4-1-20 | 7-1-17 | 1-0-24 | 1-0-24 |
| 11 | mix(simple/medium,0.5) | 25-0-0 | 25-0-0 | 12-13-0 | 13-12-0 | 14-11-0 | 19-4-2 | 6-16-3 | 17-4-4 | 12-10-3 | 6-13-6 | · | 16-0-9 | 15-1-9 | 11-0-14 | 12-0-13 |
| 12 | medium | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 22-0-3 | 24-0-1 | 20-0-5 | 19-1-5 | 13-0-12 | 14-1-10 | · | 23-0-2 | 22-0-3 | 21-0-4 |
| 13 | mix(medium/advanced,0.5) | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 24-0-1 | 23-0-2 | 22-2-1 | 24-0-1 | 17-2-6 | 20-1-4 | 23-0-2 | · | 22-0-3 | 20-0-5 |
| 14 | advanced | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 22-0-3 | 11-0-14 | 18-0-7 | · | 17-0-8 |
| 15 | master | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 20-0-5 | 15-0-10 | 18-0-7 | 18-0-7 | · |

### maps/1v1/corner_points.csv, max_turns 200

Seat 1 (the player who moves first) scores 51.4% over every game on this board (1634-2126-1490 W-D-L).

Strength ordering, weakest first. Rating: Bradley-Terry, Elo scale, centred on 0 (± one standard error).
Mean score: the bot's score against each other bot (win 1, draw 0.5), averaged.
Beaten: how often each other bot beat it (wins only), averaged; what a win-rate gate sees.

| # | Bot | Rating | Mean score | Beaten | W-D-L |
|---|---|---|---|---|---|
| 1 | noop | -245 ± 17 | 0.21 | 0.59 | 0-287-413 |
| 2 | balanced_random | -187 ± 16 | 0.27 | 0.50 | 18-335-347 |
| 3 | random_10 | -131 ± 15 | 0.33 | 0.34 | 0-463-237 |
| 4 | random_20 | -125 ± 15 | 0.34 | 0.33 | 1-470-229 |
| 5 | random_15 | -125 ± 15 | 0.34 | 0.32 | 0-473-227 |
| 6 | mix(balanced_random/simple,0.25) | -90 ± 15 | 0.38 | 0.36 | 82-369-249 |
| 7 | mix(random_20/simple,0.25) | -42 ± 14 | 0.44 | 0.24 | 88-444-168 |
| 8 | mix(balanced_random/simple,0.5) | -6 ± 14 | 0.49 | 0.25 | 161-365-174 |
| 9 | mix(random_20/simple,0.5) | +24 ± 14 | 0.53 | 0.18 | 170-404-126 |
| 10 | simple | +139 ± 15 | 0.68 | 0.07 | 299-352-49 |
| 11 | mix(medium/advanced,0.5) | +143 ± 15 | 0.68 | 0.30 | 466-24-210 |
| 12 | mix(simple/medium,0.5) | +148 ± 15 | 0.69 | 0.17 | 384-196-120 |
| 13 | master | +155 ± 15 | 0.70 | 0.29 | 481-14-205 |
| 14 | advanced | +159 ± 15 | 0.70 | 0.29 | 488-6-206 |
| 15 | medium | +183 ± 16 | 0.73 | 0.23 | 486-50-164 |

Crosstable: the row bot's W-D-L against the column bot over both seats (columns numbered as the rows).

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-34-16 | 0-50-0 | 0-49-1 | 0-50-0 | 0-31-19 | 0-36-14 | 0-17-33 | 0-20-30 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 | 0-0-50 |
| 2 | balanced_random | 16-34-0 | · | 0-50-0 | 1-49-0 | 1-49-0 | 0-41-9 | 0-45-5 | 0-20-30 | 0-33-17 | 0-7-43 | 0-0-50 | 0-7-43 | 0-0-50 | 0-0-50 | 0-0-50 |
| 3 | random_10 | 0-50-0 | 0-50-0 | · | 0-50-0 | 0-50-0 | 0-48-2 | 0-49-1 | 0-48-2 | 0-50-0 | 0-43-7 | 0-0-50 | 0-25-25 | 0-0-50 | 0-0-50 | 0-0-50 |
| 4 | random_20 | 1-49-0 | 0-49-1 | 0-50-0 | · | 0-50-0 | 0-50-0 | 0-50-0 | 0-50-0 | 0-47-3 | 0-50-0 | 0-0-50 | 0-25-25 | 0-0-50 | 0-0-50 | 0-0-50 |
| 5 | random_15 | 0-50-0 | 0-49-1 | 0-50-0 | 0-50-0 | · | 0-50-0 | 0-48-2 | 0-50-0 | 0-50-0 | 0-46-4 | 0-0-50 | 0-30-20 | 0-0-50 | 0-0-50 | 0-0-50 |
| 6 | mix(balanced_random/simple,0.25) | 19-31-0 | 9-41-0 | 2-48-0 | 0-50-0 | 0-50-0 | · | 1-40-9 | 8-25-17 | 1-40-9 | 1-29-20 | 10-2-38 | 10-7-33 | 4-0-46 | 10-1-39 | 7-5-38 |
| 7 | mix(random_20/simple,0.25) | 14-36-0 | 5-45-0 | 1-49-0 | 0-50-0 | 2-48-0 | 9-40-1 | · | 4-45-1 | 1-48-1 | 0-50-0 | 14-1-35 | 9-29-12 | 15-0-35 | 7-0-43 | 7-3-40 |
| 8 | mix(balanced_random/simple,0.5) | 33-17-0 | 30-20-0 | 2-48-0 | 0-50-0 | 0-50-0 | 17-25-8 | 1-45-4 | · | 0-43-7 | 3-23-24 | 20-8-22 | 9-17-24 | 16-6-28 | 20-0-30 | 10-13-27 |
| 9 | mix(random_20/simple,0.5) | 30-20-0 | 17-33-0 | 0-50-0 | 3-47-0 | 0-50-0 | 9-40-1 | 1-48-1 | 7-43-0 | · | 0-48-2 | 21-2-27 | 18-16-16 | 16-3-31 | 22-1-27 | 26-3-21 |
| 10 | simple | 50-0-0 | 43-7-0 | 7-43-0 | 0-50-0 | 4-46-0 | 20-29-1 | 0-50-0 | 24-23-3 | 2-48-0 | · | 30-8-12 | 21-25-4 | 35-5-10 | 32-2-16 | 31-16-3 |
| 11 | mix(medium/advanced,0.5) | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 38-2-10 | 35-1-14 | 22-8-20 | 27-2-21 | 12-8-30 | · | 12-3-35 | 26-0-24 | 24-0-26 | 20-0-30 |
| 12 | mix(simple/medium,0.5) | 50-0-0 | 43-7-0 | 25-25-0 | 25-25-0 | 20-30-0 | 33-7-10 | 12-29-9 | 24-17-9 | 16-16-18 | 4-25-21 | 35-3-12 | · | 39-0-11 | 31-2-17 | 27-10-13 |
| 13 | master | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 46-0-4 | 35-0-15 | 28-6-16 | 31-3-16 | 10-5-35 | 24-0-26 | 11-0-39 | · | 26-0-24 | 20-0-30 |
| 14 | advanced | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 39-1-10 | 43-0-7 | 30-0-20 | 27-1-22 | 16-2-32 | 26-0-24 | 17-2-31 | 24-0-26 | · | 16-0-34 |
| 15 | medium | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 50-0-0 | 38-5-7 | 40-3-7 | 27-13-10 | 21-3-26 | 3-16-31 | 30-0-20 | 13-10-27 | 30-0-20 | 34-0-16 | · |

By seat: the row bot's W-D-L in seat 1 (moving first) against the column bot in seat 2; the column bot's own seat-1 games are in its row. The diagonal is the mirror match, when played.

| # | Bot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | noop | · | 0-17-8 | 0-25-0 | 0-25-0 | 0-25-0 | 0-16-9 | 0-18-7 | 0-9-16 | 0-10-15 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 | 0-0-25 |
| 2 | balanced_random | 8-17-0 | · | 0-25-0 | 0-25-0 | 1-24-0 | 0-21-4 | 0-22-3 | 0-10-15 | 0-16-9 | 0-3-22 | 0-0-25 | 0-4-21 | 0-0-25 | 0-0-25 | 0-0-25 |
| 3 | random_10 | 0-25-0 | 0-25-0 | · | 0-25-0 | 0-25-0 | 0-24-1 | 0-25-0 | 0-24-1 | 0-25-0 | 0-20-5 | 0-0-25 | 0-12-13 | 0-0-25 | 0-0-25 | 0-0-25 |
| 4 | random_20 | 1-24-0 | 0-24-1 | 0-25-0 | · | 0-25-0 | 0-25-0 | 0-25-0 | 0-25-0 | 0-24-1 | 0-25-0 | 0-0-25 | 0-12-13 | 0-0-25 | 0-0-25 | 0-0-25 |
| 5 | random_15 | 0-25-0 | 0-25-0 | 0-25-0 | 0-25-0 | · | 0-25-0 | 0-24-1 | 0-25-0 | 0-25-0 | 0-23-2 | 0-0-25 | 0-15-10 | 0-0-25 | 0-0-25 | 0-0-25 |
| 6 | mix(balanced_random/simple,0.25) | 10-15-0 | 5-20-0 | 1-24-0 | 0-25-0 | 0-25-0 | · | 1-19-5 | 5-12-8 | 1-19-5 | 1-11-13 | 3-2-20 | 4-4-17 | 2-0-23 | 4-0-21 | 4-2-19 |
| 7 | mix(random_20/simple,0.25) | 7-18-0 | 2-23-0 | 1-24-0 | 0-25-0 | 1-24-0 | 4-21-0 | · | 2-23-0 | 0-25-0 | 0-25-0 | 7-0-18 | 5-15-5 | 7-0-18 | 2-0-23 | 4-1-20 |
| 8 | mix(balanced_random/simple,0.5) | 17-8-0 | 15-10-0 | 1-24-0 | 0-25-0 | 0-25-0 | 9-13-3 | 1-22-2 | · | 0-20-5 | 2-12-11 | 11-4-10 | 4-10-11 | 6-5-14 | 14-0-11 | 3-9-13 |
| 9 | mix(random_20/simple,0.5) | 15-10-0 | 8-17-0 | 0-25-0 | 2-23-0 | 0-25-0 | 4-21-0 | 1-23-1 | 2-23-0 | · | 0-25-0 | 12-0-13 | 10-8-7 | 8-0-17 | 11-1-13 | 13-1-11 |
| 10 | simple | 25-0-0 | 21-4-0 | 2-23-0 | 0-25-0 | 2-23-0 | 7-18-0 | 0-25-0 | 13-11-1 | 2-23-0 | · | 17-5-3 | 10-11-4 | 19-1-5 | 16-1-8 | 13-9-3 |
| 11 | mix(medium/advanced,0.5) | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 18-0-7 | 17-1-7 | 12-4-9 | 14-2-9 | 9-3-13 | · | 10-0-15 | 16-0-9 | 16-0-9 | 17-0-8 |
| 12 | mix(simple/medium,0.5) | 25-0-0 | 22-3-0 | 12-13-0 | 12-13-0 | 10-15-0 | 16-3-6 | 7-14-4 | 13-7-5 | 9-8-8 | 0-14-11 | 20-3-2 | · | 20-0-5 | 16-2-7 | 19-4-2 |
| 13 | master | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 23-0-2 | 17-0-8 | 14-1-10 | 14-3-8 | 5-4-16 | 15-0-10 | 6-0-19 | · | 19-0-6 | 13-0-12 |
| 14 | advanced | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 18-1-6 | 20-0-5 | 19-0-6 | 14-0-11 | 8-1-16 | 17-0-8 | 10-0-15 | 18-0-7 | · | 11-0-14 |
| 15 | medium | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 25-0-0 | 19-3-3 | 20-2-3 | 14-4-7 | 10-2-13 | 0-7-18 | 22-0-3 | 11-6-8 | 18-0-7 | 20-0-5 | · |
