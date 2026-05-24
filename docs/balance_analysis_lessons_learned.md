# Balance Analysis — Lessons Learned

Retrospective on the bot-tournament balance-analysis pipeline
(`notebooks/balance_analysis.ipynb`, `reinforcetactics/tournament/`,
scripted bots in `reinforcetactics/game/bot.py`). Captures what surfaced
when the `baseline_20260524_034403` run showed the top three bots
bunched within a 12% winrate spread and the underlying signal turned
out to be much weaker than the sample size suggested.

Read this before drawing conclusions from a balance run, tuning unit
costs against tournament results, or extending the scripted-bot
hierarchy.

## TL;DR

1. **Deterministic bots + deterministic engine = duplicate trajectories.**
   `games_per_side > 1` against a fixed matchup-and-map writes N copies
   of the same game, not N independent samples. Wilson CIs computed on
   N look right but reflect only `unique / N` actual data points. The
   baseline run's 96 games were really ~12 unique games × 8 duplicates.
2. **Stochastic tiebreak (`rng_seed`) restores effective sample size.**
   Without changing any scoring logic — every bot still picks among
   its top-rated options. Just shuffle ties before sort/max/min so
   equivalent decisions resolve to different picks across episodes.
3. **Capability telemetry beats endstate diagnostics for "why bot X
   beat bot Y."** `endstate_per_game` records *what happened in the
   game state* (builds, gold, captures). It doesn't tell you *which
   heuristic the bot fired*. Per-game capability counters
   (`knight_charge`, `sorcerer_haste`, `retreat_to_heal`, …) close
   that gap.
4. **Priority + cost sort produces monocultures.** SimpleBot's
   "buy the highest-priority affordable unit" loop always picked
   Warrior (priority=1, cost=200 — strictly dominant) and built 100%
   W in production. Composition caps are required for any
   purchase logic that uses a strict priority sort, otherwise the
   bot stat-checks itself out of the game's roster diversity.
5. **Score-based attack evaluation can still recommend suicide.** A
   `value = damage_dealt - counter_damage + cost_term` function goes
   positive on fatal-but-net-favourable trades (a 200g Warrior
   trading 100 damage for 80 counter-damage *and dying* still scored
   +4). An explicit suicide guard — "if you'd die without killing,
   abort" — is required separately from the score function.
6. **PR #371's stochastic tiebreak only helps if every ranking site
   shuffles.** Incomplete coverage means some decision types stay
   deterministic while others go stochastic, biasing per-bot
   capability counts. The audit found 9 missing sites; the
   most-hit was `find_best_move_position`.
7. **Tournament infrastructure has latent bugs that hide in
   deterministic mode.** The replay filename collision
   (timestamp-only, second-precision) silently overwrote replays
   of the same matchup completing within the same second. Invisible
   until stochastic mode started producing distinct games per
   matchup; trivially fixed by including `game_id` in the filename.

## The duplicate-trajectory problem

The single largest source of measurement error in baseline_20260524_034403.

### Symptom

`bot_winrate_by_bucket.csv` showed Wilson 95% CIs of ±0.20 at N=12 per
bucket. Top three bots clustered within 12% winrate (MasterBot 70.8%,
AdvancedBot 60.4%, MediumBot 58.3%). The natural read: "we need more
games — sample size is too small."

### What was actually happening

The default config (`games_per_side=1`) writes one game per matchup
direction per map. But the bots and the game engine are *both*
deterministic — same starting state + same bot policies → byte-identical
trajectories. Stepping up to `games_per_side=4` doesn't help: all 4
games of `MediumBot vs SimpleBot on starter.csv` produce the same
action stream, the same winner, the same final state. The Wilson CI
formula assumes N independent samples; it has no way to know it's
being fed duplicates.

### Quantitative evidence

A 4-game/side smoke run of MediumBot vs SimpleBot on `starter.csv`,
before any other fixes:

```
rng_seed=None  (deterministic, baseline behaviour)
  4 replays saved, 4 byte-identical trajectories
  → 1 unique game, recorded 4 times

rng_seed=42  (stochastic tiebreak enabled)
  4 replays saved, 4 distinct trajectories
  Winners:    p1, p2, p1, p1
  Actions:    40, 144, 102, 126
  → 4 unique games

rng_seed=42, re-run
  Same 4 trajectories as the first stochastic run (reproducible)
```

So the baseline_20260524_034403's "96 games" were really 12 unique
games × 8 replicas. The CIs were narrow because the sample looked
large; the *information content* was just 12 outcomes per bucket.

### Mechanism

Every scripted-bot decision site is a sort/max/min over candidates
(reachable tiles, attackable enemies, affordable buys). When two
candidates score equally, Python's `sort`/`max`/`min` returns the
first one — insertion order, not a tiebreak. The deterministic engine
serves candidates in the same insertion order on every replay of the
same starting state. Net effect: the bot makes the same choice every
time.

PR #371 introduced `BotUnitMixin._maybe_shuffle` as the infrastructure
fix: a no-op when `_rng is None`, an `rng.shuffle(items)` when set.
Drop it before every sort/max/min and ties resolve randomly. But the
PR only added the rng *plumbing*; nothing in the tournament runner
*used* it. `rng_seed=None` stayed the implicit default.

### Fix

Three pieces, none of which work alone:

1. **`TournamentConfig.rng_seed: Optional[int] = None`**, defaulting
   to None so existing behaviour is preserved.
2. **Runner derives per-game per-side seeds** via
   `SHA-256(rng_seed, game_id, map_stem, bot1_name, bot2_name, player, bot_name)`.
   Must be a stable hash — Python's built-in `hash()` is salted per
   process, so re-running the same tournament would produce different
   seeds.
3. **`create_bot_instance` threads `rng` into each scripted bot's
   constructor.** Simple/Medium/Advanced/Master all already accepted
   `rng=None` from PR #371; the missing piece was the runner passing
   a non-None value.

With these, `rng_seed=42, games_per_side=4` produces 96 unique games
on the default 8-map pool — actually 96 independent samples — and a
re-run with the same seed reproduces the same set.

### Implication for past results

`baseline_20260524_034403`'s numbers are correct as a measurement of
*the single deterministic trajectory* on each matchup × map, but should
not be interpreted as statistical samples. The bunching between
AdvancedBot and MediumBot might be real, might be one specific
trajectory pair that happened to produce close winrates — there's no
way to tell from the existing data. Conclusions about *which* bots
need balance changes have to wait for a stochastic re-run.

## The Wilson CI display trap

Pre-fix, `bot_winrate_by_bucket.csv` would still print

```
AdvancedBot, large,  12, 8, 4, 0, 0.667, [0.39, 0.86]
```

for runs where the 12 games were actually 4 unique × 3 duplicates.
The CI formula doesn't know it's being lied to about N.

Mitigation shipped: a one-time warning print in the run cell when
`RNG_SEED is None and GAMES_PER_SIDE > 1`. Doesn't fix the underlying
CI display (that would need per-game-uniqueness deduplication and a
recomputed Wilson interval), but at least surfaces the issue before
anyone draws conclusions off the misleadingly narrow bars.

A more principled fix would be: notebook computes
`unique_trajectory_hashes / N` ratio per matchup, displays it next to
the CI columns. Open work item.

## SimpleBot monoculture (the priority-sort failure mode)

`bot_unit_gold_share.csv` in baseline_20260524_034403:

```
SimpleBot, W, Warrior, 1.000, 1.000, 34.27, 48
SimpleBot, A, Archer,  0.000
SimpleBot, K, Knight,  0.000
SimpleBot, M, Mage,    0.000
... all other unit types: 0.000
```

SimpleBot built **only Warriors** across all 48 games.

### Mechanism

`SimpleBot.purchase_units` loops:
1. Fetch affordable units (cost ≤ available gold).
2. Sort by `(UNIT_PRIORITIES[unit_type], cost)`.
3. Buy the first.
4. Repeat until gold is exhausted.

`UNIT_PRIORITIES` puts Warrior at priority 1 (highest). Warrior costs
200, the cheapest. So the sort key is (1, 200) — *strictly dominant*
over every other unit, no matter how much gold the bot has accumulated.
The loop just keeps buying Warriors.

There's no anti-dominance mechanism: no "diminishing returns" weighting,
no army-composition target, no tier-gate on gold-on-hand. The intuition
"the bot should diversify because it has more gold now" never holds —
Warrior wins the sort every time.

MediumBot has the same bug with W priority 0. AdvancedBot uses a
different (target-ratio-based) purchase function that diversifies
naturally; not affected.

### Fix

A composition cap, gated on minimum army size to preserve early-game
tempo:

```python
WARRIOR_SHARE_CAP = 0.5      # SimpleBot; 0.6 on MediumBot
WARRIOR_CAP_MIN_UNITS = 3    # 4 on MediumBot

if total_units >= WARRIOR_CAP_MIN_UNITS:
    w_share = w_count / total_units
    if w_share >= WARRIOR_SHARE_CAP:
        affordable = [a for a in affordable if a["unit_type"] != "W"]
```

When the cap triggers, Warriors drop out of the affordable set and
the next-priority unit (Barbarian, then Archer) fills the slot. The
bot still defaults to Warrior on the first 2-3 buys and any time
non-Warrior options aren't available.

### Smoke evidence

Pre-fix:
```
SimpleBot: cap_buy_W ≈ 7/game, cap_buy_B=0, cap_buy_A=0
MediumBot: cap_buy_W ≈ 7/game, cap_buy_B=0, cap_buy_A=0.5
```

Post-fix:
```
SimpleBot: cap_buy_W ≈ 6.5, cap_buy_B = 0.5, warrior_cap_hit = 0.5
MediumBot: cap_buy_W ≈ 6.3, cap_buy_B = 0.5, warrior_cap_hit = 0.5
```

The cap fires once per long game on average; light touch but
sufficient to break the monoculture. May need tightening
(`WARRIOR_SHARE_CAP=0.4`?) if the new baseline shows the bots still
playing as W-swarms.

### Generalised lesson

Any "rank candidates by N criteria, take the top one" loop is
vulnerable to this if one option is *jointly* dominant on all N
criteria simultaneously. The fix isn't "more criteria" (the dominant
option still wins) — it's a cap, a quota, or stochastic selection
proportional to score. For balance tooling specifically, audit every
`while True: sort → take[0] → buy` loop in the scripted bot hierarchy
the next time a new unit is added.

## The suicide-guard hole

`AdvancedBot.calculate_attack_value` (and its callers in MediumBot)
scored attacks by:

```python
value = damage_dealt - counter_damage
value += target_cost / 100.0
value -= (counter_damage * attacker_cost) / 1000.0
```

with a short-circuit at the top: if `damage_dealt >= target.health`
(kill confirm), return `1000 + damage_dealt`.

### The failure case

Consider a 200g Warrior at 100 HP attacking a 300g Knight at 110 HP
where the Warrior would deal 100 damage and take 80 counter-damage.

- `damage_dealt = 100`, `target.health = 110` — doesn't trigger kill
  confirm.
- `counter_damage = 80`, `attacker.health = 100` — Warrior survives
  with 20 HP. So far so good.

Now the same Warrior at 70 HP attacking the same Knight at 110 HP,
dealing 100 damage and taking 80 counter:

- `damage_dealt = 100 < 110`, no kill confirm.
- `counter_damage = 80 ≥ 70`, attacker **dies**.
- `value = 100 - 80 + 3.0 - 16.0 = +7.0` — positive.

The caller's gate is `if best_value > 0: attack`. So the bot **attacks
and dies**, dealing 100 damage to a 300g unit that survives — losing
200g for 80 damage that didn't kill anything. From the cost-weighted
score function's perspective this is a slightly positive trade. From
the actual game's perspective it's "throw away a 200g unit for nothing."

### Why the cost penalty doesn't catch this

The `(counter_damage * attacker_cost) / 1000.0` term is calibrated
for *risk* (small chance of bad outcome on each attack), not for
*certainty* (the attacker is mathematically dead). The cost penalty
scales with attacker_cost/1000, which is small (0.2 to 0.4 for the
common units) compared to the raw damage exchange (50-100 points).

### Fix

An explicit guard, separate from the score function:

```python
if counter_damage >= attacker.health and damage_dealt < target.health:
    self._record("suicide_eval_rejected")
    return -1000.0 - counter_damage
```

Returns a value strictly below the caller's `> 0` gate, so the bot
treats this option as worse than any other action — including doing
nothing this turn.

### Metric semantics gotcha

The naïve name for the counter was `suicide_blocked`. But the
recording fires inside `calculate_attack_value`, which is called
*per evaluation*, not per actual attack chosen. A bot considering
five attack candidates might trip the guard on three of them, then
attack a fourth (non-suicidal) target — `suicide_blocked: 3` would
overstate the guard's behavioural impact ("we declined three
attacks") when really it just rejected three options from the
candidate pool.

Renamed to `suicide_eval_rejected` to match what's actually counted.
Doesn't fix the semantic (the metric still counts evaluations, not
declined attacks); just makes the name honest. The more accurate
"declined attack" counter would need recording at the caller after
target selection, not inside the per-candidate evaluator.

## Capability telemetry vs end-state diagnostics

The pre-existing `endstate_per_game.csv` recorded the *game state
deltas* per game:

| Column | Records |
|---|---|
| builds_p1, gold_spent_p1 | What the player built |
| captures_p1, attacks_p1, damage_p1 | What happened |
| structures_p1_final | Final position |

These are descriptive — they tell you what *happened*. They don't tell
you *why*. If MasterBot beats AdvancedBot 10-6, was it because:

- MasterBot's threat-aware retreat preserved more units?
- MasterBot's HQ-snipe priority got there first in CONQUER phase?
- MasterBot's HP-ascending focus-fire kept its chipped units swinging?
- MasterBot's Sorcerer haste followthrough chained captures?

You can't tell from build/gold/damage numbers. They're effects, not
causes.

### The fix

Per-game counters on each scripted bot, lazily-created so subclasses
that bypass `BaseBot.__init__` still work:

```python
class BaseBot(ABC):
    def _record(self, name: str, n: int = 1) -> None:
        counters = getattr(self, "capabilities_fired", None)
        if counters is None:
            counters = {}
            self.capabilities_fired = counters
        counters[name] = counters.get(name, 0) + n
```

Each bot decision site records a named event:

```python
# AdvancedBot._try_knight_charge
if best_charge and best_value > 0:
    ...
    self._record("knight_charge")
    return True
```

Tournament runner snapshots `bot.capabilities_fired` into the replay's
`game_info`. Notebook ingests, builds `capabilities_per_game.csv`
(long format, one row per player-game) and `capabilities_per_bot.csv`
(mean firings per game).

### What this surfaces (smoke run, 12 games)

```
bot         | knight_charge | hq_snipe | retreat_to_heal | sorcerer_haste
AdvancedBot | 2.17          | 0.00     | 0.50            | 0.00
MasterBot   | 3.00          | 1.50     | 1.17            | (varies)
MediumBot   | 0.00          | 0.00     | 0.33            | 0.00
SimpleBot   | 0.00          | 0.00     | 0.00            | 0.00
```

The 10% Elo gap between AdvancedBot and MasterBot now has a
mechanistic story: ~3× more retreats-to-heal, exclusive use of
HQ-snipe priority, more Knight charges. Whether that *causes* the
winrate gap is a separate question (correlation, not causation), but
at least the differentiating behaviours are visible.

### Lesson

For any system where you need to debug *behaviour* (not state),
instrument the decision points, not just the outcomes. The data
volume cost is small (a dict of counters), the analytic value is
large.

## The PR #371 audit lesson

PR #371 ("optional stochastic tiebreak at every sort / max /
best-tracking site") added `_maybe_shuffle` at 21 ranking sites. The
audit found 9 more it missed:

- `find_best_move_position` (the most-hit site — every
  move-toward-target call across every bot)
- `try_cleric_abilities` heal-target `min(HP)`
- `try_mage_paralyze` paralyze-target `max(cost)` on both paths
- `try_use_special_ability` mage `for enemy in self.game_state.units`
- `_try_sorcerer_abilities` Priority 3 attack_buff, Priority 4
  defence_buff, Priority 5 defence_buff
- `try_ranged_attack` lowest-HP target picks (both branches)
- `_try_sorcerer_abilities` MasterBot double-capture combo

### Mechanistic concern

Partial coverage is worse than no coverage for a telemetry-oriented
goal. If 21 sites stochasticize and 9 stay deterministic, the
*shape* of the stochastic distribution depends on which decision the
bot made — a deterministic site that fires often (like `find_best_move`)
produces consistent patterns that bias the apparent capability rate
of every downstream metric. The naïve interpretation "MasterBot
charges 3× per game vs Advanced's 2.17×" might be inflated by
deterministic move-toward-target choices that consistently put
MasterBot's Knight in a chargeable position.

### Why it happened

The PR was scoped by `git grep "max\(|min\(|sort"` which misses
`for x in collection` first-match-wins loops (`try_use_special_ability`
mage), best-tracking via strict `<` (`find_best_move_position` —
no `max`/`min`/`sort` call at all), and helper functions in
`bot_base.py` (the PR focused on `bot.py`).

### Generalised lesson

When introducing a cross-cutting infrastructure change ("every X
should do Y"), the scoping query has to match the semantic intent,
not the syntactic surface. For "every ranking site": grep for the
output pattern (`best_*`, `*_target`, `*_best`), not the input
pattern (`max`, `min`, `sort`). Better still: add a code review
checklist or a CI lint rule that flags new sort/max/min/strict-comparison
ranking sites in the bot files. Otherwise the audit gap re-opens as
soon as someone adds a new heuristic.

## The replay filename collision

Latent bug for the entire lifetime of `games_per_side > 1`, but
invisible in deterministic mode (the duplicate-trajectory replays
overwrote with identical content — no functional impact). Surfaced
hard once stochastic mode produced distinct games per matchup that
the filename couldn't disambiguate.

### Mechanism

```python
replay_filename = (
    f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
    f"{bot1_desc.name}_vs_{bot2_desc.name}_{map_config.stem}.json"
)
```

Second-precision timestamp. Two games of the same matchup completing
within the same second produce identical filenames. `FileIO.save_replay`
just writes; no collision detection.

### Fix

Include `game_id` (already present on `ScheduledGame`) in the filename:

```python
replay_filename = (
    f"game_{...timestamp...}_id{game_id:04d}_{bot1}_vs_{bot2}_{map_stem}.json"
)
```

### Lesson

Defaults that "just happened to work" because of an unrelated
property (deterministic replays produced identical bytes) are
ticking bombs. Anything that derives a unique identifier from a
non-unique source (timestamps, random ints with default `random()`,
hash truncations) needs to either *prove* uniqueness or *handle*
collisions at the write site.

This pattern shows up elsewhere in the codebase — e.g.
`game_session_id` in the runner uses `datetime.now()` without
disambiguation. Worth a sweep next time someone touches the
tournament-runner write paths.

## Cross-cutting: how to do a balance run from now on

The shipping-default checklist for any balance-analysis run:

1. **Set `RNG_SEED` to a fixed int.** `42` is fine; any int is fine
   as long as you record it. The notebook now warns when this is
   None and `GAMES_PER_SIDE > 1`.
2. **`GAMES_PER_SIDE = 4`** as the minimum (default is 1, which is
   fine for spot checks but gives N=12 per bucket — too narrow for
   anything beyond top-line standings).
3. **Inspect `capabilities_per_bot.csv` after the run.** If a bot's
   characteristic abilities (e.g. MasterBot's `hq_snipe`,
   `haste_followthrough`) aren't firing at the rate you expect,
   investigate before drawing conclusions from win/loss numbers.
4. **Cross-reference standings with `bot_unit_gold_share.csv`.**
   If a bot's gold-share-by-unit looks like the monoculture pattern
   (one unit at ≥0.9), you're measuring a degenerate strategy, not
   the bot's design.
5. **Don't compare runs across code commits without checking
   `engine_constants_hash`.** Balance changes that live in
   `constants.py` (Unit stats, starting gold, structure income)
   silently confound balance comparisons. The bootstrap doc's
   "engine-constant confound class" section is the relevant
   precedent — same lesson applies here.

## What's *not* worth chasing again

- **"More games will fix the CIs."** No — without `rng_seed`, more
  games are duplicates. The fix is the seed, not the count.
- **Chasing the Medium↔Advanced bunching as a design issue before
  running the stochastic baseline.** The baseline_20260524_034403
  data can't distinguish "they're actually close" from "we only have
  3 unique games per bucket." Wait for the post-`rng_seed` numbers.
- **More criteria in the priority sort to fix monocultures.** SimpleBot
  has 2 criteria (priority, cost); both are dominant for Warrior. Adding
  a 3rd dominant-for-Warrior criterion doesn't help. Caps, quotas, or
  proportional sampling are the only fixes.
- **Lowering the suicide guard's threshold to "preserve aggressive
  play."** The guard only fires when the attacker *certainly dies*
  AND the target *certainly survives*. Both conditions are fatal-trade
  diagnostics, not preference signals. There's no "more aggressive"
  knob to turn — only "ignore the guard and accept the unit losses,"
  which is what the bot was doing before.

## Future work

- **CI accuracy column in standings tables.** Compute and display
  `unique_trajectory_count / N` next to Wilson CIs so the reader can
  see when the displayed CI overstates the information content.
- **Skill-ablation tournaments.** Re-run with one capability disabled
  per pass (MasterBot with `_threat_map` off, MasterBot with
  `haste_followthrough` off, AdvancedBot with `counter_matrix` off).
  The Elo delta from each ablation gives a quantitative answer to
  "what differentiates bot N from bot N+1." Capability telemetry
  already provides the *observational* version; ablation is the
  *causal* version. Expensive (one tournament per ablation) but
  decisive.
- **Threat-aware-retreat as a separate capability counter.**
  MasterBot's `find_retreat_tile` picks a different (safer) tile than
  AdvancedBot's. Currently both record `retreat_to_heal`. Adding
  `retreat_threat_avoided` (incremented when the threat-map score
  differed from the parent's pick) would give the "MasterBot retreats
  safer than Advanced" claim direct evidence.
- **Asymmetric-quality bot tiers.** If the design goal is "Medium
  should be visibly worse than Advanced," the current implementation
  achieves it mostly by *roster restriction* (MediumBot can't build
  Cleric/Sorcerer/Rogue/Barbarian/Mage). With stochastic mode and
  the new capability telemetry, it's now possible to design a
  MediumBot that has the same roster but *worse heuristics* (e.g.
  shorter threat horizon, no charge-bonus exploitation, naive heal
  targeting). That would be a more interesting design space than
  "enabled_units differs."

## Cross-reference

The PPO bootstrap document
([`docs/bootstrap_lessons_learned.md`](bootstrap_lessons_learned.md))
has a "Knight buff: invisible to scripted bots, visible to RL" aside
that anticipated some of these findings — specifically, that scripted
bots' static heuristic priorities make them blind to stat changes
that RL would discover. With stochastic mode now available, the
*statistical* reliability of bot-tournament outputs is restored, but
the *blind-to-stat-changes* claim still stands: shuffling ties
doesn't make a fixed-priority bot start preferring a buffed unit.
For RL-relevant stat tuning, replay-level metrics (build counts,
gold shares, survival rates) remain the right signal, not bot
win/loss. The stochastic mode makes those replay-level metrics
statistically meaningful, which is the new contribution.
