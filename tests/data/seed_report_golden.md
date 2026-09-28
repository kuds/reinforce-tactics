# Seed validation report: 20260928_120000_val

Generated 2026-09-21T14:13:20+00:00 by scripts/eval/summarize_seeds.py (schema 1).

## Provenance and gate settings

|  |  |
|---|---|
| runs | s42 (20260928_120000_val_s42), s1042 (20260928_120000_val_s1042), s2042 (20260928_120000_val_s2042) |
| config | resolved_config.yaml (digest —) |
| git | abc1234 |
| gate | stochastic policy; both modes recorded: yes; criterion point; eval_freq 100; n_eval_episodes 10; seats [1]; n_eval_envs 1 |
| replicate check | ok: the runs' resolved configs differ only in seed, device, logging and labels |
| numbers | the final (promoting) eval of each stage, 'at gate' (gate-selected, biased upward near the threshold); rates with two-sided 95% Wilson intervals |

## 1. Per-seed outcome

| seed | status | cleared | deepest stage | stalled at | env steps | wall h | resumes | retries | meta fails | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| s42 | completed | 4/4 | D | — | 1.3k | 0.2 | 0 | 0 | 0 | stochastic |
| s1042 | interrupted | 3/4 | D | — | 1.4k | 0.2 | 1 | 0 | 0 | stochastic |
| s2042 | stalled | 2/4 | C | C | 2.5k | 0.4 | 0 | 1 | 0 | stochastic |

## 2. Per stage across seeds

Mean [min–max] over the seeds that reached the stage; steps: median [min–max] over the seeds that cleared it. Captures per episode are tower/building/HQ in the gate-mode eval. t-intervals are in per_stage.csv and summary.json.

| # | stage | cleared/reached | steps to promote | stoch WR | greedy WR | stoch draw | captures/ep | shaping abs | stoch WR per seed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 3/3 | 300 | 90% | 100% | 10% | 1.0/0.5/0.0 | 0.21 | s42=90% s1042=90% s2042=90% |
| 2 | B | 3/3 | 200 | 90% | 80% | 10% | 1.0/0.5/0.0 | 0.21 | s42=90% s1042=90% s2042=90% |
| 3 | C | 2/3 ⚠ | 700 [600–800] | 70% [30%–90%] | 43% [10%–60%] | 27% [10%–60%] | 1.0/0.5/0.0 | 0.36 [0.21–0.68] | s42=90% s1042=90% s2042=30% |
| 4 | D | 1/2 | 200 | 55% [30%–80%] | 55% [20%–90%] | 45% [20%–70%] | 1.0/0.5/0.0 | 0.38 [0.23–0.53] | s42=80% s1042=30% |

## 3. Per-seed detail

### s42 — 20260928_120000_val_s42 (completed, new layout)

| # | stage | outcome | steps | stoch W/D/L | greedy W/D/L | retries | captures/ep T/B/H | shaping abs | draw return/ep | cum steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | cleared | 300 | 9/1/0 90% [60%–98%] | 10/0/0 100% [72%–100%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 300 |
| 2 | B | cleared | 200 | 9/1/0 90% [60%–98%] | 8/2/0 80% [49%–94%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 500 |
| 3 | C | cleared | 600 | 9/1/0 90% [60%–98%] | 6/3/1 60% [31%–83%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 1.1k |
| 4 | D | cleared | 200 | 8/2/0 80% [49%–94%] | 9/1/0 90% [60%–98%] | 0 | 1.0/0.5/0.0 | 0.23 | -5.5 | 1.3k |

Steps/h by map: beginner 6.0k

### s1042 — 20260928_120000_val_s1042 (interrupted, new layout)

| # | stage | outcome | steps | stoch W/D/L | greedy W/D/L | retries | captures/ep T/B/H | shaping abs | draw return/ep | cum steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | cleared | 300 | 9/1/0 90% [60%–98%] | 10/0/0 100% [72%–100%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 300 |
| 2 | B | cleared | 200 | 9/1/0 90% [60%–98%] | 8/2/0 80% [49%–94%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 500 |
| 3 | C | cleared | 800 | 9/1/0 90% [60%–98%] | 6/3/1 60% [31%–83%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 1.3k |
| 4 | D | interrupted | 0 (censored) ~ | 3/7/0 30% [11%–60%] | 2/8/0 20% [6%–51%] | 0 | 1.0/0.5/0.0 | 0.53 | -5.5 | 1.4k |

Steps/h by map: beginner 6.0k

### s2042 — 20260928_120000_val_s2042 (stalled, new layout)

| # | stage | outcome | steps | stoch W/D/L | greedy W/D/L | retries | captures/ep T/B/H | shaping abs | draw return/ep | cum steps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | cleared | 300 | 9/1/0 90% [60%–98%] | 10/0/0 100% [72%–100%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 300 |
| 2 | B | cleared | 200 | 9/1/0 90% [60%–98%] | 8/2/0 80% [49%–94%] | 0 | 1.0/0.5/0.0 | 0.21 | -5.5 | 500 |
| 3 | C | stalled | 2.0k (censored) | 3/6/1 30% [11%–60%] | 1/8/1 10% [2%–40%] | 1 | 1.0/0.5/0.0 | 0.68 | -5.5 | 2.5k |

Steps/h by map: beginner 6.0k

## 4. Comparison

### Against v52a (1 run(s))

- baseline code: git 078313e (group: abc1234)
- the baseline is a legacy record (no per-row gate mode): it predates the eval-gate change, and the opponent-freeze fix, engine and bot legality and the MDP fixes; created 2026-06-01
- the baseline gated on the greedy policy: compare greedy win rates only
- the baseline resampled its eval set every eval block
- the baseline's stage steps are approximate (last minus first eval row: no stage step bounds recorded)
- numbers are the promoting (gate-selected) evals, 'at gate'; treat the comparison as qualitative
- deepest shared stage reached: D; stages cleared among the 4 shared: baseline s42=3; group s42=4, s1042=3, s2042=2

Config deltas (baseline → group):

| field | baseline | group |
|---|---|---|
| env.reward_config.turn_penalty | 0.0 | -0.5 |
| env.max_steps | 3000 | 2000 |
| env.max_actions_per_turn | null | 40 |
| gate mode | "greedy" | "stochastic" |
| eval.resample_eval_seeds | true | false |

| stage | base outcome | base steps | base greedy WR | base greedy draw | base captures/ep | base shaping | cleared | steps median | greedy WR | stoch WR | greedy draw | captures/ep | shaping | Δ greedy WR | steps ratio | Δ shaping |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A ✗ | cleared | 200 | 100% | 0% | 1.5 | 0.19 | 3/3 | 300 | 100% | 90% | 0% | 1.5 | 0.21 | +0 pp | 1.50× | +0.01 |
| B | cleared | 100 | 100% | 0% | 1.5 | 0.19 | 3/3 | 200 | 80% | 90% | 20% | 1.5 | 0.21 | -20 pp | 2.00× | +0.01 |
| C | cleared | 200 | 100% | 0% | 1.5 | 0.19 | 2/3 | 700 [600–800] | 43% [10%–60%] | 70% [30%–90%] | 47% | 1.5 | 0.36 | -57 pp | 3.50× | +0.17 |
| D | interrupted | 0 (censored) | 20% | 80% | 1.5 | 0.81 | 1/3 | 200 | 55% [20%–90%] | 55% [30%–80%] | 45% | 1.5 | 0.38 | +35 pp | — | -0.43 |

✗ not comparable (settings differ):
- A: opponent: 'mixed' vs 'random'

## 5. Flags

- **skip_ahead** s42 / B: cleared in 200 env steps (<= patience x eval_freq: the carried-in policy passed the gate almost at once)
- **skip_ahead** s42 / D: cleared in 200 env steps (<= patience x eval_freq: the carried-in policy passed the gate almost at once)
- **resumed** s1042: 1 resume(s); not bit-reproducible
- **skip_ahead** s1042 / B: cleared in 200 env steps (<= patience x eval_freq: the carried-in policy passed the gate almost at once)
- **skip_ahead** s2042 / B: cleared in 200 env steps (<= patience x eval_freq: the carried-in policy passed the gate almost at once)
- **max_steps_truncate** s2042 / C: 20.0% of final-eval episodes truncated at max_steps
- **seed_sensitive** C: s42: cleared, s1042: cleared, s2042: stalled
