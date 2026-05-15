# Experiment A — faithful reproduction of the 17-stage deep run

## Purpose

Every config variant v17–v24 stalled at `beginner_random_10`/`_15`. v24
reverted the engine economy to the rt-0.2.5 values *but still stalled at
stage 5* — proving economy reversion alone does not recover the deep
progression. Two confounds remained uncontrolled: the **curriculum
structure** (v24 uses the 33-stage v18 curriculum; the deep run used a
24-stage rt-0.2.5 curriculum) and **engine code drift** (we reverted
constants.py but not the 0.2.5→0.2.7 behavioral code).

Experiment A removes *all* confounds by running the deep run's exact code
and config, unmodified, straight from git history. No porting, no
reverting, no new config.

## What the deep run actually was

Recovered from the corrected `config.json` audit + git archaeology:

| Property | Value | Source |
|----------|-------|--------|
| Run | `20260511_132922` → `skirmish_simple` (~16 stages cleared of 24) | scan |
| Commit | **`6eb0566`** (2026-05-11, rt 0.2.5) | `meta.git.short` |
| Config | `configs/bootstrap.yaml` **at that commit** (24 stages, `turn_penalty: -0.2`) | git |
| Roster | `[W, M, C, A, K]` — **not** in the YAML; applied by an *active* override in `ppo_bootstrap.ipynb` cell 11 | git |
| Economy | `STARTING_GOLD=250`, `HEADQUARTERS_INCOME=150`, Warrior atk 10, Knight def 5 | `constants.py` @ 6eb0566 |

The roster, turn_penalty, economy, AND the (shorter, different)
curriculum all come **for free** from checking out `6eb0566`. The
notebook at that commit already has the `[W,M,C,A,K]` override active in
cell 11 — running it unchanged reproduces the run exactly.

## Procedure (Colab — the real run)

The deep run was a Colab session. Reproduce it there:

```python
# In a fresh Colab notebook cell, before anything else:
!git clone https://github.com/kuds/reinforce-tactics.git
%cd reinforce-tactics
!git checkout 6eb0566639ab18c1d40f99af80cfbaa71bc57c28
!pip install -e . -q
```

Then open `notebooks/ppo_bootstrap.ipynb` **from that checked-out
commit** (not main) and **Run All**. Do not edit any cell:

- It loads `configs/bootstrap.yaml` (24-stage rt-0.2.5 curriculum,
  `turn_penalty: -0.2`).
- Cell 11 applies `apply_overrides(cfg, {"env.enabled_units":
  ["W","M","C","A","K"]})` — already uncommented at this commit.
- `constants.py` at this commit has the rt-0.2.5 economy.
- It is `seed: 42` like every other run.

Outputs land in the usual `benchmarks/bootstrap/<timestamp>/` layout.
(Note: this commit predates `run_status.json` / the engine-economy
metadata — that's expected; the whole point is to run the *old* code.)

## Local sanity check (optional, no GPU training)

`scripts/experiment_a_setup.sh` creates an isolated git worktree at
`6eb0566` so you can diff/inspect the exact code + config without
disturbing the working branch. It does **not** train (training is a
Colab/GPU job) — it just stages the faithful tree and prints the
provenance so you can confirm economy/roster/curriculum before
committing GPU hours.

## Decision criteria (this is the load-bearing part)

Let it run to at least `beginner_random_15` (the stage every v17–v24
run died on). One of two outcomes, both decisive:

1. **It reproduces deep progression** (clears `beginner_random_10`,
   `_15`, ideally reaches intermediate/skirmish like the original):
   → The deep config was *real*. The regression is in the
   **0.2.5 → 0.2.7 code path** (gym_env reward computation, RandomBot
   logic, masking, curriculum accretion) — *not* the economy (v24
   already ruled that out) and *not* a phantom. Next step: bisect the
   code path between 6eb0566 and HEAD on the random_10 transition.

2. **It stalls at `beginner_random_10`/`_15`** like v24:
   → The "17 stages" was a short-curriculum / Colab-stop artifact —
   the 24-stage rt-0.2.5 curriculum is genuinely easier per-stage
   and/or the run simply stopped (disconnect) rather than clearing.
   The wall was always there. Stop all config/economy/curriculum
   archaeology; the next lever is **model capacity / BC warm-start**
   (Experiment B), not reproduction.

Either way this closes the two-week reproduction thread with a
definitive answer instead of another partial revert.

## Why this wasn't run first

We didn't know (until the corrected roster audit + git archaeology in
this investigation) that (a) the deep run's roster came from a notebook
override, not the YAML, and (b) its curriculum was 24 stages, not 33.
v24 was built to *port* the deep economy onto the current curriculum —
which conflated the very variables Experiment A isolates. Running the
original commit as-is is the experiment that should have been run
before any porting.
