# Experiment B — bisect the rt 0.2.5 → 0.2.7 code regression

## Why we're here

Experiment A (run `20260515_213159`) ran commit `6eb0566`
unedited and **reproduced the deep run**: `beginner_random_10`
cleared on the first two evals (WR 1.0, 1.0), 16 stages cleared,
stalled only at `skirmish_simple` (stage 17). Every v16–v24 run on
current code dies at `beginner_random_10`.

Confounds eliminated:

- **Economy** — v24 reverted constants.py to rt-0.2.5 and still
  stalled at stage 5.
- **Roster / turn_penalty / curriculum length** — Experiment A
  held all of them at the deep-run values and progressed deep.
- **Phantom / Colab-stop artifact** — the progression is real,
  end-to-end, repeatable.

What's left: the engine/training **code** that drifted between
`6eb0566` and `HEAD`. This doc bisects it.

## The probe

`configs/bootstrap_sweep/v25_bisect_random10_repro.yaml` — a
self-contained 5-stage curriculum
(`starter_random → starter_simple → starter_medium →
beginner_balanced_random → beginner_random_10`) that walks the
deep-run path to the decision point and stops. Deep-run invariants
(`enabled_units=[W,M,C,A,K]`, `turn_penalty=-0.2`) are set in the
YAML, so it needs **no prior run files, no checkpoints, and no
notebook override**. Total budget ≈ 1–1.5M env steps → tractable
per commit.

**Binary verdict per commit:**

- **GOOD** — `beginner_random_10` promotes (WR ≥ 0.75 for
  patience=2 consecutive evals) within its 800K budget, like
  `6eb0566` (~2 evals / ~50K steps).
- **BAD** — fails to promote; WR collapses toward the
  draw-with-shaping attractor (W/L/D ≈ 0/0/80), like every
  v16–v24 run.

The first commit that flips GOOD → BAD is the regression.

## ⚠️ Critical confound control — pin the economy at every commit

`constants.py` drifts inside `6eb0566..HEAD` (`a596c15`
"Rebalance Warrior, Barbarian, and starting gold", plus the
Knight-defence / HQ-income changes). A bisect that doesn't pin the
economy measures code+economy jointly and is worthless.

**Preferred (current):** use the `env.engine_overrides` config
overlay — it forces the deep economy/stats *in YAML* regardless of
the checked-out `constants.py`, and `config.json` records the
resolved `effective_engine_economy` + `engine_constants_hash` so
each probe self-proves its economy. `v26_faithful_deep_reward_on_head.yaml`
already carries the byte-faithful `[W,M,C,A,K]` block; reuse that
overlay in the bisect config and **no git pin is needed at any
candidate commit** (it only requires the `engine_overrides` feature,
present from this branch onward — fine for the forward bisect, which
runs on/near HEAD).

**Legacy (pre-`engine_overrides` commits only):** if a candidate
commit predates the feature, fall back to the whole-file pin:

```bash
git checkout <candidate>
git checkout 6eb0566 -- reinforcetactics/constants.py   # WHOLE file
pip install -q -e .
python -c "from reinforcetactics import constants as c; \
  assert (c.STARTING_GOLD, c.HEADQUARTERS_INCOME) == (250, 150)"
```

Either way the economy is held at the Experiment-A-verified
`6eb0566` values for all candidates; the overlay is preferred
because it is auditable from the run's own `config.json`.

## Candidate commits (`6eb0566..HEAD`, behavioral only)

94 commits in range; ruff-format / mypy-only commits (`4711236`,
`7b9db78`, `f943c03`, `a0bb68b`, `736dcc2`) are no-ops for behavior
and are **not** bisect points. The 7 behavioral candidates touching
the env / reward / observation / extractor / bot path, in
chronological order:

| # | Commit | Date (UTC) | Change | Suspicion |
|---|--------|-----------|--------|-----------|
| 1 | `922aa29` | 05-11 19:31 | Add 7×7 intermediate map + **opponent-capture penalty** | Med — reward event added |
| 2 | `0e7fbf1` | 05-11 23:34 | **Normalize global_features with tanh** + wire CNN extractor | Med — obs distribution shift |
| 3 | **`c7001bf`** | **05-11 23:45** | **Replace per-end_turn cost with terminal speed bonus; unify max_steps=3000** | **HIGH — reward landscape over the draw/win incentive** |
| 4 | `c43f68d` | 05-12 15:18 | **Cap agent actions per game-turn** to prevent PPO stalling | Med-High — action dynamics |
| 5 | `4c1e695` | 05-12 18:03 | **Slim PPO observation**, add per-unit acted flag, enforce 1v1 | Med — obs change |
| 6 | `b665dca` | 05-12 18:57 | Add masked_avg pool, coord-conv, extra conv block to extractor | Low-Med — arch (capacity, not attractor) |
| 7 | `485dbd6` | 05-12 19:53 | Bump beginner stages to `max_turns=75` | Low — config-ish env param |

### Why `c7001bf` is the prime suspect

The deep config's `turn_penalty=-0.2` was the load-bearing fix for
the **draw-with-shaping equilibrium** (without per-turn pressure
the value function estimates ~+45 from drawing and the policy
commits to it — see `bootstrap_lessons_learned.md`). `c7001bf`
*replaces the per-end_turn cost mechanism with a terminal speed
bonus* — i.e. it removes the exact lever the deep config used to
break the draw attractor and substitutes a different shaping
signal. Experiment A's mechanistic tell (the agent **escapes** the
collapse→draw attractor at `6eb0566`, never escapes it on current
code) points squarely at a change to the draw/win reward landscape.

## Recommended procedure

Don't blind-`git bisect` all 7 — test by suspicion to converge in
1–2 runs:

1. **Test `c7001bf` first** (and its parent). If parent = GOOD and
   `c7001bf` = BAD, **done** — that's the regression.
2. If `c7001bf` parent is already BAD, the break is earlier:
   test `0e7fbf1`, then `922aa29`.
3. If `c7001bf` is GOOD, the break is later: test `c43f68d`, then
   `4c1e695`, then `b665dca` / `485dbd6`.

Equivalent `git bisect` form (≈3 runs, log₂7):

```bash
git bisect start
git bisect good 6eb0566
git bisect bad HEAD
# at each step: checkout is automatic, then:
#   pip install -q -e .
#   <train v25 — but with the v26 env.engine_overrides block so the
#    economy/stats are pinned in-config; no constants.py pin needed.
#    Pre-feature commits only: git checkout 6eb0566 -- .../constants.py>
#   <eval beginner_random_10>
#   git bisect good   # if random_10 promoted
#   git bisect bad    # if it stalled
# restrict to behavioral commits only:
git bisect skip 4711236 7b9db78 f943c03 a0bb68b 736dcc2
```

## After the regression commit is found

1. Read its diff; isolate the single behavioral hunk that flips
   the verdict (reward term, mask rule, obs channel, or env param).
2. Confirm by reverting **only that hunk** on current `HEAD` and
   re-running v25 → expect GOOD.
3. Decide the fix forward: either restore the old behavior on HEAD,
   or re-tune the new mechanism so it doesn't reopen the draw
   attractor (the new code may be desirable for other reasons —
   e.g. the speed bonus — but must not regress random_10).
4. Re-run the full curriculum (v24-style, current code + the fix)
   to confirm deep progression returns. Only then is the v17–v23
   curriculum-gate sweep formally closed.
