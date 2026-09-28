# The §2.1 validation run: runbook

`configs/ppo/bootstrap.yaml` with the back-port, **three seeds**, on the fixed
code ([`REVIEW_full_2026-09-26.md`](REVIEW_full_2026-09-26.md) §2.1). This page
is how to run it and how to report it. Which config values the run uses, and
why, is in [`validation_run_config.md`](validation_run_config.md): the reward
back-port, the 24 ladder-ordered stages, the Wilson gate on the stochastic
policy in both seats, the budgets and the compute estimate. This page refers
to the config only by its path.

The tooling:

| Piece | What it does |
|---|---|
| `scripts/train/run_seeds.py` | Launches one `train_bootstrap.py` per seed (locally, sequentially or in parallel; or one Vertex AI job per seed), records the group and its launch settings in `seed_group.json`, and continues it with `--group <id>`. `status` shows progress; `fetch` downloads Vertex runs. |
| `scripts/train/train_bootstrap.py --seed N --output-dir D --resume-if-exists` | The per-seed command. The same command starts a seed, resumes it after any interruption, and exits 0 (finished) or 3 (stalled) without training once it is done. It holds a lock on its run dir while it runs and refuses (exit 1) a dir another live trainer holds. |
| `scripts/eval/summarize_seeds.py` | The report: per seed and across seeds, per stage; optionally against v52a. |
| `notebooks/ppo_bootstrap.ipynb` §10 | The Colab path (sequential, on Drive). |

## 1. Goal and exit criterion

Per seed and per stage, report the win / draw / loss rates of the **stochastic**
policy, the captures by structure type, and the shaping share of the return;
compare with the archived v52a run. Until this exists, no new sweep variant is
interpretable.

Exit criterion (review §8, week 2): **three seeds finish or stall, each with a
resumable run record.** A stall is a result, not a failure of the run.

## 2. Preconditions

1. The back-port is merged into `configs/ppo/bootstrap.yaml`
   ([`validation_run_config.md`](validation_run_config.md) §2–5). Record the
   commit the run uses: every run dir records it again (`meta.git` in each
   stage's `config.json`, `git` in `seed_group.json`). A Vertex image has no
   `.git`; its jobs record the commit that the image's `TAG` names (§4.3).
2. The fast suite passes: `pytest -m "not slow"`.
3. The config loads, validates and reads every field it sets:

   ```bash
   python3 scripts/train/train_bootstrap.py --config configs/ppo/bootstrap.yaml --strict --check-only
   ```

   It prints the stage table and the worst-case budget, and writes nothing.
4. For Vertex: an image built from the merge commit (`TAG=<short sha>`, §4.3).
5. The slice has run (§3.1): it calibrates the training and eval throughput
   the compute estimate assumes.

## 3. Smoke test (about 30–60 min)

Every stage promotes at its first eval, so the whole 24-stage pipeline runs:
each stage's envs and evals are built (late-stage env or config errors show up
now, not after three days), both seeds run in parallel, and the report is
written.

```bash
python3 scripts/train/run_seeds.py --config configs/ppo/bootstrap.yaml --seeds 42,1042 --tag smoke --parallel 2 -- \
    --strict --skip-videos --sanity-episodes 0 \
    --set 'curriculum.stages[*].promotion_win_rate=0' \
    --set 'curriculum.stages[*].patience=1' \
    --set 'curriculum.stages[*].min_timesteps_before_promotion=0' \
    --set eval.n_eval_episodes=4
python3 scripts/eval/summarize_seeds.py \
    --group-manifest benchmarks/bootstrap/_groups/<smoke group>/seed_group.json
```

`bootstrap.yaml` sets `min_timesteps_before_promotion: 25_000` on every stage,
so the stage-entry eval cannot promote; the smoke test zeroes it, or every
stage would train to its first in-stage eval (up to 100k steps). `n_eval_episodes`
is set once, under `eval` (per seat: 4 × 2 seats here). Expect exit 0 from
both commands and, in the report, both seeds `completed` with 24/24 stages
cleared. The launcher prints the group id; delete the smoke group's run dirs
afterwards. Expected here, and not a problem:

- every stage flagged `skip_ahead` (it promotes at its first eval);
- `flat_truncated` on `corner_points_balanced_random` and
  `corner_points_mixed_50`: an untrained policy meets 770–805 legal actions
  on that map, and 15–17% of its decision points were cut to 512 in the
  2026-09-28 smoke test (§5 says when to act on it);
- once per stage in each seed's log, `[warn] no best_model.zip for
  '<stage>'; carrying end-of-stage policy forward`: a stage that promotes
  at its first eval never saves a best model.

The 2026-09-28 smoke test (2 seeds, `--parallel 2`, 4 CPUs) took 685 s.

### 3.1 The slice (about 2 h)

`configs/ppo/bootstrap_validation.yaml` is the first 8 stages of
`bootstrap.yaml` with about half the budgets, `eval_freq` 50k and no retries
([`validation_run_config.md`](validation_run_config.md) §6). Run it once, at
the first seed, with the real gate:

```bash
python3 scripts/train/run_seeds.py --config configs/ppo/bootstrap_validation.yaml --seeds 42 --tag slice -- \
    --strict --skip-videos
```

Calibrate the compute model ([`validation_run_config.md`](validation_run_config.md)
§5) from its report (`summarize_seeds.py --group-manifest …`, §8):

- **Eval throughput:** §3's "Throughput by map" line gives eval agent
  steps/s, `sum(lengths) / eval_seconds` over the eval rows of
  `eval_results.jsonl`. The model assumes 675/s (270 serial × 2.5 for the 8
  subprocess eval envs).
- **Training throughput:** the same line's train env steps/s,
  Δtimesteps / (Δwall_time − eval_seconds) between consecutive eval rows.
  The model assumes 1000/s.
- **Wall clock:** §1's `active h` (the launcher's session times), not
  `wall h`, which includes any time between sessions.

TensorBoard's `time/fps` and the stage wall-times mix training, PPO updates
and evals, so neither measures either rate, and the slice never runs a
serial eval to compare the ×2.5 against. Also read the per-seat win rates on
`starter_mixed_random_medium` and `starter_medium`, `avg_turns` against
`max_turns`, and `flat_truncated_rate`. A stall ends the slice (exit 3);
that is a finding to diagnose before the three-seed launch, not a reason to
extend its budget.

## 4. Launch

Pick one venue. In every case pass `-- --strict --skip-videos` (videos can be
rendered later from the checkpoints) and keep the printed group id: `--group
<id>` alone continues the group with everything it was launched with (§6).
Run the commands from the repository root (the default `--root` is
`benchmarks/bootstrap` there; the children always run from the root, where
the config's map paths resolve).

Seeds default to `--n-seeds 3` with a stride of 1000 from the config's seed:
**42, 1042, 2042** (42 for continuity with the archive). Consecutive seeds
(42, 43, 44) would share 7 of 8 training env streams and 59 of 60 gate-eval
episode seeds per seat (policy-sampling streams included); the launcher
refuses seeds whose streams overlap unless `--allow-seed-overlap`.

### 4.1 Local workstation

The launcher counts 17 CPUs per run for this config (the main process, 8 env
workers and 8 subprocess eval workers) and warns when a slot is smaller. The
env workers idle while an eval runs and the eval workers idle while PPO
trains, so a slot of about 10 CPUs shares cores only during evals: about 30
vCPUs for three seeds in parallel; otherwise run them one after another
(`--parallel 1`).

```bash
# three seeds at once, each pinned to its own third of the CPUs, one GPU each
python3 scripts/train/run_seeds.py --config configs/ppo/bootstrap.yaml --n-seeds 3 --tag val \
    --parallel 3 --gpus 0,1,2 -- --strict --skip-videos
# or sequentially on one GPU
python3 scripts/train/run_seeds.py --config configs/ppo/bootstrap.yaml --n-seeds 3 --tag val \
    --device cuda -- --strict --skip-videos
```

With `--parallel K > 1` each slot gets a contiguous CPU set (`--cpu-sets
"0-9,10-19,20-29"` to choose them, `--no-pin` to not pin) and
`--torch-threads` of its size; every child runs with one OMP/MKL/OpenBLAS
thread per process. A slot gets GPU `i mod len(--gpus)`; `--device cpu` hides
the GPUs. The logs are in `<run dir>/logs/train.<UTC stamp>.log`, one per
launch session (and on the console with `--parallel 1`). The group records
`--parallel`, `--cpu-sets`, `--no-pin`, `--gpus` and `--device`, and a
continuation reuses them.

Run the launcher inside `tmux` or `screen` (or under `nohup`). A hangup
(an SSH disconnect) now stops the group cleanly: every child gets SIGTERM,
checkpoints and exits, and the group continues with `--group <id>`. But a
launcher killed outright (SIGKILL, the OOM killer) leaves its children
training with nobody recording them; `status` then shows them as
`running`, and a relaunch leaves them alone until they exit.

### 4.2 Colab

`notebooks/ppo_bootstrap.ipynb`, section 10: run the setup cells (1–2b), set
`RUN_SEED_GROUP = True` in 10a, run 10a and 10b. The seeds run one after
another on the runtime's GPU, into
`MyDrive/reinforce-tactics/benchmarks/bootstrap/<group>_s<seed>/`. Pin `GROUP`
in 10a after the first launch (10a prints the line to paste). After a
disconnect, run 10a and 10b again: with `GROUP` unpinned, 10a picks the
newest group with the same `TAG` on Drive rather than starting a new one.
Expect about 5–10 days in all, over many sessions.

### 4.3 Vertex AI (one job per seed)

Recommended machine: 16 vCPUs + one T4 (`n1-standard-16`).

```bash
# once per code version: the image, tagged with the merge commit
TAG=$(git rev-parse --short HEAD) PROJECT_ID=my-project REGION=us-central1 ./scripts/cloud/build_image.sh

BUCKET=my-bucket PROJECT_ID=my-project REGION=us-central1 TAG=<short sha> \
MACHINE_TYPE=n1-standard-16 ACCELERATOR_TYPE=NVIDIA_TESLA_T4 \
    python3 scripts/train/run_seeds.py --backend vertex --config configs/ppo/bootstrap.yaml \
    --n-seeds 3 --tag val -- --strict --skip-videos
```

Each seed's job runs `python3 scripts/train/train_bootstrap.py --config
configs/ppo/bootstrap.yaml --seed S --output-dir benchmarks/bootstrap/<run>
--resume-if-exists --device cuda --strict --skip-videos`, with
`OUTPUT_URI=gs://<bucket>/jobs/<group>` (every seed lands at
`gs://<bucket>/jobs/<group>/<group>_s<seed>/`) and
`RESTORE_DIRS=benchmarks/bootstrap/<run>=<run>`, so a resubmitted job first
restores the run its predecessor synced. The config must be a path inside the
repository (it is read from the image). `--dry-run` prints the submissions
without submitting.

The group records the bucket, the image (`IMAGE_URI`, or `TAG`) and the
machine settings (`MACHINE_TYPE`, `ACCELERATOR_TYPE`, `ACCELERATOR_COUNT`,
`REPLICA_COUNT`, `SERVICE_ACCOUNT`, `RESTART_ON_WORKER_RESTART`,
`SYNC_INTERVAL`); `--group <id>` alone reuses them (the submit script's own
default is a smaller `n1-highmem-8`). A different bucket or image is refused
unless `--force`: a new bucket restores nothing, so every seed would start
over, and a new image runs one group on two code versions. When `TAG` (or
the tag of `IMAGE_URI`) names a commit of the launching checkout, each job
gets it as `RT_GIT_COMMIT`, and the run's records name it; otherwise the
launcher says the jobs will record no commit.

## 5. Monitor

```bash
python3 scripts/train/run_seeds.py status --group <group>            # local (or --group-manifest <path>)
python3 scripts/train/run_seeds.py status --group <group> --jobs     # Vertex: plus each job's state
tensorboard --logdir benchmarks/bootstrap                             # then filter the runs by "<group>_s"
```

`status` shows, per seed: its state, the current stage (index of 24), the
env steps so far, the last gate win rate, stages cleared, resumes, sessions and
the last exit code, and steps per hour. The states: `pending`, `running` (a
live trainer holds the run dir's lock), `interrupted` (resumable), `failed`,
`stalled`, `completed`, and on Vertex `submitted` (the job's own state is the
`--jobs` column; `fetch` brings the run dir, and a finished one then reads
`completed`).

In the first two hours check:

- **steps/h** (`status`): for scale, the archive's v52a ran at about 420k
  env steps/h on the beginner and intermediate stages and 200k/h on skirmish
  (Colab L4, 12 vCPUs, greedy-only serial evals).
  This config evaluates the stochastic policy only, every 100k steps, in 8
  batched subprocess envs; the compute model in
  [`validation_run_config.md`](validation_run_config.md) §5 and the slice
  (§3.1) say what to expect.
- **the eval share of wall clock**: every eval row carries `eval_seconds`
  and `wall_time`; `summarize_seeds.py` reports `eval_share` per stage.
- **`flat_truncated_rate`** in the eval rows: 0 through skirmish. On
  corner_points an untrained policy meets up to 805 legal actions and is cut
  at 15–17% of its decision points (the smoke test, §3), so a nonzero rate
  there is expected at first; act (raise `max_flat_actions`, a sweep axis)
  only if it persists once the stage is trained.
- **the first stages clearing**, with plausible win rates in both seats
  (`win_rate_by_seat` in the eval rows).

## 6. Interruptions

- **Killed, preempted, disconnected:** continue with `--group <id>` alone
  (plus `--root` if it is not the default), locally or on Vertex. It reuses
  the group's config, seeds, train_bootstrap.py args and launch settings
  (`--parallel`, `--cpu-sets`, `--no-pin`, `--gpus`, `--device`; on Vertex
  the bucket, image and machine); options you give again replace the stored
  ones. Do not re-run the original launch command: without `--group` it
  would start a new group from step 0, so the launcher refuses while an
  unfinished group with the same tag and config exists, and names it
  (`--new-group` overrides).
- Finished and stalled seeds are left alone; an interrupted seed resumes
  from its rolling `latest.zip`. A seed whose trainer still runs (`running`
  in `status`: it holds its run dir's lock) is left alone, and a second
  trainer in the same run dir would refuse to start. A seed killed with
  SIGKILL or by the OOM killer (exit 137) counts as interrupted and resumes;
  if it is killed again the same way, look at memory before relaunching.
- On Vertex nothing is submitted twice: a seed whose job is still active,
  or whose run in GCS finished or stalled, is skipped (checking GCS needs
  `google-cloud-storage`; without it, name the seeds to resubmit with
  `--only-seeds`). A job that ended `JOB_STATE_FAILED` is resubmitted only
  when its error names an interrupt exit status (130, 137, 143: preemption,
  a cancel); after exit 1, or a failure whose error names no exit status
  (a timeout, say), read its log and pass `--retry-failed`.
- **Stalls are results.** Never resume or retry a stalled seed; report it.
- **A failed seed (exit 1):** read `<run dir>/logs/train.*.log` (Vertex:
  `gcloud ai custom-jobs stream-logs <job>`), fix the cause, then re-run
  with `--retry-failed`.
- **`--force`** (continuing a group with a different config, seeds, args,
  bucket or image) is recorded in `seed_group.json`'s `history`; report any
  use of it. Changed launch settings are recorded there too.
- A resumed seed is valid but not bit-reproducible (the RNG streams are not
  restored); the report flags resumed seeds.

## 7. Budget

The numbers and the model behind them are in
[`validation_run_config.md`](validation_run_config.md) §5; the slice (§3.1)
calibrates them.

- **Env steps:** a prior of about 21M per seed (each stage stopping at 40%
  of its budget, 20% on map-entry stages). The archive's deepest runs
  cleared 20 of 33 of the old stages in about 5.9M, against bots that were
  weaker then.
- **Wall clock:** about 24 h per seed at the expected length (14–38 h) on
  an L4 with 12 vCPUs; about 58 h if every stage runs out its budget.
- **Worst case:** the stage budgets sum to 55.25M env steps; with
  `curriculum.max_retries: 1` every stage but the capstone can use its budget
  twice (106.5M). A single stall on a 4M-step map top with its retry is 8M
  steps. `train_bootstrap.py --check-only` prints both totals.
- **Vertex:** a custom job times out after 7 days (the GCP default);
  resubmitting continues it (`--retry-failed` if the timed-out job reads
  `JOB_STATE_FAILED` without an interrupt exit status, §6). Cost is roughly $1/h per job for
  `n1-highmem-8` + T4 (verify against current GCP pricing; a 16-vCPU machine
  costs more per hour and runs faster), times the wall clock above.
- The eval-throughput knobs (`eval.n_eval_envs`, `eval.eval_use_subprocess`,
  `eval.eval_both_modes`) belong to the config decision
  ([`validation_run_config.md`](validation_run_config.md) §4–5), not to this
  runbook.

## 8. Aggregate and report

```bash
python3 scripts/eval/summarize_seeds.py \
    --group-manifest benchmarks/bootstrap/_groups/<group>/seed_group.json \
    --compare v52a=/content/drive/MyDrive/reinforce-tactics/benchmarks/bootstrap/20260601_172412
```

The `--compare` path is the Drive copy of v52a, so it exists only on Colab.
Elsewhere, download `MyDrive/reinforce-tactics/benchmarks/bootstrap/20260601_172412`
(its `v52a_maxturn_scaled_draw.yaml`, `bootstrap_results.csv` and the stage
folders; at least the CSV and the YAML) and pass that local path.

It writes `report.md`, `runs.csv`, `per_seed_stage.csv`, `per_stage.csv`,
`comparison.csv` and `summary.json` (with every metric's definition) to
`<root>/_groups/<group>/report/`. It exits 1 when the runs' resolved configs
differ beyond seed, device and logging (`--allow-mixed` to report anyway).
Vertex runs: `run_seeds.py fetch --group <group>` first (add
`--include-checkpoints` for the zips).

What to report, in this order:

1. **The seed spread first** (report §1–2): stages cleared per seed, where
   each stalled, and per stage the stochastic win rate mean [min–max] with
   the per-seed values. With n = 3 a t-interval is very wide (t = 4.30), so
   the spread is the honest summary; the intervals are in `per_stage.csv`.
2. **Seed-sensitive stages** (§5 flags): cleared by some seeds, stalled by
   others. These say more about the curriculum than any mean.
3. Per stage and seed (§3): stochastic W/D/L (with Wilson intervals, both
   seats pooled as the gate pools them) and the gate win rate per seat,
   captures per episode (tower/building/HQ), the shaping share, the draw
   return, steps to promotion, retries. Two shaping shares:
   - `dense_share_abs` ("dense share") is the one
     [`validation_run_config.md`](validation_run_config.md) §1 asks for: the
     action stream without the turn penalty, against the terminal. The turn
     penalty (`reward_per_ep_turn_penalty`) and the potential term
     (`reward_per_ep_shaping_delta`) are separate columns in
     `per_seed_stage.csv`.
   - `shaping_share_abs` ("shaping abs") is the archive-comparable headline.
     It counts the turn penalty and the potential term's drain as shaping,
     so on this config it rises with game length even when nothing is
     farmed.
4. The flags: `skip_ahead`, `draw_breakeven` (a draw that pays at least
   break-even without the potential term, the draw-farming signature; the
   potential term's undiscounted eval sum drifts with game length, +393 per
   draw on an untrained corner_points policy, so it is left out, and
   `draw_return_raw_per_ep` keeps the full return), `flat_truncated`,
   `max_steps_truncate` above 5%, metadata write failures, resumed seeds.
   Report `active h` (the launcher's session times) as each seed's wall
   clock: `wall h` also counts the time between sessions.
5. **v52a, qualitatively** (§4): only the stages marked comparable (same map,
   opponent and kwargs, max_turns, threshold and patience), and only greedy
   win rates (v52a gated and recorded the greedy policy only). The config
   deltas and caveats are listed with it. On this config the comparison is
   thinner than that:
   - The run records the stochastic policy only (`eval_both_modes: false`),
     so the group's greedy columns are empty. The greedy seat-1 numbers come
     from a post-run eval of each stage's `best_model.zip`
     ([`validation_run_config.md`](validation_run_config.md) §1, §4); there
     is no script for that eval yet.
   - 14 of the 24 stages share a name with a v52a stage, and none is marked
     comparable: every Wilson threshold T differs from v52a's point
     threshold (often by construction, since a stage's intended rate t is
     its old point threshold), and `beginner_random_10` / `_15` also differ
     in opponent (v52a used MixedBot random pairs there). Read the listed
     differences rather than the ✗.

Afterwards: archive the run dirs and the `_groups/<group>/` folder to Drive
(`MyDrive/reinforce-tactics/benchmarks/bootstrap/`), re-run
`notebooks/bootstrap_run_analysis.ipynb` (it lists the seeds with `seed` and
`seed_group` columns), and update the status table in
`docs/REVIEW_full_2026-09-26.md`.

## 9. Caveats

- **Cross-code comparison.** v52a (git `078313e`, 2026-06-01) predates the
  opponent-freeze fix, engine and bot legality, the MDP fixes and the
  eval-gate change, gated on the greedy policy, and resampled its eval set
  every block. Its numbers are context, not a control.
- **Gate-selected numbers.** A cleared stage's numbers come from the eval
  that promoted it, chosen because it passed: biased upward near the
  threshold. The report labels them "at gate"; a held-out re-eval of each
  stage's handed-forward checkpoint is future work.
- **Resumed seeds are not bit-reproducible** (the RNG streams restart).
- **The eval set is fixed per run** (`resample_eval_seeds: false`): a
  seed's evals all replay the same episodes, and different seeds use
  disjoint ones (the seed stride guarantees it).
- **Stochastic samples depend on the eval batching.** `n_eval_envs`
  changes which stochastic samples are drawn (greedy numbers are identical),
  so keep it fixed across the seeds of a group; the replicate check enforces
  it.
