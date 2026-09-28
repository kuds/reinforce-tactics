# The §2.1 validation run: runbook

`configs/ppo/bootstrap.yaml` with the back-port, **three seeds**, on the fixed
code ([`REVIEW_full_2026-09-26.md`](REVIEW_full_2026-09-26.md) §2.1). This page
is how to run it and how to report it. Which config values the run uses is
decided separately, in `docs/validation_run_config.md`; this page refers to
the config only by its path.

The tooling:

| Piece | What it does |
|---|---|
| `scripts/train/run_seeds.py` | Launches one `train_bootstrap.py` per seed (locally, sequentially or in parallel; or one Vertex AI job per seed), records the group in `seed_group.json`, and continues it when re-run. `status` shows progress; `fetch` downloads Vertex runs. |
| `scripts/train/train_bootstrap.py --seed N --output-dir D --resume-if-exists` | The per-seed command. The same command starts a seed, resumes it after any interruption, and exits 0 (finished) or 3 (stalled) without training once it is done. |
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

1. The back-port is merged into `configs/ppo/bootstrap.yaml`. Record the
   commit the run uses: every run dir records it again (`meta.git` in each
   stage's `config.json`, `git` in `seed_group.json`).
2. The fast suite passes: `pytest -m "not slow"`.
3. The config loads, validates and reads every field it sets:

   ```bash
   python3 scripts/train/train_bootstrap.py --config configs/ppo/bootstrap.yaml --strict --check-only
   ```

   It prints the stage table and the worst-case budget, and writes nothing.
4. For Vertex: an image built from the merge commit (`TAG=<short sha>`, §4.3).

## 3. Smoke test (about 30–60 min)

Every stage promotes at its first eval, so the whole 33-stage pipeline runs:
each stage's envs and evals are built (late-stage env or config errors show up
now, not after three days), both seeds run in parallel, and the report is
written.

```bash
python3 scripts/train/run_seeds.py --config configs/ppo/bootstrap.yaml --seeds 42,1042 --tag smoke --parallel 2 -- \
    --strict --skip-videos --sanity-episodes 0 \
    --set 'curriculum.stages[*].promotion_win_rate=0' \
    --set 'curriculum.stages[*].patience=1' \
    --set 'curriculum.stages[*].n_eval_episodes=4'
python3 scripts/eval/summarize_seeds.py \
    --group-manifest benchmarks/bootstrap/_groups/<smoke group>/seed_group.json
```

`bootstrap.yaml` sets `n_eval_episodes: 80` on every stage, which wins over
`eval.n_eval_episodes`, hence the per-stage `--set`. Expect exit 0 from both
commands and, in the report, both seeds `completed` with 33/33 stages cleared
(every stage flagged `skip_ahead`, as it should be here). The launcher prints
the group id; delete the smoke group's run dirs afterwards.

## 4. Launch

Pick one venue. In every case pass `-- --strict --skip-videos` (videos can be
rendered later from the checkpoints) and keep the printed group id: re-running
with `--group <id>` continues the group. Run the commands from the repository
root (the default `--root` is `benchmarks/bootstrap` there; the children always
run from the root, where the config's map paths resolve).

Seeds default to `--n-seeds 3` with a stride of 1000 from the config's seed:
**42, 1042, 2042** (42 for continuity with the archive). Consecutive seeds
(42, 43, 44) would share 7 of 8 training env streams and 79 of 80 gate-eval
episodes (policy-sampling streams included); the launcher refuses seeds whose
streams overlap unless `--allow-seed-overlap`.

### 4.1 Local workstation

At least ~30 vCPUs for three seeds in parallel (each run keeps `n_envs + 1`
cores busy, plus `n_eval_envs` with subprocess evals); otherwise run them one
after another (`--parallel 1`).

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
launch session (and on the console with `--parallel 1`).

### 4.2 Colab

`notebooks/ppo_bootstrap.ipynb`, section 10: run the setup cells (1–2b), set
`RUN_SEED_GROUP = True` in 10a, run 10a and 10b. The seeds run one after
another on the runtime's GPU, into
`MyDrive/reinforce-tactics/benchmarks/bootstrap/<group>_s<seed>/`. Pin `GROUP`
in 10a after the first launch; after a disconnect, run 10a and 10b again.
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

## 5. Monitor

```bash
python3 scripts/train/run_seeds.py status --group <group>            # local (or --group-manifest <path>)
python3 scripts/train/run_seeds.py status --group <group> --jobs     # Vertex: plus each job's state
tensorboard --logdir benchmarks/bootstrap                             # then filter the runs by "<group>_s"
```

`status` shows, per seed: its state, the current stage (index of 33), the
env steps so far, the last gate win rate, stages cleared, resumes, sessions and
the last exit code, and steps per hour.

In the first two hours check:

- **steps/h** (`status`): for scale, the archive's v52a ran at about 420k
  env steps/h on the beginner and intermediate stages and 200k/h on
  intermediate and skirmish (Colab L4, 12 vCPUs, greedy-only serial evals).
  Both-mode evals cost more unless the eval is batched.
- **the eval share of wall clock**: every eval row carries `eval_seconds`
  and `wall_time`; `summarize_seeds.py` reports `eval_share` per stage.
- **`flat_truncated_rate`** in the eval rows: should stay 0.
- **the first stages clearing**, with plausible win rates in both modes.

## 6. Interruptions

- **Killed, preempted, disconnected:** re-run the identical launch command
  (or `--group <id>` alone, which reuses the group's config, seeds and
  train_bootstrap.py args), or re-submit on Vertex the same way. Finished and
  stalled seeds are left alone; an interrupted seed resumes from its rolling
  `latest.zip`. Nothing is ever submitted twice: a seed whose Vertex job is
  still active, or whose run in GCS finished or stalled, is skipped
  (checking GCS needs `google-cloud-storage`; without it, name the seeds to
  resubmit with `--only-seeds`).
- **Stalls are results.** Never resume or retry a stalled seed; report it.
- **A failed seed (exit 1):** read `<run dir>/logs/train.*.log`, fix the
  cause, then re-run with `--retry-failed`.
- **`--force`** (continuing a group with a different config, seeds or args)
  is recorded in `seed_group.json`'s `history`; report any use of it.
- A resumed seed is valid but not bit-reproducible (the RNG streams are not
  restored); the report flags resumed seeds.

## 7. Budget

- **Env steps:** expect roughly 10–20M per seed (the archive's deepest runs
  cleared 20 of 33 stages in about 5.9M). That is about 1.5–3.5 days per
  seed on Colab-class hardware with the default serial eval.
- **Worst case:** the stage budgets sum to 87.5M env steps; with
  `curriculum.max_retries: 1` every stage can use its budget twice (175M).
  A single stall on a 3M-step skirmish stage with its retry is 6M steps
  (about 30–50 h at the rates above).
- **Vertex:** a custom job times out after 7 days (the GCP default);
  resubmitting continues it. Cost is roughly $1/h per job for
  `n1-highmem-8` + T4 (verify against current GCP pricing; a 16-vCPU machine
  costs more per hour and runs faster), i.e. on the order of $40–90 per seed
  at the expected length.
- The eval-throughput knobs (`eval.n_eval_envs`, `eval.eval_use_subprocess`,
  `eval.eval_both_modes`) belong to the config decision, not to this runbook.

## 8. Aggregate and report

```bash
python3 scripts/eval/summarize_seeds.py \
    --group-manifest benchmarks/bootstrap/_groups/<group>/seed_group.json \
    --compare v52a=/content/drive/MyDrive/reinforce-tactics/benchmarks/bootstrap/20260601_172412
```

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
3. Per stage and seed (§3): stochastic and greedy W/D/L (with Wilson
   intervals), captures per episode (tower/building/HQ), `shaping_share_abs`,
   the draw return, steps to promotion, retries.
4. The flags: `skip_ahead`, `draw_breakeven` (a draw that pays at least
   break-even, the draw-farming signature), `flat_truncated`,
   `max_steps_truncate` above 5%, metadata write failures, resumed seeds.
5. **v52a, qualitatively** (§4): only the stages marked comparable (same map,
   opponent and kwargs, max_turns, threshold and patience), and only greedy
   win rates (v52a gated and recorded the greedy policy only). The config
   deltas and caveats are listed with it.

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
