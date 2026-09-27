# Cloud Training on Vertex AI

Run a single, long-running training job on Google Cloud using the project's
Docker image and Vertex AI **custom jobs**. You build the image once, push it to
Artifact Registry, then submit a job with a CLI command. Vertex provisions a
GPU machine, runs your training command to completion (which can take hours or
days), streams logs, and tears the machine down afterwards.

Because that machine is ephemeral, the image's entrypoint
([`scripts/cloud/vertex_train.py`](../scripts/cloud/vertex_train.py)) uploads the
output directories (`models/`, `checkpoints/`, `tensorboard/`, `logs/`, and the
curriculum runs under `benchmarks/bootstrap/`) to Google Cloud Storage
**periodically and on exit**, so your trained model survives the job ending (or
being preempted/cancelled). A preempted or cancelled job keeps everything up to
its last periodic sync (every `SYNC_INTERVAL` seconds) plus whatever the final
upload manages in Vertex's shutdown grace period.

> Prefer the managed approach below over the legacy GCE-VM launcher
> (`scripts/gcp_launch.sh`), which manages raw Compute Engine instances by hand.

## Prerequisites

- A GCP project with billing enabled.
- The [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and
  authenticated (`gcloud auth login`, `gcloud config set project PROJECT_ID`).
- These APIs enabled:
  ```bash
  gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
  ```
- A GCS bucket for outputs (in the same region as the job for best performance):
  ```bash
  gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1
  ```
- GPU quota for the region (e.g. `NVIDIA_TESLA_T4` in `us-central1`). Check under
  *IAM & Admin → Quotas* and request an increase if needed.

## 1. Build and push the image

```bash
PROJECT_ID=your-project REGION=us-central1 ./scripts/cloud/build_image.sh
```

This uses **Cloud Build** (no local Docker required), creates the Artifact
Registry repo `reinforce-tactics` on first run, and pushes:

```
us-central1-docker.pkg.dev/your-project/reinforce-tactics/rl-trainer:latest
```

Override `AR_REPO`, `IMAGE_NAME`, `TAG`, or `BUILD_TIMEOUT` via environment
variables. To build locally instead:

```bash
docker build -t us-central1-docker.pkg.dev/your-project/reinforce-tactics/rl-trainer:latest .
docker push   us-central1-docker.pkg.dev/your-project/reinforce-tactics/rl-trainer:latest
```

## 2. Submit a training job

Everything after the script name is the training command run inside the
container. The image entrypoint wraps it with GCS sync.

```bash
# PPO via main.py
BUCKET=YOUR_BUCKET ./scripts/cloud/submit_vertex_job.sh \
  python3 main.py --mode train --algorithm ppo --timesteps 10000000 --opponent bot

# Feudal RL via the advanced script, with action masking and W&B
WANDB_API_KEY=$WANDB_API_KEY BUCKET=YOUR_BUCKET \
  ./scripts/cloud/submit_vertex_job.sh \
  python3 scripts/train/train_feudal_rl.py --mode feudal --total-timesteps 20000000 \
    --n-envs 8 --device cuda --use-action-masking --wandb

# Use a YAML config from configs/
BUCKET=YOUR_BUCKET ./scripts/cloud/submit_vertex_job.sh \
  python3 scripts/train/train_feudal_rl.py --config configs/feudal/feudal_rl.yaml --device cuda
```

`BUCKET` is required — it's where artifacts land. Outputs for a job go to
`gs://BUCKET/jobs/<JOB_NAME>/{models,checkpoints,tensorboard,logs}/`.

### Choosing a config on the CLI

Pass the training command's own flags through the submit script. Which entry
points accept `--config <yaml>`:

| Entry point | `--config`? |
|---|---|
| `scripts/train/train_bootstrap.py` (curriculum) | ✅ |
| `scripts/train/train_feudal_rl.py` | ✅ |
| `scripts/train/train_alphazero.py` | ✅ |
| `scripts/train/train_self_play.py` | ✅ |
| `main.py` (`--mode train`) | ❌ — individual flags only (`--algorithm`, `--timesteps`, …) |

### Curriculum bootstrap — charts & videos

`scripts/train/train_bootstrap.py` is a headless CLI mirror of
`notebooks/ppo_bootstrap.ipynb`: it runs the curriculum (`run_curriculum`),
writes all the diagnostic **charts** (`viz.plot_*`), and records a per-stage
replay **video** (`.mp4`) — the same artifacts the notebook produces, minus the
Colab/Drive bits. The image already bundles the libraries (matplotlib, pygame,
opencv, imageio+ffmpeg) and forces headless rendering (`SDL_VIDEODRIVER=dummy`,
`MPLBACKEND=Agg`), so it runs unattended.

```bash
# Reproduce the ppo_bootstrap pipeline on Vertex; pick the YAML with --config
BUCKET=YOUR_BUCKET ./scripts/cloud/submit_vertex_job.sh \
  python3 scripts/train/train_bootstrap.py \
    --config configs/ppo/bootstrap.yaml --device cuda

# With a BC warm-start (needs a multi_discrete config), e.g. the v33 sweep config
BUCKET=YOUR_BUCKET ./scripts/cloud/submit_vertex_job.sh \
  python3 scripts/train/train_bootstrap.py \
    --config configs/ppo/bootstrap_sweep/v33_production_bc_warmstart.yaml \
    --build-bc --device cuda
```

The script writes everything under one run directory
(`benchmarks/bootstrap/<timestamp>/` by default) — `charts/`, `videos/`,
`checkpoints/`, the config snapshot, `bootstrap_results.csv`, `final_model.zip` —
and that whole tree ends up in `gs://BUCKET/jobs/<JOB_NAME>/<timestamp>/`: on
success, on a stall, on an error, and on the `SIGTERM` Vertex sends when a job
is cancelled or preempted. While the run is in progress, the entrypoint's
periodic sync mirrors `benchmarks/bootstrap/` to that location, and its final
sync after the script exits uploads whatever changed since, so the script
leaves the upload to it rather than re-sending every checkpoint inside the
shutdown grace period. Run outside the entrypoint, or with a `--gcs-output` or
`--output-dir` the entrypoint does not sync to that same place, the script
uploads the tree itself on the way out. Checkpoints are written to a
`.partial` file and renamed into place, so a run stopped mid-save never
replaces a good `best_model.zip` (locally or in the bucket) with a truncated
one.

Useful flags: `--skip-videos`, `--skip-plots`, `--sanity-episodes N`,
`--set dotted.key=value` (config overrides), `--gcs-output gs://...`
(explicit destination). Run `python3 scripts/train/train_bootstrap.py --help`
for the full list. Fetch the results with:

```bash
gcloud storage cp -r gs://YOUR_BUCKET/jobs/JOB_NAME ./bootstrap_run
```

#### Exit codes

`train_bootstrap.py` exits with a code that says how the run ended. Vertex marks
a job **Failed** for any non-zero exit, so a stalled curriculum no longer shows
up as a success; the entrypoint passes the code through unchanged.

| Code | Meaning | Artifacts |
|---|---|---|
| `0` | Every curriculum stage promoted | Complete, uploaded |
| `1` | Failure: an exception during the run (see the traceback in the log), or an invalid `--config` / `--set` value at startup | Whatever the run wrote, uploaded (nothing for a startup error) |
| `2` | Command-line usage error (argparse) | None; the run never started |
| `3` | **Stalled**: a stage used its `max_timesteps` budget, and its retries (`curriculum.max_retries`, default 1: one more full budget from the stage's `best_model.zip`), without meeting its promotion criterion | Partial run post-processed (charts, videos, sanity eval) and uploaded; `run_status.json` says `curriculum_stalled` |
| `130` | Interrupted with Ctrl-C (`SIGINT`) | Uploaded |
| `143` | Terminated by `SIGTERM` (Vertex cancel/preemption, `docker stop`) | Uploaded on the way out, within the grace period |

A `SIGTERM` that arrives once the run has ended, while its upload is in
progress, is ignored so the upload can finish; the exit code then still reports
how the run ended.

#### Resuming an interrupted run

A run that ended with `143` (or `130`, or was killed outright) can continue
where it stopped. Download its directory (or run in the same container) and
pass it to `--resume`:

```bash
python3 scripts/train/train_bootstrap.py --resume ./bootstrap_run --device cuda
```

The stages whose `config.json` says `promoted: true` are skipped (so are
stages whose `config.json` write failed but whose promotion `run_manifest.json`
recorded); the stage that was running continues from its rolling
`<stage>/latest.zip` (saved every `eval.checkpoint_freq` stage steps, on
promotion, and on `SIGTERM`) with the rest of its budget, its eval timeline,
promotion streak, best-model record and schedule positions (see
`run_manifest.json`). A stage whose checkpoint was taken on its promoting eval
is finished rather than trained again, and a retry killed before its first
checkpoint restarts from the checkpoint it began from. `num_timesteps` and the
TensorBoard curves continue. Without `--config` the run's own
`resolved_config.yaml` is used (a record written before the eval-gate change
keeps its greedy gate, greedy-only evals and no stall retries); a config that
differs from it in what is trained or measured is refused unless `--force`.
`--force` is also needed to resume a run whose records stop short of stages
that already have output (resuming would overwrite them), or a `--build-bc`
run stopped before its warm start was built (it then resumes without one). A
stalled run (`run_status.json` says `curriculum_stalled`) is not resumed: start
a new run with `warm_start_path` set to the stalled stage's `best_model.zip`.
`metadata_write_failures` in `run_status.json` counts the best-effort writes
that failed in every session of the run.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PROJECT_ID` | current gcloud project | GCP project |
| `REGION` | `us-central1` | Region for the job and image |
| `BUCKET` | *(required)* | GCS bucket for outputs (name or `gs://` URI) |
| `JOB_NAME` | `rt-train-<timestamp>` | Display name and output subfolder |
| `IMAGE_URI` | derived | Full image URI (overrides `AR_REPO`/`IMAGE_NAME`/`TAG`) |
| `MACHINE_TYPE` | `n1-highmem-8` | Worker machine type |
| `ACCELERATOR_TYPE` | `NVIDIA_TESLA_T4` | GPU type |
| `ACCELERATOR_COUNT` | `1` | GPUs per replica (`0` = CPU-only) |
| `REPLICA_COUNT` | `1` | Worker replicas |
| `SYNC_INTERVAL` | `300` | Seconds between GCS syncs (`0` = only on exit) |
| `SYNC_DIRS` | *(unset)* | Extra local dirs to sync, comma-separated: `dir` goes to `gs://.../jobs/<name>/dir/`, `dir=prefix` to `.../prefix/`, and `dir=` straight into `gs://.../jobs/<name>/`. A file under two entries (e.g. `benchmarks` and the default `benchmarks/bootstrap`) is uploaded to both places, and a warning is logged. Sets `GCS_SYNC_DIRS` in the container |
| `SERVICE_ACCOUNT` | *(unset)* | Run the job as this service account |
| `WANDB_API_KEY` | *(unset)* | Passed through to the container when set |

## 3. Monitor the job

```bash
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

You can also watch it in the Cloud Console under *Vertex AI → Training → Custom
jobs*.

## 4. Retrieve the trained model

```bash
gcloud storage ls   gs://YOUR_BUCKET/jobs/JOB_NAME/
gcloud storage cp -r gs://YOUR_BUCKET/jobs/JOB_NAME/models ./models
```

Then evaluate locally:

```bash
python main.py --mode evaluate --model models/ppo_final.zip --episodes 20
```

## How artifact persistence works

The container entrypoint is the wrapper, not the training command directly:

```dockerfile
ENTRYPOINT ["python3", "scripts/cloud/vertex_train.py"]
CMD ["python3", "main.py", "--mode", "train"]
```

The wrapper:

1. Resolves the GCS destination from `GCS_OUTPUT_URI` (set by the submit script),
   falling back to Vertex's `AIP_MODEL_DIR`. With neither set it just runs
   locally — the same image works on your laptop.
2. Runs the training command as a child process, with `GCS_WRAPPER_SYNC` in its
   environment describing what the final sync will upload.
3. Every `GCS_SYNC_INTERVAL` seconds, uploads `models/`, `checkpoints/`,
   `tensorboard/`, and `logs/` to `gs://.../jobs/<name>/<dir>/`, each run
   directory under `benchmarks/bootstrap/` to `gs://.../jobs/<name>/<run>/` (the
   same place `train_bootstrap.py` uploads it to when run on its own), and any
   `GCS_SYNC_DIRS` entries. Unchanged files, and `*.partial` files still being
   written, are skipped.
4. Forwards `SIGTERM`/`SIGINT` (Vertex sends `SIGTERM` on cancel/preemption) to
   the trainer so it can checkpoint, then performs a **final sync** before exit.
   `train_bootstrap.py` turns the `SIGTERM` into a clean exit (code 143); seeing
   `GCS_WRAPPER_SYNC` cover its run directory, it leaves the upload to this final
   sync, which only sends what changed since the last periodic one.
5. Exits with the trainer's exit code, or `128 + N` when the trainer was killed by
   signal `N`.

Uploads are best-effort: a transient storage hiccup is logged, never fatal.

## IAM / permissions

By default a custom job runs as the **Vertex AI Custom Code Service Agent**. For
the GCS upload to succeed, that identity (or a `SERVICE_ACCOUNT` you pass) needs
write access to the bucket:

```bash
# Grant the job's service account object access to the bucket
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
  --role="roles/storage.objectAdmin"
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `google-cloud-storage not installed; skipping GCS sync` | The image wasn't built with the `[cloud]` extra. Rebuild with `build_image.sh` (the `Dockerfile` installs it). |
| Job runs but bucket stays empty | Service account lacks `storage.objectAdmin` on the bucket (see IAM above). |
| Bootstrap job marked *Failed* with exit code 3 | The curriculum stalled (see [Exit codes](#exit-codes)); the partial run is in the bucket. |
| `Quota exceeded` on submit | Request GPU quota for the region, or set `ACCELERATOR_COUNT=0` for a CPU smoke test. |
| Cloud Build times out | Raise `BUILD_TIMEOUT` (e.g. `BUILD_TIMEOUT=7200s`). |
| Want a shell in the image | `docker run --entrypoint bash -it IMAGE_URI` (bypasses the wrapper). |
