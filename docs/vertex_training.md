# Cloud Training on Vertex AI

Run a single, long-running training job on Google Cloud using the project's
Docker image and Vertex AI **custom jobs**. You build the image once, push it to
Artifact Registry, then submit a job with a CLI command. Vertex provisions a
GPU machine, runs your training command to completion (which can take hours or
days), streams logs, and tears the machine down afterwards.

Because that machine is ephemeral, the image's entrypoint
([`scripts/cloud/vertex_train.py`](../scripts/cloud/vertex_train.py)) uploads the
output directories (`models/`, `checkpoints/`, `tensorboard/`, `logs/`) to Google
Cloud Storage **periodically and on exit**, so your trained model survives the
job ending (or being preempted/cancelled).

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
2. Runs the training command as a child process.
3. Every `GCS_SYNC_INTERVAL` seconds, uploads `models/`, `checkpoints/`,
   `tensorboard/`, and `logs/` to `gs://.../jobs/<name>/<dir>/`.
4. Forwards `SIGTERM`/`SIGINT` (Vertex sends `SIGTERM` on cancel/preemption) to
   the trainer so it can checkpoint, then performs a **final sync** before exit.

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
| `Quota exceeded` on submit | Request GPU quota for the region, or set `ACCELERATOR_COUNT=0` for a CPU smoke test. |
| Cloud Build times out | Raise `BUILD_TIMEOUT` (e.g. `BUILD_TIMEOUT=7200s`). |
| Want a shell in the image | `docker run --entrypoint bash -it IMAGE_URI` (bypasses the wrapper). |
