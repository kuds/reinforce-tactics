# Cloud training scripts (Vertex AI)

Submit a single, long-running training run on Google Cloud as a **Vertex AI
custom job**, using the project's Docker image. Artifacts are synced to GCS so
they survive the ephemeral job.

| File | Purpose |
|---|---|
| `build_image.sh` | Build the training image and push it to Artifact Registry (via Cloud Build). |
| `submit_vertex_job.sh` | Submit a Vertex AI custom job running a training command. |
| `vertex_train.py` | Container entrypoint: runs the training command and syncs `models/`, `checkpoints/`, `tensorboard/`, `logs/` to GCS (periodic + on exit). |

## Quickstart

```bash
# 1. Build + push the image (creates the Artifact Registry repo on first run)
PROJECT_ID=your-project REGION=us-central1 ./scripts/cloud/build_image.sh

# 2. Submit a job (BUCKET is required — where outputs are uploaded)
BUCKET=your-bucket ./scripts/cloud/submit_vertex_job.sh \
  python3 main.py --mode train --algorithm ppo --timesteps 10000000

# ...or reproduce the ppo_bootstrap notebook (curriculum + charts + videos):
BUCKET=your-bucket ./scripts/cloud/submit_vertex_job.sh \
  python3 scripts/train/train_bootstrap.py --config configs/ppo/bootstrap.yaml --device cuda

# 3. Monitor
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# 4. Fetch the trained model
gcloud storage cp -r gs://your-bucket/jobs/JOB_NAME/models ./models
```

See **[docs/vertex_training.md](../../docs/vertex_training.md)** for prerequisites
(API enablement, GPU quota, IAM), all configuration variables, and
troubleshooting.
