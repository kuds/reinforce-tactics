#!/usr/bin/env bash
#
# Submit a single Vertex AI custom training job.
#
# Everything passed as positional arguments becomes the training command run
# inside the container (it is appended to the image ENTRYPOINT, which is the
# GCS-sync wrapper scripts/cloud/vertex_train.py). If no command is given, a
# default PPO run is used.
#
# Configuration via environment variables (defaults shown):
#   PROJECT_ID         GCP project              (default: current gcloud project)
#   REGION             Region                   (default: us-central1)
#   IMAGE_URI          Full image URI           (default: derived from AR_REPO/IMAGE_NAME/TAG)
#   AR_REPO            Artifact Registry repo    (default: reinforce-tactics)
#   IMAGE_NAME         Image name                (default: rl-trainer)
#   TAG                Image tag                 (default: latest)
#   BUCKET             GCS bucket for outputs    (REQUIRED; name or gs:// URI)
#   JOB_NAME           Display name / output dir (default: rt-train-<timestamp>)
#   MACHINE_TYPE       Machine type              (default: n1-highmem-8)
#   ACCELERATOR_TYPE   GPU type                  (default: NVIDIA_TESLA_T4)
#   ACCELERATOR_COUNT  GPU count (0 = CPU only)  (default: 1)
#   REPLICA_COUNT      Worker replicas           (default: 1)
#   SYNC_INTERVAL      Seconds between GCS syncs (default: 300)
#   SERVICE_ACCOUNT    Run-as service account    (optional)
#
# Example:
#   BUCKET=my-bucket ./scripts/cloud/submit_vertex_job.sh \
#     python3 main.py --mode train --algorithm ppo --timesteps 10000000

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-reinforce-tactics}"
IMAGE_NAME="${IMAGE_NAME:-rl-trainer}"
TAG="${TAG:-latest}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${TAG}}"

MACHINE_TYPE="${MACHINE_TYPE:-n1-highmem-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_TESLA_T4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
REPLICA_COUNT="${REPLICA_COUNT:-1}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
JOB_NAME="${JOB_NAME:-rt-train-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: PROJECT_ID is not set and no default gcloud project is configured." >&2
  exit 1
fi
if [[ -z "${BUCKET:-}" ]]; then
  echo "ERROR: BUCKET is required (where trained models/checkpoints/logs are uploaded)." >&2
  echo "       export BUCKET=my-bucket" >&2
  exit 1
fi

# Training command (defaults to a PPO run if none supplied).
if [[ "$#" -gt 0 ]]; then
  TRAIN_CMD=("$@")
else
  TRAIN_CMD=(python3 main.py --mode train --algorithm ppo --timesteps 1000000)
fi

# Normalise the bucket into a gs:// base output directory for this job.
case "${BUCKET}" in
  gs://*) BASE_OUTPUT="${BUCKET%/}/jobs/${JOB_NAME}" ;;
  *)      BASE_OUTPUT="gs://${BUCKET%/}/jobs/${JOB_NAME}" ;;
esac

echo "=================================================="
echo "Vertex AI custom job"
echo "  Project:    ${PROJECT_ID}"
echo "  Region:     ${REGION}"
echo "  Image:      ${IMAGE_URI}"
echo "  Machine:    ${MACHINE_TYPE} + ${ACCELERATOR_COUNT} x ${ACCELERATOR_TYPE}"
echo "  Job name:   ${JOB_NAME}"
echo "  Output dir: ${BASE_OUTPUT}"
echo "  Command:    ${TRAIN_CMD[*]}"
echo "=================================================="

# Generate the job config. A YAML config cleanly expresses the container args,
# environment, and the worker pool — avoiding gcloud --args quoting pitfalls.
CONFIG_FILE="$(mktemp /tmp/rt-vertex-job.XXXXXX.yaml)"
trap 'rm -f "${CONFIG_FILE}"' EXIT

{
  echo "workerPoolSpecs:"
  echo "  - replicaCount: ${REPLICA_COUNT}"
  echo "    machineSpec:"
  echo "      machineType: ${MACHINE_TYPE}"
  if [[ "${ACCELERATOR_COUNT}" -gt 0 ]]; then
    echo "      acceleratorType: ${ACCELERATOR_TYPE}"
    echo "      acceleratorCount: ${ACCELERATOR_COUNT}"
  fi
  echo "    containerSpec:"
  echo "      imageUri: \"${IMAGE_URI}\""
  echo "      args:"
  for arg in "${TRAIN_CMD[@]}"; do
    # Quote every token so flags (--mode) and numbers (1000000) stay strings.
    printf '        - "%s"\n' "${arg//\"/\\\"}"
  done
  echo "      env:"
  echo "        - name: GCS_OUTPUT_URI"
  echo "          value: \"${BASE_OUTPUT}\""
  echo "        - name: GCS_SYNC_INTERVAL"
  echo "          value: \"${SYNC_INTERVAL}\""
  # Pass W&B credentials through when present so --wandb works on the worker.
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "        - name: WANDB_API_KEY"
    echo "          value: \"${WANDB_API_KEY}\""
  fi
  echo "baseOutputDirectory:"
  echo "  outputUriPrefix: \"${BASE_OUTPUT}\""
} > "${CONFIG_FILE}"

echo "Job config:"
sed 's/^/  /' "${CONFIG_FILE}"
echo "--------------------------------------------------"

SA_FLAG=()
if [[ -n "${SERVICE_ACCOUNT:-}" ]]; then
  SA_FLAG=(--service-account="${SERVICE_ACCOUNT}")
fi

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --display-name="${JOB_NAME}" \
  --config="${CONFIG_FILE}" \
  "${SA_FLAG[@]}"

echo ""
echo "✅ Submitted '${JOB_NAME}'. Trained artifacts will appear under:"
echo "     ${BASE_OUTPUT}/{models,checkpoints,tensorboard,logs}/"
echo ""
echo "Track it:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=${REGION}"
