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
#   SYNC_DIRS          Extra dirs to sync, comma-separated (optional; see
#                      GCS_SYNC_DIRS in vertex_train.py). models/, checkpoints/,
#                      tensorboard/, logs/ and benchmarks/bootstrap/ always are.
#   OUTPUT_URI         gs:// output base (optional; default
#                      gs://<BUCKET>/jobs/<JOB_NAME>). run_seeds.py sets
#                      gs://<BUCKET>/jobs/<group> so every seed of a group
#                      lands under one prefix.
#   RESTORE_DIRS       Dirs to download from the output base before the command
#                      starts (optional; passed as GCS_RESTORE_DIRS, same
#                      dir=prefix syntax as SYNC_DIRS). A resubmitted seed job
#                      continues its run this way.
#   RESTART_ON_WORKER_RESTART  1 = restart the job when its worker restarts
#                      (scheduling.restartJobOnWorkerRestart; optional)
#   SERVICE_ACCOUNT    Run-as service account    (optional)
#
# On success the script prints JOB_RESOURCE=projects/.../customJobs/<id>
# (scripts/train/run_seeds.py reads it to check the job before resubmitting).
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

# Validate numeric worker-pool fields up front so a typo (e.g. ACCELERATOR_COUNT=t4)
# fails clearly here instead of crashing the arithmetic comparison below under
# `set -u`, or silently producing an invalid job spec.
for _numvar in REPLICA_COUNT ACCELERATOR_COUNT SYNC_INTERVAL; do
  if ! [[ "${!_numvar}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${_numvar} must be a non-negative integer, got '${!_numvar}'." >&2
    exit 1
  fi
done

# Emit a value as a single-quoted YAML scalar. A single-quoted scalar only needs
# embedded single quotes doubled, so values containing backslashes, ':', '$',
# leading '-', etc. can never corrupt the generated config.
yaml_squote() { printf "'%s'" "${1//\'/\'\'}"; }

# Training command (defaults to a PPO run if none supplied).
if [[ "$#" -gt 0 ]]; then
  TRAIN_CMD=("$@")
else
  TRAIN_CMD=(python3 main.py --mode train --algorithm ppo --timesteps 1000000)
fi

# Normalise the bucket into a gs:// base output directory for this job, unless
# OUTPUT_URI names one (a seed group shares one base).
if [[ -n "${OUTPUT_URI:-}" ]]; then
  case "${OUTPUT_URI}" in
    gs://*) BASE_OUTPUT="${OUTPUT_URI%/}" ;;
    *) echo "ERROR: OUTPUT_URI must be a gs:// URI, got '${OUTPUT_URI}'." >&2; exit 1 ;;
  esac
else
  case "${BUCKET}" in
    gs://*) BASE_OUTPUT="${BUCKET%/}/jobs/${JOB_NAME}" ;;
    *)      BASE_OUTPUT="gs://${BUCKET%/}/jobs/${JOB_NAME}" ;;
  esac
fi

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
  echo "      machineType: $(yaml_squote "${MACHINE_TYPE}")"
  if [[ "${ACCELERATOR_COUNT}" -gt 0 ]]; then
    echo "      acceleratorType: $(yaml_squote "${ACCELERATOR_TYPE}")"
    echo "      acceleratorCount: ${ACCELERATOR_COUNT}"
  fi
  echo "    containerSpec:"
  echo "      imageUri: $(yaml_squote "${IMAGE_URI}")"
  echo "      args:"
  for arg in "${TRAIN_CMD[@]}"; do
    # Single-quote every token so flags (--mode), numbers (1000000), and any
    # value with backslashes/colons stay literal strings.
    printf '        - %s\n' "$(yaml_squote "${arg}")"
  done
  echo "      env:"
  echo "        - name: GCS_OUTPUT_URI"
  echo "          value: $(yaml_squote "${BASE_OUTPUT}")"
  echo "        - name: GCS_SYNC_INTERVAL"
  echo "          value: $(yaml_squote "${SYNC_INTERVAL}")"
  if [[ -n "${SYNC_DIRS:-}" ]]; then
    echo "        - name: GCS_SYNC_DIRS"
    echo "          value: $(yaml_squote "${SYNC_DIRS}")"
  fi
  if [[ -n "${RESTORE_DIRS:-}" ]]; then
    echo "        - name: GCS_RESTORE_DIRS"
    echo "          value: $(yaml_squote "${RESTORE_DIRS}")"
  fi
  # Pass W&B credentials through when present so --wandb works on the worker.
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "        - name: WANDB_API_KEY"
    echo "          value: $(yaml_squote "${WANDB_API_KEY}")"
  fi
  echo "baseOutputDirectory:"
  echo "  outputUriPrefix: $(yaml_squote "${BASE_OUTPUT}")"
  if [[ "${RESTART_ON_WORKER_RESTART:-0}" == "1" ]]; then
    # CustomJobSpec.scheduling.restartJobOnWorkerRestart (Vertex AI v1 API).
    echo "scheduling:"
    echo "  restartJobOnWorkerRestart: true"
  fi
} > "${CONFIG_FILE}"

echo "Job config:"
sed 's/^/  /' "${CONFIG_FILE}"
echo "--------------------------------------------------"

SA_FLAG=()
if [[ -n "${SERVICE_ACCOUNT:-}" ]]; then
  SA_FLAG=(--service-account="${SERVICE_ACCOUNT}")
fi

# gcloud reports the new job's resource name (on stderr, as "CustomJob
# [projects/.../customJobs/<id>] is submitted successfully."); keep its output
# to print that name on a line of its own.
if ! SUBMIT_OUTPUT="$(gcloud ai custom-jobs create \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --display-name="${JOB_NAME}" \
  --config="${CONFIG_FILE}" \
  "${SA_FLAG[@]}" 2>&1)"; then
  printf '%s\n' "${SUBMIT_OUTPUT}" >&2
  echo "ERROR: gcloud ai custom-jobs create failed." >&2
  exit 1
fi
printf '%s\n' "${SUBMIT_OUTPUT}"
JOB_RESOURCE="$(printf '%s\n' "${SUBMIT_OUTPUT}" | grep -oE 'projects/[^] /]+/locations/[^] /]+/customJobs/[0-9]+' | head -n 1 || true)"

echo ""
echo "JOB_RESOURCE=${JOB_RESOURCE}"
echo "✅ Submitted '${JOB_NAME}'. Trained artifacts will appear under:"
echo "     ${BASE_OUTPUT}/{models,checkpoints,tensorboard,logs}/"
echo "   (train_bootstrap.py runs: ${BASE_OUTPUT}/<run timestamp>/)"
echo ""
echo "Track it:"
echo "  gcloud ai custom-jobs list --region=${REGION} --project=${PROJECT_ID}"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=${REGION}"
