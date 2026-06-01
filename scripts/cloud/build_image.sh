#!/usr/bin/env bash
#
# Build the Reinforce Tactics training image and push it to Artifact Registry
# using Cloud Build (no local Docker daemon required).
#
# Configuration is via environment variables (sensible defaults shown). The
# image tag may also be passed as the first positional argument.
#
#   PROJECT_ID    GCP project       (default: current gcloud project)
#   REGION        Region            (default: us-central1)
#   AR_REPO       Artifact Registry repo name (default: reinforce-tactics)
#   IMAGE_NAME    Image name         (default: rl-trainer)
#   TAG           Image tag          (default: latest, or $1)
#   BUILD_TIMEOUT Cloud Build timeout (default: 3600s — CUDA images are large)
#
# Example:
#   PROJECT_ID=my-proj REGION=us-central1 ./scripts/cloud/build_image.sh v1

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-reinforce-tactics}"
IMAGE_NAME="${IMAGE_NAME:-rl-trainer}"
TAG="${1:-${TAG:-latest}}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-3600s}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: PROJECT_ID is not set and no default gcloud project is configured." >&2
  echo "       export PROJECT_ID=your-project   (or run: gcloud config set project ...)" >&2
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${TAG}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=================================================="
echo "Building training image"
echo "  Project: ${PROJECT_ID}"
echo "  Image:   ${IMAGE_URI}"
echo "  Context: ${REPO_ROOT}"
echo "=================================================="

# Create the Artifact Registry repo on first use (idempotent).
if ! gcloud artifacts repositories describe "${AR_REPO}" \
      --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Creating Artifact Registry repository '${AR_REPO}' in ${REGION}..."
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="Reinforce Tactics training images"
fi

# Build remotely and push. Cloud Build reads .dockerignore for the build context
# and .gcloudignore for what to upload.
gcloud builds submit "${REPO_ROOT}" \
  --tag "${IMAGE_URI}" \
  --project="${PROJECT_ID}" \
  --timeout="${BUILD_TIMEOUT}"

echo ""
echo "✅ Pushed ${IMAGE_URI}"
echo "Next: submit a job with"
echo "  IMAGE_URI=${IMAGE_URI} BUCKET=your-bucket ./scripts/cloud/submit_vertex_job.sh \\"
echo "    python3 main.py --mode train --algorithm ppo --timesteps 10000000"
