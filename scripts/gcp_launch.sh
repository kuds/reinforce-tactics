#!/bin/bash
# Launch training on Google Cloud Platform

set -e

# Configuration
PROJECT_ID="reinforcetactics-rl"
ZONE="us-central1-a"
MACHINE_TYPE="n1-highmem-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# Parse arguments
MODE=${1:-flat}  # flat or feudal
TIMESTEPS=${2:-10000000}
SEEDS=${3:-5}

echo "=================================="
echo "Launching GCP Training"
echo "=================================="
echo "Mode: $MODE"
echo "Timesteps: $TIMESTEPS"
echo "Seeds: $SEEDS"
echo "=================================="

# Build and push Docker image
echo "Building Docker image..."
docker build -t gcr.io/${PROJECT_ID}/rl-trainer:latest .
docker push gcr.io/${PROJECT_ID}/rl-trainer:latest

# Launch training instances (one per seed)
for seed in $(seq 0 $((SEEDS-1))); do
    INSTANCE_NAME="rl-trainer-${MODE}-seed${seed}-$(date +%s)"
    
    echo "Launching instance: $INSTANCE_NAME"
    
    gcloud compute instances create $INSTANCE_NAME \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=100GB \
        --boot-disk-type=pd-ssd \
        --maintenance-policy=TERMINATE \
        --metadata=startup-script="#!/bin/bash
            # Install Docker
            apt-get update
            apt-get install -y docker.io
            
            # Pull and run training container
            docker pull gcr.io/${PROJECT_ID}/rl-trainer:latest
            docker run --gpus all \
                -e MODE=$MODE \
                -e TIMESTEPS=$TIMESTEPS \
                -e SEED=$seed \
                gcr.io/${PROJECT_ID}/rl-trainer:latest \
                python train_feudal_rl.py \
                    --mode $MODE \
                    --total-timesteps $TIMESTEPS \
                    --seed $seed \
                    --n-envs 4 \
                    --device cuda \
                    --wandb
            
            # Shutdown instance after training
            shutdown -h now
        "
    
    echo "✅ Instance $INSTANCE_NAME launched"
done

echo ""
echo "=================================="
echo "All instances launched!"
echo "=================================="
echo "Monitor progress:"
echo "  gcloud compute instances list"
echo "  gcloud compute ssh <instance-name>"
