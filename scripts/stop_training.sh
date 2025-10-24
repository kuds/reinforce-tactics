#!/bin/bash
# Stop all training instances

PROJECT_ID="reinforcetactics-rl"

echo "Stopping all training instances..."

# Get list of training instances
INSTANCES=$(gcloud compute instances list \
    --project=$PROJECT_ID \
    --filter="name~'rl-trainer-*'" \
    --format="value(name,zone)")

if [ -z "$INSTANCES" ]; then
    echo "No training instances found"
    exit 0
fi

# Stop each instance
while IFS=$'\t' read -r name zone; do
    echo "Stopping $name in $zone..."
    gcloud compute instances delete $name --zone=$zone --quiet
done <<< "$INSTANCES"

echo "✅ All training instances stopped"
