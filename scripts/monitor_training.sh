#!/bin/bash
# Monitor training progress on GCP

PROJECT_ID="reinforcetactics-rl"

echo "Active Training Instances:"
echo "=================================="
gcloud compute instances list --project=$PROJECT_ID --filter="name~'rl-trainer-*'"

echo ""
echo "To SSH into an instance:"
echo "  gcloud compute ssh <instance-name> --zone=<zone>"
echo ""
echo "To view logs:"
echo "  gcloud compute ssh <instance-name> --zone=<zone> --command='docker logs <container-id>'"
echo ""
echo "To stop all training instances:"
echo "  ./scripts/stop_training.sh"
