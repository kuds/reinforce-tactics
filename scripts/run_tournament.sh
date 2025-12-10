#!/bin/bash
# Example tournament run script
#
# This script demonstrates how to run a tournament with custom settings.
# Adjust the parameters as needed for your use case.

# Set working directory
cd "$(dirname "$0")/.." || exit

# Configuration
MAP="maps/1v1/6x6_beginner.csv"
OUTPUT_DIR="tournament_results_$(date +%Y%m%d_%H%M%S)"
GAMES_PER_SIDE=2

echo "==================================="
echo "Reinforce Tactics Tournament Runner"
echo "==================================="
echo ""
echo "Map: $MAP"
echo "Output: $OUTPUT_DIR"
echo "Games per side: $GAMES_PER_SIDE"
echo ""
echo "==================================="
echo ""

# Run tournament
python3 scripts/tournament.py \
    --map "$MAP" \
    --output-dir "$OUTPUT_DIR" \
    --games-per-side "$GAMES_PER_SIDE"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "Tournament completed successfully!"
    echo "==================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  CSV:  cat $OUTPUT_DIR/tournament_results.csv"
    echo "  JSON: cat $OUTPUT_DIR/tournament_results.json"
    echo ""
    echo "Replay files: $OUTPUT_DIR/replays/"
else
    echo ""
    echo "==================================="
    echo "Tournament failed!"
    echo "==================================="
fi
