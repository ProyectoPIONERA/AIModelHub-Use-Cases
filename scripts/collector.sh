#!/bin/bash

# --- Configuration ---
DEFAULT_INTERVAL=15
SCRIPT_NAME=$(basename "$0")

# --- Argument Parsing ---
# Allows setting interval via command line: ./script.sh 5
if [[ -n "$1" ]]; then
    INTERVAL="$1"
elif [[ -n "$INTERVAL_SECONDS" ]]; then
    # Allows setting via environment variable: INTERVAL_SECONDS=5 ./script.sh
    INTERVAL="$INTERVAL_SECONDS"
else
    INTERVAL="$DEFAULT_INTERVAL"
fi

# --- Validation ---
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]]; then
    echo "Error: Interval must be a positive integer."
    exit 1
fi

echo "Starting loop with interval: ${INTERVAL} seconds..."

# --- Main Loop ---
while true; do
    # 1. Execute your task here
    # Example: Log the current time
    python3 ./src/utils/mobility/collector.py 
    
    # Simulate work (optional - remove if your task is fast)
    # sleep 1 
    
    # 2. Wait for the configured duration
    sleep "$INTERVAL"
done