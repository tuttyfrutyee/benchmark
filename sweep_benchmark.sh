#!/bin/bash

# Function to handle Ctrl+C
cleanup() {
    echo "Benchmark interrupted. Exiting..."
    exit 1
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Check if description is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <description>"
    echo "Please provide a brief description for the benchmark results"
    exit 1
fi

# Get the description from command line argument
DESCRIPTION=$1

# Create a folder with the current date and description
DATE=$(date +%Y-%m-%d-%H-%M)
RESULTS_DIR="Results_${DATE}_${DESCRIPTION}"
mkdir -p "$RESULTS_DIR"

# Number of times to run the benchmark
NUM_RUNS=12

# Loop to run the benchmark script 48 times
for i in $(seq 0 $(($NUM_RUNS-1))); do
    echo "Running benchmark iteration $i"
    ./run_benchmark.sh $i "$RESULTS_DIR"
    echo "Completed benchmark iteration $i"
done 

# Run tabular analysis on the results
echo "Running tabular analysis..."
python tabular_analysis.py --folder_path "$RESULTS_DIR"
echo "Analysis complete. Results saved in $RESULTS_DIR"

