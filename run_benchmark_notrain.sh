#!/bin/bash

# Check if an index and results directory are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <index> <results_directory>"
    exit 1
fi

INDEX=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RESULTS_DIR=$2/results_notrain_run_${INDEX}

# Create timestamp for this run

# Function to cleanup processes
cleanup() {
    echo "Cleaning up..."
    
    # Signal the monitoring script to stop gracefully
    if [ ! -z "$MONITOR_PID" ]; then
        echo "Sending SIGTERM to monitoring process..."
        kill -TERM $MONITOR_PID 2>/dev/null
        
        # Wait for monitoring process to finish (max 10 seconds)
        wait $MONITOR_PID 2>/dev/null
    fi
    
    # Kill any remaining processes
    pkill -P $$
    jobs -p | xargs -r kill
    
    echo "Saving final results..."
    sleep 2
    
    echo "Results directory contents for run ${TIMESTAMP}:"
    ls -l ${RESULTS_DIR}/
    
    # # Create a symlink to the latest results
    # ln -sfn ${RESULTS_DIR} ${RESULTS_DIR%/*}/results_latest
    # echo "Created symlink 'results_latest' pointing to ${RESULTS_DIR}"

    # Analyze results
    python3 analyze_results.py --results_dir ${RESULTS_DIR}
    
    exit 0
}

# Set up trap for Ctrl+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

# Number of parallel runs
N_RUNS=5

# Create results directory with full permissions
mkdir -p ${RESULTS_DIR}
chmod 777 ${RESULTS_DIR}

echo "Created results directory: ${RESULTS_DIR}"

# Launch N parallel training processes
TRAIN_PIDS_STR=""
for i in $(seq 0 $((N_RUNS-1))); do
    python3 test_multisessionloader_notrain.py \
        --output_file "${RESULTS_DIR}" \
        --outer_index "${INDEX}" \
        --inner_index "${i}" &
    TRAIN_PIDS[$i]=$!
    # Build comma-separated string of PIDs
    TRAIN_PIDS_STR="${TRAIN_PIDS_STR}${TRAIN_PIDS[$i]},"
done

# Remove trailing comma and update monitor with PIDs
TRAIN_PIDS_STR=${TRAIN_PIDS_STR%,}

# Start resource monitoring in background with empty PID list initially
python3 monitor_resources.py --output_file ${RESULTS_DIR}/resource_usage.json --pids ${TRAIN_PIDS_STR} &
MONITOR_PID=$!

# Print initial directory state
echo "Initial results directory contents:"
ls -l ${RESULTS_DIR}/

# Wait for all training processes to complete
for pid in ${TRAIN_PIDS[@]}; do
    wait $pid
done

echo "All training processes completed. Stopping monitor..."

# Call cleanup to handle the monitoring process
cleanup

