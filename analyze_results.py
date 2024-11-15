import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils import load_results

def plot_combined_metrics(resource_data, run_data, pid, results_dir, config_content):
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(hspace=0.1)  # Reduce space between subplots
    
    # Extract timestamps and metrics for the specific PID
    timestamps = [pd.to_datetime(d['timestamp']) for d in resource_data]
    cpu_percent = [d['cpu_percent'] for d in resource_data]
    memory_usage = [d['process_memory'].get(pid, {}).get('rss', 0) / (1024*1024*1024) for d in resource_data]  # Convert to GB
    
    # Plot CPU and Memory usage
    ax1_twin = ax1.twinx()
    
    # CPU usage in blue
    l1 = ax1.plot(timestamps, cpu_percent, 'b-', label='CPU Usage')
    ax1.set_ylabel('CPU Usage (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Memory usage in red
    l2 = ax1_twin.plot(timestamps, memory_usage, 'r-', label='Memory Usage')
    ax1_twin.set_ylabel('Memory Usage (GB)', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Add legend for first subplot
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')
    
    # Plot FPS
    fps_timestamps = [pd.to_datetime(d['timestamp']) for d in run_data]
    fps = [d['avg_fps'] for d in run_data]
    
    ax2.plot(fps_timestamps, fps, 'g-', label='Average FPS')
    ax2.set_ylabel('FPS')
    ax2.set_xlabel('Time')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    plt.title(f'PID {pid} - Config: {config_content}')
    plt.savefig(f'{results_dir}/combined_metrics.png')
    plt.close()

def analyze_results(results_dir):
    # Load resource usage data
    resource_data = load_results(f'{results_dir}/resource_usage.json')

    import random
    import re

    # List all files in the results_latest directory
    results_dir = Path(results_dir)
    all_files = list(results_dir.glob('run_*.json'))

    # Filter files that have 'run_' in their name
    run_files = [f for f in all_files if re.match(r'run_\d+\.json', f.name)]

    # Randomly select one of the run files
    selected_run_file = random.choice(run_files)

    # Extract the PID from the selected file name
    selected_pid = re.search(r'run_(\d+)\.json', selected_run_file.name).group(1)
    
    # Load single run results (46524)
    run_data = load_results(f'{results_dir}/run_{selected_pid}.json')
    
    # Read outer_config.txt
    config_file = results_dir / 'outer_config.txt'
    if config_file.exists():
        with open(config_file, 'r') as file:
            config_content = file.read().strip()
    else:
        config_content = 'No Config Found'
    
    # Create combined plot with config in title
    plot_combined_metrics(resource_data, run_data, selected_pid, results_dir, config_content)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze results')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to the results directory')
    args = parser.parse_args()

    results_dir = args.results_dir

    print(f'Analyzing results from {results_dir}')

    analyze_results(results_dir)