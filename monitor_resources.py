import psutil
import time
import json
from datetime import datetime
from experanto.ram_usage import MemoryMonitor
import os
import argparse
from utils import save_results
import signal

# Add global variable to store PIDs to monitor
PIDS_TO_MONITOR = set()

def update_monitored_pids(signum, frame):
    """Signal handler to update monitored PIDs from file"""
    global PIDS_TO_MONITOR
    try:
        with open("results_latest/pids_to_monitor", 'r') as f:
            pids_str = f.read().strip()
            PIDS_TO_MONITOR = {int(pid) for pid in pids_str.split(',') if pid}
        print(f"Updated PIDs to monitor: {PIDS_TO_MONITOR}")
    except Exception as e:
        print(f"Error updating PIDs: {e}")

def get_valid_python_processes():
    """Get list of valid (non-zombie) Python processes that match our monitored PIDs."""
    global PIDS_TO_MONITOR
    processes = []
    for proc in psutil.process_iter(['name', 'pid', 'status']):
        try:
            if ('python' in proc.info['name'] and 
                proc.info['status'] != 'zombie' and 
                (not PIDS_TO_MONITOR or proc.info['pid'] in PIDS_TO_MONITOR)):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def save_stats(stats, output_file):
    save_results(stats, output_file)
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
            
    except Exception as e:
        print(f"Error saving stats to {output_file}: {e}")

def monitor_resources(output_file, initial_pids=""):
    global PIDS_TO_MONITOR
    
    # Set up signal handler for USR1
    signal.signal(signal.SIGUSR1, update_monitored_pids)
    
    # Initialize PIDs if provided
    if initial_pids:
        PIDS_TO_MONITOR = {int(pid) for pid in initial_pids.split(',') if pid}
    
    stats = []
    memory_monitor = MemoryMonitor()
    last_save_time = time.time()
    
    print(f"Monitoring resources, will save to: {output_file}")
    print(f"Initial PIDs to monitor: {PIDS_TO_MONITOR}")
    
    while True:
        try:
            # Update monitored processes
            python_processes = get_valid_python_processes()
            
            # Reset memory monitor with current valid PIDs
            memory_monitor = MemoryMonitor([proc.info['pid'] for proc in python_processes])
            
            try:
                memory_data = memory_monitor._refresh()
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied) as e:
                print(f"Error refreshing memory data: {e}")
                memory_data = {}
            
            current_stats = {
                'timestamp': datetime.now().isoformat(),
                'timestamp_number': datetime.now().timestamp(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'system_memory_percent': psutil.virtual_memory().percent,
                'process_memory': {
                    str(pid): {
                        k: v for k, v in data.items()
                    } for pid, data in memory_data.items()
                },
                'active_python_processes': len(python_processes)
            }
            
            # Print current state
            # print(f"\n{datetime.now().strftime('%H:%M:%S')}")
            # print(memory_monitor.table())
            
            # Save every 30 seconds instead of every minute
            current_time = time.time()
            if current_time - last_save_time >= 1:
                save_stats(current_stats, output_file)
                last_save_time = current_time
                # print(f"Saved {len(stats)} records")
            
            time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            save_stats(stats, output_file)
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            save_stats(stats, output_file)  # Try to save before exiting
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to save the monitoring results')
    parser.add_argument('--pids', type=str, default="",
                      help='Comma-separated list of PIDs to monitor')
    args = parser.parse_args()
    
    monitor_resources(args.output_file, args.pids)