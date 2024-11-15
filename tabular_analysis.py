import json
import os
import pandas as pd
import numpy as np
from ast import literal_eval
import argparse

def analyze_resource_usage(resource_file):
    with open(resource_file, 'r') as f:
        data = json.load(f)
    
    cpu_percents = [entry['cpu_percent'] for entry in data]
    mem_percents = [entry['system_memory_percent'] for entry in data]
    
    return {
        'cpu_mean': np.mean(cpu_percents),
        'cpu_std': np.std(cpu_percents),
        'memory_mean': np.mean(mem_percents),
        'memory_std': np.std(mem_percents)
    }

def analyze_results_folder(folder_path):
    results = []
    
    # Walk through all run folders
    for run_folder in sorted(os.listdir(folder_path)):

        run_path = os.path.join(folder_path, run_folder)
        if not os.path.isdir(run_path):
            continue
            
        # Read config
        config_path = os.path.join(run_path, 'outer_config.txt')
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            config = literal_eval(f.read())
        
        # Read resource usage
        resource_file = os.path.join(run_path, 'resource_usage.json')
        if os.path.exists(resource_file):
            resource_metrics = analyze_resource_usage(resource_file)
        else:
            resource_metrics = {
                'cpu_mean': None, 'cpu_std': None,
                'memory_mean': None, 'memory_std': None
            }
            
        # Find and process a run file
        run_files = [f for f in os.listdir(run_path) if f.startswith('run_') and f.endswith('.json')]
        if not run_files:
            continue
            
        # Take the first run file
        run_file = run_files[0]
        with open(os.path.join(run_path, run_file), 'r') as f:
            run_data = json.load(f)
            
        # Calculate FPS statistics
        fps_values = [entry['avg_fps'] for entry in run_data]
        avg_fps = f"{np.mean(fps_values):.2f}±{np.std(fps_values):.2f}"
        
        # Combine all results
        result = {
            **config,
            'fps': avg_fps,
            'cpu_util': f"{resource_metrics['cpu_mean']:.2f}±{resource_metrics['cpu_std']:.2f}" if resource_metrics['cpu_mean'] is not None else None,
            'memory_util': f"{resource_metrics['memory_mean']:.2f}±{resource_metrics['memory_std']:.2f}" if resource_metrics['memory_mean'] is not None else None
        }
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze FPS results from run folders')
    parser.add_argument('--folder_path', type=str, help='Path to the results folder')
    parser.add_argument('--output', type=str, default='results.xlsx', help='Output Excel file path (default: results.xlsx)')
    args = parser.parse_args()
    
    results_df = analyze_results_folder(args.folder_path)
    print("\nResults Analysis:")
    print(results_df.to_string(float_format=lambda x: '{:.2f}'.format(x)))
    
    # Create Excel writer with xlsxwriter engine
    with pd.ExcelWriter(os.path.join(args.folder_path, args.output), engine='xlsxwriter') as writer:
        # Write the DataFrame to Excel
        results_df.to_excel(writer, sheet_name='Results', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Results']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '0.00',
            'border': 1
        })
        
        # Set column widths and format
        for idx, col in enumerate(results_df.columns):
            # Get maximum length of column content
            max_length = max(
                results_df[col].astype(str).apply(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(idx, idx, max_length)
            
            # Apply number format only to numeric columns (not the combined mean±std columns)
            if results_df[col].dtype in ['float64', 'int64']:
                worksheet.set_column(idx, idx, max_length, number_format)
        
        # Write headers with format
        for col_num, value in enumerate(results_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Add autofilter
        worksheet.autofilter(0, 0, len(results_df), len(results_df.columns) - 1)
        
        # Freeze top row
        worksheet.freeze_panes(1, 0)
    
    print(f"\nResults saved to {args.output}")
