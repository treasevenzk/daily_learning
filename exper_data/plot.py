import ast
import matplotlib.pyplot as plt
import sys
import os
import glob
import platform
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

def analyze_perf_values(file_path):
    total_perf_count = 0
    zero_perf_count = 0
    
    # Track best perf value evolution
    line_numbers = []
    best_perf_values = []
    current_best_perf = float('-inf')  # Initialize to negative infinity
    
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data_dict = ast.literal_eval(line)
                    
                    if 'perf' in data_dict:
                        perf_value = data_dict['perf']
                        total_perf_count += 1
                        
                        if perf_value == 0:
                            zero_perf_count += 1
                        
                        if perf_value > current_best_perf:
                            current_best_perf = perf_value
                        
                        line_numbers.append(line_num)
                        best_perf_values.append(current_best_perf)
                
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error message: {e}")
                    continue
        
        return total_perf_count, zero_perf_count, line_numbers, best_perf_values
    
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return 0, 0, [], []
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0, 0, [], []

def plot_best_perf(line_numbers, best_perf_values, output_path="best_perf_evolution.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(line_numbers, best_perf_values, 'b-', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Line Count', fontsize=12)
    plt.ylabel('Best Perf Value', fontsize=12)
    plt.title('Best Perf Value Evolution Over Lines', fontsize=14)
    
    if len(line_numbers) > 0:
        plt.scatter([line_numbers[0]], [best_perf_values[0]], color='red', s=50, label='Start')
        plt.scatter([line_numbers[-1]], [best_perf_values[-1]], color='green', s=50, label='End')
    
    plt.legend()
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Chart saved to: {output_path}")
    
    # Close the figure to avoid memory leaks
    plt.close()

def find_all_records_files(ops_dir):
    """Find all records.txt files in the OPS directory"""
    records_files = []
    
    # Use os.walk to traverse all subdirectories of the OPS directory
    for root, dirs, files in os.walk(ops_dir):
        for file in files:
            if file == "records.txt" or file == ".records.txt":
                full_path = os.path.join(root, file)
                records_files.append(full_path)
    
    return records_files

def group_records_by_top_folder(ops_dir, records_files):
    """Group record files by top-level folder"""
    groups = {}
    
    for file_path in records_files:
        # Get relative path
        rel_path = os.path.relpath(file_path, ops_dir)
        # Get top-level folder name
        top_folder = rel_path.split(os.sep)[0]
        
        if top_folder not in groups:
            groups[top_folder] = []
        
        groups[top_folder].append(file_path)
    
    return groups

def process_files(record_files):
    """Process all found record files"""
    results = {}
    file_data = {}  # Store data for each file for later chart generation
    
    for file_path in record_files:
        print(f"\nAnalyzing file: {file_path}")
        
        # Create appropriate output directory
        output_dir = os.path.join(os.path.dirname(file_path), "perf_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output chart path
        rel_path = os.path.relpath(file_path, ops_dir).replace('/', '_').replace('\\', '_')
        output_path = os.path.join(output_dir, f"{rel_path}_perf_evolution.png")
        
        # Analyze file
        total_count, zero_count, line_numbers, best_perf_values = analyze_perf_values(file_path)
        
        # Store file data for later chart generation
        if line_numbers and best_perf_values:
            file_data[file_path] = {
                'line_numbers': line_numbers,
                'best_perf_values': best_perf_values,
                'rel_path': os.path.relpath(file_path, ops_dir)
            }
        
        # Print statistics
        print(f"Statistics:")
        print(f"Total perf values: {total_count}")
        print(f"Perf values = 0: {zero_count}")
        print(f"Percentage of zero perf values: {zero_count/total_count:.2%}" if total_count > 0 else "No perf values found")
        
        # Generate chart
        if line_numbers and best_perf_values:
            plot_best_perf(line_numbers, best_perf_values, output_path)
        else:
            print("Not enough data to generate chart")
        
        # Store results for later comparison
        results[file_path] = {
            'total_count': total_count,
            'zero_count': zero_count,
            'zero_ratio': zero_count/total_count if total_count > 0 else 0,
            'best_perf': best_perf_values[-1] if best_perf_values else None
        }
    
    return results, file_data

def plot_grouped_perf_evolution(file_groups, file_data, output_dir):
    """Plot combined chart of best perf value evolution by group"""
    for group_name, file_paths in file_groups.items():
        # Calculate number of rows and columns for subplots
        num_files = len(file_paths)
        if num_files == 0:
            continue
            
        # Filter files with data
        valid_files = [f for f in file_paths if f in file_data]
        if not valid_files:
            continue
            
        num_files = len(valid_files)
        rows = int(np.ceil(np.sqrt(num_files)))
        cols = int(np.ceil(num_files / rows))
        
        # Create chart
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), squeeze=False)
        fig.suptitle(f'"{group_name}" Folder - Best Perf Value Evolution', fontsize=16)
        
        # Plot each file's data in a subplot
        for i, file_path in enumerate(valid_files):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            data = file_data[file_path]
            line_numbers = data['line_numbers']
            best_perf_values = data['best_perf_values']
            rel_path = data['rel_path']
            
            # Get filename, without path
            file_name = os.path.basename(file_path)
            # Get directory name
            dir_name = os.path.basename(os.path.dirname(file_path))
            
            # Plot line chart
            ax.plot(line_numbers, best_perf_values, 'b-', linewidth=1.5)
            
            # Mark start and end points
            ax.scatter([line_numbers[0]], [best_perf_values[0]], color='red', s=30, label='Start')
            ax.scatter([line_numbers[-1]], [best_perf_values[-1]], color='green', s=30, label='End')
            
            # Set title and labels - use directory/filename as title
            title = f'{dir_name}/{file_name}'
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Line Count', fontsize=8)
            ax.set_ylabel('Best Perf Value', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Only show legend in the first subplot
            if i == 0:
                ax.legend(fontsize=8)
        
        # Hide empty subplots
        for i in range(num_files, rows * cols):
            row = i // cols
            col = i % cols
            if row < len(axes) and col < len(axes[row]):
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.92)  # Leave space for the main title
        
        # Save chart
        output_path = os.path.join(output_dir, f"{group_name}_combined_perf_evolution.png")
        plt.savefig(output_path)
        print(f"Combined chart saved to: {output_path}")
        plt.close(fig)

def plot_all_in_one(file_groups, file_data, output_dir):
    """Plot all groups' data in one chart, using different colors for each group"""
    plt.figure(figsize=(12, 8))
    
    # Choose different colors for each group
    colors = plt.cm.tab10.colors
    
    # Record the maximum perf value for each group, for comparison
    max_perfs = {}
    
    # Plot curves by group
    for i, (group_name, file_paths) in enumerate(file_groups.items()):
        color = colors[i % len(colors)]
        
        # Find the file with the highest best perf value in the group
        best_file = None
        best_perf = float('-inf')
        
        for file_path in file_paths:
            if file_path in file_data:
                perf_values = file_data[file_path]['best_perf_values']
                if perf_values and perf_values[-1] > best_perf:
                    best_perf = perf_values[-1]
                    best_file = file_path
        
        if best_file:
            data = file_data[best_file]
            line_numbers = data['line_numbers']
            best_perf_values = data['best_perf_values']
            
            # Plot the best file's curve for this group
            plt.plot(line_numbers, best_perf_values, 
                    color=color, linewidth=2, label=f"{group_name} (Best: {best_perf:.2f})")
            
            # Record the maximum perf value
            max_perfs[group_name] = best_perf
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Line Count', fontsize=12)
    plt.ylabel('Best Perf Value', fontsize=12)
    plt.title('Best Perf Value Comparison Across Groups', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    output_path = os.path.join(output_dir, "all_groups_best_perf_comparison.png")
    plt.savefig(output_path)
    print(f"Group comparison chart saved to: {output_path}")
    plt.close()
    
    # Return the maximum perf values for each group, for ranking report
    return max_perfs

def group_all_files_in_one(file_groups, file_data, output_dir):
    """Plot all files from each group in a single chart per group"""
    for group_name, file_paths in file_groups.items():
        # Filter valid files
        valid_files = [f for f in file_paths if f in file_data]
        if not valid_files:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Choose different colors for each file
        colors = plt.cm.tab10.colors
        
        # Plot each file in this group
        for i, file_path in enumerate(valid_files):
            color = colors[i % len(colors)]
            
            data = file_data[file_path]
            line_numbers = data['line_numbers']
            best_perf_values = data['best_perf_values']
            
            # Get filename, without path
            file_name = os.path.basename(file_path)
            # Get directory name
            dir_name = os.path.basename(os.path.dirname(file_path))
            
            # Plot the file's curve
            plt.plot(line_numbers, best_perf_values, 
                    color=color, linewidth=2, 
                    label=f"{dir_name}/{file_name} (Best: {best_perf_values[-1]:.2f})")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Line Count', fontsize=12)
        plt.ylabel('Best Perf Value', fontsize=12)
        plt.title(f'"{group_name}" Folder - All Files Best Perf Value Evolution', fontsize=14)
        plt.legend(fontsize=8, loc='best')
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(output_dir, f"{group_name}_all_files_in_one.png")
        plt.savefig(output_path)
        print(f"Group all-in-one chart saved to: {output_path}")
        plt.close()

def plot_comparison(results, output_path="perf_comparison.png"):
    """Compare zero perf percentages across different files"""
    if not results:
        print("No data for comparison")
        return
    
    # Extract data
    file_names = []
    zero_ratios = []
    
    for file_path, result in results.items():
        # Use only directory name as chart label, without records.txt
        dir_name = os.path.basename(os.path.dirname(file_path))
        file_names.append(dir_name)
        
        # Calculate zero perf percentage
        zero_ratios.append(result['zero_ratio'] * 100)  # Convert to percentage
    
    # Create a single chart for zero perf percentage
    plt.figure(figsize=(12, 8))
    
    # Zero perf percentage comparison
    bars = plt.bar(range(len(file_names)), zero_ratios, color='salmon')
    plt.ylabel('Percentage of Zero Perf Values (%)', fontsize=12)
    plt.title('Zero Perf Percentage Comparison Across Folders', fontsize=14)
    plt.xticks(range(len(file_names)), file_names, rotation=90, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Leave space for rotated labels
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Zero perf percentage chart saved to: {output_path}")
    
    plt.close()

def save_summary_to_csv(results, output_path):
    """Save summary data to CSV file"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("File Path,Total Perf Values,Zero Perf Values,Zero Perf Percentage(%),Best Perf Value\n")
            
            # Write data for each file
            for file_path, result in results.items():
                rel_path = os.path.relpath(file_path, ops_dir)
                f.write(f"{rel_path},{result['total_count']},{result['zero_count']},{result['zero_ratio']*100:.2f},{result['best_perf'] if result['best_perf'] is not None else 'N/A'}\n")
        
        print(f"Summary data saved to: {output_path}")
    
    except Exception as e:
        print(f"Error saving summary data: {e}")

def save_group_ranking(max_perfs, output_path):
    """Save group ranking by best perf value to CSV file"""
    try:
        # Sort groups by best perf value
        sorted_perfs = sorted(max_perfs.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Rank,Group,Best Perf Value\n")
            
            # Write ranking data
            for i, (group, perf) in enumerate(sorted_perfs, 1):
                f.write(f"{i},{group},{perf:.6f}\n")
        
        print(f"Group ranking data saved to: {output_path}")
    
    except Exception as e:
        print(f"Error saving group ranking data: {e}")

def main():
    global ops_dir
    
    if len(sys.argv) > 1:
        # Accept command line parameter as OPS directory
        ops_dir = sys.argv[1]
    else:
        # Default to OPS directory in the current directory
        ops_dir = "OPS"
    
    # Ensure OPS directory exists
    if not os.path.isdir(ops_dir):
        print(f"Error: Directory {ops_dir} does not exist")
        return
    
    print(f"Searching for records.txt files in {ops_dir} directory")
    
    # Find all record files
    record_files = find_all_records_files(ops_dir)
    
    if not record_files:
        print("No records.txt files found")
        return
    
    print(f"Found {len(record_files)} records.txt files:")
    for file in record_files:
        print(f"  - {file}")
    
    # Group record files by top-level folder
    file_groups = group_records_by_top_folder(ops_dir, record_files)
    print(f"Grouping result: {len(file_groups)} groups")
    for group, files in file_groups.items():
        print(f"  - {group}: {len(files)} files")
    
    # Process all files
    results, file_data = process_files(record_files)
    
    # Create summary directory
    summary_dir = os.path.join(ops_dir, "perf_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Generate combined charts for each group
    plot_grouped_perf_evolution(file_groups, file_data, summary_dir)
    
    # Generate all-in-one charts for each group
    group_all_files_in_one(file_groups, file_data, summary_dir)
    
    # Generate comparison chart for all groups
    max_perfs = plot_all_in_one(file_groups, file_data, summary_dir)
    
    # Save group ranking to CSV file
    save_group_ranking(max_perfs, os.path.join(summary_dir, "group_ranking.csv"))
    
    # Generate comparison chart
    plot_comparison(results, os.path.join(summary_dir, "perf_comparison.png"))
    
    # Save summary data to CSV file
    save_summary_to_csv(results, os.path.join(summary_dir, "perf_summary.csv"))

if __name__ == "__main__":
    main()