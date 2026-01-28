#!/usr/bin/env python3
"""
Script to merge two experiment output JSON files.
"""

import json
import sys
from pathlib import Path

def merge_json_files(file1_path, file2_path, output_path):
    """
    Merge two JSON files by combining their dictionaries.
    
    Args:
        file1_path (str): Path to first JSON file
        file2_path (str): Path to second JSON file  
        output_path (str): Path for merged output file
    """
    # Load first file
    with open(file1_path, 'r') as f:
        data1 = json.load(f)
    
    # Load second file
    with open(file2_path, 'r') as f:
        data2 = json.load(f)
    
    # Merge dictionaries
    merged_data = {**data1, **data2}
    
    # Write merged data to output file
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"Merged {len(data1)} entries from file 1 and {len(data2)} entries from file 2")
    print(f"Total merged entries: {len(merged_data)}")
    print(f"Output written to: {output_path}")

if __name__ == "__main__":
    # File paths
    file1 = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/scripts/experiments/experiment_outputs/steps_dict_exp4_012626_scene_1.json"
    file2 = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/scripts/experiments/experiment_outputs/steps_dict_exp4_012626_scene_2.json"
    output = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/scripts/experiments/experiment_outputs/steps_dict_exp4_012626_merged.json"
    
    # Merge the files
    merge_json_files(file1, file2, output)