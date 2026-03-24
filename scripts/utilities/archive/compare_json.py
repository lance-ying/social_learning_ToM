#!/usr/bin/env python3
import json
import sys

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_dicts(dict1, dict2, path="", differences=None):
    if differences is None:
        differences = []
    
    # Get all keys from both dicts
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key
        
        if key not in dict1:
            differences.append(f"Key '{current_path}' only in file2: {dict2[key]}")
        elif key not in dict2:
            differences.append(f"Key '{current_path}' only in file1: {dict1[key]}")
        else:
            val1 = dict1[key]
            val2 = dict2[key]
            
            if isinstance(val1, dict) and isinstance(val2, dict):
                compare_dicts(val1, val2, current_path, differences)
            elif val1 != val2:
                differences.append(f"Key '{current_path}':")
                differences.append(f"  file1: {val1}")
                differences.append(f"  file2: {val2}")
    
    return differences

if __name__ == "__main__":
    file1 = "scripts/experiments/experiment_outputs/debug_again.json"
    file2 = "steps_dict_exp3.json"
    
    print(f"Comparing:")
    print(f"  file1: {file1}")
    print(f"  file2: {file2}")
    print()
    
    try:
        data1 = load_json(file1)
        data2 = load_json(file2)
        
        differences = compare_dicts(data1, data2)
        
        if differences:
            print("=== DIFFERENCES ===")
            for diff in differences:
                print(diff)
        else:
            print("Files are identical!")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

