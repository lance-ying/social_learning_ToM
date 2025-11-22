import csv
import sys
import os
import glob

def parse_pilot_csv(filename, excluded_levels=None):
    """
    Parse a pilot CSV file and calculate total remaining steps per user.
    
    Args:
        filename: Path to the CSV file
        excluded_levels: List of level names to exclude (default: ['mod_s111', 'mod_s112', 's211'])
    
    Returns:
        Dictionary with userID, prolificID, and total remaining steps
    """
    if excluded_levels is None:
        excluded_levels = ['new_s111_1', 'new_s112_1', 'new_s111_2', 'new_s112_2']
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Extract userID from first line (Session ID)
    user_id = lines[0].strip().split(',')[1]
    
    # Extract Prolific ID from User Data section
    prolific_id = None
    in_user_data = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if we're entering User Data section
        if line_stripped == 'User Data':
            in_user_data = True
            continue
        
        # If we're in User Data section, look for prolificId
        if in_user_data:
            if line_stripped.startswith('prolificId,'):
                prolific_id = line_stripped.split(',', 1)[1]
                break
            # Exit User Data section if we hit an empty line
            elif not line_stripped or line_stripped.startswith('Comprehension Check'):
                in_user_data = False
    
    # Parse the file by level sections
    current_level = None
    steps_data = {}
    
    for line in lines:
        line = line.strip()
        
        # Check if this is a level marker
        if line.startswith('Level:'):
            current_level = line.replace('Level:', '').strip()
            steps_data[current_level] = []
            continue
        
        # If we're in a level section and it's not the header line
        if current_level and not line.startswith('Timestamp') and line:
            parts = line.split(',')
            
            # Steps Remaining is at index 7 (8th column)
            if len(parts) > 7 and parts[7]:
                try:
                    steps = float(parts[7].strip())
                    steps_data[current_level].append(steps)
                except ValueError:
                    pass  # Skip non-numeric values
    
    # Calculate totals
    results = {
        'user_id': user_id,
        'prolific_id': prolific_id,
        'levels': {},
        'total_steps': 0
    }
    
    for level, steps_list in steps_data.items():
        if steps_list:
            level_total = sum(steps_list)
            is_excluded = level in excluded_levels
            
            results['levels'][level] = {
                'total_steps': level_total,
                'num_entries': len(steps_list),
                'excluded': is_excluded
            }
            
            # Add to total if not excluded
            if not is_excluded:
                results['total_steps'] += level_total
    
    return results


def main():
    # Get all CSV files in the pilot_100625+092925 directory
    csv_files = glob.glob('pilot_101725_exp2/*.csv')
    
    if not csv_files:
        print("No CSV files found in pilot_101625_exo2/ directory")
        return
    
    print(f"Processing {len(csv_files)} CSV files...")
    print("="*80)
    
    all_results = []
    
    for filename in sorted(csv_files):
        try:
            # Parse the CSV
            results = parse_pilot_csv(filename)
            all_results.append(results)
            
            # Print results for this file
            print(f"Session ID:   {results['user_id']}")
            print(f"Prolific ID:  {results['prolific_id'] if results['prolific_id'] else 'N/A'}")
            print(f"File:         {filename}")
            print(f"Total Steps:  {results['total_steps']:.0f} (excluding mod_s111, mod_s112, s211)")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            print("-" * 80)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files processed: {len(all_results)}")
    
    if all_results:
        total_steps_all = sum(r['total_steps'] for r in all_results)
        print(f"Combined total steps (all users): {total_steps_all:.0f}")
        print(f"Average steps per user: {total_steps_all/len(all_results):.0f}")
        
        # Count how many have Prolific IDs
        with_prolific = sum(1 for r in all_results if r['prolific_id'])
        print(f"Files with Prolific ID: {with_prolific}/{len(all_results)}")
    
    # Optional: Export to CSV
    export_choice = input("\nExport results to CSV? (y/n): ").lower()
    if export_choice == 'y':
        output_file = 'pilot_results_summary_101625.csv'
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Session ID', 'Prolific ID', 'Total Steps', 'Filename'])
            for result, filename in zip(all_results, sorted(csv_files)):
                writer.writerow([
                    result['user_id'],
                    result['prolific_id'] if result['prolific_id'] else 'N/A',
                    f"{result['total_steps']:.0f}",
                    filename
                ])
        print(f"Results exported to {output_file}")


if __name__ == "__main__":
    main()