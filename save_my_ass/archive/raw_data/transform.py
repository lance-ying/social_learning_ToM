import pandas as pd

def create_simplified_csv():
    """
    Create a simplified CSV with just pid and amount columns
    Includes all PIDs from Prolific export, assigning 0.5 to those not in pilot results
    """
    # Read the files
    pilot_file = '/Users/heyodogo/Documents/labs/lance_lab/social_learning_ToM/raw_data/pilot_100625+092925/pilot_results_summary.csv'
    prolific_file = '/Users/heyodogo/Documents/labs/lance_lab/social_learning_ToM/raw_data/prolific_export_68e443f68b7d9eb18571fd0c.csv'
    output_file = 'pilot_results_simplified_1006_total.csv'
    
    try:
        # Read the pilot results CSV
        pilot_df = pd.read_csv(pilot_file)
        
        # Read the Prolific export CSV
        prolific_df = pd.read_csv(prolific_file)
        
        # Create transformation function
        def transform_steps(steps):
            if pd.isna(steps) or steps == '':
                return steps
            try:
                steps_num = float(steps)
                # If negative, return 0.0
                if steps_num < 0:
                    return 0.0
                # Linear scaling: steps/200, capped at 1.00
                scaled_value = min(steps_num / 200, 1.00)
                return round(scaled_value, 2)
            except (ValueError, TypeError):
                return steps
        
        # Create a dictionary of pid -> amount from pilot results
        pilot_pids = {}
        for _, row in pilot_df.iterrows():
            pid = row['Prolific ID']
            if pd.notna(pid) and pid != 'N/A':
                amount = transform_steps(row['Total Steps'])
                pilot_pids[pid] = amount
        
        # Get all PIDs from Prolific export (only APPROVED participants)
        prolific_pids = prolific_df[prolific_df['Status'] == 'APPROVED']['Participant id'].tolist()
        
        # Create the final dataframe
        final_data = []
        for pid in prolific_pids:
            if pid in pilot_pids:
                # Use the transformed amount from pilot results
                amount = pilot_pids[pid]
            else:
                # Assign 0.5 for PIDs not in pilot results
                amount = 0.5
            
            # Only include PIDs with non-zero amounts
            if amount != 0.0:
                final_data.append({'pid': pid, 'amount': amount})
        
        # Create simplified dataframe
        simplified_df = pd.DataFrame(final_data)
        
        # Save the simplified CSV
        simplified_df.to_csv(output_file, index=False)
        
        print(f"Simplified CSV created!")
        print(f"Output file: {output_file}")
        print(f"Total PIDs from Prolific: {len(prolific_pids)}")
        print(f"PIDs with pilot data: {len(pilot_pids)}")
        print(f"PIDs assigned 0.5: {len(prolific_pids) - len(pilot_pids)}")
        print(f"PIDs with zero scores (excluded): {len([pid for pid in prolific_pids if pid in pilot_pids and pilot_pids[pid] == 0.0])}")
        print(f"Final PIDs included: {len(simplified_df)}")
        print("\nData preview:")
        print(simplified_df.to_string(index=False))
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_simplified_csv()