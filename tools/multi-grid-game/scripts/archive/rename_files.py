#!/usr/bin/env python3
import os
import re

def rename_old_to_mod_files():
    # Directory containing the files
    mod_dir = "/Users/heyodogo/Documents/labs/lance_lab/multi-grid-game-stuff/multi_grid_game 2/src/data/levels/mod"
    
    # Get all old_s*.ts files
    files = [f for f in os.listdir(mod_dir) if f.startswith('old_s') and f.endswith('.ts')]
    
    for filename in files:
        old_path = os.path.join(mod_dir, filename)
        
        # Special handling for old_s112.ts since mod_s112.ts already exists
        if filename == 'old_s112.ts':
            new_filename = 'mod_s112_backup.ts'
            new_export_name = 'mod_s112_backup'
        else:
            # Replace old_s with mod_s
            new_filename = filename.replace('old_s', 'mod_s')
            new_export_name = new_filename[:-3]  # Remove .ts extension
        
        new_path = os.path.join(mod_dir, new_filename)
        
        # Read the file content
        with open(old_path, 'r') as f:
            content = f.read()
        
        # Update the content
        old_export_name = filename[:-3]  # Remove .ts extension
        content = content.replace(f'export const {old_export_name}', f'export const {new_export_name}')
        content = content.replace(f"id: '{old_export_name}'", f"id: '{new_export_name}'")
        content = content.replace(f"name: '{old_export_name}'", f"name: '{new_export_name}'")
        
        # Write to new file
        with open(new_path, 'w') as f:
            f.write(content)
        
        # Remove old file
        os.remove(old_path)
        
        print(f"Renamed {filename} to {new_filename}")

if __name__ == "__main__":
    rename_old_to_mod_files()