#!/usr/bin/env python3
import os
import glob

def fix_import_paths():
    # Directory containing the files
    mod_dir = "/Users/heyodogo/Documents/labs/lance_lab/multi-grid-game-stuff/multi_grid_game 2/src/data/levels/mod"
    
    # Get all .ts files in the mod directory
    pattern = os.path.join(mod_dir, "*.ts")
    files = glob.glob(pattern)
    
    for file_path in files:
        # Skip .DS_Store and other non-TypeScript files
        if not file_path.endswith('.ts'):
            continue
            
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix the import path
        updated_content = content.replace("import { LevelConfig } from './types';", "import { LevelConfig } from '../types';")
        
        # Only write if there was a change
        if updated_content != content:
            with open(file_path, 'w') as f:
                f.write(updated_content)
            print(f"Fixed import in {os.path.basename(file_path)}")

if __name__ == "__main__":
    fix_import_paths()