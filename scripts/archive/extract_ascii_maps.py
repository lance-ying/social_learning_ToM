#!/usr/bin/env python3
"""
Extract ASCII maps from TypeScript level files and save them as individual text files.

This script reads all .ts files in the src/data/levels/backup/ directory,
extracts the asciiMap content from each file, and saves them as .txt files.
"""

import os
import re
import glob
from pathlib import Path

def extract_ascii_map(file_path):
    """Extract ASCII map from a TypeScript file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match asciiMap content between backticks
        # This looks for asciiMap: ` ... `.trim(),
        pattern = r'asciiMap:\s*`\s*\n(.*?)\n\s*`\.trim\(\)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            ascii_map = match.group(1)
            # Clean up any extra whitespace but preserve the map structure
            return ascii_map
        else:
            print(f"Warning: No ASCII map found in {file_path}")
            return None
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    """Main function to process all backup files and extract ASCII maps."""

    # Define paths
    backup_dir = "src/data/levels/backup"
    output_dir = "extracted_ascii_maps/backup"

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all .ts files in the backup directory
    pattern = os.path.join(backup_dir, "*.ts")
    ts_files = glob.glob(pattern)

    if not ts_files:
        print(f"No .ts files found in {backup_dir}")
        return
    
    print(f"Found {len(ts_files)} TypeScript files to process...")
    
    processed_count = 0
    failed_count = 0
    
    for ts_file in sorted(ts_files):
        # Extract filename without extension
        filename = Path(ts_file).stem
        
        # Extract ASCII map
        ascii_map = extract_ascii_map(ts_file)
        
        if ascii_map:
            # Create output filename
            output_filename = f"{filename}_ascii.txt"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save ASCII map to text file
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(ascii_map)
                
                print(f"✓ Extracted: {filename} -> {output_filename}")
                processed_count += 1
                
            except Exception as e:
                print(f"✗ Failed to write {output_filename}: {e}")
                failed_count += 1
        else:
            print(f"✗ Failed to extract ASCII map from: {filename}")
            failed_count += 1
    
    print(f"\n--- Summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {output_dir}")
    
    if processed_count > 0:
        print(f"\nASCII maps have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()