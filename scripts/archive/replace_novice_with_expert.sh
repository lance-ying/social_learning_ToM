#!/bin/bash

# Replace all instances of "Novice" with "Expert" in exp2 TypeScript files

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP2_DIR="$SCRIPT_DIR/../src/data/levels/exp2"

# Check if the directory exists
if [ ! -d "$EXP2_DIR" ]; then
    echo "Error: Directory $EXP2_DIR not found"
    exit 1
fi

echo "Replacing 'Novice' with 'Expert' in exp2 levels..."
echo "Directory: $EXP2_DIR"
echo ""

# Find all .ts files (excluding types.ts and other non-level files)
files=$(find "$EXP2_DIR" -name "s*.ts" -type f)

if [ -z "$files" ]; then
    echo "No s*.ts files found in $EXP2_DIR"
    exit 1
fi

count=0

for file in $files; do
    # Check if file contains "Novice"
    if grep -q "Novice" "$file"; then
        # Replace Novice with Expert using sed
        # On macOS, sed requires an empty string for in-place editing
        sed -i '' 's/Novice/Expert/g' "$file"
        echo "✓ Updated: $(basename "$file")"
        ((count++))
    fi
done

echo ""
echo "Successfully updated $count files"

