#!/bin/bash
# Quick wrapper script to run the analysis pipeline

# Default paths
JSON_FILE="pilot_exp4.json"
REFERENCE_FILE="results/dictionaries/steps_dict_exp4_point5_updated.json"
OUTPUT_DIR="analysis_results"

# Run the analysis
echo "Running observation analysis pipeline..."
echo "JSON: $JSON_FILE"
echo "Reference: $REFERENCE_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$(dirname "$0")"
python3 scripts/process_and_compare.py "$JSON_FILE" "$REFERENCE_FILE" "$OUTPUT_DIR"
