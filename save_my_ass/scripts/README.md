# Scripts Documentation

## Simple CSV to Results Dictionary

### `csv_to_results_dict.py`

**Purpose**: Simple script that takes a directory of CSV files and generates a `results_dict`.

**Usage**:
```bash
uv run scripts/csv_to_results_dict.py <csv_directory> [output_file] [description]
```

**Examples**:
```bash
# Generate EXP2 results from combined CSV directory
uv run scripts/csv_to_results_dict.py data_processed/exp2_combined results/current/results_dict_merged_exp2.py "EXP2"

# Generate EXP1 results
uv run scripts/csv_to_results_dict.py data_processed/exp1_combined results/current/results_dict_exp1_50.py "EXP1"
```

**Features**:
- Automatically deduplicates CSV files by user ID (keeps first occurrence)
- Uses the same statistics calculation as the original pipeline
- Only processes CSV files (assumes they already have prolificId filtering)

**Output**:
- Generates a Python file with `results_dict` containing statistics for each level
- Includes metadata about the source directory and file counts

## Other Scripts

### `json_to_statistics.py`
Direct JSON → Statistics parser (for future use if needed)

### `regenerate_results_dicts_v2.py`
Complex pipeline for processing JSON files directly (archived approach)

### `simple_correlation.py`
Correlation analysis between model predictions and human data

