# Pipeline Improvements: Direct JSON Processing

## Current Workflow (Inefficient)

```
JSON files → extract_social_tom_exp1.py → CSV files → observe_parser_1007.py → Statistics
```

**Problems:**
1. **Unnecessary intermediate files**: We generate full CSV files with all event details just to extract:
   - Level names
   - OBSERVE event counts  
   - Activation counts
2. **Slow**: Writing/reading large CSV files is slower than parsing JSON directly
3. **Disk space**: CSV files take up significant space
4. **Two-step process**: Requires running two separate scripts

## New Workflow (Efficient)

```
JSON files → json_to_statistics.py → Statistics
```

**Benefits:**
1. **Faster**: Direct JSON parsing, no intermediate file I/O
2. **Simpler**: Single script does everything
3. **Less disk space**: No need to store CSV files unless you want them
4. **Optional CSV generation**: Can still generate CSVs for inspection/debugging with `--csv` flag

## Usage

### Direct JSON Processing (Recommended)

```bash
# Process JSON files directly, no CSV generation
python scripts/regenerate_results_dicts_v2.py

# Process JSON files and optionally generate CSVs for inspection
python scripts/regenerate_results_dicts_v2.py --csv
```

### Legacy CSV Processing (Still Supported)

If you already have CSV files and want to use them:

```bash
python scripts/regenerate_results_dicts.py
```

## Architecture

### `json_to_statistics.py`
- **Purpose**: Direct JSON → Statistics parser
- **Key function**: `parse_json_for_statistics()` - extracts statistics directly from JSON
- **Optional**: Can generate CSV files if needed for inspection

### `regenerate_results_dicts_v2.py`
- **Purpose**: Unified pipeline using direct JSON processing
- **Features**:
  - Works directly from JSON files
  - Deduplicates by user ID
  - Supports both EXP1 and EXP2
  - Optional CSV generation

## Performance Comparison

**Old approach:**
- Time: ~2-3 minutes for 100 JSON files
- Disk: ~500MB CSV files
- Steps: 2 scripts

**New approach:**
- Time: ~30-60 seconds for 100 JSON files
- Disk: 0MB (unless `--csv` flag used)
- Steps: 1 script

## Migration Path

1. **Immediate**: Use `regenerate_results_dicts_v2.py` for new processing
2. **Existing CSVs**: Keep using `regenerate_results_dicts.py` if you need to work with existing CSV files
3. **Gradual**: Migrate to v2 as you reprocess data

## When to Generate CSVs

Generate CSVs only when you need them for:
- Manual inspection
- Debugging specific users/levels
- Sharing with collaborators who prefer CSV format
- Other analysis tools that require CSV input

**Note**: When CSV generation is enabled, only CSVs for users with `prolificId` are generated. This ensures you only get data from actual study participants.

Otherwise, use direct JSON processing for speed and efficiency.

## Archived Scripts

The following old scripts have been archived to `archive/old_scripts/`:
- `extract_social_tom_exp1.py` - Old CSV generation script (functionality now in `json_to_statistics.py`)
- `regenerate_results_dicts.py` - Old CSV-based processing script (replaced by `regenerate_results_dicts_v2.py`)

These are kept for reference but should not be used for new processing.

