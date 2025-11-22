# save_my_ass Directory

This directory contains data processing scripts, raw data, processed outputs, and analysis results for the social learning Theory of Mind experiments.

## Directory Structure

```
save_my_ass/
├── data_raw/              # Raw JSON data files from experiments
│   ├── data_YYYY-MM-DD_*.json  # Dated raw data files
│   ├── 111925.json, 112025.json, etc.  # Large aggregated JSON files
│   ├── agent_observes_results.json  # Processed agent observation results
│   ├── steps_exp2.json
│   └── steps.dict.json
│
├── data_processed/        # Processed CSV files and derived data
│   ├── csv_reports/       # Individual CSV reports per data file
│   ├── exp1_combined/     # Combined EXP1 CSV files
│   ├── exp2_combined/     # Combined EXP2 CSV files
│   ├── exp3/             # EXP3 CSV files
│   ├── exp3_final/       # Final EXP3 CSV files
│   └── date_based/        # Date-based CSV directories
│       ├── 111925_csvs/
│       ├── 112025_csvs/
│       └── 112025_2_csv_combined/
│
├── scripts/               # Python processing scripts
│   ├── extract_json.py   # Extract data from JSON files
│   ├── extract_social_tom_exp1.py  # EXP1-specific extraction
│   ├── csv_to_dict.py    # Convert CSV to dictionary format
│   ├── csv_to_results_dict.py  # Generate results dictionaries
│   ├── json_to_statistics.py  # Direct JSON to statistics
│   ├── correlation_exp3.py  # Correlation analysis for EXP3
│   ├── observe_parser_multi_npc.py  # Multi-NPC observation parser
│   ├── convert_json_to_csv.py  # Convert JSON to CSV
│   ├── regenerate_results_dicts_v2.py  # Regenerate results from JSON
│   ├── simple_correlation.py  # Simple correlation analysis
│   ├── remove_duplicate_csvs.py  # Remove duplicate CSV files
│   ├── agent_observes_results.py  # Agent observation analysis
│   ├── observe_parser_1007.py  # Observation parser
│   ├── README.md  # Scripts documentation
│   └── PIPELINE_IMPROVEMENTS.md  # Pipeline documentation
│
├── results/               # Final results and outputs
│   ├── current/          # Current results dictionaries (Python files)
│   ├── old/              # Old/archived results dictionaries
│   ├── dictionaries/    # JSON result dictionaries
│   │   ├── steps_dict_exp3_optimized_good.json
│   │   └── steps_dict_exp3_optimized_good_old.json
│   └── presentations/    # Presentation files
│       ├── path_comparison_presentation.pptx
│       └── create_presentation.py
│
├── outputs/               # Generated visualizations
│   └── plots/            # Correlation plots and other visualizations
│
├── archive/               # Old/unused content
│   ├── old_scripts/      # Deprecated scripts
│   ├── raw_data/         # Archived raw data
│   ├── regenerated_results/  # Regenerated results (archived)
│   ├── test_output/      # Test outputs
│   └── date_combined/    # Date-based combined directories
│       └── 111925_+_112025_combined/
│
└── .gitignore            # Git ignore configuration
```

## Data Flow

1. **Raw Data**: JSON files are stored in `data_raw/`
2. **Processing**: Scripts in `scripts/` process raw data into CSV files
3. **Processed Data**: CSV files are organized in `data_processed/` by experiment or date
4. **Results**: Final results dictionaries and statistics are in `results/`
5. **Visualizations**: Generated plots are in `outputs/plots/`

## Key Scripts

- **`scripts/csv_to_results_dict.py`**: Generate results dictionaries from CSV files
- **`scripts/json_to_statistics.py`**: Direct JSON to statistics conversion
- **`scripts/correlation_exp3.py`**: Correlation analysis for EXP3
- **`results/presentations/create_presentation.py`**: Generate presentation slides

See `scripts/README.md` for detailed script documentation.

## Notes

- Large JSON files may be tracked with Git LFS
- Some scripts may need path updates if run from different directories
- Archive directory contains deprecated or test content

