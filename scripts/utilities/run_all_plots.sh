#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
INTERACTIVE=0
EXP4_FILTERED=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/utilities/run_all_plots.sh [--python python3] [--interactive] [--exp4-filtered]

Options:
  --python <bin>   Python executable to use. Default: python3
  --interactive    Use the interactive matplotlib backend instead of Agg.
  --exp4-filtered  Use data_processing/outputs/human_costs/exp4_human_costs_filtered.json for exp4 cost plots.
  -h, --help       Show this help message.

Environment:
  PYTHON_BIN       Alternative way to set the Python executable.
  MPLCONFIGDIR     Matplotlib cache dir. Default: /tmp/mplcache
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=1
            shift
            ;;
        --exp4-filtered)
            EXP4_FILTERED=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

mkdir -p "$MPLCONFIGDIR"

if [[ "$INTERACTIVE" -eq 1 ]]; then
    export MPLCONFIGDIR
else
    export MPLCONFIGDIR
    export MPLBACKEND=Agg
fi

if [[ "$EXP4_FILTERED" -eq 1 ]]; then
    export EXP4_HUMAN_COSTS_FILE="$ROOT_DIR/data_processing/outputs/human_costs/exp4_human_costs_filtered.json"
    echo "Using filtered exp4 human costs: $EXP4_HUMAN_COSTS_FILE"
fi

run_plot() {
    echo "==> $*"
    "$@"
}

run_plot "$PYTHON_BIN" data_processing/scripts/cost_scatter_exp1234_mega.py --metric total_cost
run_plot "$PYTHON_BIN" data_processing/scripts/cost_scatter_exp1234_mega.py --metric planning_cost
run_plot "$PYTHON_BIN" data_processing/scripts/cost_scatter_exp1234_mega.py --metric observe_cost
run_plot "$PYTHON_BIN" data_processing/scripts/cost_scatter_exp1234_mega.py --metric move_cost
run_plot "$PYTHON_BIN" data_processing/scripts/cost_scatter_exp1234_mega.py --metric interaction_cost

run_plot "$PYTHON_BIN" data_processing/scripts/cost_barplot_exp1234_multi.py --metrics total_cost planning_cost observe_cost
run_plot "$PYTHON_BIN" data_processing/scripts/steps_barplot_exp1234_multi.py
run_plot "$PYTHON_BIN" data_processing/scripts/run_all_correlation_4panel.py --python "$PYTHON_BIN"

echo
echo "Plots written under data_processing/outputs/plots"
