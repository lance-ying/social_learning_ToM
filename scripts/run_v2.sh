#!/usr/bin/env bash
# Run the v2 experiments and/or baselines.
#
# Usage:
#   ./run_v2.sh              # run everything (experiments then baselines)
#   ./run_v2.sh experiments  # run only experiments
#   ./run_v2.sh baselines    # run only baselines
#   ./run_v2.sh exp1         # run experiment + baselines for one family
#   ./run_v2.sh exp2 exp3    # multiple families
#
# Each Julia script is invoked from its own directory so relative paths work.
# Failures in one script do not abort the rest; a summary is printed at the end.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$SCRIPT_DIR/experiments_v2"
BASE_DIR="$SCRIPT_DIR/baselines_v2"
LOG_DIR="$SCRIPT_DIR/_v2_logs"
mkdir -p "$LOG_DIR"

JULIA="${JULIA:-julia}"

declare -a RAN
declare -a FAILED

run_julia() {
    local label="$1"
    local script="$2"
    local cwd="$(dirname "$script")"
    local log="$LOG_DIR/${label}.log"

    echo ""
    echo "=========================================="
    echo "Running: $label"
    echo "Script:  $script"
    echo "Log:     $log"
    echo "=========================================="

    if (cd "$cwd" && "$JULIA" "$script") 2>&1 | tee "$log"; then
        # exit status of the pipe is the tee's; capture julia's via PIPESTATUS
        if [[ "${PIPESTATUS[0]}" -eq 0 ]]; then
            RAN+=("$label")
            return 0
        fi
    fi
    FAILED+=("$label")
    return 1
}

run_experiment() {
    local fam="$1"
    case "$fam" in
        exp1) run_julia "exp_exp1" "$EXP_DIR/run_experiment_exp1.jl" ;;
        exp2) run_julia "exp_exp2" "$EXP_DIR/run_experiment_exp2.jl" ;;
        exp3) run_julia "exp_exp3" "$EXP_DIR/run_experiment_exp3.jl" ;;
        exp4) run_julia "exp_exp4" "$EXP_DIR/run_experiment_exp4_wrapper.jl" ;;
        *) echo "unknown experiment family: $fam" >&2; return 1 ;;
    esac
}

run_baselines_for() {
    local fam="$1"
    local dir="$BASE_DIR/$fam"
    if [[ ! -d "$dir" ]]; then
        echo "no baseline dir for $fam at $dir" >&2
        return 1
    fi
    local f
    for f in "$dir"/*.jl; do
        local label="base_$(basename "$f" .jl)"
        run_julia "$label" "$f"
    done
}

run_family() {
    local fam="$1"
    run_experiment "$fam"
    run_baselines_for "$fam"
}

run_all_experiments() {
    local fam
    for fam in exp1 exp2 exp3 exp4; do
        run_experiment "$fam"
    done
}

run_all_baselines() {
    local fam
    for fam in exp1 exp2 exp3 exp4; do
        run_baselines_for "$fam"
    done
}

main() {
    if [[ $# -eq 0 ]]; then
        run_all_experiments
        run_all_baselines
    else
        local arg
        for arg in "$@"; do
            case "$arg" in
                experiments) run_all_experiments ;;
                baselines)   run_all_baselines ;;
                exp1|exp2|exp3|exp4) run_family "$arg" ;;
                *) echo "unknown arg: $arg" >&2; exit 2 ;;
            esac
        done
    fi

    echo ""
    echo "=========================================="
    echo "Summary"
    echo "=========================================="
    echo "Succeeded (${#RAN[@]}):"
    printf '  %s\n' "${RAN[@]:-<none>}"
    echo "Failed (${#FAILED[@]}):"
    printf '  %s\n' "${FAILED[@]:-<none>}"
    echo "Logs: $LOG_DIR"

    [[ "${#FAILED[@]}" -eq 0 ]]
}

main "$@"
