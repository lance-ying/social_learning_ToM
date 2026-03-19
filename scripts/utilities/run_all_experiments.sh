#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_all_experiments.sh [--no-bootstrap]

Runs the active experiment generators:
  - exp1
  - exp2
  - exp3_debug
  - exp4 wrapper
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bootstrap)
      BOOTSTRAP_ENV=0
      shift
      ;;
    --help|-h)
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

if [[ "$BOOTSTRAP_ENV" == "1" ]]; then
  echo "==> bootstrapping Julia environment"
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
fi

run_experiment() {
  local label="$1"
  local script_path="$2"

  echo "==> ${label}"
  echo "    script: ${script_path}"
  julia --project=. "$script_path"
}

run_experiment "exp1 / full experiment" "scripts/experiments/run_experiment_exp1.jl"
run_experiment "exp2 / full experiment" "scripts/experiments/run_experiment_exp2.jl"
run_experiment "exp3 / debug runner" "scripts/experiments/run_experiment_exp3_debug.jl"
run_experiment "exp4 / wrapper" "scripts/experiments/run_experiment_exp4_wrapper.jl"

echo "Done."
