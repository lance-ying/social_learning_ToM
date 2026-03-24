#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/run_regenerate_fixed_baselines.sh [--no-bootstrap]

Runs the baseline generators whose step-dict semantics changed after the recent fixes:
  - exp1 rational_non_mentalizing
  - exp1 social_mentalizing
  - exp2 social_mentalizing
  - exp3 naive_observer
  - exp4 naive_observer
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

run_baseline() {
  local label="$1"
  local script_path="$2"

  echo "==> ${label}"
  echo "    script: ${script_path}"
  julia --project=. "$script_path"
}

run_baseline "exp1 / rational_non_mentalizing" "scripts/baselines/exp1/baseline_non_mentalizing_exp1.jl"
run_baseline "exp1 / social_mentalizing" "scripts/baselines/exp1/baseline_mentalize_exp1.jl"
run_baseline "exp2 / social_mentalizing" "scripts/baselines/exp2/baseline_mentalize_exp2.jl"
run_baseline "exp3 / naive_observer" "scripts/baselines/exp3/baseline_naive_exp3.jl"
run_baseline "exp4 / naive_observer" "scripts/baselines/exp4/baseline_naive_exp4.jl"

echo "Done."
