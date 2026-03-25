#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

JULIA_BIN="${JULIA_BIN:-julia +1.11.9}"
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
RUN_EXPERIMENTS=1
RUN_BASELINES=1
RUN_RECONSTRUCT=1
RECONSTRUCT_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/utilities/bash_scripts/run_all_model_outputs.sh [options] [-- <reconstruct args>]

Runs the active end-to-end model pipeline:
  1. experiments
  2. baselines
  3. reconstructed costs

Options:
  --no-bootstrap         Skip the initial Julia Pkg.instantiate()
  --skip-experiments     Do not run experiment generation
  --skip-baselines       Do not run baseline generation
  --skip-reconstruct     Do not run reconstruction
  --help, -h             Show this help

Anything after `--` is forwarded to:
  bash scripts/utilities/bash_scripts/run_reconstruct_all.sh

Examples:
  bash scripts/utilities/bash_scripts/run_all_model_outputs.sh
  bash scripts/utilities/bash_scripts/run_all_model_outputs.sh -- --exp3-4
  bash scripts/utilities/bash_scripts/run_all_model_outputs.sh --skip-experiments -- --exp2
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-bootstrap)
      BOOTSTRAP_ENV=0
      shift
      ;;
    --skip-experiments)
      RUN_EXPERIMENTS=0
      shift
      ;;
    --skip-baselines)
      RUN_BASELINES=0
      shift
      ;;
    --skip-reconstruct)
      RUN_RECONSTRUCT=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      RECONSTRUCT_ARGS=("$@")
      break
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
  bash -lc "$JULIA_BIN --project=. -e 'using Pkg; Pkg.instantiate()'"
fi

if [[ "$RUN_EXPERIMENTS" == "1" ]]; then
  echo "==> running experiments"
  JULIA_BIN="$JULIA_BIN" bash scripts/utilities/bash_scripts/run_all_experiments.sh --no-bootstrap
fi

if [[ "$RUN_BASELINES" == "1" ]]; then
  echo "==> running baselines"
  JULIA_BIN="$JULIA_BIN" bash scripts/utilities/bash_scripts/run_all_baselines.sh --no-bootstrap
fi

if [[ "$RUN_RECONSTRUCT" == "1" ]]; then
  echo "==> running reconstruction"
  if [[ ${#RECONSTRUCT_ARGS[@]} -gt 0 ]]; then
    JULIA_BIN="$JULIA_BIN" bash scripts/utilities/bash_scripts/run_reconstruct_all.sh --no-bootstrap "${RECONSTRUCT_ARGS[@]}"
  else
    JULIA_BIN="$JULIA_BIN" bash scripts/utilities/bash_scripts/run_reconstruct_all.sh --no-bootstrap
  fi
fi

echo "Pipeline complete."
