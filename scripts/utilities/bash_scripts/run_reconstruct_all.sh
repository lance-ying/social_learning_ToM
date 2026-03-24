#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Backward-compatible wrapper.
# The maintained reconstruction driver is run_reconstruct_replay_all.sh; this
# script keeps the historical name but defaults to the non-replay output folder.

: "${OUTPUT_DIR:=scripts/experiments/experiment_outputs/reconstructed_costs}"
export OUTPUT_DIR

exec bash scripts/utilities/run_reconstruct_replay_all.sh "$@"
