#!/bin/bash
# Shared by importance_cpp.cmd and evaluate_cpp.cmd (frozen network, compiled CPU).
#   DATASET         mnist | fashion                                    (required)
#   POISSON_SEED    seed of the input spikes                           (importance 1, evaluate 0)
#   INPUT_TOTAL_HZ  must match the trained model (0 for all our runs)  (default 0)
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
cd "${PBS_O_WORKDIR:?Submit with qsub from the project folder}"
frozen_error() { echo "SETTINGS ERROR: $*" >&2; exit 2; }
case "${DATASET:-}" in mnist|fashion) ;; *) frozen_error "DATASET must be mnist or fashion (got '${DATASET:-}')" ;; esac
INPUT_TOTAL_HZ="${INPUT_TOTAL_HZ:-0}"
for var in POISSON_SEED INPUT_TOTAL_HZ; do
    [[ "${!var}" =~ ^[0-9]+$ ]] || frozen_error "$var must be a whole number (got '${!var}')"
done
export FASHION_DATA="$PBS_O_WORKDIR/data/$DATASET"
export FASHION_RESULTS="$PBS_O_WORKDIR/results/$DATASET"
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
job="${PBS_JOBID%%.*}"
