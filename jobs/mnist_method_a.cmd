#!/bin/bash
#PBS -N mnist_method_a
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=12:00:00

# Submit from repository root. This evaluates existing checkpoints; no training.
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
export FASHION_DATA="${FASHION_DATA:-${PBS_O_WORKDIR:?}/data/mnist}"
export FASHION_RESULTS="${FASHION_RESULTS:-$PBS_O_WORKDIR/results/mnist_method_a}"
# The archive-staging version of aqua_common.sh defaults to
# $FASHION_RESULTS/environments. Method A has its own result folder, but must
# reuse the existing training environment instead of looking in that new folder.
export FASHION_ENV_ARCHIVES="${FASHION_ENV_ARCHIVES:-$PBS_O_WORKDIR/results/fashion/environments}"
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
request="${REQUEST:-class6_9}"
args=(--request "$request" --data-dir "$FASHION_DATA"
      --eval-limit "${EVAL_LIMIT:-0}" --inference-seed "${INFERENCE_SEED:-12345}"
      --out-dir "$FASHION_RESULTS/${request}_${PBS_JOBID}")
if [ -n "${BASELINE_DIR:-}" ]; then args+=(--baseline-dir "$BASELINE_DIR"); fi
if [ -n "${GOLD_DIR:-}" ]; then args+=(--gold-dir "$GOLD_DIR"); fi
if [ "${NO_GOLD:-0}" = 1 ]; then args+=(--no-gold); fi
# The transfer bundle isolates matching dependencies from older Aqua src/ files.
entry=run_method_a.py
if [ -f "$PBS_O_WORKDIR/method_a_runtime/run_method_a.py" ]; then
    entry="$PBS_O_WORKDIR/method_a_runtime/run_method_a.py"
fi
echo "Method A entry point: $entry"
"$FASHION_PYTHON" -u "$entry" "${args[@]}" 2>&1 | tee "$FASHION_RESULTS/${request}_${PBS_JOBID}.log"
