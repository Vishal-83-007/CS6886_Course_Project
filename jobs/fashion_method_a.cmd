#!/bin/bash
#PBS -N fashion_method_a
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=12:00:00

# Submit from the project root; evaluates existing checkpoints without training.
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
export FASHION_DATA="${FASHION_DATA:-${PBS_O_WORKDIR:?}/data/fashion}"
export FASHION_RESULTS="${FASHION_RESULTS:-$PBS_O_WORKDIR/results/fashion_method_a}"
export FASHION_ENV_ARCHIVES="${FASHION_ENV_ARCHIVES:-$PBS_O_WORKDIR/results/fashion/environments}"
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
request="${REQUEST:-class6_9}"
args=(--request "$request" --data-dir "$FASHION_DATA"
      --eval-limit "${EVAL_LIMIT:-0}" --inference-seed "${INFERENCE_SEED:-12345}"
      --out-dir "$FASHION_RESULTS/${request}_${PBS_JOBID}")
if [ -n "${BASELINE_DIR:-}" ]; then args+=(--baseline-dir "$BASELINE_DIR"); fi
if [ -n "${GOLD_DIR:-}" ]; then args+=(--gold-dir "$GOLD_DIR"); fi
if [ "${NO_GOLD:-0}" = 1 ]; then args+=(--no-gold); fi
entry=run_method_a_fashion.py
if [ -f "$PBS_O_WORKDIR/method_a_fashion_runtime/run_method_a_fashion.py" ]; then
    entry="$PBS_O_WORKDIR/method_a_fashion_runtime/run_method_a_fashion.py"
fi
echo "Fashion Method A entry point: $entry"
"$FASHION_PYTHON" -u "$entry" "${args[@]}" 2>&1 | tee "$FASHION_RESULTS/${request}_${PBS_JOBID}.log"
