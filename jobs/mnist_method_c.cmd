#!/bin/bash
#PBS -N mnist_method_c
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=24:00:00

set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
export FASHION_DATA="${FASHION_DATA:-${PBS_O_WORKDIR:?}/data/mnist}"
export FASHION_RESULTS="${FASHION_RESULTS:-$PBS_O_WORKDIR/results/mnist_method_c}"
export FASHION_ENV_ARCHIVES="${FASHION_ENV_ARCHIVES:-$PBS_O_WORKDIR/results/fashion/environments}"
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
variant="${VARIANT:-C1}"
scale="${SCALE:-0.1}"
args=(--variants "$variant" --scales "$scale" --passes 1 2 5
      --data-dir "$FASHION_DATA" --eval-limit "${EVAL_LIMIT:-0}"
      --replay-per-class "${REPLAY_PER_CLASS:-0}"
      --inference-seed "${INFERENCE_SEED:-12345}" --replay-seed "${REPLAY_SEED:-2026}"
      --out-dir "$FASHION_RESULTS/${variant}_s${scale}_${PBS_JOBID}")
if [ -n "${BASELINE_DIR:-}" ]; then args+=(--baseline-dir "$BASELINE_DIR"); fi
if [ -n "${GOLD_DIR:-}" ]; then args+=(--gold-dir "$GOLD_DIR"); fi
if [ "${NO_GOLD:-0}" = 1 ]; then args+=(--no-gold); fi
if [ "${ADAPT_THRESHOLDS:-0}" = 1 ]; then args+=(--adapt-thresholds); fi
if [ "${SKIP_EVALUATION:-0}" = 1 ]; then args+=(--skip-evaluation); fi
entry="$PBS_O_WORKDIR/run_method_c.py"
if [ -f "$PBS_O_WORKDIR/method_c_runtime/run_method_c.py" ]; then
    entry="$PBS_O_WORKDIR/method_c_runtime/run_method_c.py"
fi
"$FASHION_PYTHON" -u "$entry" "${args[@]}" 2>&1 | tee "$FASHION_RESULTS/${variant}_s${scale}_${PBS_JOBID}.log"
