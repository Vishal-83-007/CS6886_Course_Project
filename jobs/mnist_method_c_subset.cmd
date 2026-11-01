#!/bin/bash
#PBS -N mnist_c_subset
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=24:00:00

set -euo pipefail
export USER="${USER:-$(id -un)}"
project="${PBS_O_WORKDIR:?Submit from the project root}"
runtime="$project/method_c_subset_runtime"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
export FASHION_DATA="${FASHION_DATA:-$project/data/mnist}"
export FASHION_RESULTS="${FASHION_RESULTS:-$project/results/mnist_method_c_subset}"
export FASHION_ENV_ARCHIVES="${FASHION_ENV_ARCHIVES:-$project/results/fashion/environments}"
# Reuse the installed environment helper, including archive staging on older AQUA setups.
if [ -f "$project/jobs/aqua_common.sh" ]; then
    source "$project/jobs/aqua_common.sh"
else
    source "$runtime/jobs/aqua_common.sh"
fi
read -r -a passes <<< "${PASSES:-1 2 5}"
request="${REQUEST:-sub6_9_p10}"
variant="${VARIANT:-C1}"
scale="${SCALE:-0.1}"
args=(--dataset mnist --request "$request" --variants "$variant" --scales "$scale"
      --passes "${passes[@]}" --data-dir "$FASHION_DATA"
      --forget-csv "${FORGET_CSV:-$runtime/forget_set_mnist_sub6_9_p10.csv}"
      --eval-limit "${EVAL_LIMIT:-0}" --retain-per-class "${RETAIN_PER_CLASS:-100}"
      --assign-per-class "${ASSIGN_PER_CLASS:-500}" --assignment-seed "${ASSIGNMENT_SEED:-45}"
      --inference-seed "${INFERENCE_SEED:-12345}" --replay-seed "${REPLAY_SEED:-2026}"
      --out-dir "$FASHION_RESULTS/${request}_${variant}_s${scale}_${PBS_JOBID}")
if [ -n "${BASELINE_DIR:-}" ]; then args+=(--baseline-dir "$BASELINE_DIR"); fi
if [ -n "${GOLD_DIR:-}" ]; then args+=(--gold-dir "$GOLD_DIR"); fi
if [ "${SKIP_EVALUATION:-0}" = 1 ]; then args+=(--skip-evaluation); fi
if [ "${PREPARE_ONLY:-0}" = 1 ]; then args+=(--prepare-only); fi
"$FASHION_PYTHON" -u "$runtime/run_method_c_subset.py" "${args[@]}" 2>&1 |
    tee "$FASHION_RESULTS/${request}_${variant}_s${scale}_${PBS_JOBID}.log"
