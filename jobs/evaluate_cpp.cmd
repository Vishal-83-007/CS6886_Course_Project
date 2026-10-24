#!/bin/bash
#PBS -N evaluate
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

# Frozen evaluation of any weights (baseline, Gold, unlearned) with several
# read-outs (src/frozen.py): own labels from all classes, own labels from the
# retained classes only, and optionally fixed labels from a file.
#   WEIGHTS       npz in baseline_weights.npz format                      (required)
#   FORGET        removed classes, e.g. class6_9                          (default none)
#   FIXED_LABELS  neuron_labels.npy to score as a fixed read-out          (default none)
#   EVAL_SPLIT    tune (500/class, for choosing settings) | final | all   (default tune)
#   EVAL_LIMIT    balanced subset of the split for quick checks; 0 = all  (default 0)
#   NAME          output folder name                                      (default from WEIGHTS)
#   POISSON_SEED  input-spike seed; keep it equal for models you compare  (default 0)
# Output: results/<DATASET>/evaluate/<NAME>_<EVAL_SPLIT>_<job>/metrics.json
POISSON_SEED="${POISSON_SEED:-0}"
EVAL_SPLIT="${EVAL_SPLIT:-tune}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/frozen_common.sh"
[ -f "${WEIGHTS:-}" ] || frozen_error "WEIGHTS must be an existing .npz file (got '${WEIGHTS:-}')"
case "$EVAL_SPLIT" in tune|final|all) ;; *) frozen_error "EVAL_SPLIT must be tune, final or all" ;; esac
[[ "$EVAL_LIMIT" =~ ^[0-9]+$ ]] && [ $((EVAL_LIMIT % 10)) -eq 0 ] || frozen_error "EVAL_LIMIT must be a multiple of 10"
if [ -n "${FORGET:-}" ]; then
    [[ "$FORGET" =~ ^class[0-9](_[0-9])*$ ]] || frozen_error "FORGET must look like class6 or class6_9"
fi
[ -z "${FIXED_LABELS:-}" ] || [ -f "$FIXED_LABELS" ] || frozen_error "FIXED_LABELS file not found: $FIXED_LABELS"
if [ -z "${NAME:-}" ]; then
    NAME="$(basename "$(dirname "$WEIGHTS")")"
    [ "$(basename "$WEIGHTS")" = baseline_weights.npz ] || NAME+="_$(basename "$WEIGHTS" .npz)"
fi
out="$FASHION_RESULTS/evaluate/${NAME}_${EVAL_SPLIT}_${job}"
mkdir -p "$(dirname "$out")"
echo "Evaluate $WEIGHTS -> $out"
"$FASHION_PYTHON" -u run_frozen.py evaluate --dataset "$DATASET" --data-dir "$FASHION_DATA" \
    --weights "$WEIGHTS" --eval-split "$EVAL_SPLIT" --eval-limit "$EVAL_LIMIT" \
    --poisson-seed "$POISSON_SEED" --input-total-hz "$INPUT_TOTAL_HZ" \
    ${FORGET:+--forget "$FORGET"} ${FIXED_LABELS:+--fixed-labels "$FIXED_LABELS"} \
    --out-dir "$out" 2>&1 | tee "$out.log"
