#!/bin/bash
#PBS -N replay
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

# Method D (src/replay.py): replay R retained training images with STDP on
# while the gated weights leak, then evaluate the result (src/frozen.py) with
# the baseline's fixed labels and the own-label read-outs.
#   BASELINE      run folder of the trained baseline (baseline_weights.npz,
#                 neuron_labels.npy)                                      (required)
#   FORGET        removed classes                                    (default class6_9)
#   GATE          neuron | synapse                                         (required)
#   ALPHA         synapse gate: I(Df) > ALPHA * I(D)                     (default 2)
#                 importance: every results/<DATASET>/importance/<baseline>_seed*/importance.npz
#   LEAK          lambda_u, fraction of a gated weight removed per image  (required)
#   REPLAY        R, number of retained images replayed                   (required)
#   THETA         frozen | updated                                   (default frozen)
#   WEIGHT_NORM   as in training, ungated neurons only: 39 for the Fashion
#                 baseline, 0 for MNIST                                  (default 0)
#   EVAL_SPLIT    tune | final | all                                   (default tune)
#   EVAL_LIMIT    0 = whole split                                         (default 0)
# Output: results/<DATASET>/replay/<name>_<job>/ and results/<DATASET>/evaluate/<name>_<split>_<job>/
POISSON_SEED="${POISSON_SEED:-0}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/frozen_common.sh"
FORGET="${FORGET:-class6_9}"; ALPHA="${ALPHA:-2}"; THETA="${THETA:-frozen}"
WEIGHT_NORM="${WEIGHT_NORM:-0}"; EVAL_SPLIT="${EVAL_SPLIT:-tune}"; EVAL_LIMIT="${EVAL_LIMIT:-0}"
[ -f "${BASELINE:-}/baseline_weights.npz" ] && [ -f "$BASELINE/neuron_labels.npy" ] \
    || frozen_error "BASELINE must be a run folder with baseline_weights.npz and neuron_labels.npy"
[[ "$FORGET" =~ ^class[0-9](_[0-9])*$ ]] || frozen_error "FORGET must look like class6 or class6_9"
case "${GATE:-}" in neuron|synapse) ;; *) frozen_error "GATE must be neuron or synapse" ;; esac
case "$THETA" in frozen|updated) ;; *) frozen_error "THETA must be frozen or updated" ;; esac
case "$EVAL_SPLIT" in tune|final|all) ;; *) frozen_error "EVAL_SPLIT must be tune, final or all" ;; esac
[[ "${LEAK:-}" =~ ^0?\.[0-9]+$|^0$ ]] || frozen_error "LEAK must be a number in [0, 1), e.g. 0.001"
[[ "${REPLAY:-}" =~ ^[1-9][0-9]*$ ]] || frozen_error "REPLAY must be a positive whole number"
[[ "$WEIGHT_NORM" =~ ^[0-9]+$ ]] || frozen_error "WEIGHT_NORM must be a whole number"
[[ "$ALPHA" =~ ^[0-9]+(\.[0-9]+)?$ ]] || frozen_error "ALPHA must be a number"
importance=()
if [ "$GATE" = synapse ]; then
    importance=( "$FASHION_RESULTS"/importance/"$(basename "$BASELINE")"_seed*/importance.npz )
    [ -f "${importance[0]}" ] || frozen_error "no importance.npz for $(basename "$BASELINE"); run importance_cpp.cmd first"
fi
name="D_${FORGET}_${GATE}"; [ "$GATE" = synapse ] && name+="_a${ALPHA}"
name+="_l${LEAK}_r${REPLAY}_theta${THETA}"; [ "$WEIGHT_NORM" != 0 ] && name+="_wn${WEIGHT_NORM}"
replay_dir="$FASHION_RESULTS/replay/${name}_${job}"
eval_dir="$FASHION_RESULTS/evaluate/${name}_${EVAL_SPLIT}_${job}"
mkdir -p "$(dirname "$replay_dir")" "$(dirname "$eval_dir")"
echo "Replay $BASELINE ($GATE gate, leak $LEAK, $REPLAY images, theta $THETA) -> $replay_dir"
"$FASHION_PYTHON" -u run_replay.py --dataset "$DATASET" --data-dir "$FASHION_DATA" \
    --weights "$BASELINE/baseline_weights.npz" --labels "$BASELINE/neuron_labels.npy" \
    --forget "$FORGET" --gate "$GATE" --alpha "$ALPHA" --leak "$LEAK" --replay-images "$REPLAY" \
    --theta "$THETA" --weight-norm "$WEIGHT_NORM" --input-total-hz "$INPUT_TOTAL_HZ" \
    ${importance[@]+--importance "${importance[@]}"} --out-dir "$replay_dir" 2>&1 | tee "$replay_dir.log"
echo "Evaluate -> $eval_dir"
"$FASHION_PYTHON" -u run_frozen.py evaluate --dataset "$DATASET" --data-dir "$FASHION_DATA" \
    --weights "$replay_dir/replayed_weights.npz" --eval-split "$EVAL_SPLIT" --eval-limit "$EVAL_LIMIT" \
    --poisson-seed "$POISSON_SEED" --input-total-hz "$INPUT_TOTAL_HZ" --forget "$FORGET" \
    --fixed-labels "$BASELINE/neuron_labels.npy" --out-dir "$eval_dir" 2>&1 | tee "$eval_dir.log"
