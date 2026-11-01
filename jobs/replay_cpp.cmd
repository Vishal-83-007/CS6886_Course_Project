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
#   GATE          neuron | synapse | none (no leak: repair replay)        (required)
#   START_WEIGHTS weights to start from        (default BASELINE/baseline_weights.npz;
#                 e.g. an SSD-dampened .npz for B + repair replay)
#   ALPHA         synapse gate: I(Df) > ALPHA * I(D)                     (default 2)
#                 importance: every results/<DATASET>/importance/<baseline>_seed*/importance.npz
#   LEAK          lambda_u, fraction of a gated weight removed per image  (required)
#   REPLAY        R, number of retained images replayed                   (required)
#   THETA         frozen | updated                                   (default frozen)
#   WEIGHT_NORM   as in training, never on gated synapses: 39 for the Fashion
#                 baseline, 0 for MNIST                                  (default 0)
#   NORM_TARGET   weight-norm (ungated neurons to WEIGHT_NORM, as in training) |
#                 start (every neuron keeps its sum in START_WEIGHTS; use after SSD)
#                 (default weight-norm; repair_sweep.sh uses start)
#   NORM_GATED    partial (ungated synapses of partly gated neurons keep their
#                 baseline sum) | none (first design: not normalised) (default partial)
#   EVAL_SPLIT    tune | final | all                                   (default tune)
#   EVAL_LIMIT    0 = whole split                                         (default 0)
# Output: results/<DATASET>/replay/<name>_<job>/ and results/<DATASET>/evaluate/<name>_<split>_<job>/
POISSON_SEED="${POISSON_SEED:-0}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/frozen_common.sh"
FORGET="${FORGET:-class6_9}"; ALPHA="${ALPHA:-2}"; THETA="${THETA:-frozen}"
WEIGHT_NORM="${WEIGHT_NORM:-0}"; NORM_TARGET="${NORM_TARGET:-weight-norm}"; NORM_GATED="${NORM_GATED:-partial}"; EVAL_SPLIT="${EVAL_SPLIT:-tune}"; EVAL_LIMIT="${EVAL_LIMIT:-0}"
[ -f "${BASELINE:-}/baseline_weights.npz" ] && [ -f "$BASELINE/neuron_labels.npy" ] \
    || frozen_error "BASELINE must be a run folder with baseline_weights.npz and neuron_labels.npy"
[[ "$FORGET" =~ ^class[0-9](_[0-9])*$ ]] || frozen_error "FORGET must look like class6 or class6_9"
case "${GATE:-}" in neuron|synapse|none) ;; *) frozen_error "GATE must be neuron, synapse or none" ;; esac
START_WEIGHTS="${START_WEIGHTS:-$BASELINE/baseline_weights.npz}"
[ -f "$START_WEIGHTS" ] || frozen_error "START_WEIGHTS not found: $START_WEIGHTS"
[ "$GATE" = none ] && LEAK="${LEAK:-0}"
case "$THETA" in frozen|updated) ;; *) frozen_error "THETA must be frozen or updated" ;; esac
case "$NORM_GATED" in partial|none) ;; *) frozen_error "NORM_GATED must be partial or none" ;; esac
case "$NORM_TARGET" in weight-norm|start) ;; *) frozen_error "NORM_TARGET must be weight-norm or start" ;; esac
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
if [ "$GATE" = none ]; then
    name="repair_$(basename "$START_WEIGHTS" .npz)"     # e.g. repair_a2_l0.5 (from ssd_sweep.sh)
    [ "$(basename "$START_WEIGHTS")" = baseline_weights.npz ] && name="repair_baseline"
    name+="_r${REPLAY}_theta${THETA}"; [ "$WEIGHT_NORM" != 0 ] && name+="_wn${WEIGHT_NORM}"
    [ "$WEIGHT_NORM" != 0 ] && [ "$NORM_TARGET" = start ] && name+="ns"
else
    name="D_${FORGET}_${GATE}"; [ "$GATE" = synapse ] && name+="_a${ALPHA}"
    name+="_l${LEAK}_r${REPLAY}_theta${THETA}"; [ "$WEIGHT_NORM" != 0 ] && name+="_wn${WEIGHT_NORM}"
    [ "$START_WEIGHTS" != "$BASELINE/baseline_weights.npz" ] && name+="_from_$(basename "$START_WEIGHTS" .npz)"
fi
# partial normalisation only differs from the first design for the synapse gate
[ "$WEIGHT_NORM" != 0 ] && [ "$GATE" = synapse ] && [ "$NORM_GATED" = partial ] && name+="pg"
replay_dir="$FASHION_RESULTS/replay/${name}_${job}"
eval_dir="$FASHION_RESULTS/evaluate/${name}_${EVAL_SPLIT}_${job}"
mkdir -p "$(dirname "$replay_dir")" "$(dirname "$eval_dir")"
echo "Replay $START_WEIGHTS ($GATE gate, leak $LEAK, $REPLAY images, theta $THETA) -> $replay_dir"
"$FASHION_PYTHON" -u run_replay.py --dataset "$DATASET" --data-dir "$FASHION_DATA" \
    --weights "$START_WEIGHTS" --labels "$BASELINE/neuron_labels.npy" \
    --forget "$FORGET" --gate "$GATE" --alpha "$ALPHA" --leak "$LEAK" --replay-images "$REPLAY" \
    --theta "$THETA" --weight-norm "$WEIGHT_NORM" --norm-target "$NORM_TARGET" --norm-gated "$NORM_GATED" --input-total-hz "$INPUT_TOTAL_HZ" \
    ${importance[@]+--importance "${importance[@]}"} --out-dir "$replay_dir" 2>&1 | tee "$replay_dir.log"
echo "Evaluate -> $eval_dir"
"$FASHION_PYTHON" -u run_frozen.py evaluate --dataset "$DATASET" --data-dir "$FASHION_DATA" \
    --weights "$replay_dir/replayed_weights.npz" --eval-split "$EVAL_SPLIT" --eval-limit "$EVAL_LIMIT" \
    --poisson-seed "$POISSON_SEED" --input-total-hz "$INPUT_TOTAL_HZ" --forget "$FORGET" \
    --fixed-labels "$BASELINE/neuron_labels.npy" --out-dir "$eval_dir" 2>&1 | tee "$eval_dir.log"
