#!/bin/bash
#PBS -N importance
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

# Hypothetical-STDP importance of a trained model (Method B, src/frozen.py):
# every training image is shown with learning OFF and the per-synapse update
# STDP would have made is squared and summed per class.
#   MODEL         run folder of the trained model, relative to the project folder
#                 (must contain baseline_weights.npz)                     (required)
#   POISSON_SEED  input-spike seed; use 1, 2, ... for repeated passes      (default 1)
#   TRAIN_LIMIT   0 = all 60,000 training images; a multiple of 10 for a quick pilot
# Example:
#   qsub -v DATASET=mnist,MODEL=results/mnist/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_cpp_2461985 jobs/importance_cpp.cmd
# Output: results/<DATASET>/importance/<model folder>_seed<POISSON_SEED>[_train<N>]_<job>/importance.npz
POISSON_SEED="${POISSON_SEED:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/frozen_common.sh"
[ -f "${MODEL:-}/baseline_weights.npz" ] || frozen_error "MODEL must be a run folder with baseline_weights.npz (got '${MODEL:-}')"
[[ "$TRAIN_LIMIT" =~ ^[0-9]+$ ]] && [ $((TRAIN_LIMIT % 10)) -eq 0 ] || frozen_error "TRAIN_LIMIT must be a multiple of 10"
suffix="_seed${POISSON_SEED}"; [ "$TRAIN_LIMIT" != 0 ] && suffix+="_train${TRAIN_LIMIT}"
out="$FASHION_RESULTS/importance/$(basename "$MODEL")${suffix}_${job}"
mkdir -p "$(dirname "$out")"
echo "Importance of $MODEL -> $out"
"$FASHION_PYTHON" -u run_frozen.py importance --dataset "$DATASET" --data-dir "$FASHION_DATA" \
    --weights "$MODEL/baseline_weights.npz" --poisson-seed "$POISSON_SEED" \
    --train-limit "$TRAIN_LIMIT" --input-total-hz "$INPUT_TOTAL_HZ" --out-dir "$out" 2>&1 | tee "$out.log"
