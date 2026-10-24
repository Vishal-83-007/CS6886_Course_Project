#!/bin/bash
#PBS -N fashion_cpp_train
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

# Primary production path: one independent model per allocated CPU core.
# C++ snapshots support inference recovery, NOT exact training resume.
set -euo pipefail
source "${PBS_O_WORKDIR:?}/jobs/aqua_common.sh"
exec > >(tee "$FASHION_RESULTS/train_cpp_${PBS_JOBID}.log") 2>&1
request="${REQUEST:-baseline}"
seed="${SEED:-0}"
validation="${VALIDATION_SIZE:-0}"
epochs="${EPOCHS:-3}"
echo "Production C++ job $PBS_JOBID; request=$request seed=$seed epochs=$epochs validation=$validation"
echo "Allocation: queue=${PBS_QUEUE:-unknown}; host=$(hostname); one core, one thread"
"$FASHION_PYTHON" --version
g++ --version
echo "Snapshots are inference-only. An interrupted run must restart from initialization."
"$FASHION_PYTHON" run_fashion.py --data-dir "$FASHION_DATA" \
    --out-dir "$FASHION_RESULTS/${request}_seed${seed}_val${validation}_cpp_${PBS_JOBID}" \
    --request "$request" --seed "$seed" --backend cpp --epochs "$epochs" --threads 1 \
    --validation-size "$validation"
