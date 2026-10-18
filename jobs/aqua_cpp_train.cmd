#!/bin/bash
#PBS -N fashion_cpp_train
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

# Use only after the CPU pilot establishes that the full run fits the walltime.
# C++ snapshots support inference recovery, NOT exact training resume.
set -euo pipefail
source "${PBS_O_WORKDIR:?}/jobs/aqua_common.sh"
request="${REQUEST:-baseline}"
seed="${SEED:-0}"
validation="${VALIDATION_SIZE:-5000}"
"$FASHION_PYTHON" run_fashion.py --data-dir "$FASHION_DATA" \
    --out-dir "$FASHION_RESULTS/${request}_seed${seed}_val${validation}_cpp_${PBS_JOBID}" \
    --request "$request" --seed "$seed" --backend cpp --epochs 3 --threads 1 \
    --validation-size "$validation"
