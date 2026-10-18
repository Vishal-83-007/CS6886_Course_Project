#!/bin/bash
#PBS -N fashion_cpu_pilot
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=02:00:00

set -euo pipefail
source "${PBS_O_WORKDIR:?}/jobs/aqua_common.sh"
# One reference pilot, then the same inputs with compiled CPU control.
# Actual Fashion-MNIST, not synthetic data. No final-test tuning.
for backend in cython cpp; do
    "$FASHION_PYTHON" run_fashion.py --data-dir "$FASHION_DATA" \
        --backend "$backend" --epochs 1 --train-limit 1000 \
        --assign-per-class 10 --eval-limit 100 --n-excitatory 400 \
        --out-dir "$FASHION_RESULTS/pilot_${backend}_${PBS_JOBID}" --threads 1
done
