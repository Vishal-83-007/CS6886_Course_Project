#!/bin/bash
#PBS -N pilot_gpu
#PBS -j oe
#PBS -q gpuq
#PBS -l select=1:ncpus=2:ngpus=2
#PBS -l walltime=02:00:00

# GPU (Brian2CUDA) pilot. REQUIRED once before train_gpu.cmd: it is the first
# time the CUDA program actually runs. N on DATASET on one V100, and N2 (default
# N) on DATASET2 (default DATASET) on the other.
# Pilot defaults: DATASET=mnist DATASET2=fashion N=800 EPOCHS=1 TRAIN_LIMIT=1000
# ASSIGN_PER_CLASS=10 EVAL_LIMIT=100 (a pilot must keep TRAIN_LIMIT<=1000 and
# EPOCHS=1). Examples:
#   qsub jobs/pilot_gpu.cmd
#   qsub -v DATASET=mnist,DATASET2=mnist,N=400,N2=800 jobs/pilot_gpu.cmd
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-gpu/bin/python}"
N="${N:-800}"
DATASET2="${DATASET2:-fashion}"
EPOCHS="${EPOCHS:-1}"; TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
ASSIGN_PER_CLASS="${ASSIGN_PER_CLASS:-10}"; EVAL_LIMIT="${EVAL_LIMIT:-100}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/settings.sh"
source "$PBS_O_WORKDIR/jobs/gpu_common.sh"
run_gpu_workers "pilot_" "_gpu_${PBS_JOBID%%.*}"
