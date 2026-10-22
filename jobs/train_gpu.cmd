#!/bin/bash
#PBS -N train_gpu
#PBS -j oe
#PBS -q gpuq
#PBS -l select=1:ncpus=2:ngpus=2
#PBS -l walltime=48:00:00

# Full run(s) on GPU (Brian2CUDA). Run only after pilot_gpu.cmd succeeded.
# N on DATASET on the first V100; a second model on the other V100 at the same
# time if N2 and/or DATASET2 is given (N2 defaults to N, DATASET2 to DATASET).
# No checkpoint/resume: each run must finish inside this 48 h job. All settings
# in jobs/settings.sh can be given with qsub -v:
#   qsub -v DATASET=mnist,N=800 jobs/train_gpu.cmd
#   qsub -v DATASET=mnist,DATASET2=fashion,N=800 jobs/train_gpu.cmd
#   qsub -v DATASET=mnist,N=400,N2=800 jobs/train_gpu.cmd
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-gpu/bin/python}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/settings.sh"
source "$PBS_O_WORKDIR/jobs/gpu_common.sh"
run_gpu_workers "" "_gpu_${PBS_JOBID%%.*}" --allow-long-cuda
