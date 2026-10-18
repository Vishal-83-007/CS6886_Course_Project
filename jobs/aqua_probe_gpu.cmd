#!/bin/bash
#PBS -N fashion_gpu_probe
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq
#PBS -l walltime=00:10:00

# Submit from the repository root: qsub jobs/aqua_probe_gpu.cmd
# Load/activate your intended compiler/CUDA/Python environment here if needed.
set -uo pipefail
cd "${PBS_O_WORKDIR:?Submit with qsub from the repository root}" || exit 1
bash jobs/aqua_probe.sh gpu
