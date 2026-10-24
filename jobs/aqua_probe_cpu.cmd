#!/bin/bash
#PBS -N fashion_cpu_probe
#PBS -j oe
#PBS -l select=1:ncpus=2
#PBS -q small20
#PBS -l walltime=00:10:00

# Submit from the repository root: qsub jobs/aqua_probe_cpu.cmd
# Load/activate your intended compiler/Python environment here if needed.
set -uo pipefail
cd "${PBS_O_WORKDIR:?Submit with qsub from the repository root}" || exit 1
bash jobs/aqua_probe.sh
