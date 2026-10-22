#!/bin/bash
#PBS -N train_cpp
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

# Full run, COMPILED CPU backend (whole training loop in C++). Measured locally
# ~10x faster than train_cpu.cmd at N=400 with identical results. No
# checkpoint/resume: the run must finish inside this 48 h job. All settings in
# jobs/settings.sh can be given with qsub -v:
#   qsub -v DATASET=mnist,N=400,VALIDATION_SIZE=0 jobs/train_cpp.cmd
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/settings.sh"
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
print_settings
run_dir="$FASHION_RESULTS/$(run_name "$N")_cpp_${PBS_JOBID%%.*}"
echo "Run folder: $run_dir"
"$FASHION_PYTHON" -u run_fashion.py "${common_args[@]}" --n-excitatory "$N" \
    --backend cpp --threads 1 --out-dir "$run_dir" 2>&1 | tee "$run_dir.log"
