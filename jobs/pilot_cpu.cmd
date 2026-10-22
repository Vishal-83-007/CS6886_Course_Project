#!/bin/bash
#PBS -N pilot_cpu
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=02:00:00

# Short timing run with the Cython backend. All settings in jobs/settings.sh can
# be given with qsub -v; pilot defaults: EPOCHS=1 TRAIN_LIMIT=1000
# ASSIGN_PER_CLASS=10 EVAL_LIMIT=100. Example:
#   qsub -v DATASET=mnist,N=800 jobs/pilot_cpu.cmd
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
EPOCHS="${EPOCHS:-1}"; TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
ASSIGN_PER_CLASS="${ASSIGN_PER_CLASS:-10}"; EVAL_LIMIT="${EVAL_LIMIT:-100}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/settings.sh"
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
print_settings
run_dir="$FASHION_RESULTS/pilot_$(run_name "$N")_cython_${PBS_JOBID%%.*}"
echo "Run folder: $run_dir"
"$FASHION_PYTHON" -u run_fashion.py "${common_args[@]}" --n-excitatory "$N" \
    --backend cython --out-dir "$run_dir" 2>&1 | tee "$run_dir.log"
