#!/bin/bash
#PBS -N train_cpu
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

# Full run, Cython backend: slowest, but saves a checkpoint every 15 min. If not
# finished after 46 h it saves and exits (status 75): resubmit the SAME command
# and it resumes. All settings in jobs/settings.sh can be given with qsub -v:
#   qsub -v DATASET=mnist,N=400,VALIDATION_SIZE=0 jobs/train_cpu.cmd
set -euo pipefail
export USER="${USER:-$(id -un)}"
export FASHION_PYTHON="${FASHION_PYTHON:-$HOME/fashion-cpu/bin/python}"
source "${PBS_O_WORKDIR:?Submit with qsub from the project folder}/jobs/settings.sh"
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
print_settings
run_dir="$FASHION_RESULTS/$(run_name "$N")_cython"
resume=()
if [ -f "$run_dir/checkpoints/CURRENT.json" ]; then
    resume=(--resume)
    echo "Resuming from $run_dir"
fi
echo "Run folder: $run_dir"
set +e
"$FASHION_PYTHON" -u run_fashion.py "${common_args[@]}" --n-excitatory "$N" \
    --backend cython --out-dir "$run_dir" --checkpoint-seconds 900 --max-seconds 165600 \
    ${resume[@]+"${resume[@]}"} 2>&1 | tee -a "$run_dir.log"
status=${PIPESTATUS[0]}
set -e
if [ "$status" -eq 75 ]; then
    echo "Checkpointed before walltime. Resubmit the same qsub command to continue."
fi
exit "$status"
