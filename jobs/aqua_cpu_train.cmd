#!/bin/bash
#PBS -N fashion_cpu_train
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00

set -euo pipefail
source "${PBS_O_WORKDIR:?}/jobs/aqua_common.sh"
request="${REQUEST:-baseline}"
seed="${SEED:-0}"
validation="${VALIDATION_SIZE:-5000}"
run_dir="$FASHION_RESULTS/${request}_seed${seed}_val${validation}_cython"
resume=()
if [ -f "$run_dir/checkpoints/CURRENT.json" ]; then
    resume=(--resume)
fi
# 46h leaves time for a slow final example and checkpoint before PBS kills at 48h.
set +e
"$FASHION_PYTHON" run_fashion.py --data-dir "$FASHION_DATA" --out-dir "$run_dir" \
    --request "$request" --seed "$seed" --backend cython --epochs 3 \
    --validation-size "$validation" --checkpoint-seconds 900 --max-seconds 165600 \
    ${resume[@]+"${resume[@]}"}
status=$?
set -e
if [ "$status" -eq 75 ]; then
    echo "Checkpointed before walltime. Resubmit this job with the same parameters to continue."
fi
exit "$status"
