#!/bin/bash
# Sourced by training jobs. All paths can be passed via qsub -v.
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
module load anaconda3_2024.10
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
FASHION_DATA="${FASHION_DATA:-$PBS_O_WORKDIR/data/fashion}"
FASHION_RESULTS="${FASHION_RESULTS:-$PBS_O_WORKDIR/results/fashion}"
FASHION_WORK="${FASHION_WORK:-${TMPDIR:-/tmp}/fashion_${USER}_${PBS_JOBID}}"
mkdir -p "$FASHION_WORK" "$FASHION_RESULTS"
export MPLCONFIGDIR="$FASHION_WORK/matplotlib"
export FASHION_BUILD_ROOT="$FASHION_WORK/builds"
export TMPDIR="$FASHION_WORK/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
env_kind="${FASHION_ENV_KIND:-cpu}"
if [ "$env_kind" != cpu ]; then
    echo "Only FASHION_ENV_KIND=cpu is supported" >&2
    exit 1
fi
# Reuse the verified CPU archive; expand it into this job's local storage.
FASHION_PYTHON="$(python -u scripts/stage_aqua_env.py \
    --archive-dir "${FASHION_ENV_ARCHIVES:-$FASHION_RESULTS/environments}" \
    --destination "$FASHION_WORK/envs" --kind cpu)"
"$FASHION_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13); import brian2, numpy'
echo "Job $PBS_JOBID; host $(hostname); Python $FASHION_PYTHON"
echo "Data $FASHION_DATA; results $FASHION_RESULTS; work $FASHION_WORK"
"$FASHION_PYTHON" -m pip freeze > "$FASHION_RESULTS/environment_${PBS_JOBID}.txt"
