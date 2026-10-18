#!/bin/bash
# Sourced by training jobs. All paths can be passed via qsub -v.
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
module load anaconda3_2024.10
: "${FASHION_PYTHON:?Set FASHION_PYTHON to an absolute environment executable via qsub -v}"
unset PYTHONHOME PYTHONPATH
"$FASHION_PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10–3.12"; import brian2, numpy'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
FASHION_DATA="${FASHION_DATA:-$PBS_O_WORKDIR/data/fashion}"
FASHION_RESULTS="${FASHION_RESULTS:-$PBS_O_WORKDIR/results/fashion}"
FASHION_WORK="${FASHION_WORK:-${TMPDIR:-/tmp}/fashion_${USER}_${PBS_JOBID}}"
mkdir -p "$FASHION_WORK" "$FASHION_RESULTS"
export MPLCONFIGDIR="$FASHION_WORK/matplotlib"
export FASHION_BUILD_ROOT="$FASHION_WORK/builds"
export TMPDIR="$FASHION_WORK/tmp"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
echo "Job $PBS_JOBID; host $(hostname); Python $FASHION_PYTHON"
echo "Data $FASHION_DATA; results $FASHION_RESULTS; work $FASHION_WORK"
"$FASHION_PYTHON" -m pip freeze > "$FASHION_RESULTS/environment_${PBS_JOBID}.txt"
