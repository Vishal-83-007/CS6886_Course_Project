#!/bin/bash
#PBS -N fashion_setup
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=2
#PBS -l walltime=02:00:00

# Offline setup using the module's Python and transferred Linux wheels.
# Submit from repository root: qsub jobs/aqua_setup.cmd
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit from the repository root}"
mkdir -p results/fashion
exec > >(tee "results/fashion/setup_${PBS_JOBID}.log") 2>&1
getconf GNU_LIBC_VERSION
module load anaconda3_2024.10
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
python --version
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
setup_work="${FASHION_SETUP_WORK:-${TMPDIR:-/tmp}/fashion_setup_${USER}_${PBS_JOBID}}"
archive_dir="${FASHION_ENV_ARCHIVES:-$PBS_O_WORKDIR/results/fashion/environments}"
export MPLCONFIGDIR="$setup_work/matplotlib" TMPDIR="$setup_work/tmp"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR"
echo "Node-local setup work: $setup_work"
df -h "$setup_work" "$PBS_O_WORKDIR"
echo "Persistent environment archives: $archive_dir"
setup_kind="${FASHION_SETUP_KIND:-cpu}"
if [ "$setup_kind" != cpu ]; then
    echo "Only FASHION_SETUP_KIND=cpu is supported" >&2
    exit 1
fi
setup_kinds=(cpu)
echo "Environment selection: $setup_kind"
python -u scripts/install_aqua_offline.py \
    --wheelhouse "${FASHION_WHEELHOUSE:-$PBS_O_WORKDIR/wheelhouse}" \
    --env-root "$setup_work/envs" --stage-dir "$setup_work/wheels" \
    --pack-dir "$archive_dir" --kind "$setup_kind" --step-timeout 1200 \
    --pack-timeout 1800
for kind in "${setup_kinds[@]}"; do
    prefix="$setup_work/envs/fashion-$kind"
    "$prefix/bin/python" -m pip freeze > "results/fashion/environment_${kind}_${PBS_JOBID}.txt"
done
"$setup_work/envs/fashion-${setup_kinds[0]}/bin/python" prepare_fashion.py \
    --data-dir "${FASHION_DATA:-$PBS_O_WORKDIR/data/fashion}"
echo "Setup completed. Packed environments are in $archive_dir."
echo "Submit training with: bash scripts/submit_aqua_cpu_suite.sh"
