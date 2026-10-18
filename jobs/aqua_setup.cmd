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
python --version
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fashion_setup_${PBS_JOBID}"
mkdir -p "$MPLCONFIGDIR"
python scripts/install_aqua_offline.py --wheelhouse "${FASHION_WHEELHOUSE:-$PBS_O_WORKDIR/wheelhouse}"
for kind in cpu gpu; do
    prefix="$HOME/fashion-$kind"
    "$prefix/bin/python" -m pip freeze > "results/fashion/environment_${kind}_${PBS_JOBID}.txt"
done
"$HOME/fashion-cpu/bin/python" prepare_fashion.py --data-dir data/fashion
echo "CPU Python: $HOME/fashion-cpu/bin/python"
echo "GPU Python: $HOME/fashion-gpu/bin/python"
echo "Setup completed. Submit the CPU and GPU pilot jobs next."
