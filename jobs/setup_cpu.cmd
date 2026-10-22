#!/bin/bash
#PBS -N setup_cpu
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=24:00:00

# One-time OFFLINE setup of the CPU environment $HOME/fashion-cpu (Python 3.12,
# Brian2 2.9) from wheelhouse/cp312/cpu, plus data checks and built-in tests.
# Needed by pilot_cpu.cmd, train_cpu.cmd and train_cpp.cmd. Skip it if
# ~/fashion-cpu already exists and works (it is the same environment as before).
#   qsub jobs/setup_cpu.cmd
# Log: results/setup_cpu_<jobid>.log   (last line: SETUP OK)
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit with qsub from the project folder}"
mkdir -p results
exec > >(tee "results/setup_cpu_${PBS_JOBID%%.*}.log") 2>&1
echo "Host $(hostname)  glibc $(getconf GNU_LIBC_VERSION)  $(gcc --version | head -1)"
module load anaconda3_2024.10
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/setup_cpu_${PBS_JOBID%%.*}"
mkdir -p "$MPLCONFIGDIR"
python scripts/install_aqua_offline.py --kind cpu --wheelhouse "$PBS_O_WORKDIR/wheelhouse"
PY="$HOME/fashion-cpu/bin/python"
for dataset in mnist fashion; do
    echo "== Checking data/$dataset"
    "$PY" prepare_fashion.py --dataset "$dataset"
done
echo "== Compiling and running the built-in tests (a few minutes)"
"$PY" -m unittest -v test_baseline test_fashion.MnistDataChecks \
    test_fashion.FashionExecutionChecks.test_exact_cython_resume_across_processes
echo "SETUP OK. CPU Python: $PY"
