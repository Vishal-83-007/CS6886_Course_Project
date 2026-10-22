#!/bin/bash
#PBS -N setup_gpu
#PBS -j oe
#PBS -q small20
#PBS -l select=1:ncpus=1
#PBS -l walltime=24:00:00

# One-time OFFLINE setup of the GPU environment $HOME/fashion-gpu (Brian2 2.8.0 +
# Brian2CUDA 1.0a7, kept separate because Brian2CUDA needs the older Brian2).
# Needed by pilot_gpu.cmd and train_gpu.cmd.
#   qsub jobs/setup_gpu.cmd
# Log: results/setup_gpu_<jobid>.log   (last line: GPU SETUP OK)
set -euo pipefail
cd "${PBS_O_WORKDIR:?Submit with qsub from the project folder}"
mkdir -p results
exec > >(tee "results/setup_gpu_${PBS_JOBID%%.*}.log") 2>&1
module load anaconda3_2024.10
unset PYTHONHOME PYTHONPATH
python scripts/install_aqua_offline.py --kind gpu --wheelhouse "$PBS_O_WORKDIR/wheelhouse"
PY="$HOME/fashion-gpu/bin/python"
"$PY" -c 'import brian2, brian2cuda; print("brian2", brian2.__version__, "brian2cuda", brian2cuda.__version__)'
for dataset in mnist fashion; do
    "$PY" prepare_fashion.py --dataset "$dataset"
done
echo "GPU SETUP OK. GPU Python: $PY"
