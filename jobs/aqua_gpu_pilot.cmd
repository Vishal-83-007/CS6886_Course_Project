#!/bin/bash
#PBS -N fashion_gpu_pilot
#PBS -j oe
#PBS -q gpuq
#PBS -l select=1:ncpus=2:ngpus=2
#PBS -l walltime=02:00:00

# Both GPUs are requested because the probe showed no per-GPU visibility mask.
# Each worker is restricted to one of this node's two allocated V100s.
set -euo pipefail
source "${PBS_O_WORKDIR:?}/jobs/aqua_common.sh"
module unload cuda10.1
module load cuda11.4
"$FASHION_PYTHON" -c 'import brian2cuda'
nvidia-smi
nvcc --version
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a gpu_devices <<< "$CUDA_VISIBLE_DEVICES"
else
    gpu_devices=(0 1)
fi
if [ "${#gpu_devices[@]}" -ne 2 ]; then
    echo "Expected exactly two allocated devices; check PBS allocation/mask." >&2
    exit 1
fi
pids=()
for index in 0 1; do
    request=baseline
    if [ "$index" -eq 1 ]; then request=class6; fi
    CUDA_VISIBLE_DEVICES="${gpu_devices[$index]}" "$FASHION_PYTHON" run_fashion.py \
        --data-dir "$FASHION_DATA" --backend cuda --gpu-id 0 --epochs 1 \
        --request "$request" --train-limit 1000 --assign-per-class 10 \
        --eval-limit 100 --n-excitatory 400 --threads 1 \
        --out-dir "$FASHION_RESULTS/pilot_cuda_${request}_${PBS_JOBID}" \
        > "$FASHION_RESULTS/pilot_cuda_${request}_${PBS_JOBID}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then :; else status=1; fi
done
exit "$status"
