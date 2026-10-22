#!/bin/bash
# Shared by pilot_gpu.cmd and train_gpu.cmd (sourced after settings.sh).
# Requests are for BOTH V100s of a node: an AQUA one-GPU allocation was observed
# to expose both devices without CUDA_VISIBLE_DEVICES, so each worker is pinned
# to one allocated GPU instead of guessing a free one.
#   First GPU:  N on DATASET.
#   Second GPU: only if N2 or DATASET2 is given; N2 (default N) on DATASET2
#               (default DATASET).
source "$PBS_O_WORKDIR/jobs/aqua_common.sh"
module unload cuda10.1 2>/dev/null || true
module load cuda11.4
"$FASHION_PYTHON" -c 'import brian2cuda'
nvidia-smi || true
nvcc --version | tail -1
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
else
    gpus=(0 1)
fi
[ "${#gpus[@]}" -eq 2 ] || { echo "Expected two allocated GPUs, got: ${gpus[*]}" >&2; exit 1; }
sizes=("$N"); datasets=("$DATASET")
if [ -n "${N2:-}" ] || [ -n "${DATASET2:-}" ]; then
    sizes+=("${N2:-$N}"); datasets+=("${DATASET2:-$DATASET}")
fi
print_settings

# run_gpu_workers <folder prefix> <folder suffix> <extra args...>
run_gpu_workers() {
    local prefix="$1" suffix="$2"; shift 2
    local pids=() i n ds out status=0
    for i in "${!sizes[@]}"; do
        n="${sizes[$i]}"; ds="${datasets[$i]}"
        mkdir -p "$PBS_O_WORKDIR/results/$ds"
        out="$PBS_O_WORKDIR/results/$ds/${prefix}$(run_name "$n")${suffix}"
        echo "GPU ${gpus[$i]}: $ds, N=$n -> $out"
        CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$FASHION_PYTHON" -u run_fashion.py \
            --dataset "$ds" --data-dir "$PBS_O_WORKDIR/data/$ds" "${run_args[@]}" \
            --n-excitatory "$n" --backend cuda --gpu-id 0 --threads 1 \
            ${COMPILE_ONLY:+--compile-only} "$@" --out-dir "$out" > "$out.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    echo "GPU workers finished with status $status (0 = all succeeded)"
    return "$status"
}
