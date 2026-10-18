#!/bin/bash
# Compute-node inspection; invoked by the two PBS probe jobs.
set -uo pipefail
mode="${1:?Expected cpu or gpu}"
case "$mode" in
    cpu|gpu) ;;
    *) echo "Expected cpu or gpu" >&2; exit 2 ;;
esac
: "${PBS_JOBID:?Run this probe through qsub}"
: "${PBS_O_WORKDIR:?Missing PBS submission directory}"
probe_dir="$PBS_O_WORKDIR/results/aqua_diagnostics/${PBS_JOBID}"
mkdir -p "$probe_dir" || exit 1
exec > >(tee "$probe_dir/probe.log") 2>&1

inspect() {
    echo
    printf 'Inspect:'
    printf ' %q' "$@"
    echo
    "$@"
    local inspect_status=$?
    if [ "$inspect_status" -ne 0 ]; then
        echo "Inspection unavailable or failed (status $inspect_status); continuing."
    fi
    return 0
}

echo "Fashion-MNIST Aqua diagnostic: $mode"
inspect date -Is
inspect hostname
printf 'PBS_JOBID=%s\nPBS_QUEUE=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
    "$PBS_JOBID" "${PBS_QUEUE:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
    inspect cat "$PBS_NODEFILE"
fi
inspect qstat -f "$PBS_JOBID"
inspect qstat -Qf gpuq
inspect qstat -Qf small20
inspect qstat -Bf
inspect qstat -u "${USER:-$(id -un)}"
inspect lscpu
inspect getconf GNU_LIBC_VERSION
inspect free -h
inspect df -h "$PBS_O_WORKDIR"
inspect module list
inspect module avail cuda
inspect command -v g++
inspect g++ --version
inspect command -v nvcc
inspect nvcc --version

probe_status=0
if [ "$mode" = gpu ]; then
    echo
    echo "GPU inventory is metadata, not authorization to use unallocated GPUs."
    nvidia-smi
    if [ "$?" -ne 0 ]; then
        echo "ERROR: nvidia-smi failed inside the GPU allocation."
        probe_status=1
    fi
    inspect nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
fi

probe_python=""
if command -v python3 >/dev/null 2>&1; then
    probe_python=python3
elif command -v python >/dev/null 2>&1; then
    probe_python=python
fi
if [ -n "$probe_python" ]; then
    inspect "$probe_python" --version
    inspect "$probe_python" -c 'import sys; print(sys.executable); print(sys.version)'
    inspect "$probe_python" -c '
import importlib.metadata
import platform
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Platform:", platform.platform())
for name in ("brian2", "brian2cuda", "numpy", "scipy", "Cython", "setuptools"):
    try:
        print(name, importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        print(name, "not installed in this environment")
'
else
    echo "No Python interpreter found in this job environment."
fi
inspect module avail python
inspect module avail anaconda
echo
echo "Probe log: $probe_dir/probe.log"
echo "Diagnostic exit status: $probe_status"
exit "$probe_status"
