#!/bin/bash
# Compute-node inspection for the CPU workflow.
set -uo pipefail
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

echo "Fashion-MNIST Aqua diagnostic: cpu"
inspect date -Is
inspect hostname
printf 'PBS_JOBID=%s\nPBS_QUEUE=%s\n' \
    "$PBS_JOBID" "${PBS_QUEUE:-unset}"
if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
    inspect cat "$PBS_NODEFILE"
fi
inspect qstat -f "$PBS_JOBID"
inspect qstat -Qf small20
inspect qstat -Bf
inspect qstat -u "${USER:-$(id -un)}"
inspect lscpu
inspect getconf GNU_LIBC_VERSION
inspect free -h
inspect df -h "$PBS_O_WORKDIR"
inspect module list
inspect command -v g++
inspect g++ --version

probe_status=0
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
for name in ("brian2", "numpy", "scipy", "Cython", "setuptools"):
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
