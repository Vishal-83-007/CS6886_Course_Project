#!/bin/bash
# Retire obsolete source files left behind when extracting the CPU-only patch.
# Historical results, environments, wheels and earlier transfer archives remain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for retired in jobs/aqua_gpu_pilot.cmd jobs/aqua_probe_gpu.cmd \
    requirements-aqua-gpu.txt test_cuda_codegen.py aqua_job.cmd; do
    if [ -f "$retired" ]; then
        rm -- "$retired"
        echo "Removed retired source: $retired"
    fi
done
