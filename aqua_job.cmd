#!/bin/bash
#PBS -N cs6023_a3
#PBS -o logfile.log
#PBS -e errorfile.err
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -q gpuq
#PBS -l walltime=00:20:00

# Submit from this assignment directory with:
#   qsub aqua_job.cmd
#
# The job compiles submit.cu and runs the supplied public test harness on one
# Aqua GPU. Failed outputs are copied back to temp_output for inspection.
# The test report is saved as results.txt and also appears in logfile.log.

job_id="${PBS_JOBID%%.*}"
scratch_dir="$HOME/scratch/cs6023_a3_${job_id}"

if ! mkdir -p "$scratch_dir"; then
    echo "ERROR: could not create scratch directory: $scratch_dir" >&2
    exit 1
fi

copy_results_back() {
    if [ -f "$scratch_dir/results.txt" ]; then
        cp -f "$scratch_dir/results.txt" "$PBS_O_WORKDIR/results.txt"
    fi
    if [ -d "$scratch_dir/temp_output" ]; then
        cp -R "$scratch_dir/temp_output" "$PBS_O_WORKDIR/temp_output_${job_id}"
    fi
    if [ -f "$scratch_dir/compile.log" ]; then
        cp -f "$scratch_dir/compile.log" "$PBS_O_WORKDIR/compile_${job_id}.log"
    fi
}

cleanup() {
    status=$?
    copy_results_back
    rm -rf "$scratch_dir"
    exit "$status"
}
trap cleanup EXIT

if ! cp -R "$PBS_O_WORKDIR"/. "$scratch_dir"/; then
    echo "ERROR: could not copy assignment files to scratch" >&2
    exit 1
fi
cd "$scratch_dir" || exit 1

echo "Job $PBS_JOBID running on $(hostname)"
echo "Working directory: $scratch_dir"

module load cuda11.4 || exit 1
echo "CUDA compiler: $(command -v nvcc)"
nvcc -O3 -std=c++14 EP23B048.cu -o main > compile.log 2>&1
compile_status=$?
cat compile.log
if [ "$compile_status" -ne 0 ]; then
    echo "ERROR: CUDA compilation failed; see compile.log and errorfile.err" >&2
    exit "$compile_status"
fi

bash run_testcases.sh 2>&1 | tee results.txt
test_status=${PIPESTATUS[0]}
echo "Test harness exit status: $test_status"
exit "$test_status"
