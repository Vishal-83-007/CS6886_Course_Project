#!/bin/bash
# Run on the Aqua login node from the project root. Preview by default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:---dry-run}"
if [ "$#" -gt 1 ] || { [ "$mode" != --dry-run ] && [ "$mode" != --submit ]; }; then
    echo "Usage: bash scripts/submit_aqua_cpu_suite.sh [--dry-run|--submit]" >&2
    exit 2
fi
read -r -a requests <<< "${REQUESTS:-baseline class6 class0 class9 random10}"
read -r -a seeds <<< "${SEEDS:-0}"
validation="${VALIDATION_SIZE:-0}"
epochs="${EPOCHS:-3}"
if [ "${#requests[@]}" -eq 0 ] || [ "${#seeds[@]}" -eq 0 ]; then
    echo "REQUESTS and SEEDS must be nonempty" >&2; exit 2
fi
for request in "${requests[@]}"; do
    case "$request" in
        baseline|class6|class0|class9|random10) ;;
        *) echo "Unknown request: $request" >&2; exit 2 ;;
    esac
done
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done
path_variables=""
for name in FASHION_DATA FASHION_RESULTS FASHION_ENV_ARCHIVES; do
    value="${!name:-}"
    if [ -n "$value" ]; then
        if [[ "$value" = *','* || "$value" = *$'\n'* ]]; then
            echo "$name must not contain commas or newlines for qsub -v" >&2; exit 2
        fi
        path_variables+=",$name=$value"
    fi
done
[[ "$validation" =~ ^[0-9]+$ ]] && [[ "$epochs" =~ ^[1-9][0-9]*$ ]] || {
    echo "VALIDATION_SIZE must be nonnegative; EPOCHS must be positive" >&2; exit 2;
}
if (( 10#$validation % 10 != 0 || 10#$validation >= 60000 )); then
    echo "VALIDATION_SIZE must be a multiple of ten below 60000" >&2; exit 2
fi
archive_dir="${FASHION_ENV_ARCHIVES:-${FASHION_RESULTS:-$PWD/results/fashion}/environments}"
if [ "$mode" = --submit ]; then
    command -v qsub >/dev/null
    for suffix in tar.gz json; do
        if [ ! -f "$archive_dir/fashion-cpu-cp312.$suffix" ]; then
            echo "Missing CPU environment: $archive_dir/fashion-cpu-cp312.$suffix" >&2
            exit 1
        fi
    done
    log_dir="${FASHION_RESULTS:-$PWD/results/fashion}"
    mkdir -p "$log_dir"
    submission_log="$log_dir/cpu_submissions_$(date +%Y%m%d_%H%M%S)_$$.tsv"
    printf 'request\tseed\tvalidation_size\tepochs\tjob_id\n' > "$submission_log"
    echo "Submission record: $submission_log"
fi
echo "One core per model; baseline and Gold jobs are independent. Scheduler controls concurrency."
for seed in "${seeds[@]}"; do
    for request in "${requests[@]}"; do
        variables="FASHION_ENV_KIND=cpu,REQUEST=$request,SEED=$seed,VALIDATION_SIZE=$validation,EPOCHS=$epochs$path_variables"
        command=(qsub -v "$variables" jobs/aqua_cpp_train.cmd)
        printf '%q ' "${command[@]}"; printf '\n'
        if [ "$mode" = --submit ]; then
            job_id="$("${command[@]}")"
            printf '%s\t%s\t%s\t%s\t%s\n' "$request" "$seed" "$validation" "$epochs" "$job_id" >> "$submission_log"
            echo "Submitted: $job_id"
        fi
    done
done
if [ "$mode" = --dry-run ]; then
    echo "Preview only. Add --submit to submit these jobs."
fi
