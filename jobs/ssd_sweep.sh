#!/bin/bash
# Method B sweep (run on the AQUA login node from the project folder; dampening
# itself is numpy only and takes seconds). For every alpha x lambda it writes a
# dampened copy of the baseline weights and submits one evaluate_cpp.cmd job
# (tuning half of the test set, fixed baseline labels + own labels).
# Safe to run again: settings that already have a result or were already
# submitted are skipped, and it stops cleanly when qsub refuses a job (AQUA's
# per-user job limit), so just run the same command again later.
#   DATASET     mnist | fashion
#   BASELINE    run folder of the trained baseline
#   IMPORTANCE  importance.npz files (space separated; several passes are averaged)
#   FORGET      removed classes, e.g. class6_9
#   ALPHAS      default "1.5 2 3 4". I(D_f)/I(D) <= |D|/|D_f| (about 5 for two
#               classes of ten, 10 for one), so the SSD paper's 5-50 (one class
#               of 20 or 100) selects nothing here; see dampen.py
#   LAMBDAS     default "0.1 0.5 1"       (SSD paper range 0.1-1)
#   PROTECTS    percent of synapses with the highest retain importance I(D_r) that
#               are never dampened, default "0" (plain SSD); e.g. "5 10 20"
#   EVAL_LIMIT  default 0 (all 5,000 tuning images)
# Example:
#   DATASET=mnist BASELINE=results/mnist/n800_baseline_..._cpp_2461985 \
#   IMPORTANCE="$(ls results/mnist/importance/*/importance.npz)" FORGET=class6_9 bash jobs/ssd_sweep.sh
set -euo pipefail
: "${DATASET:?}" "${BASELINE:?}" "${IMPORTANCE:?}" "${FORGET:?}"
ALPHAS="${ALPHAS:-1.5 2 3 4}"; LAMBDAS="${LAMBDAS:-0.1 0.5 1}"; PROTECTS="${PROTECTS:-0}"; EVAL_LIMIT="${EVAL_LIMIT:-0}"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh 2>/dev/null || true; fi
module load anaconda3_2024.10 >/dev/null 2>&1 || true
out="results/$DATASET/ssd/$(basename "$BASELINE")_$FORGET"
marks="results/$DATASET/submitted"
mkdir -p "$out" "$marks"
for alpha in $ALPHAS; do for lam in $LAMBDAS; do for protect in $PROTECTS; do
    tag="a${alpha}_l${lam}"; [ "$protect" != 0 ] && tag+="_p${protect}"
    name="ssd_${tag}"
    mark="$marks/$(basename "$BASELINE")_${FORGET}_${name}"
    if compgen -G "results/$DATASET/evaluate/${name}_tune_*/metrics.json" >/dev/null; then
        echo "$tag: already evaluated, skipped"; continue
    fi
    if [ -e "$mark" ]; then echo "$tag: already submitted, skipped"; continue; fi
    file="$out/${tag}.npz"
    python dampen.py --importance $IMPORTANCE --weights "$BASELINE/baseline_weights.npz" \
        --forget "$FORGET" --alpha "$alpha" --lam "$lam" --protect-top "$protect" --out "$file" > /dev/null
    echo "$tag: $(python -c "import json;s=json.load(open('${file%.npz}.json'));print('selected %.2f%% of synapses (%.2f%% before protection), %d neurons touched' % (100*s['selected_fraction'], 100*s['selected_without_protection']/s['selected_synapses']*s['selected_fraction'] if s['selected_synapses'] else 0, s['neurons_touched']))")"
    if ! qsub -v "DATASET=$DATASET,WEIGHTS=$file,FORGET=$FORGET,FIXED_LABELS=$BASELINE/neuron_labels.npy,EVAL_SPLIT=tune,EVAL_LIMIT=$EVAL_LIMIT,NAME=$name" jobs/evaluate_cpp.cmd; then
        echo "qsub refused this job (probably AQUA's per-user job limit). Run the same command again"
        echo "when some of your jobs have finished; submitted and finished settings are skipped."
        exit 0
    fi
    date > "$mark"
done; done; done
echo "All settings submitted or finished."
