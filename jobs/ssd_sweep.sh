#!/bin/bash
# Method B sweep (run on the AQUA login node from the project folder; dampening
# itself is numpy only and takes seconds). For every alpha x lambda it writes a
# dampened copy of the baseline weights and submits one evaluate_cpp.cmd job
# (tuning half of the test set, fixed baseline labels + own labels).
#   DATASET     mnist | fashion
#   BASELINE    run folder of the trained baseline
#   IMPORTANCE  importance.npz files (space separated; several passes are averaged)
#   FORGET      removed classes, e.g. class6_9
#   ALPHAS      default "1.5 2 3 4". I(D_f)/I(D) <= |D|/|D_f| (about 5 for two
#               classes of ten, 10 for one), so the SSD paper's 5-50 (one class
#               of 20 or 100) selects nothing here; see dampen.py
#   LAMBDAS     default "0.1 0.5 1"       (SSD paper range 0.1-1)
#   EVAL_LIMIT  default 0 (all 5,000 tuning images)
# Example:
#   DATASET=mnist BASELINE=results/mnist/n800_baseline_..._cpp_2461985 \
#   IMPORTANCE="$(ls results/mnist/importance/*/importance.npz)" FORGET=class6_9 bash jobs/ssd_sweep.sh
set -euo pipefail
: "${DATASET:?}" "${BASELINE:?}" "${IMPORTANCE:?}" "${FORGET:?}"
ALPHAS="${ALPHAS:-1.5 2 3 4}"; LAMBDAS="${LAMBDAS:-0.1 0.5 1}"; EVAL_LIMIT="${EVAL_LIMIT:-0}"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh 2>/dev/null || true; fi
module load anaconda3_2024.10 >/dev/null 2>&1 || true
out="results/$DATASET/ssd/$(basename "$BASELINE")_$FORGET"
mkdir -p "$out"
for alpha in $ALPHAS; do for lam in $LAMBDAS; do
    file="$out/a${alpha}_l${lam}.npz"
    python dampen.py --importance $IMPORTANCE --weights "$BASELINE/baseline_weights.npz" \
        --forget "$FORGET" --alpha "$alpha" --lam "$lam" --out "$file" > /dev/null
    echo "a=$alpha l=$lam: $(python -c "import json;s=json.load(open('${file%.npz}.json'));print('selected %.2f%% of synapses, %d neurons touched' % (100*s['selected_fraction'], s['neurons_touched']))")"
    qsub -v "DATASET=$DATASET,WEIGHTS=$file,FORGET=$FORGET,FIXED_LABELS=$BASELINE/neuron_labels.npy,EVAL_SPLIT=tune,EVAL_LIMIT=$EVAL_LIMIT,NAME=ssd_a${alpha}_l${lam}" jobs/evaluate_cpp.cmd
done; done
