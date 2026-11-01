#!/bin/bash
# Method B (SSD) for random-subset forgetting (run on the AQUA login node from
# the project folder; dampening is numpy only and takes seconds). For every
# alpha x lambda it writes a dampened copy of the baseline and submits one
# evaluate_cpp.cmd job that scores the forgotten training images, an equal
# sample of retained same-class training images and the test set (with the
# baseline's fixed labels and own labels from images that exclude D_f).
# Safe to run again: finished or submitted settings are skipped, and it stops
# cleanly when qsub refuses a job (AQUA's per-user job limit).
#   DATASET            mnist | fashion
#   BASELINE           run folder of the trained baseline
#   IMPORTANCE         importance.npz of the full training-set passes (I(D))
#   FORGET_IMPORTANCE  importance.npz of the pass(es) over the forgotten images (I(D_f));
#                      importance_cpp.cmd with FORGET=sub6_9_p10
#   FORGET             default sub6_9_p10
#   ALPHAS             default "2 5 10 25". I(D_f)/I(D) is about 5 at most for synapses
#                      that serve classes 6 and 9 as a whole, so alpha up to 5 behaves like
#                      forgetting the classes; larger alpha keeps only synapses where the
#                      forgotten images matter more than their classes
#   LAMBDAS            default "1" (SSD paper value for random forgetting)
#   EVAL_SPLIT         default all (10,000 test images; no setting is chosen on it, the
#                      whole sweep is reported)
set -euo pipefail
: "${DATASET:?}" "${BASELINE:?}" "${IMPORTANCE:?}" "${FORGET_IMPORTANCE:?}"
FORGET="${FORGET:-sub6_9_p10}"; ALPHAS="${ALPHAS:-2 5 10 25}"; LAMBDAS="${LAMBDAS:-1}"; EVAL_SPLIT="${EVAL_SPLIT:-all}"
[[ "$FORGET" =~ ^sub[0-9](_[0-9])*_p[0-9]+$ ]] || { echo "FORGET must be a subset request, e.g. sub6_9_p10"; exit 1; }
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh 2>/dev/null || true; fi
module load anaconda3_2024.10 >/dev/null 2>&1 || true
out="results/$DATASET/ssd/$(basename "$BASELINE")_$FORGET"
marks="results/$DATASET/submitted"
mkdir -p "$out" "$marks"
for alpha in $ALPHAS; do for lam in $LAMBDAS; do
    tag="a${alpha}_l${lam}"
    name="subssd_${FORGET}_${tag}"
    mark="$marks/$(basename "$BASELINE")_${name}"
    if compgen -G "results/$DATASET/evaluate/${name}_${EVAL_SPLIT}_*/metrics.json" >/dev/null; then
        echo "$tag: already evaluated, skipped"; continue
    fi
    if [ -e "$mark" ]; then echo "$tag: already submitted, skipped"; continue; fi
    file="$out/${tag}.npz"
    python dampen.py --importance $IMPORTANCE --forget-importance $FORGET_IMPORTANCE \
        --weights "$BASELINE/baseline_weights.npz" --forget "$FORGET" --alpha "$alpha" --lam "$lam" --out "$file" > /dev/null
    echo "$tag: $(python -c "import json;s=json.load(open('${file%.npz}.json'));print('selected %d synapses (%.2f%%), %d neurons touched; class-level importance would select %d, overlap %d; corr(I(Df), I(classes)) = %.3f' % (s['selected_synapses'], 100*s['selected_fraction'], s['neurons_touched'], s['selected_by_class_importance'], s['selected_by_both'], s['corr_forget_vs_class_importance']))")"
    if ! qsub -v "DATASET=$DATASET,WEIGHTS=$file,FORGET=$FORGET,FORGET_SET=1,FIXED_LABELS=$BASELINE/neuron_labels.npy,EVAL_SPLIT=$EVAL_SPLIT,NAME=$name" jobs/evaluate_cpp.cmd; then
        echo "qsub refused this job (probably AQUA's per-user job limit). Run the same command again"
        echo "when some of your jobs have finished; submitted and finished settings are skipped."
        exit 0
    fi
    date > "$mark"
done; done
echo "All settings submitted or finished."
