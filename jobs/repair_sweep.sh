#!/bin/bash
# Repair replay after unlearning (run on the login node from the project folder):
# for every start weights file and replay length, one replay_cpp.cmd job with
# GATE=none (no leak): R retained training images with STDP on, so features the
# retained classes share with the removed ones can recover; then evaluation.
# Safe to run again (finished/submitted settings are skipped; stops cleanly if
# qsub refuses a job).
#   DATASET, BASELINE, FORGET (default class6_9)
#   STARTS       weights files to start from, e.g. SSD outputs
#                "results/<ds>/ssd/<baseline>_class6_9/a2_l0.5.npz"   (required)
#   REPLAYS      default "200 1000"
#   THETAS       default "frozen" (e.g. "frozen updated")
#   WEIGHT_NORM  39 for the Fashion baseline, 0 for MNIST (default 0). Every
#                neuron keeps its weight sum from the START weights
#                (NORM_TARGET=start), so the dampening is not rescaled back up.
set -euo pipefail
: "${DATASET:?}" "${BASELINE:?}" "${STARTS:?}"
FORGET="${FORGET:-class6_9}"; REPLAYS="${REPLAYS:-200 1000}"; THETAS="${THETAS:-${THETA:-frozen}}"; WEIGHT_NORM="${WEIGHT_NORM:-0}"
marks="results/$DATASET/submitted"
mkdir -p "$marks"
for start in $STARTS; do for r in $REPLAYS; do for THETA in $THETAS; do
    [ -f "$start" ] || { echo "not found: $start"; exit 1; }
    # same folder name as jobs/replay_cpp.cmd
    name="repair_$(basename "$start" .npz)"
    [ "$(basename "$start")" = baseline_weights.npz ] && name="repair_baseline"
    name+="_r${r}_theta${THETA}"; [ "$WEIGHT_NORM" != 0 ] && name+="_wn${WEIGHT_NORM}ns"
    mark="$marks/$(basename "$BASELINE")_${FORGET}_${name}"
    if compgen -G "results/$DATASET/evaluate/${name}_tune_*/metrics.json" >/dev/null; then
        echo "$name: already evaluated, skipped"; continue
    fi
    if [ -e "$mark" ]; then echo "$name: already submitted, skipped"; continue; fi
    echo "$name"
    if ! qsub -v "DATASET=$DATASET,BASELINE=$BASELINE,FORGET=$FORGET,GATE=none,START_WEIGHTS=$start,LEAK=0,REPLAY=$r,THETA=$THETA,WEIGHT_NORM=$WEIGHT_NORM,NORM_TARGET=start" jobs/replay_cpp.cmd; then
        echo "qsub refused this job (probably AQUA's per-user job limit). Run the same command again"
        echo "when some of your jobs have finished; submitted and finished settings are skipped."
        exit 0
    fi
    date > "$mark"
done; done; done
echo "All settings submitted or finished."
