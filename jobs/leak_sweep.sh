#!/bin/bash
# Method D sweep (run on the login node from the project folder): one
# replay_cpp.cmd job (replay + evaluation on the tuning half) per setting.
# Safe to run again: settings that already have a result or were already
# submitted are skipped, and it stops cleanly when qsub refuses a job (AQUA's
# per-user job limit), so just run the same command again later.
#   DATASET, BASELINE, FORGET (default class6_9)
#   GATES        default "neuron synapse"
#   LEAKS        lambda_u values, default "0.0002 0.001 0.005"
#   REPLAYS      R values, default "1000 5000"
#                (an untouched gated weight keeps (1 - lambda_u)^R: R=1000 gives
#                 0.82 / 0.37 / 0.0067, R=5000 gives 0.37 / 0.0067 / ~0)
#   ALPHA        synapse gate, default 2;  THETA default frozen
#   WEIGHT_NORM  39 for the Fashion baseline, 0 for MNIST (default 0)
#   NORM_GATED   partial (default) | none; see jobs/replay_cpp.cmd
set -euo pipefail
: "${DATASET:?}" "${BASELINE:?}"
FORGET="${FORGET:-class6_9}"; GATES="${GATES:-neuron synapse}"; LEAKS="${LEAKS:-0.0002 0.001 0.005}"
REPLAYS="${REPLAYS:-1000 5000}"; ALPHA="${ALPHA:-2}"; THETA="${THETA:-frozen}"; WEIGHT_NORM="${WEIGHT_NORM:-0}"; NORM_GATED="${NORM_GATED:-partial}"
marks="results/$DATASET/submitted"
mkdir -p "$marks"
for gate in $GATES; do for leak in $LEAKS; do for r in $REPLAYS; do
    # same folder name as jobs/replay_cpp.cmd
    name="D_${FORGET}_${gate}"; [ "$gate" = synapse ] && name+="_a${ALPHA}"
    name+="_l${leak}_r${r}_theta${THETA}"; [ "$WEIGHT_NORM" != 0 ] && name+="_wn${WEIGHT_NORM}"
    [ "$WEIGHT_NORM" != 0 ] && [ "$gate" = synapse ] && [ "$NORM_GATED" = partial ] && name+="pg"
    mark="$marks/$(basename "$BASELINE")_${name}"
    if compgen -G "results/$DATASET/evaluate/${name}_tune_*/metrics.json" >/dev/null; then
        echo "$name: already evaluated, skipped"; continue
    fi
    if [ -e "$mark" ]; then echo "$name: already submitted, skipped"; continue; fi
    echo "$name"
    if ! qsub -v "DATASET=$DATASET,BASELINE=$BASELINE,FORGET=$FORGET,GATE=$gate,ALPHA=$ALPHA,LEAK=$leak,REPLAY=$r,THETA=$THETA,WEIGHT_NORM=$WEIGHT_NORM,NORM_GATED=$NORM_GATED" jobs/replay_cpp.cmd; then
        echo "qsub refused this job (probably AQUA's per-user job limit). Run the same command again"
        echo "when some of your jobs have finished; submitted and finished settings are skipped."
        exit 0
    fi
    date > "$mark"
done; done; done
echo "All settings submitted or finished."
