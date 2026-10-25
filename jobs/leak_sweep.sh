#!/bin/bash
# Method D sweep (run on the login node from the project folder): one
# replay_cpp.cmd job (replay + evaluation on the tuning half) per setting.
#   DATASET, BASELINE, FORGET (default class6_9)
#   GATES        default "neuron synapse"
#   LEAKS        lambda_u values, default "0.0002 0.001 0.005"
#   REPLAYS      R values, default "1000 5000"
#                (an untouched gated weight keeps (1 - lambda_u)^R: R=1000 gives
#                 0.82 / 0.37 / 0.0067, R=5000 gives 0.37 / 0.0067 / ~0)
#   ALPHA        synapse gate, default 2;  THETA default frozen
#   WEIGHT_NORM  39 for the Fashion baseline, 0 for MNIST (default 0)
set -euo pipefail
: "${DATASET:?}" "${BASELINE:?}"
FORGET="${FORGET:-class6_9}"; GATES="${GATES:-neuron synapse}"; LEAKS="${LEAKS:-0.0002 0.001 0.005}"
REPLAYS="${REPLAYS:-1000 5000}"; ALPHA="${ALPHA:-2}"; THETA="${THETA:-frozen}"; WEIGHT_NORM="${WEIGHT_NORM:-0}"
for gate in $GATES; do for leak in $LEAKS; do for r in $REPLAYS; do
    echo "gate=$gate leak=$leak replay=$r"
    qsub -v "DATASET=$DATASET,BASELINE=$BASELINE,FORGET=$FORGET,GATE=$gate,ALPHA=$ALPHA,LEAK=$leak,REPLAY=$r,THETA=$THETA,WEIGHT_NORM=$WEIGHT_NORM" jobs/replay_cpp.cmd
done; done; done
