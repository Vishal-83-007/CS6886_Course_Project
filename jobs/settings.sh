#!/bin/bash
# Run settings shared by every job script. All of them can be given on the
# qsub command line, e.g.
#   qsub -v DATASET=mnist,N=800,TRAIN_LIMIT=20000,EVAL_LIMIT=2000 jobs/train_cpp.cmd
#
#   DATASET           mnist | fashion                          (default mnist)
#   N                 excitatory neurons                       (default 400)
#   EPOCHS            passes over the training images          (default 3)
#   TRAIN_LIMIT       unique training images, multiple of 10,
#                     equal per class; 0 = all available       (default 0)
#   VALIDATION_SIZE   training images held out for accuracy; 0 = none,
#                     accuracy is then measured on the TEST set (default 0)
#   EVAL_LIMIT        images used to measure accuracy (validation or test),
#                     multiple of 10; 0 = all                  (default 0)
#   ASSIGN_PER_CLASS  training images per class used to label neurons (default 500)
#   REQUEST           baseline | classK | classK_L_... | random10 (default baseline)
#                     classK: Gold model, trained from scratch without any
#                     training image of class K (also not used for labels);
#                     classK_L_...: same without several classes, in increasing
#                     order joined by _ (e.g. class6_9; a comma would split the
#                     qsub -v list); random10: without a fixed random 10% of
#                     each class
#   SEED              model seed                                (default 0)
#   WEIGHT_NORM       0 = off; e.g. 78: before every training presentation
#                     scale each neuron's input weights to sum to 78 (Diehl &
#                     Cook released code). Compiled CPU jobs only (default 0)
#   INPUT_TOTAL_HZ    0 = off; e.g. 6500: scale every image so its input rates
#                     sum to this (MNIST mean is about 6,500 Hz). Compiled CPU
#                     jobs only (default 0)
# GPU jobs only (train_gpu.cmd, pilot_gpu.cmd) run a second model on the node's
# second GPU when either of these is given:
#   N2                neurons of the second model           (default: same as N)
#   DATASET2          dataset of the second model     (default: same as DATASET)
# Image counts: MNIST and Fashion-MNIST have 60,000 training and 10,000 test
# images. Training pool = 60,000 - VALIDATION_SIZE. ASSIGN_PER_CLASS x 10 must
# fit inside the training pool (it is drawn from it).
# A job script may set its own defaults BEFORE sourcing this file.

settings_error() { echo "SETTINGS ERROR: $*" >&2; exit 2; }
check_dataset_name() {
    case "$1" in mnist|fashion) ;; *) settings_error "$2 must be mnist or fashion (got '$1')" ;; esac
}

DATASET="${DATASET:-mnist}"
check_dataset_name "$DATASET" DATASET
[ -n "${DATASET2:-}" ] && check_dataset_name "$DATASET2" DATASET2
N="${N:-400}"
EPOCHS="${EPOCHS:-3}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
VALIDATION_SIZE="${VALIDATION_SIZE:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
ASSIGN_PER_CLASS="${ASSIGN_PER_CLASS:-500}"
REQUEST="${REQUEST:-baseline}"
SEED="${SEED:-0}"
WEIGHT_NORM="${WEIGHT_NORM:-0}"
INPUT_TOTAL_HZ="${INPUT_TOTAL_HZ:-0}"
request_help="REQUEST must be baseline, random10, classK (K = 0..9) or several classes in increasing order joined by _, e.g. class6_9 (got '$REQUEST')"
if [[ "$REQUEST" =~ ^class[0-9](_[0-9])*$ ]]; then
    previous=-1; removed=0
    for class in ${REQUEST//[!0-9]/ }; do
        [ "$class" -gt "$previous" ] || settings_error "$request_help"
        previous=$class; removed=$((removed + 1))
    done
    [ "$removed" -le 9 ] || settings_error "$request_help"
elif [ "$REQUEST" != baseline ] && [ "$REQUEST" != random10 ]; then
    settings_error "$request_help"
fi

# Check the numbers now, with a plain message, instead of failing deep inside
# Python. Image subsets are balanced (equal per class), so they are limited by
# the rarest class: MNIST has 5,421 training / 892 test images of digit 5.
for var in N EPOCHS TRAIN_LIMIT VALIDATION_SIZE EVAL_LIMIT ASSIGN_PER_CLASS SEED WEIGHT_NORM INPUT_TOTAL_HZ; do
    [[ "${!var}" =~ ^[0-9]+$ ]] || settings_error "$var must be a whole number (got '${!var}')"
done
if [ -n "${N2:-}" ]; then
    [[ "$N2" =~ ^[0-9]+$ ]] && [ "$N2" -ge 1 ] || settings_error "N2 must be a whole number of at least 1 (got '$N2')"
fi
[ "$N" -ge 1 ] && [ "$EPOCHS" -ge 1 ] && [ "$ASSIGN_PER_CLASS" -ge 1 ] \
    || settings_error "N, EPOCHS and ASSIGN_PER_CLASS must be at least 1"
[ "$TRAIN_LIMIT" = 60000 ] && [ "$VALIDATION_SIZE" = 0 ] && TRAIN_LIMIT=0   # 60000 = all
[ "$EVAL_LIMIT" = 10000 ] && [ "$VALIDATION_SIZE" = 0 ] && EVAL_LIMIT=0     # 10000 = all
for var in TRAIN_LIMIT VALIDATION_SIZE EVAL_LIMIT; do
    [ $(( ${!var} % 10 )) -eq 0 ] || settings_error "$var must be a multiple of 10 (equal per class)"
done

check_dataset_limits() {  # $1 = dataset
    local ds="$1" min_train min_test pool per_class
    if [ "$ds" = mnist ]; then min_train=5421; min_test=892; else min_train=6000; min_test=1000; fi
    pool=$(( min_train - VALIDATION_SIZE / 10 ))   # per class left for training
    [ "$pool" -ge 1 ] || settings_error "VALIDATION_SIZE=$VALIDATION_SIZE is too large"
    if [ "$TRAIN_LIMIT" -gt 0 ]; then
        [ $(( TRAIN_LIMIT / 10 )) -le "$pool" ] || settings_error "TRAIN_LIMIT=$TRAIN_LIMIT needs $(( TRAIN_LIMIT / 10 )) per class; $ds has $pool per class available (max $(( pool * 10 )), or 0 = all)"
        per_class=$(( TRAIN_LIMIT / 10 ))
    else
        per_class=$pool
    fi
    [ "$ASSIGN_PER_CLASS" -le "$per_class" ] || settings_error "ASSIGN_PER_CLASS=$ASSIGN_PER_CLASS is more than the $per_class training images per class ($ds)"
    if [ "$VALIDATION_SIZE" = 0 ] && [ "$EVAL_LIMIT" -gt 0 ]; then
        [ $(( EVAL_LIMIT / 10 )) -le "$min_test" ] || settings_error "EVAL_LIMIT=$EVAL_LIMIT needs $(( EVAL_LIMIT / 10 )) test images per class; $ds has $min_test (max $(( min_test * 10 )), or 0 = all)"
    elif [ "$EVAL_LIMIT" -gt "$VALIDATION_SIZE" ]; then
        settings_error "EVAL_LIMIT=$EVAL_LIMIT is more than the $VALIDATION_SIZE held-out images"
    fi
}
check_dataset_limits "$DATASET"
[ -n "${DATASET2:-}" ] && check_dataset_limits "$DATASET2"

export FASHION_DATA="${FASHION_DATA:-${PBS_O_WORKDIR:?Submit with qsub from the project folder}/data/$DATASET}"
export FASHION_RESULTS="${FASHION_RESULTS:-$PBS_O_WORKDIR/results/$DATASET}"

# Run arguments that do not depend on the dataset (neuron count added per run).
run_args=(--request "$REQUEST" --seed "$SEED" --epochs "$EPOCHS"
          --validation-size "$VALIDATION_SIZE" --train-limit "$TRAIN_LIMIT"
          --eval-limit "$EVAL_LIMIT" --assign-per-class "$ASSIGN_PER_CLASS"
          --weight-norm "$WEIGHT_NORM" --input-total-hz "$INPUT_TOTAL_HZ")
# Full arguments for a run on DATASET.
common_args=(--dataset "$DATASET" --data-dir "$FASHION_DATA" "${run_args[@]}")

# Folder name describing a run; identical settings give an identical name.
run_name() {  # $1 = neuron count
    local train="$TRAIN_LIMIT" eval="$EVAL_LIMIT"
    [ "$train" = 0 ] && train=all
    [ "$eval" = 0 ] && eval=all
    local extra=""   # only non-default input settings, so older names stay unchanged
    [ "$WEIGHT_NORM" != 0 ] && extra+="_wn${WEIGHT_NORM}"
    [ "$INPUT_TOTAL_HZ" != 0 ] && extra+="_in${INPUT_TOTAL_HZ}"
    echo "n${1}_${REQUEST}_seed${SEED}_ep${EPOCHS}_train${train}_val${VALIDATION_SIZE}_eval${eval}_assign${ASSIGN_PER_CLASS}${extra}"
}

print_settings() {
    echo "Settings: DATASET=$DATASET N=$N${DATASET2:+ DATASET2=$DATASET2}${N2:+ N2=$N2}" \
         "EPOCHS=$EPOCHS TRAIN_LIMIT=$TRAIN_LIMIT VALIDATION_SIZE=$VALIDATION_SIZE" \
         "EVAL_LIMIT=$EVAL_LIMIT ASSIGN_PER_CLASS=$ASSIGN_PER_CLASS REQUEST=$REQUEST SEED=$SEED" \
         "WEIGHT_NORM=$WEIGHT_NORM INPUT_TOTAL_HZ=$INPUT_TOTAL_HZ"
}
