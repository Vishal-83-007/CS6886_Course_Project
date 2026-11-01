# Method C: exact CSV subset forgetting on AQUA

This runner reuses `src/method_c.py`'s existing compiled C1/C2 replay engine.
C1 reverses the baseline STDP increment; C2 applies depression only to its
positive component. There is no importance-estimation stage. Only the CSV
training images are replayed, once each per pass, shuffled by a separate
seed-2026 generator shared across trials. Every trial starts from the original
baseline weights and theta. Thresholds and inhibition stay fixed, normalization
is disabled, and plastic updates are disabled during the original 150 ms rest.
The existing Method A and whole-class Method C entry points are unchanged.

The supplied `train_index,label` CSVs use official training IDs, not positions
inside a split or replay sequence. Their sets match the shared
`numpy.random.default_rng(46)` class-6-then-class-9 selection: 1,185 MNIST
(591 sixes, 594 nines), 1,200 Fashion (600 Shirts, 600 Ankle boots).
The runner validates unique in-range IDs, official labels, checksummed data,
the baseline dataset fingerprint, and actual baseline training membership.
It uses the CSV directly; it does not regenerate or expand it to whole classes.
Other CSV sets with this header are supported; set a distinct `--request` name.
That name is metadata, not a selector. IDs are canonicalized into ascending
order before replay permutations are generated.

## Readout and evaluation

Forgotten images must never label neurons. The original baseline's saved
assignments may have used them, so this experiment explicitly changes the
readout protocol: frozen baseline responses to 500 retained training images
per class fit new labels using the existing mean-response/argmax rule (silent
neurons get -1). Selection uses seed 45 and excludes every CSV ID.
The resulting labels are fixed for every C1/C2 checkpoint. They are also used
for the unchanged baseline reference, providing a control for the readout
change. `readout_change.json` records how many labels changed. Gold gets an
independent assignment on the same retained IDs. No class output is suppressed;
6 and 9 remain eligible. The supplied baseline artifacts are never edited.

Default evaluation for each model covers:

- Every CSV forgotten training image.
- 100 retained training images per class, disjoint from assignment. Retained
  images from the requested classes are also reported separately.
- All 10,000 official test images, with separate requested-class and other-class
  metrics. Test sixes/nines are generalization probes, not forgotten members.

`--retain-per-class 0` evaluates all retained training images except assignment.
`--eval-limit 100` limits only test evaluation to a balanced diagnostic sample;
it never shrinks the forget/replay set. `--assign-per-class` and
`--retain-per-class` can reduce the readout and retain-evaluation cost for smoke
runs. Such runs are diagnostic and should not be mixed with full-run results.
Insufficient class examples cause an error instead of silently reducing counts.

Each model's evaluation is one continuous inference trajectory in order:
forget train, retain train, test. The original state/rest and retry protocol is
preserved. Models share initial seed/order; differing retries consume differing
random streams. No hard reset or exact per-image Poisson pairing is claimed.

`replay_manifest.npz` saves exact forget, retain, assignment, evaluation IDs and
every replay permutation in official training coordinates. Responses save
`dataset_ids` **and** `dataset_split`; train/test IDs occupy distinct namespaces.
`summary.json` includes per-group accuracy, per-class accuracy, confusion,
spike measures, and disagreements with baseline and optional Gold. Per-image
responses and replay update diagnostics are retained. The sweep is prespecified;
do not choose parameters using test or forgotten-image accuracy. Low forgotten
accuracy alone is not proof of successful unlearning, especially when a matched
subset Gold may classify these images correctly through generalization.

## Package and transfer

Build from any directory:

```bash
python scripts/package_method_c_subset.py
# Optional alternative with the existing Python 3.12 offline CPU wheels:
python scripts/package_method_c_subset.py --include-wheels --output method_c_subset_aqua_offline_cp312.tar.gz
```

The standard archive includes both CSVs, both PBS jobs, matching isolated source
modules, tests, pinned requirements, a fallback environment helper and SHA256
manifest. It contains no datasets, source checkpoints, or Gold weights.
Everything extracts inside `method_c_subset_runtime/`; it does not overwrite
existing jobs, training sources, or whole-class experiments.

Copy `method_c_subset_aqua.tar.gz` to AQUA's existing project root and extract:

```bash
cd ~/CS6886
tar -xzf method_c_subset_aqua.tar.gz
```

Keep the existing official data in `data/mnist` and `data/fashion`, baseline
checkpoint folders, `jobs/aqua_common.sh`, and CPU environment. Defaults use
the same 800-neuron baselines as the existing Method C jobs. The installed
environment helper is preferred so older archive-staging setups keep working;
a fallback helper is bundled for projects without one. All relative model/data
paths resolve from the project root. Do not submit from inside the runtime.

If a new environment is needed, the optional offline archive supports:

```bash
python3.12 -m venv ~/method-c-subset-cpu
~/method-c-subset-cpu/bin/python -m pip install --no-index \
  --find-links method_c_subset_runtime/wheelhouse \
  -r method_c_subset_runtime/requirements-aqua-cpu.txt
```

Then pass `FASHION_PYTHON=/absolute/path/to/method-c-subset-cpu/bin/python`
to `qsub -v`. A C++ compiler is required, as for existing jobs.

## Submit

Gold is **not selected automatically**. While subset Gold trains, prepare plans
or run baseline/C trials without it:

```bash
qsub -v PREPARE_ONLY=1 method_c_subset_runtime/jobs/mnist_method_c_subset.cmd
qsub -v PREPARE_ONLY=1 method_c_subset_runtime/jobs/fashion_method_c_subset.cmd

# Bounded diagnostic evaluation; still replay the entire fixed forget set.
qsub -v VARIANT=C2,SCALE=0.1,PASSES=1,EVAL_LIMIT=100,ASSIGN_PER_CLASS=10,RETAIN_PER_CLASS=10 method_c_subset_runtime/jobs/mnist_method_c_subset.cmd

# Full evaluation, snapshots after passes 1, 2 and 5.
qsub -v VARIANT=C1,SCALE=0.1 method_c_subset_runtime/jobs/mnist_method_c_subset.cmd
qsub -v VARIANT=C2,SCALE=0.1 method_c_subset_runtime/jobs/mnist_method_c_subset.cmd
qsub -v VARIANT=C1,SCALE=0.1 method_c_subset_runtime/jobs/fashion_method_c_subset.cmd
qsub -v VARIANT=C2,SCALE=0.1 method_c_subset_runtime/jobs/fashion_method_c_subset.cmd
```

Submit each desired scale separately (`0.1`, `0.5`, `1.0` match the old sweep).
Each job requests one core and 24 hours; completion time has not been benchmarked
for this experiment. Results go under `results/{dataset}_method_c_subset/` with
the request, variant, scale and PBS job ID. Available overrides include
`BASELINE_DIR`, `GOLD_DIR`, `FORGET_CSV`, `REQUEST`, `PASSES` (space-separated),
`VARIANT`, `SCALE`, `FASHION_DATA`, `FASHION_RESULTS`, `FASHION_PYTHON`,
`FASHION_ENV_ARCHIVES`, `ASSIGNMENT_SEED`, `INFERENCE_SEED`, `REPLAY_SEED`,
`ASSIGN_PER_CLASS`, `RETAIN_PER_CLASS`, `EVAL_LIMIT`, `PREPARE_ONLY`, and
`SKIP_EVALUATION`. `SKIP_EVALUATION=1` writes edited weights without a fitted
readout, clearly marked `readout_available=false`; it does not claim accuracy.

## Completed subset Gold

Once training finishes, supply `GOLD_DIR=/absolute/path/to/subset_gold` to a new
job. This runs replay/evaluation afresh; it does not resume or modify prior runs.
Do not point to the existing whole-class `class6_9` Gold.

Gold must have `config.json`, `baseline_weights.npz`, `neuron_labels.npy` and
`manifest.npz`, with request `sub6_9_p10` (or the exact custom request name).
Training settings, encoding and data fingerprint must match baseline.
The manifest must contain `train`, `forget`, `validation`, and `assignment`:
`train` must equal baseline training IDs minus the complete CSV set; `forget`
must equal the CSV IDs; validation must equal baseline validation; assignment
must be nonempty and entirely retained. Full-data runs therefore require
58,815 MNIST / 58,800 Fashion training IDs. Subset Gold neuron labels may
include classes 6 and 9. Membership checks document provenance but cannot
independently prove how a checkpoint was trained.

## Local use and validation

```bash
python run_method_c_subset.py --dataset mnist \
  --forget-csv forget_set_mnist_sub6_9_p10.csv \
  --prepare-only --out-dir results/mnist_subset_plan
python -m unittest test_method_c_subset test_method_c_fashion -v
python -m unittest test_method_c test_method_a -v
```

Preparation needs NumPy and official local data, not Brian2 or simulation.
The tests check malformed/wrong-label CSVs, actual training membership,
retain-only labels, unchanged labels across edited models, exact Gold matching,
member-versus-class metrics, the supplied seed-46 sets when local data exists,
and the existing compiled-vs-runtime event rules, frozen thresholds and C2
monotonicity. No long training or full 800-neuron sweep runs during validation.
