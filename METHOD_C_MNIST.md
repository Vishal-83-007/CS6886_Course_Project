# MNIST Method C on Aqua

Joint deletion request: **digits 6 and 9**. C1 reverses the baseline's raw
STDP increment; C2 negates only its positive part. Each variant/scale starts
from a fresh copy of the original 800-neuron baseline. Gold is used only in
evaluation. Existing training files, Method A, checkpoints and results are
not modified. No baseline or Gold retraining is performed.

## Transfer and run

Transfer `method_c_mnist_code.tar.gz` to Aqua's existing project directory.
From that directory:

```bash
tar -xzf method_c_mnist_code.tar.gz
qsub jobs/mnist_method_c.cmd
```

The default job runs C1, scale 0.1, saves passes 1, 2 and 5, and evaluates each
checkpoint, Baseline and joint Gold on all 10,000 test images. Results go to
`results/mnist_method_c/C1_s0.1_JOBID/`. Submit from the project root, where
`jobs/aqua_common.sh`, `data/mnist`, `mnist/` and `gold_weights/` already exist.
The job uses the same environment helper and environment archive location as
Method A. Set `FASHION_PYTHON` if the established environment has another path:

```bash
qsub -v FASHION_PYTHON=/absolute/path/to/environment/bin/python,VARIANT=C2,SCALE=0.5 jobs/mnist_method_c.cmd
```

Run a pilot first (two training images and ten test images total):

```bash
qsub -v VARIANT=C2,SCALE=0.1,REPLAY_PER_CLASS=1,EVAL_LIMIT=10 jobs/mnist_method_c.cmd
```

Submit the prescribed full sweep as six independent jobs:

```bash
for variant in C1 C2; do
    for scale in 0.1 0.5 1.0; do
        qsub -v VARIANT="$variant",SCALE="$scale" jobs/mnist_method_c.cmd
    done
done
```

Each job has its own outputs and scratch build directories. It compiles once
for replay and captures all three pass checkpoints from that trajectory.
Each evaluation has a fresh compiled simulation. Full replay has 11,867
images per pass (5,918 sixes and 5,949 nines). Allow several hours per job;
the requested 24-hour walltime is a starting allocation, not a measured
completion guarantee. Pilot runs deliberately use less data and are marked
in metadata. Scratch must have space for Brian's generated builds.

## Dependencies and isolation

The small archive includes all required project Python modules under
`method_c_runtime/`, the PBS job, documentation and the pinned CPU package
list. It reuses Aqua's existing Python 3.10–3.12 environment with Brian2 2.9.0,
NumPy and a C++ compiler/make. It does not download data or install packages
at job startup. Extracting it does not replace any existing `src/` module,
training job, environment helper or model file.

`method_c_mnist_offline_cp312.tar.gz` additionally contains the existing CPU
wheelhouse for **CPython 3.12, Linux x86_64**. Use this larger alternative only
if an offline environment must be created. Extract it instead of the small
archive, then use the Python 3.12 interpreter available on Aqua:

```bash
python3.12 -m venv "$HOME/method-c-cpu"
"$HOME/method-c-cpu/bin/python" -m pip install --no-index \
    --find-links method_c_runtime/wheelhouse \
    -r method_c_runtime/requirements-aqua-cpu.txt
qsub -v FASHION_PYTHON="$HOME/method-c-cpu/bin/python" jobs/mnist_method_c.cmd
```

Both archives contain `method_c_runtime/bundle_sha256.json`, mapping packaged
files to SHA256 hashes. The larger archive requires no new packages beyond
the baseline/Method A CPU dependency set.

## Direct execution and preparation

After activating the existing CPU environment and staying in the project root:

```bash
python method_c_runtime/run_method_c.py --prepare-only --out-dir results/mnist_method_c/plan
python method_c_runtime/run_method_c.py --variants C2 --scales 0.5 \
    --passes 1 2 5 --out-dir results/mnist_method_c/c2_s05
```

The Python default runs both variants and all three scales sequentially.
Use `--baseline-dir`, `--gold-dir` and `--data-dir` to override source locations.
The defaults are the same artifacts used by Method A:

- `mnist/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_cpp_2461985`
- `gold_weights/mnist_class6_9`
- `data/mnist`

`--prepare-only` verifies checkpoints, dataset fingerprint and source training
IDs, and records replay IDs/order without Brian simulation. `--skip-evaluation`
runs replay and saves checkpoints only. Existing output directories are
refused. `--adapt-thresholds` (PBS: `ADAPT_THRESHOLDS=1`) is the separately
named homeostasis ablation; both increment and decay are enabled during
replay, and its resulting thresholds are frozen during inference.

## Scientific protocol

The unchanged network builder supplies dt=0.5 ms, inhibition, membrane model,
STDP constants, connectivity and scheduling. The adapter extracts the raw
baseline event expression before clipping:

```
delta_base = eta * (pretrace - x_tar) * (w_max - w)**mu
C1: delta = -scale * delta_base
C2: delta = -scale * max(delta_base, 0)
w = clip(w + delta, 0, w_max)
```

Replay uses only forget images from the baseline's saved **training manifest**,
with original IDs and recorded seed/order. The current edited weights generate
spikes; weights update online at the original post-spike event. No retain
images or Gold weights enter this process. There is no normalization.

Each replay presentation lasts 350 ms at pixel/4 Hz (preserving optional saved
input-total scaling). There are **no rate boosts or retries**, even for silent
inputs. It is followed by the baseline's original continuous 150 ms zero-input
rest; no new hard resets are introduced. Membranes, conductances, refractory
state and event-driven traces relax/evolve under this existing protocol.
Erasure updates are explicitly disabled during rest; traces remain active.
This exposure policy differs from baseline training, which allows retries
and keeps learning enabled during rest. Primary thresholds remain exactly
at saved baseline values throughout replay.

Evaluation reuses Method A's frozen compiled CPU inference, including its
original retry policy, fixed initial seed, fresh simulator and original saved
mean-per-label voting. Retried inputs can consume different RNG stream lengths
across models, so the initial seed does not imply matched per-image Poisson
trains. No labels are deleted from the unlearned readout.

The actual baseline has `validation_size=0` and trained on all 60,000 training
images. These runs are a **prespecified sweep with test reporting, not a
validation-selected optimum**. Do not pick scale/pass by these test results.
If using a compatible baseline with an existing held-out validation manifest,
the runner evaluates that split instead. A new split of this baseline's
training images would not be independent validation.

## Outputs and checks

- `config.json`: source/code hashes, seeds, protocol, split and pilot flag.
- `replay_manifest.npz`: original forget IDs, labels, every pass's replay order,
  and fixed evaluation IDs.
- `baseline/`, `gold/`: frozen inference metrics, confusion matrices and responses.
- `C1_s0.1/pass_1/` (and other variant/scale/pass paths): standalone NPZ weights,
  unchanged neuron labels, config, intervention statistics, and evaluation.
- `presentations.npz`: per-presentation original dataset ID, active/rest spikes,
  positive/negative effective updates and out-of-bounds proposals clipped.
- `summary.json`: retain/forget accuracy and firing, disagreement with Baseline
  and Gold, total-spike response differences, weight changes, silent
  presentations, and final-pass fraction with no applied updates. Zero-update
  fractions indicate possible stalling; silent images are not retried.
- `status.json`: preparation, replay/evaluation progress, completion or failure.

Every snapshot is checked for legal weights, finite thresholds and exact frozen
theta preservation. C2 is checked for no positive event updates and no weight
increases between saved checkpoints. Source checkpoint files are hashed and
checked again on completion/failure. Weight arrays retain receptive fields for
inspection; response files retain full per-neuron spike counts. Gold neurons
are independently trained, so response comparisons use total firing rather
than treating their neuron indices as matched representations. Zero forget
accuracy alone is not evidence of erasure.

Run the included scientific tests from the isolated runtime directory:

```bash
cd method_c_runtime
python -m unittest test_method_c -v
```

Tests exercise positive/negative baseline events, C1/C2 transformations,
clipping, frozen thresholds, rest without updates, and compiled-versus-runtime
agreement across two passes with the same seeds. Tests use small synthetic
networks; a full MNIST/Aqua sweep is a separate experiment.

To rebuild bundles from the development checkout:

```bash
python scripts/package_method_c.py
python scripts/package_method_c.py --include-wheels --output method_c_mnist_offline_cp312.tar.gz
```

## Local validation completed

Validated using Python 3.12, Brian2 2.9.0 and NumPy 1.26.4 installed from the
existing CPU wheelhouse without network access. The entire pinned CPU
requirements file also installs from those wheels. All four scientific tests
pass, including C++ versus runtime replay and the threshold-adaptation ablation.

An end-to-end C2 pilot with the actual saved 800-neuron baseline and joint Gold,
scale 0.1, one pass, one forget image per class and ten balanced test images
completed replay, checkpoint writing and all three model evaluations. This
checks integration only; it is not an estimate of unlearning effectiveness.
Preparation with the complete original manifest verified all 11,867 forget
images. Python syntax and PBS shell syntax were checked. The full sweep has
not been run locally or submitted to Aqua.
