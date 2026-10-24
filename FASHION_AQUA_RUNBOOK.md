# Fashion-MNIST on Aqua: compiled C++ CPU training

Updated 19 October 2026. All production models use Brian2 C++ standalone on CPU.
The accelerator execution path, dependency file and jobs have been removed.
Historical results and environment archives remain available for provenance.
Cython/NumPy remain reference backends; they are not launched by the production suite.

## Transfer the production package

The working CPU environment from setup and pilot 2417846 is reused unchanged.
No wheels, dataset or environment rebuild is needed on your existing Aqua checkout.
From the local project directory:

```bash
scp -P 40826 aqua_cpu_training_20261019.tar.gz \
  ep23b048@aqua.iitm.ac.in:/lfs/usrhome/btech/ep23b048/CS6886/
```

On Aqua:

```bash
cd /lfs/usrhome/btech/ep23b048/CS6886
tar -xzf aqua_cpu_training_20261019.tar.gz
bash scripts/apply_cpu_only_cleanup.sh
bash scripts/submit_aqua_cpu_suite.sh --dry-run
bash scripts/submit_aqua_cpu_suite.sh --submit
qstat -u "$USER"
```

The cleanup removes only five retired source files left by older tar extractions.
It does not remove data, results, wheelhouses, environments or previous bundles.
The submit command queues five independent seed-0 jobs. Run it once; another
invocation submits new jobs rather than resuming existing ones. A timestamped TSV
submission record is written under results/fashion. On partial submission failure,
check that record and qstat before submitting the remaining requests.

## What the jobs do

Each job requests one CPU core and 48h via jobs/aqua_cpp_train.cmd. Actual queue
routing/admission is controlled by Aqua. The observed small-resource limits
suggest up to five such running jobs, including other jobs; capacity is not
reserved. All five models are independent and can execute concurrently.
One-core-per-model throughput is measured; multicore scaling has not been selected.

| Request | Unique train images | Three-epoch presentations | Assignment images | Test images |
| --- | ---: | ---: | ---: | ---: |
| baseline | 60,000 | 180,000 | 5,000 | 10,000 |
| class6 | 54,000 | 162,000 | 4,500 | 10,000 |
| class0 | 54,000 | 162,000 | 4,500 | 10,000 |
| class9 | 54,000 | 162,000 | 4,500 | 10,000 |
| random10 | 54,000 | 162,000 | 5,000 | 10,000 |

N=400, model seed 0, split/deletion seed 42, three shuffled epochs, existing
STDP settings. Gold starts fresh and excludes deleted IDs from training and
assignment. It does not load baseline weights. Assignment is balanced among
allowed classes and uses training images. Assignment and test observation freeze
weights and adaptive thresholds. Class Gold evaluates all test classes, including
the removed class. Its zero removed-class accuracy is partly a readout constraint,
not evidence that an unlearning method worked. No unlearning method runs here.

This is the selected initial production configuration. Final test metrics must
not guide hyperparameter changes. To develop/tune instead, explicitly use
VALIDATION_SIZE=5000 for every paired run and later retrain the final suite.

## Submit selected models or later seeds

Start only the first pair if preferred:

```bash
REQUESTS="baseline class6" bash scripts/submit_aqua_cpu_suite.sh --submit
```

Submit the remaining seed-0 models without duplicating that pair:

```bash
REQUESTS="class0 class9 random10" bash scripts/submit_aqua_cpu_suite.sh --submit
```

After reviewing seed 0:

```bash
SEEDS="1 2" bash scripts/submit_aqua_cpu_suite.sh --submit
```

Individual submission is also supported:

```bash
qsub -v FASHION_ENV_KIND=cpu,REQUEST=baseline,SEED=0,VALIDATION_SIZE=0,EPOCHS=3 \
  jobs/aqua_cpp_train.cmd
```

Launcher overrides: REQUESTS (space-separated), SEEDS (space-separated), EPOCHS,
VALIDATION_SIZE, FASHION_DATA, FASHION_RESULTS and FASHION_ENV_ARCHIVES.
Default paths use the current project checkout. Do not mix configurations or
training splits across baseline and Gold. Build inputs and compilation stay in
job-local temporary storage; persistent artifacts stay in results/fashion.

## Monitor progress and collect results

```bash
qstat -u "$USER"
tail -n 50 results/fashion/train_cpp_JOBID.hn1.log
ls results/fashion/baseline_seed0_val0_cpp_JOBID.hn1/
```

The job log prints progress every 100 presentations. Output directories include
request, seed, validation size, backend and PBS job ID. Save the job's PBS output,
training log and complete model directory when collecting results locally.
Inspect status.json for complete and metrics.json for all counts, accuracy and
phase timing. Missing completion or a stale building status after PBS termination
requires inspection of PBS output; a scheduler kill cannot update Python status.

Each model saves final weights/theta, neuron labels, manifest/order and hashes,
assignment/evaluation spikes, predictions/class scores, confusion, per-class
metrics, retry/failure statistics and timings. Inference snapshots are saved
every 5,000 training presentations and at each epoch boundary. Earlier epoch
snapshots are raw float64 arrays; final weights are packaged in baseline_weights.npz
for both baseline and Gold (the filename is generic; config.json identifies the request).

## Runtime and restart limitation

The measured C++ baseline pilot projects approximately 10.5h for a final run,
or 12.6h with 20% headroom, excluding additional setup costs. Later-epoch rates
can differ, so this is a planning estimate, not a completion guarantee.
The job requests 48h; inspect rejection messages if site limits differ.

C++ snapshots are inference-only. They do not include all network/RNG state and
cannot resume interrupted training exactly. A failed C++ run must restart fresh;
resubmitting creates a separate output directory. Cython remains available via
jobs/aqua_cpu_train.cmd for exact-resume experiments, with its slower runtime.
It is not part of the production launcher.

## Fresh CPU setup, only when needed

The existing CPU archive and requirements are unchanged. If starting from a fresh
checkout, transfer checked data/fashion and wheelhouse/cp312/cpu separately. The
production source bundle intentionally excludes data/wheels/results.

Regenerate offline CPU wheels on an internet-enabled local machine if necessary:

```bash
python3 scripts/prepare_aqua_wheels.py --python-version 3.12
```

On Aqua, CPU-only setup uses the module Python 3.12.7, node-local installation,
--no-index, interpreter symlinks, bounded installation/packing, transfer checksums
and heartbeat logs:

```bash
qsub jobs/aqua_setup.cmd
```

After Setup completed, the common job loader verifies the archive and expands it
locally. Call Python modules via the relocated interpreter rather than venv
console-script shebangs. Keep exact Python/package versions for runtime resume.

## Validation and scope

Aqua CPU pilot 2417846 matched Cython spike responses/predictions exactly, with
maximum weight difference 2.78e-16 and about 4.85x faster simulation phases.
Both scored 43% on 100 validation images after only 1,000 training examples.
This does not promise final accuracy. Accelerator pilot 2422956 completed but
failed the behavior/performance acceptance comparison; accelerator work is closed.
See AQUA_TRAINING_PLAN.md for the scientific suite and remaining evaluation work.
