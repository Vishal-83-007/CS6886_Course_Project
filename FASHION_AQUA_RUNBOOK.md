# Fashion-MNIST: execute on Aqua

The uploaded logs establish two running GPU jobs per generic user, 48h GPU
walltime, 32GB V100s, driver 470.57.02 and available CUDA 11.4. The default CUDA
is 10.1. Both GPUs appeared in a one-GPU allocation with no mask; our pilot
requests both and binds two separate workers explicitly.

## 1. Offline Python/package setup (fix for job 2406770)

The compute job could not resolve `conda.anaconda.org`. Do not retry the old
Conda-create setup: compute nodes cannot currently reach that package server.
The user confirmed `module load anaconda3_2024.10` provides Python 3.12.7, and
the job reports glibc 2.17. Reuse that existing interpreter with isolated venvs;
no Conda solve or Python download is needed.

The prepared bundle lives in ignored `wheelhouse/cp312/`. To regenerate it on
an internet-enabled LOCAL machine (any local Python with modern pip):

```bash
python3 scripts/prepare_aqua_wheels.py --python-version 3.12
```

This requests only Linux x86_64/glibc-2.17-compatible wheels, includes pip and
wheel for bootstrap, and writes SHA256 readiness manifests. CPU and GPU packages
are resolved separately because their Brian versions differ.

A ready-to-transfer archive is provided as `aqua_offline_setup_cp312.tar.gz`.
It contains updated source/setup scripts, both wheel sets and the checked dataset;
no Git metadata or training results. From the local inner repository directory:

```bash
scp -P 40826 aqua_offline_setup_cp312.tar.gz ep23b048@aqua.iitm.ac.in:/lfs/usrhome/btech/ep23b048/CS6886/
```

Then on Aqua:

```bash
cd /lfs/usrhome/btech/ep23b048/CS6886
tar -xzf aqua_offline_setup_cp312.tar.gz
qsub jobs/aqua_setup.cmd
```

Alternatively, transfer the updated source, requirements, `wheelhouse/` AND `data/fashion/`
to Aqua. From the LOCAL inner repository directory, using scp:

```bash
scp -P 40826 -r scripts jobs wheelhouse data requirements-aqua-cpu.txt requirements-aqua-gpu.txt \
  ep23b048@aqua.iitm.ac.in:/lfs/usrhome/btech/ep23b048/CS6886/
```

The complete-source transfer command below is also suitable if other code has
not yet been copied. Then from the Aqua project directory:

```bash
qsub jobs/aqua_setup.cmd
```

Setup loads the confirmed module, checks wheel hashes/requirements/Python minor
version, creates `$HOME/fashion-cpu` and `$HOME/fashion-gpu` with module Python
3.12.7, and installs with `--no-index` from local wheels. It does not modify the
Anaconda module environment. No compute-node package-server access is needed.
It finishes with dependency/import/data checks and writes
`results/fashion/setup_<job-id>.log`. Use environments only after success.

If a complete environment of a DIFFERENT Python minor version already exists at
these paths, setup stops without deleting it. A failed Conda-create attempt
usually left no interpreter, in which case the venv can be created there.
Exact training resume still requires the original exact Python/package versions.
Both wheel sets were installed locally without index access into fresh Python
3.12 environments, with passing dependency/import checks. This is not a test on
Aqua itself; run its pilots after setup to check the actual compiler/GPU toolchain.

Brian2CUDA 1.0a7 requires Brian2 2.8.0; CPU uses Brian2 2.9.0. Keep them separate.
CUDA controller execution remains unvalidated on Aqua; use the bounded pilot.
No installation or qsub command has been executed remotely by this assistant.

## 2. Stage checked data

The official gzip files have been downloaded and verified locally. Copy this
repository INCLUDING the ignored `data/fashion/` directory to Aqua. If transferring
source through Git, transfer data separately. Example from this repo directory,
using the account/path visible in the uploaded diagnostic:

```bash
rsync -av -e 'ssh -p 40826' --exclude=.git --exclude=results --exclude=aqua_diagnostics \
  ./ ep23b048@aqua.iitm.ac.in:/lfs/usrhome/btech/ep23b048/CS6886/
```

Alternatively, on any machine with Python/NumPy and internet:

```bash
python prepare_fashion.py --download --data-dir data/fashion
```

Then transfer those files. Compute jobs load offline and verify all four official
checksums on each run. No TensorFlow/PyTorch download fallback is needed.

## 3. Submit the first work

From `/lfs/usrhome/btech/ep23b048/CS6886` on Aqua:

```bash
qsub -v FASHION_PYTHON=$HOME/fashion-cpu/bin/python jobs/aqua_cpu_pilot.cmd
qsub -v FASHION_PYTHON=$HOME/fashion-gpu/bin/python jobs/aqua_gpu_pilot.cmd
qstat -u "$USER"
```

CPU pilot: N=400, 1,000 training images, 100 assignment and 100 validation images;
compares Cython and compiled CPU sequentially. GPU pilot requests TWO GPUs on
ONE node, uses one worker each for baseline and class6 Gold, and logs each
separately. These are timing/semantic pilots, not final accuracy measurements.
The runner rejects long CUDA runs until this path is validated. No nvcc/GPU
execution has occurred locally. If the two-GPU request is not admitted, inspect
the PBS reason; do not select an arbitrary visible GPU in a one-GPU allocation.

Logs and version captures are under `results/fashion/`; CPU/GPU run dirs include
PBS job IDs. Optional `FASHION_DATA`, `FASHION_RESULTS` and `FASHION_WORK` can be
passed via `qsub -v`. Paths must be accessible on compute nodes. Input binaries and build/cache I/O are isolated per run under the job TMPDIR
(or node /tmp by default); override FASHION_WORK for site-approved scratch.
Checkpoints, inference snapshots, logs and final metrics stay in persistent results.

Inspect `metrics.json`: phase seconds, attempt histogram, exhausted retries,
unassigned neurons, silence/ties, accuracy and class confusion. Compare against
Cython before enabling long compiled/GPU experiments. Do not infer convergence
from a 1,000-image pilot. Include compilation separately when projecting time.

## 4. Production CPU execution (available now)

Exact runtime checkpoint/resume with periodic persistent saves:

```bash
qsub -v FASHION_PYTHON=$HOME/fashion-cpu/bin/python,REQUEST=baseline,SEED=0 \
  jobs/aqua_cpu_train.cmd
qsub -v FASHION_PYTHON=$HOME/fashion-cpu/bin/python,REQUEST=class6,SEED=0 \
  jobs/aqua_cpu_train.cmd
```

Defaults are development runs: 55k train/5k validation, three epochs, 500
assignment examples per available class, no test-set tuning. Gold uses fresh
initialization and excludes forget IDs from learning and assignment.

Runtime exits 75 after checkpointing at the 46h budget (48h allocation).
Resubmit exactly the same request/seed/validation parameters to resume. Keep
the original source/config and exact Python/Brian/NumPy/Cython versions; mismatches
are rejected. Resume only checkpoints created by your own run: Brian stores
Python pickle state. Two generations are kept; interruption during publication
leaves the previous CURRENT generation intact.

Compiled CPU production, ONLY if the pilot predicts completion within 48h:

```bash
qsub -v FASHION_PYTHON=$HOME/fashion-cpu/bin/python,REQUEST=baseline,SEED=0 \
  jobs/aqua_cpp_train.cmd
```

Run its paired Gold models with the same backend/configuration. C++ snapshots
contain inference weights/thresholds every 5,000 examples and every epoch;
they cannot resume exact training. A killed standalone process may retain raw
snapshots but not final packaged metrics. Use runtime Cython if exact continuation
is needed. No automatic resubmission is performed.

After choosing and locking configuration, add `VALIDATION_SIZE=0` for official
60k training / 10k test runs. Repeat requests baseline/class6/class0/class9/random10
and seeds 0/1/2 as budget allows. One baseline is shared across all requests for
a seed. Do not compare a development baseline with a final-data Gold.

## Outputs and validation limits

Each run saves config/source/data/schedule hashes, environment versions, immutable
manifest/order, inference weights and thresholds, neuron labels, assignment/test
responses, class scores, predictions, confusion, per-class/retain/forget accuracy,
and timings. Assignment/test weights are frozen; unassigned neurons use -1.
With class Gold, zero removed-class accuracy is partly a readout constraint and
is not proof of synaptic erasure. Random deletion may retain high accuracy.

Runtime resume was tested bit for bit, including responses and final weights.
Compiled CPU control was tested for advancing time across loop iterations,
finite weights and unchanged weights/thresholds during frozen observation.
Real Fashion-MNIST pilot results are recorded in FASHION_LOCAL_VALIDATION.md
separately from synthetic checks.
CUDA and actual Aqua throughput remain unmeasured; return pilot logs for the
next execution decision. No remote SSH credentials/session are available here,
so these qsub commands must be run on Aqua by the user.

## SSH port and outbound network access

The user and IITM's access guide confirm SSH port 40826:
https://hpce.iitm.ac.in/content.php?navigate=supercomputing
Use ssh -p 40826, scp -P 40826, or rsync -e 'ssh -p 40826'.
This controls inbound SSH/file transfer; it does not configure outbound
DNS/HTTPS from a compute job. Job 2406770 proves that its node could not resolve
conda.anaconda.org at that time, not that all internet access is prohibited.
No public Aqua-specific outbound proxy/package-mirror configuration was found
in the accessible guidance. Some linked Google Docs could not be read.
IITM's Remote Proxy Setup page concerns browser access to internal sites/journals
from outside campus, so it should not be used as an Aqua pip/Conda proxy recipe.
Ask hpce@iitm.ac.in for the supported compute-node DNS/HTTPS/proxy/mirror policy.
Continue with offline setup while that is clarified; no training-script change
is required. The archive already transferred remains usable.
