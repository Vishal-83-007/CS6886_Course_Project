# Running the SNN experiments on AQUA

## How AQUA works

- **Login node**: where `ssh` puts you. Copy files and submit jobs there; do not train there.
- **Compute nodes**: run your *job scripts* when PBS (the scheduler) has a free slot.
  Your home folder is shared, so jobs see the same files.
- **No internet** on AQUA: the bundle contains the Python packages (`wheelhouse/`)
  and both datasets (`data/mnist/`, `data/fashion/`).
- SSH port **40826**; off campus connect to the IITM VPN first.

| Command | Meaning |
| --- | --- |
| `qsub [-v SETTINGS] jobs/<script>` | submit a job; prints a job ID like `2407123.hn1` |
| `qstat -u ep23b042` | your jobs: `Q` waiting, `R` running, gone = finished |
| `qdel <job id>` | cancel a job |
| `bash aqua/progress.sh` | progress, status and accuracy of every run |

## Settings: choose dataset, network size and image counts

Every job takes its settings on the command line with `qsub -v`, comma-separated,
no spaces. Anything you leave out uses the default.

| Setting | Meaning | Default |
| --- | --- | --- |
| `DATASET` | `mnist` or `fashion` | `mnist` |
| `N` | excitatory neurons | `400` |
| `N2` | GPU jobs only: neurons of a second model, on the node's second GPU | same as `N` |
| `DATASET2` | GPU jobs only: dataset of the second model (`mnist` or `fashion`) | same as `DATASET` |
| `EPOCHS` | passes over the training images | `3` |
| `TRAIN_LIMIT` | unique training images (multiple of 10, equal per class); `0` = all | `0` |
| `VALIDATION_SIZE` | training images held out to measure accuracy instead of the test set; `0` = none, accuracy is measured on the **test** set | `0` |
| `EVAL_LIMIT` | test images used to measure accuracy (held-out images if `VALIDATION_SIZE` > 0), multiple of 10; `0` = all | `0` |
| `ASSIGN_PER_CLASS` | training images per class used to label neurons after training | `500` |
| `REQUEST` | `baseline`, or a gold model without data: `class0`, `class6`, `class9`, `random10` | `baseline` |
| `SEED` | model seed | `0` |

Both datasets have 60,000 training and 10,000 test images. Training images used =
`TRAIN_LIMIT` (or 60,000 - `VALIDATION_SIZE` if 0), each shown `EPOCHS` times.
`ASSIGN_PER_CLASS` x 10 must fit in the training images. `TRAIN_LIMIT=60000` and
`EVAL_LIMIT=10000` also mean "all". Other counts are split equally across the 10
classes, so they are limited by the rarest class: on MNIST at most 54,210 training
and 8,920 test images (Fashion-MNIST: 60,000 / 10,000). Impossible settings stop
the job immediately with a `SETTINGS ERROR` line in its output file.

Examples:
```
# MNIST, 400 neurons, all 60k training images x 3, accuracy on the 10k test set
qsub -v DATASET=mnist,N=400 jobs/train_cpp.cmd
# Fashion-MNIST, 800 neurons, 20k training images x 3, accuracy on 2k test images
qsub -v DATASET=fashion,N=800,TRAIN_LIMIT=20000,EVAL_LIMIT=2000 jobs/train_cpp.cmd
```

Each run gets a folder in `results/<dataset>/` whose name lists its settings, e.g.
`results/mnist/n400_baseline_seed0_ep3_trainall_val0_evalall_assign500_cpp_2431234/`.

## Job scripts

| Script | Backend | Notes |
| --- | --- | --- |
| `setup_cpu.cmd` | - | once: CPU environment `~/fashion-cpu` (skip if it already works) |
| `setup_gpu.cmd` | - | once: GPU environment `~/fashion-gpu` |
| `pilot_cpu.cmd` | Cython | short timing run (defaults 1,000 train / 100 eval images) |
| `train_cpu.cmd` | Cython | slowest; saves every 15 min and **resumes** after the 48 h limit |
| `train_cpp.cmd` | compiled CPU | ~10x faster at N=400, identical results; must finish in 48 h |
| `pilot_gpu.cmd` | GPU | required once before `train_gpu.cmd`; by default N=800 on MNIST and on Fashion-MNIST, one per GPU |
| `train_gpu.cmd` | GPU | must finish in 48 h; one model per GPU: `N` on `DATASET`, and a second if `N2`/`DATASET2` is given |

Measured locally (identical images, same accuracy and spike counts):

| s/image | N=400 | N=2400 |
| --- | --- | --- |
| `train_cpu.cmd` (Cython) | 0.85 | 1.26 |
| `train_cpp.cmd` (compiled CPU) | **0.08** | **0.42** |
| `train_gpu.cmd` (GPU) | not yet measured | not yet measured |

N=800, pilot settings (1,000 training images, 1 epoch, `ASSIGN_PER_CLASS=10`,
`EVAL_LIMIT=100`), seed 0:

| | MNIST | Fashion-MNIST |
| --- | --- | --- |
| compiled CPU, local: accuracy | 44% | 48% |
| compiled CPU, local: s/training image | 0.19 | 0.26 |
| GPU on AQUA, job 2461201 (frozen-RNG bug): accuracy | 24% | 10% |
| GPU on AQUA, job 2461201: s/training image | 0.34 | 0.32 |

A GPU pilot is correct only if its accuracy is close to the compiled-CPU numbers
(within about 10 points; 100 test images, and the GPU draws different random
spike trains).

## Step by step (WSL on your laptop, then AQUA)

1. Copy and unpack:
   ```
   scp -P 40826 ./CS6886_snn_aqua.tar.gz ep23b042@aqua.iitm.ac.in:~/
   ssh -p 40826 ep23b042@aqua.iitm.ac.in
   tar -xzf CS6886_snn_aqua.tar.gz
   cd ~/CS6886_snn_aqua
   ```
2. CPU environment: `~/fashion-cpu` from the earlier setup is reused as-is. If it
   does not exist yet: `qsub jobs/setup_cpu.cmd`; its log `results/setup_cpu_*.log`
   must end with `SETUP OK`.
3. Compiled-CPU run: `qsub -v DATASET=mnist,N=400 jobs/train_cpp.cmd`
4. GPU, once: `qsub jobs/setup_gpu.cmd`; log `results/setup_gpu_*.log` must end
   with `GPU SETUP OK`.
5. GPU pilot (800 neurons, MNIST on one GPU and Fashion-MNIST on the other):
   `qsub jobs/pilot_gpu.cmd`; afterwards `bash aqua/progress.sh` must show both
   `pilot_n800_..._gpu_...` runs (one under `mnist/`, one under `fashion/`)
   `complete`, with accuracies close to the compiled-CPU pilot table above.
   Seconds per image:
   `grep -h -A1 '"phase_seconds"' results/*/pilot_*gpu*/metrics.json`
   (first number / training images, 1,000 by default).
6. GPU run, both datasets at once, one per GPU:
   `qsub -v DATASET=mnist,DATASET2=fashion,N=800 jobs/train_gpu.cmd`
   (one dataset only: `qsub -v DATASET=mnist,N=800 jobs/train_gpu.cmd`)
7. Results to your laptop (WSL):
   `scp -P 40826 -r ep23b042@aqua.iitm.ac.in:~/CS6886_snn_aqua/results .`

Each run folder has `metrics.json` (accuracy, per-class accuracy, timings),
`baseline_weights.npz` (trained weights + thresholds), `neuron_labels.npy`,
`confusion_matrix.npy`, `responses.npz`; its log is the `.log` file next to it.

## Gold models (retrained without a class) for unlearning

A Gold model is trained from scratch, exactly like the baseline (same N,
epochs, seed, initial weights and image order), except that the removed
training images are never shown. `REQUEST` chooses what is removed:

| `REQUEST` | Removed from training and from neuron labelling |
| --- | --- |
| `classK` (K = 0 ... 9) | every training image of class K (MNIST: about 6,000; Fashion: 6,000) |
| `classK_L_...` | every training image of several classes, listed once each in increasing order and joined by `_`, e.g. `class6_9` (a comma would split the `qsub -v` list) |
| `random10` | a fixed random 10% of each class (all classes kept) |

Accuracy is measured on all 10,000 test images. For class requests,
`aqua/progress.sh` also prints the accuracy on the retained-class test images
and on the removed-class test images (1,000 per removed class). No neuron can
be labelled with a removed class (none of its images is used for labelling),
so the removed-class accuracy of a Gold model is 0% by construction: that is a read-out effect, not proof that the
features are gone (see `AQUA_TRAINING_PLAN.md`).

Examples, the same settings as the N=800 MNIST baseline:
`qsub -v DATASET=mnist,N=800,REQUEST=class3 jobs/train_cpp.cmd` (without digit 3),
`qsub -v DATASET=mnist,N=800,REQUEST=class6_9 jobs/train_cpp.cmd` (without 6 and 9).

## If something goes wrong

- Job output: `cat $(ls -t <scriptname>.o* | head -1)`, e.g. `train_cpp.o*`.
- `progress.sh` shows `ERROR`: read the `.log` file it names.
- `train_cpu.cmd` shows `checkpointed`: resubmit the same command.
- `qsub` rejects the queue: send the output of `qstat -Q`.

## What was changed from the Fashion-MNIST version

- `--dataset mnist|fashion` in `run_fashion.py`, `prepare_fashion.py` and
  `src/fashion_data.py` (official MNIST checksums; data folder `data/<dataset>`).
- `src/network.py`: neurons now use `exponential_euler`. Forward Euler with
  dt = 0.5 ms is unstable when inhibitory conductance gi > 400 (~24 inhibitory
  spikes in one step): v overshoots below -100 mV and back above threshold, so
  inhibition caused spikes. At N=2400 this locked all neurons at the maximum
  rate (chance accuracy). Also affects Fashion-MNIST runs: retrain those.
- `jobs/train_cpu.cmd`, `jobs/aqua_cpu_train.cmd`: empty-array expansion fixed for bash 4.2 (CentOS 7).
- `src/fashion_standalone.py`: CUDA copies of synaptic (growable) arrays used
  `dev_array_input_to_exc_w` / `..._eta`, which Brian2CUDA does not define
  (it stores them in `thrust::device_vector dev_dynamic_array_...`). The CUDA
  build failed to compile; fixed with `thrust::raw_pointer_cast`.
- Build folders (`src/fashion_standalone.py`, `src/fashion_runner.py`) are keyed by
  dataset as well as run name, so two GPU workers in one job (MNIST and
  Fashion-MNIST with identical settings) no longer collide.
- `src/fashion_standalone.py`, CUDA only: Brian2CUDA's random-number buffer is
  indexed by run call (0 = present, 1 = rest) and is never refilled or advanced
  from the third run on. Because the compiled loop repeats those two runs for
  every image, the Poisson input reused the same 784 random numbers at every
  time step of every later image (each pixel fired at every step or never):
  pilot job 2461201 reached 24% (MNIST) and 10% (Fashion) against 44% / 48% on
  the CPU. The generated `rand.cu` is now patched at build time to cycle the
  runs and refill from the continuing cuRAND stream each run; the build stops
  if that patch cannot be applied.
- `src/fashion_standalone.py`, CUDA only: Brian2CUDA's code that runs before
  every run appends to a global memory report and prints all of it each time,
  so output grows with the square of the number of runs; pilot job 2462420 did
  fewer than 100 images in its 2 h limit. The report is now reset every run and
  printed (with the INFO lines) on the first run only.
- `REQUEST` accepts `class0` ... `class9` (was only `class0`, `class6`,
  `class9`) and several classes such as `class6_9` (training, labelling and
  removed-class accuracy all use the whole set); single-class, baseline and
  random10 image selections are unchanged; `jobs/settings.sh` rejects anything else before the job starts;
  `aqua/progress.sh` prints retained- and removed-class accuracy for Gold runs.
- `DATASET2` (GPU jobs): second model on the other GPU may use another dataset.
- `--allow-long-cuda` (runner): lifts the 1,000-image CUDA pilot limit; used
  only by `jobs/train_gpu.cmd`.
- Job scripts: `jobs/settings.sh` (all run settings via `qsub -v`), `jobs/gpu_common.sh`,
  `setup_cpu/setup_gpu/pilot_cpu/train_cpu/train_cpp/pilot_gpu/train_gpu.cmd`.
- `wheelhouse/cp312/gpu/` (Brian2 2.8.0 + Brian2CUDA 1.0a7), `data/fashion/`.
- `aqua/progress.sh`, `aqua/check_aqua.*`, `data/mnist/`, `wheelhouse/cp312/cpu/`.
