# Machine Unlearning using SNNs

**CS6886 course project — Vishal Rajesh (EP23B042), Aditya Girish (EP23B048).**
Post-hoc machine unlearning in a Diehl & Cook (2015) spiking network trained with
unsupervised STDP: remove classes 6 and 9 (MNIST digits 6/9; Fashion-MNIST Shirt
and Ankle boot), and a random 10% of their training images, without retraining,
using only local plasticity information. Every method is compared with a **Gold**
model retrained from scratch without the forgotten data.

| | Method | Docs |
|---|---|---|
| A | Label-guided neuron silencing | [METHOD_A_MNIST.md](METHOD_A_MNIST.md), [METHOD_A_FASHION.md](METHOD_A_FASHION.md) |
| B | Selective Synaptic Dampening with hypothetical-STDP importance | [METHOD_B_MNIST.md](METHOD_B_MNIST.md), [METHOD_B_FASHION.md](METHOD_B_FASHION.md), [METHOD_B_SUBSET.md](METHOD_B_SUBSET.md) |
| C | Anti-STDP (C1) / LTD-only (C2) replay of the forget set | [METHOD_C_MNIST.md](METHOD_C_MNIST.md), [METHOD_C_FASHION.md](METHOD_C_FASHION.md), [METHOD_C_SUBSET.md](METHOD_C_SUBSET.md) |
| D | Request-gated leak with retained-data replay | [METHOD_D_MNIST.md](METHOD_D_MNIST.md), [METHOD_D_FASHION.md](METHOD_D_FASHION.md) |

The design of all four is in [Unlearning-Approaches-Implementation.md](Unlearning-Approaches-Implementation.md);
Method B/D results are in [RESULTS_B_D.md](RESULTS_B_D.md); the report is [REPORT.pdf](REPORT.pdf).

## Reproducing the results

1. **Check out and install** (Python 3.10–3.12, a C++ compiler such as `g++`):

   ```bash
   git clone https://github.com/Vishal-83-007/CS6886_Course_Project.git
   cd CS6886_Course_Project
   pip install -r requirements-aqua-cpu.txt
   ```

   On a cluster without internet (we used IITM AQUA, PBS) follow
   [AQUA_RUNBOOK.md](AQUA_RUNBOOK.md), which installs from an offline wheelhouse.

2. **Data** (official IDX files, checksum-verified; `data/` is not in git):

   ```bash
   python prepare_fashion.py --dataset mnist --download
   python prepare_fashion.py --dataset fashion --download
   ```

3. **Baselines and Gold models** (N = 800, 3 epochs over all 60,000 images,
   compiled C++ backend; about 17 h (MNIST) / 21 h (Fashion) on one CPU core):

   ```bash
   python run_fashion.py --dataset mnist --n-excitatory 800 --validation-size 0 --backend cpp \
     --out-dir results/mnist/baseline                                   # 89.37% test accuracy
   python run_fashion.py --dataset mnist --n-excitatory 800 --validation-size 0 --backend cpp \
     --request class6_9 --out-dir results/mnist/gold_class6_9           # Gold: 6 and 9 removed
   python run_fashion.py --dataset fashion --n-excitatory 800 --validation-size 0 --backend cpp \
     --weight-norm 39 --out-dir results/fashion/baseline                # 74.28%
   python run_fashion.py --dataset fashion --n-excitatory 800 --validation-size 0 --backend cpp \
     --weight-norm 39 --request class6_9 --out-dir results/fashion/gold_class6_9
   ```

   `--request sub6_9_p10` trains the Gold for random-subset forgetting. On AQUA:
   `qsub -v DATASET=mnist,N=800,REQUEST=class6_9 jobs/train_cpp.cmd` (Fashion: add
   `WEIGHT_NORM=39`). Trained weights are not stored in git (`*.npz` is ignored);
   the A/C runners expect them under `gold_weights/` as described below.

4. **Unlearning:** Method A/C commands are in the next section; Method B and D:

   ```bash
   python run_frozen.py importance --dataset mnist \
     --weights results/mnist/baseline/baseline_weights.npz --out-dir results/mnist/importance   # I(D), ~7 h
   python run_method_b.py --dataset mnist --baseline-dir results/mnist/baseline \
     --importance results/mnist/importance/importance.npz --alpha 2 --lam 0.5 \
     --gold-dir results/mnist/gold_class6_9 --out-dir results/mnist_method_b
   python run_method_d.py --dataset mnist --baseline-dir results/mnist/baseline \
     --gate neuron --leak 0.001 --replay-images 5000 \
     --gold-dir results/mnist/gold_class6_9 --out-dir results/mnist_method_d
   ```

   Each run writes `summary.json` and, per evaluation, `metrics.json` with the
   accuracies reported in [RESULTS_B_D.md](RESULTS_B_D.md). The hyper-parameter
   sweeps, the relapse test and the subset experiment use the PBS scripts in
   `jobs/` (`ssd_sweep.sh`, `leak_sweep.sh`, `repair_sweep.sh`, `subset_sweep.sh`);
   `bash aqua/progress.sh` lists every result.

5. **Tests:** `python -m unittest test_method_a test_method_c test_method_b test_method_d -v`.


## Running Method A and Method C from an external baseline

The local implementation includes Method A (label-guided neuron silencing),
Method C (C1/C2 replay), and CSV-subset Method C on MNIST and Fashion-MNIST.
The older baseline-development notes below describe the initial training work.
See [METHOD_A_MNIST.md](METHOD_A_MNIST.md),
[METHOD_A_FASHION.md](METHOD_A_FASHION.md),
[METHOD_C_MNIST.md](METHOD_C_MNIST.md),
[METHOD_C_FASHION.md](METHOD_C_FASHION.md), and
[METHOD_C_SUBSET.md](METHOD_C_SUBSET.md) for the individual protocols.

Use Python 3.10–3.12 with `requirements-aqua-cpu.txt` and a C++ compiler for
compiled replay/evaluation. Run commands from the repository root. Model
weights and datasets are supplied separately; they are not installed by pip.

Pass the **baseline directory**, not a single weights file, using
`--baseline-dir /absolute/path/to/baseline`. The directory must contain:

```text
baseline/
  config.json             # original training settings, dataset fingerprint, request=baseline
  baseline_weights.npz    # weights, thresholds, and connectivity
  neuron_labels.npy       # saved original neuron-to-class assignments
  manifest.npz            # original training/validation IDs; required by Method C
```

These runners expect the checkpoint format produced by `run_fashion.py`
(which supports both datasets). An isolated weights array or the older
`run_baseline.py` output without config/manifest is insufficient. The checkpoint
must match the chosen dataset. Paths may be absolute or relative to the current
working directory. Always choose a new output directory outside the baseline
and Gold directories.

Both Fashion entry points and their PBS jobs default to `class6_9` (shirt and
ankle boot); MNIST defaults to digits 6 and 9. Use `--request class6` explicitly
for shirt-only deletion.

### Gold is optional comparison data

Gold is an independently retrained reference, not an input to the silencing or
replay update rule. The full runners also evaluate the baseline and edited
model, and evaluate Gold / record disagreements when Gold is supplied.
For historical joint-deletion workflows, omitting both Gold options selects
`gold_weights/mnist_class6_9` or `gold_weights/fashion_class6_9_wn39` automatically.
**Pass `--no-gold` when those comparison artifacts are unavailable.**
Alternatively, pass `--gold-dir /absolute/path/to/matched_gold`; it must match the
baseline training settings and exact deletion request. `--gold-dir` and
`--no-gold` are mutually exclusive. Gold is not trained by these commands.

Method A can prepare its intervention using only the baseline directory and
request; this does not run inference or need image data:

```bash
python3 run_method_a_fashion.py \
  --baseline-dir /absolute/path/to/fashion_baseline \
  --request class6_9 --no-gold --prepare-only \
  --out-dir results/fashion_a_prepared
```

The saved Method A checkpoint includes `intervention.npz`. The inference adapter
must apply it; using the copied weights alone does not silence neurons.

Method A evaluation also requires the official dataset. Method C unlearning
requires the original training manifest and forget images for replay, in addition
to the baseline. Its current runner loads and validates the full official dataset
even with `--skip-evaluation`; that flag skips accuracy evaluation, not data loading.
Place the official checksum-verified IDX gzip files in the selected data directory.
The runners do not download them implicitly.

Examples without Gold (the first command evaluates baseline/Method A; the second
performs replay and saves edited checkpoints without accuracy evaluation):

```bash
python3 run_method_a_fashion.py \
  --baseline-dir /absolute/path/to/fashion_baseline \
  --data-dir /absolute/path/to/data/fashion \
  --request class6_9 --no-gold \
  --out-dir results/fashion_a_evaluated

python3 run_method_c_fashion.py \
  --baseline-dir /absolute/path/to/fashion_baseline \
  --data-dir /absolute/path/to/data/fashion \
  --request class6_9 --no-gold --variants C2 --scales 0.5 --passes 1 2 5 \
  --skip-evaluation --out-dir results/fashion_c_checkpoints
```

Omit `--skip-evaluation` in Method C to evaluate the baseline and edited models.
For MNIST use `run_method_a.py` / `run_method_c.py` with MNIST checkpoint/data
paths. To include Gold in either method, replace `--no-gold` with
`--gold-dir /absolute/path/to/matched_gold`.
For CSV-subset forgetting use `run_method_c_subset.py` and its separate guide;
whole-class Gold is not a matched reference for a subset request.

On Aqua, the four whole-class A/C jobs accept
`BASELINE_DIR=/absolute/path/to/baseline`, `FASHION_DATA=/absolute/path/to/data`,
`REQUEST=class6_9`, and `NO_GOLD=1` through `qsub -v`. For a comparison supply
`GOLD_DIR` and leave `NO_GOLD` unset. These jobs prefer an extracted isolated
runtime when present: update that runtime after source edits using the packaging
script, or it may execute older code despite an updated repository `src/`.
Existing transfer archives are snapshots and are not regenerated automatically.


For **Fashion-MNIST Method A unlearning**, including shirt-only deletion and
shirt/ankle-boot joint deletion on Aqua, see [METHOD_A_FASHION.md](METHOD_A_FASHION.md).
For Fashion-MNIST C1/C2 replay, see [METHOD_C_FASHION.md](METHOD_C_FASHION.md).

For the Fashion-MNIST/Aqua workflow, start with
[FASHION_AQUA_RUNBOOK.md](FASHION_AQUA_RUNBOOK.md). It documents the new
`run_fashion.py` runner, reproducible baseline/Gold splits, CPU checkpoint/resume,
compiled CPU execution, and bounded CUDA pilots. The older MNIST instructions
below describe `run_baseline.py` and remain separate.

Course project: importance-based machine unlearning on a Spiking Neural
Network trained for MNIST digit recognition. The initial baseline SNN was trained with standard STDP. Method A/C
unlearning is now implemented as described above; the following sections retain
the original baseline-development context.

## Why this architecture / learning rule

The SNN architecture and the STDP learning methodology follow
**Diehl & Cook (2015), "Unsupervised learning of digit recognition using
spike-timing-dependent plasticity"** — this is also the exact baseline
the ASP paper (Panda et al.) compares itself against, using, in their
words, "identical parameters for neuron and synapse models, homeostasis,
input encoding and input image presentation time as Diehl & Cook."

**We deliberately do NOT use ASP's continuous weight-decay mechanism**
during baseline training. ASP's decay is a *training-time* forgetting
mechanism for sequential/non-stationary data streams. Our unlearning is a
*post-hoc* operation on an already-trained, stable network. Training with
ASP's decay would mean the network is already continuously forgetting
during training, which confounds any analysis of what the unlearning
step later removes. We need a clean, well-understood, stable baseline —
that's plain Diehl & Cook STDP.

### What we take from each paper

**From Diehl & Cook — used:**
- Conductance-based LIF neuron model
- 3-layer architecture: Input (784, Poisson) → Excitatory (N) → Inhibitory (N)
- Connectivity: input→exc all-to-all (plastic), exc→inh one-to-one (fixed),
  inh→exc all-to-all-except-self (fixed)
- Power-law weight-dependent STDP (their Eqn. 3), presynaptic trace +
  target-trace depression offset, updates triggered on postsynaptic spikes only
- Adaptive threshold homeostasis (theta)
- Poisson input encoding (0–63.75 Hz) with re-presentation/rate-boost if
  too few spikes are fired
- Post-hoc neuron→class label assignment (labels never touch training)

**From Diehl & Cook — ignored:**
- The other 3 STDP rule variants they benchmark (exponential weight-dependence,
  symmetric pre-and-post, triplet rule) — we use only the power-law rule,
  matching what the ASP paper itself uses as its STDP baseline
- Divisive weight normalization (only relevant to the triplet rule)

**From ASP — ignored (for this baseline):**
- The entire ASP learning rule: three-trace recovery phase (Pre_rec, Pre_acc,
  Post), significance-driven weight updates, time-dependent learning rate
- The decay/leak phase (exponential or linear), and its modulated
  time constant tied to postsynaptic trace + homeostatic threshold + weight
- Sequential/dynamic-environment training capability
- The denoising behavior — this is an *emergent property* of the weight
  decay mechanism (noise-pixel weights never get replenished by STDP and
  simply leak away); it is not separable from the decay mechanism, so it
  cannot be "borrowed" without also bringing in the decay. It may be worth
  revisiting empirically once unlearning is implemented (see Future Work).

## Repo structure

```
snn-unlearning/
├── src/
│   ├── network.py     # Brian2 network: neurons, synapses, STDP rule
│   ├── encoding.py     # Poisson input encoding + re-presentation logic
│   ├── data.py          # MNIST loading (fallback chain) + balanced subsetting
│   ├── train.py          # Unsupervised STDP training + neuron label assignment
│   ├── evaluate.py        # Test-set accuracy evaluation
│   └── utils.py             # Receptive-field plotting, weight save/load
├── run_baseline.py            # Main entry point — run this on Kaggle
├── sanity_check.py             # Synthetic-data validation of the pipeline (see below)
├── requirements.txt
└── results/                     # Output dir (weights, plots, accuracy) - gitignored
```

## Running on Kaggle

1. Create a new Kaggle Notebook, **enable internet** (Settings → Internet: On)
2. Upload this repo (or `git clone` it in a cell if the repo is public)
3. `!pip install -q -r requirements.txt`
4. Run a short timing pilot, then choose a training count for your CPU:

   ```bash
   python run_baseline.py --n-train 1000 --n-assign 200 --n-test 200 --out-dir results/pilot
   python run_baseline.py --n-train 30000 --n-assign 2000 --n-test 1000 --out-dir results/baseline_seed0
   ```

   Run `python run_baseline.py --help` for all options. Use separate output
   directories for each seed.

No GPU needed or used — see "CPU vs GPU" below.

### Time budget

The default is 400 neurons, 30,000 training presentations, 2,000 label
assignment presentations, and 1,000 test presentations. This is a budget
starting point, not a promised accuracy. Diehl & Cook reported 87% for
400 neurons after 180,000 presentations (three full MNIST passes). A
30,000-image run can be substantially worse. For a 24-hour allocation,
measure a 1,000-image pilot on the target CPU and estimate roughly
`(N_TRAIN + N_ASSIGN + N_TEST) * measured_seconds_per_image`, allowing
extra time for setup, retries, plotting, and changing spike activity.
If the estimate exceeds your budget, reduce the sample counts.

## CPU vs GPU

**This version runs on CPU.** The training script selects Brian2's compiled
Cython backend by default, so install a C++ compiler as well as the Python
dependencies. If compilation fails, fix the compiler setup before a long run;
`--backend numpy` is available for debugging but will generally be slower.
Check the measured throughput on your target CPU.

Brian2CUDA exists, but the per-image retry and immediate spike-count decision
in this script would require a larger refactor to use it efficiently. CPU
profiling is the first step; GPU work is outside this branch's scope.

## Validation performed

Before handing off this code, the full pipeline (encoding → network →
STDP → homeostasis → label assignment → evaluation) was validated on a
synthetic 3-class task (`sanity_check.py`) — three distinct 28×28 spatial
patterns with Gaussian noise — since this development environment has no
internet access to download real MNIST. With the corrected threshold and
frozen evaluation path, this check scored **95.6% test accuracy**
(chance = 33.3%). It is only a pipeline check, not an estimate of MNIST
performance. Run `python sanity_check.py` to repeat it in your environment;
`python -m unittest test_baseline.py` checks frozen evaluation and checkpoint
reload.

**Historical baseline-development note:** at that stage, real MNIST accuracy
had not yet been measured. The following original instructions describe
running `run_baseline.py` on Kaggle (or any machine with internet). Do
this before your proposal and report the actual number; don't quote the
Diehl & Cook paper's numbers as if they were yours.

## Outputs (saved to `results/` or `/kaggle/working/`)

- `baseline_weights.npz` — trained input→excitatory weights and adaptive
  thresholds (the checkpoint Phase 2 unlearning will load)
- `neuron_labels.npy` — class assigned to each excitatory neuron
- `per_class_spike_rates.npy` — mean assignment-set spike counts per class
- `confusion_matrix.npy` / `.png` — measured test confusion matrix
- `baseline_accuracy.txt` — measured accuracy, configuration, and stage times
- `receptive_fields.png` — visualization of learned digit prototypes
  (should look like Fig. 2A of Diehl & Cook / Fig. 4 of ASP once trained
  on real MNIST)
