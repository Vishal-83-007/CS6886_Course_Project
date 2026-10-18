# Machine Unlearning using SNNs

For the Fashion-MNIST/Aqua workflow, start with
[FASHION_AQUA_RUNBOOK.md](FASHION_AQUA_RUNBOOK.md). It documents the new
`run_fashion.py` runner, reproducible baseline/Gold splits, CPU checkpoint/resume,
compiled CPU execution, and bounded CUDA pilots. The older MNIST instructions
below describe `run_baseline.py` and remain separate.

Course project: importance-based machine unlearning on a Spiking Neural
Network trained for MNIST digit recognition. This repo currently
implements **Phase 1: the baseline SNN**, trained with standard STDP.
Phase 2 (importance-based unlearning) builds on the saved checkpoint
this phase produces.

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

**Real MNIST accuracy has not yet been measured** — that requires
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

## Next steps (Phase 2 — not yet implemented)

The importance-based unlearning method described in the project proposal:
- `Δw_ij(x)` hypothetical STDP updates without applying them (plasticity
  frozen, update computed and discarded) — needs a variant of the
  training loop that computes but doesn't apply `w = clip(...)`
- `I_ij(D)` and `I_ij(D_f)` importance accumulation (mean squared
  hypothetical update)
- Forget-specificity ratio `R_ij` and synapse selection/dampening
- Comparison baselines: mean pre/post activity, product of pre/post
  firing, |Δw|, (Δw)²

This will load `baseline_weights.npz` rather than retraining from
scratch.
