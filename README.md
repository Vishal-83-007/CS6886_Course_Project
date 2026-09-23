# Machine Unlearning using SNNs
something is something
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
4. Edit the config block at the top of `run_baseline.py` for your time budget
   (see below), then `!python run_baseline.py`

No GPU needed or used — see "CPU vs GPU" below.

### Time budget presets

| Preset | N_EXCITATORY | N_TRAIN | Expected runtime (CPU) | Expected accuracy |
|---|---|---|---|---|
| Quick demo (proposal) | 100 | 1000 | ~20–30 min | ~70–80% |
| Stronger baseline | 400 | 3000 | ~70–90 min | ~85–88% |
| Paper-scale | 6400 | 60000 | many hours | ~95% |

These accuracy figures are extrapolated from Diehl & Cook's published
scaling curve (Fig. 2B: 82.9% @ 100 neurons w/ 40k examples, up to 95.0%
@ 6400 neurons w/ 900k example-presentations) — actual numbers from our
implementation will differ somewhat and should be reported as measured,
not assumed.

## CPU vs GPU

**Everything here runs on CPU only, by design — no CUDA/GPU dependency
anywhere in this repo.** Brian2's default backends are a Python/numpy
runtime and a Cython-compiled C++ backend (`requirements.txt` includes
`cython` so Brian2 auto-selects the faster compiled backend when a C
compiler is available, falling back to pure numpy otherwise — either way
it runs on any machine with no extra setup).

SNN + STDP simulation is inherently sequential in time and event-driven
(sparse, irregular spikes) — it does not parallelize onto GPU the way
dense-matrix deep learning does. This is why both source papers used
Brian2 on CPU. A GPU backend (Brian2CUDA) technically exists but requires
CUDA toolkit setup and is far less maintained; using it would risk the
repo breaking on a grader's machine, so it's intentionally not used here.

## Validation performed

Before handing off this code, the full pipeline (encoding → network →
STDP → homeostasis → label assignment → evaluation) was validated on a
synthetic 3-class task (`sanity_check.py`) — three distinct 28×28 spatial
patterns with Gaussian noise — since this development environment has no
internet access to download real MNIST. Result: **97.8% test accuracy**
(chance = 33.3%), with neuron labels distributed roughly evenly across
the three classes. This confirms the pipeline is functionally correct
end-to-end. Run `python sanity_check.py` yourself to reproduce (~1 min).

**Real MNIST accuracy has not yet been measured** — that requires
running `run_baseline.py` on Kaggle (or any machine with internet). Do
this before your proposal and report the actual number; don't quote the
Diehl & Cook paper's numbers as if they were yours.

## Outputs (saved to `results/` or `/kaggle/working/`)

- `baseline_weights.npz` — trained input→excitatory weights (this is the
  checkpoint Phase 2 unlearning will load)
- `neuron_labels.npy` — class assigned to each excitatory neuron
- `baseline_accuracy.txt` — measured test accuracy + config used
- `receptive_fields.png` — visualization of learned digit prototypes
  (should look like Fig. 2A of Diehl & Cook / Fig. 4 of ASP once trained
  on real MNIST)

## Known issues found & fixed (debugging log)

Early test runs on real MNIST gave **47% accuracy at 100 neurons and
~10% (chance level) at 400 neurons** — the opposite of the expected
trend (more neurons should help, per Diehl & Cook's own scaling curve).
Root-caused via direct instrumentation (see git history), not guesswork:

1. **Threshold initialization bug (critical, universal).** The adaptive
   threshold is `theta - theta_offset + v_thresh_e`. `theta` was
   initialized to `0`, giving an effective threshold of `-72mV` —
   *below* the resting potential of `-65mV`. Every excitatory neuron was
   trivially above threshold at rest, before any input, and fired at its
   refractory-limited maximum regardless of what was presented. Fixed by
   initializing `theta = theta_offset` (`src/network.py`).

2. **Missing weight normalization.** The STDP rule bounds individual
   synapse dynamics but not the *sum* of a neuron's incoming weights.
   Combined with continuous high-rate Poisson drive, total excitatory
   conductance could grow unbounded, independently saturating neurons.
   Diehl & Cook's own implementation normalizes each neuron's total
   incoming weight after every presentation; this isn't spelled out in
   the paper's prose (Eqn. 3) and is a common trap when reproducing this
   architecture from the paper text alone. Added `normalize_weights()`,
   called automatically after every training presentation.

3. **E/I loop instability at scale (found after fixing 1 & 2).** Even
   with the above fixes, the network could still fall into a
   **self-sustaining, input-independent oscillation** — confirmed by
   feeding different random images and observing identical periodic
   output, proving the dynamics had decoupled from the input entirely.
   This got worse with more neurons (appeared after ~30 presentations at
   N=100, after only 3 at N=400). Root cause: the original E→I and I→E
   conductance strengths (`W_EI=10.4`, `W_IE=17.0`) were hand-derived
   without empirical validation and made every excitatory spike
   deterministically trigger strong, synchronized inhibitory feedback —
   a classic recipe for pathological synchrony in recurrently-coupled
   spiking circuits. Weakened to `W_EI=3.0`, `W_IE=8.0`; verified stable
   (smooth, monotonically settling firing rate, no runaway) across 120
   consecutive presentations at both N=100 and N=400.

A runtime guard was also added to `src/train.py`: if the excitatory
layer stays above 50% of its max possible firing rate for 3 consecutive
presentations during training, it prints a loud warning rather than
silently continuing to "train" on a saturated, uninformative network.
If you ever see that warning on a real run, stop and let me know —
it means these parameters need further tuning for whatever `N_EXCITATORY`
/ data regime triggered it.

**Practical implication for your numbers:** the 47%/10% results you got
were from the buggy code and should not be reported anywhere. Re-run
`run_baseline.py` with the current code for real numbers.

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
