# Machine Unlearning using SNNs
something is something
Course project: importance-based machine unlearning on a Spiking Neural
Network trained for MNIST digit recognition. This repo currently
implements **Phase 1: the baseline SNN**, trained with standard STDP.
Phase 2 (importance-based unlearning) builds on the saved checkpoint
this phase produces.

## Implementation provenance (read this first)

The code in `src/` is a **faithful Brian2 port of Peter Diehl's own
released implementation**:
https://github.com/peter-u-diehl/stdp-mnist
(`Diehl&Cook_spiking_MNIST.py` + `Diehl&Cook_MNIST_random_conn_generator.py`)

This project went through two versions, and it's worth knowing the
difference if you're citing this work:

- **v1 (superseded, kept as `src/network_v1_paperbased.py.bak` for the
  record):** built directly from the paper's written description
  (Eqn. 3's power-law STDP rule, hand-picked conductance values,
  Brian2's native refractory mechanism). This had a real bug (wrong
  theta initialization) and was missing a mechanism the paper's prose
  doesn't mention (weight normalization) — see "Debugging log" below
  for the full story of how those were found.
- **v2 (current):** every equation, parameter value, and mechanism in
  `src/network.py`, `src/encoding.py`, and `src/train.py` is copied
  directly from Diehl's actual source files, not re-derived from the
  paper text. This includes the STDP rule (a different, two-trace
  variant than the paper's Eqn. 3), the exact conductance values
  (`W_EI=10.4`, `W_IE=17.0` — confirmed from
  `Diehl&Cook_MNIST_random_conn_generator.py`, not guessed), the exact
  weight normalization target (`78.0`), and the input encoding
  mechanism (multiplicative `input_intensity` scalar starting at 2.0,
  not the paper's stated `+32Hz` additive boost). The one deliberate,
  documented substitution is the refractory mechanism: Brian 1's
  `dtimer/dt = 100.0 : ms` idiom has no literal Brian2 equivalent, so
  it's re-expressed with clear, equivalent semantics (see the docstring
  at the top of `src/network.py`).

**If you're citing accuracy figures from the Diehl & Cook paper or from
other reimplementations (e.g. the ~91-95% figures commonly quoted), be
aware those come from the two-trace STDP rule used here in v2, not the
power-law rule from the paper's Eqn. 3 that v1 used.**

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

**From Diehl & Cook (v2 = Diehl's actual released code) — used:**
- Conductance-based LIF neuron model
- 3-layer architecture: Input (784, Poisson) → Excitatory (N) → Inhibitory (N)
- Connectivity: input→exc all-to-all (plastic, with 0-10ms random delay),
  exc→inh one-to-one (fixed, w=10.4), inh→exc all-to-all-except-self (fixed, w=17.0)
- The two-trace pre/post STDP rule actually used in Diehl's code (NOT the
  power-law rule from the paper's Eqn. 3 — see "Implementation provenance"
  above): presynaptic trace `pre`, two postsynaptic traces `post1`
  (fast, 20ms) and `post2` (slow, 40ms); depression on every presynaptic
  spike (`w -= nu_pre * post1`), potentiation on postsynaptic spikes
  (`w += nu_post * pre * post2before`)
- Adaptive threshold homeostasis (theta), initialized to the offset value (20mV)
- Weight normalization: each neuron's total incoming weight rescaled to
  78.0 before every training presentation
- Poisson input encoding via a multiplicative `input_intensity` scalar
  (starts at 2.0, `pixel/8 * intensity`), incremented by 1 and the image
  re-presented if fewer than 5 excitatory spikes result
- Post-hoc neuron→class label assignment (labels never touch training)
- Timer-based manual refractory period (membrane potential keeps
  integrating throughout, unlike Brian2's native refractory clause —
  see "Implementation provenance" for why this distinction mattered)

**From Diehl & Cook — ignored:**
- The other 3 STDP rule variants the paper benchmarks (the power-law
  rule is one of these four — v1 used it, v2 does not)
- The paper's own stated "+32Hz additive" rate-boost description — the
  actual code uses a multiplicative intensity scalar instead, and v2
  follows the code

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

## Debugging log

### v1 (paper-text reconstruction) — bugs found on real MNIST results

Test runs on real MNIST gave **47% accuracy at 100 neurons and ~10%
(chance level) at 400 neurons** — backwards from the expected trend.
Root-caused via direct instrumentation:

1. **Threshold initialization bug.** `theta` was initialized to `0`
   instead of the offset value, putting the effective firing threshold
   below the resting potential — neurons fired at max rate regardless of
   input. (v2 confirms the correct value: Diehl's code sets
   `theta = 20.0*mV` explicitly, matching the fix that was applied.)
2. **Missing weight normalization.** Not mentioned in the paper's prose;
   Diehl's code normalizes every neuron's total incoming weight to a
   fixed target after every presentation. Without it, excitatory drive
   grows unbounded. (v2 confirms the exact value: `78.0`, matching the
   value used at the time by coincidence/estimation.)
3. **E/I loop instability at scale, and why it doesn't affect v2 — see
   below.** After fixing 1 & 2, the network could still fall into a
   self-sustaining, input-independent oscillation. v1's workaround was
   to detune the conductance values (10.4→3.0, 17.0→8.0) away from the
   literature-standard values.

### v2 (faithful port) — re-investigating issue 3

Once the exact original conductance values were confirmed
(`W_EI=10.4`, `W_IE=17.0`, from `Diehl&Cook_MNIST_random_conn_generator.py`),
the natural question was whether v1's real problem was its refractory
implementation (Brian2's native `(unless refractory)` freezes the
membrane potential during refractory, discarding any inhibition that
arrives in that window — unlike Diehl's manual timer-based refractory,
which lets `v` keep integrating throughout). v2 replicates the
timer-based mechanism and uses the original, undetuned conductance
values.

**Re-running the same stress test (many presentations, checking for
runaway firing) initially reproduced the same instability at N=100 —
which turned out to be a flaw in the *stress test*, not the network.**
The stress test used full-field uniform random noise images (all 784
input pixels active at once), which is a pathological worst case very
unlike real MNIST (mostly black background, one sparse bright region).
Re-running the identical test with MNIST-realistic sparse synthetic
images (small bright patch on a zero background) showed **no
instability at all** — clean, sparse, stable firing (well under 1% of
max rate) across 40-50 consecutive presentations at N=15, N=100, and
N=400, using the literature-standard, undetuned conductance values.

**Conclusion: v2's instability "bug" was a false alarm caused by
unrepresentative synthetic test data, not a real flaw.** The refractory
mechanism may still matter for robustness at the extremes, but it was
not the actual cause of what broke v1 on real MNIST — that was
entirely bugs 1 and 2. This is worth knowing if you extend the stress
tests in `sanity_check.py` yourself: **don't use full-random-noise
images as a stability probe for this architecture; use sparse,
MNIST-like input.**

A runtime guard (in `src/train.py`) still prints a warning if the
excitatory layer stays above 50% of max firing rate for 3 consecutive
presentations, in case something genuinely goes wrong on real data —
treat it as a smoke detector, not an indictment of the architecture.

**Practical implication:** the 47%/10% results from v1 should not be
reported. Run `run_baseline.py` with the current (v2) code for real
numbers.

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
