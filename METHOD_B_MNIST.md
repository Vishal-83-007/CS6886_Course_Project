# MNIST Method B: Selective Synaptic Dampening with hypothetical-STDP importance

The default request is **joint deletion of digits 6 and 9** (`class6_9`), matching
`gold_weights/mnist_class6_9`. Method B is SSD (Foster et al., 2024) with the
gradient/Fisher importance replaced by a local STDP proxy: the squared update the
network's own STDP rule *would* make to each synapse, measured with learning off.
For Fashion-MNIST see [METHOD_B_FASHION.md](METHOD_B_FASHION.md); for forgetting a
random part of a class see [METHOD_B_SUBSET.md](METHOD_B_SUBSET.md). All results
are collected in [RESULTS_B_D.md](RESULTS_B_D.md).

No training code or source model is changed. The edit multiplies selected
input→excitatory weights by a factor in [0, 1]; thresholds, inhibition and the
baseline's neuron labels are unchanged.

## Method

1. **Importance (once per baseline).** `run_frozen.py importance` presents every
   training image to the frozen baseline (STDP and θ off) and accumulates, per
   synapse and per class, the square of the hypothetical update
   `Δw_ij(x) = Σ_post spikes η (x_pre − x_tar)(w_max − w_ij)^μ` (same η = 1e-4 and rule as
   training; `dw_hyp` in `src/network.py`). `I(S)` is the mean of `Δw²` over a set `S`.
   We averaged two passes (Poisson seeds 1 and 2); one pass is enough to run the method.
2. **Selection and dampening (seconds, `dampen.py`).**
   select `I(D_f) > α·I(D)`; set `w ← β·w` with `β = min(λ·I(D)/I(D_f), 1)`.
   Because D contains D_f, `I(D_f)/I(D) ≤ |D|/|D_f| ≈ 5` for two of ten classes, so
   **α must be below about 5** (the SSD paper's α = 5–50 selects nothing here).
3. **Evaluation.** `run_frozen.py evaluate` scores the edited weights with the
   baseline's fixed labels, and also relabels the neurons with all 10 classes
   (recoverability probe) and with the 8 kept classes only.

Reported setting: **α = 2, λ = 0.5** (smallest mean gap to Gold on the tuning half of
the test set); **λ = 0.1** is the robust variant (see the relapse test in
[RESULTS_B_D.md](RESULTS_B_D.md)).

## Run locally

Python 3.12 with `requirements-aqua-cpu.txt` and a C++ compiler; official MNIST IDX
files in `data/mnist` (or `--data-dir`). From the repository root:

```bash
# 1. importance of the baseline over all 60,000 training images (hours; once)
python run_frozen.py importance --dataset mnist \
  --weights /path/to/mnist_baseline/baseline_weights.npz \
  --poisson-seed 1 --out-dir results/mnist/importance/seed1

# 2. Method B + frozen evaluation (+ Gold for comparison)
python run_method_b.py --dataset mnist \
  --baseline-dir /path/to/mnist_baseline \
  --importance results/mnist/importance/seed1/importance.npz \
  --alpha 2 --lam 0.5 --gold-dir gold_weights/mnist_class6_9 \
  --out-dir results/mnist_method_b
```

Add `--eval-limit 100` for a quick smoke run, `--skip-evaluation` to produce only the
edited weights, `--dry-run` to print the steps. `summary.json` collects the
accuracies; each evaluation folder has `metrics.json` (per readout: overall,
retained, removed and per-class accuracy, and what removed images were predicted
as) and `responses.npz` (all spike counts).

## Aqua

```bash
qsub -v DATASET=mnist,MODEL=$MB jobs/importance_cpp.cmd             # I(D), ~7 h
DATASET=mnist BASELINE=$MB IMPORTANCE="$(ls results/mnist/importance/$(basename $MB)_seed*/importance.npz)" \
  FORGET=class6_9 ALPHAS="1.5 2 3 4" LAMBDAS="0.1 0.5 1" bash jobs/ssd_sweep.sh
qsub -v "DATASET=mnist,WEIGHTS=<dampened .npz>,FORGET=class6_9,FIXED_LABELS=$MB/neuron_labels.npy,EVAL_SPLIT=final" jobs/evaluate_cpp.cmd
```

`$MB` is the baseline run folder. The sweep is resumable (it skips finished or
submitted settings and stops cleanly at AQUA's per-user job limit).
`bash aqua/progress.sh` prints every result. Details: [AQUA_RUNBOOK.md](AQUA_RUNBOOK.md).

## Interpretation

- With the baseline's labels, B removes 6 and 9 (0% on the final half) and raises
  retained accuracy from 89.1% to 91.7%, matching Gold (92.1%).
- `λ` barely matters in a frozen evaluation: winner-take-all means a neuron that
  loses its 6/9 synapses simply stops winning. It matters once thresholds adapt:
  the adaptive θ can compensate a gain reduction, so weaker dampening relapses more
  (relapse test).
- Selection by `I(D_f)/I(D)` is equivalent to selection by `I(D_f)/I(D_r)` (a
  monotone reparametrisation of α); protecting the top-p% retained-importance
  synapses (`--protect-top`) had no effect.

## Tests

```bash
python -m unittest test_method_b -v
```

Checks argument handling, the steps the entry point runs, that the edit only
shrinks weights and never changes thresholds, and that output folders are new.
