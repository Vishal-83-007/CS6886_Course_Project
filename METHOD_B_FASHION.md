# Fashion-MNIST Method B: Selective Synaptic Dampening

Same method as [METHOD_B_MNIST.md](METHOD_B_MNIST.md), on the Fashion-MNIST
baseline trained with weight normalisation 39 (74.28% test accuracy). The default
request `class6_9` removes **Shirt and Ankle boot**, matching
`gold_weights/fashion_class6_9_wn39`. Results: [RESULTS_B_D.md](RESULTS_B_D.md).

## Run locally

```bash
python run_frozen.py importance --dataset fashion \
  --weights /path/to/fashion_baseline/baseline_weights.npz \
  --poisson-seed 1 --out-dir results/fashion/importance/seed1        # ~9 h

python run_method_b.py --dataset fashion \
  --baseline-dir /path/to/fashion_baseline \
  --importance results/fashion/importance/seed1/importance.npz \
  --alpha 2 --lam 0.5 --gold-dir gold_weights/fashion_class6_9_wn39 \
  --out-dir results/fashion_method_b
```

The weight normalisation is a training-time operation; frozen evaluation and the
importance pass do not normalise. Official Fashion-MNIST IDX files go in
`data/fashion`.

## Aqua

As for MNIST with `DATASET=fashion` and the Fashion baseline folder `$FB`
(`n800_baseline_..._wn39_cpp_2507283` in our runs):

```bash
qsub -v DATASET=fashion,MODEL=$FB jobs/importance_cpp.cmd
DATASET=fashion BASELINE=$FB IMPORTANCE="$(ls results/fashion/importance/$(basename $FB)_seed*/importance.npz)" \
  FORGET=class6_9 bash jobs/ssd_sweep.sh
```

## Interpretation

- With the baseline's labels B reaches 80.9% retained / 2.1% removed (Gold 81.1 / 0).
  The 2.1% left is mostly Shirt, which the baseline itself classifies at only
  ~30% (confused with T-shirt, Pullover and Coat).
- **Relapse:** after 20,000 retained images with adaptive thresholds, Shirt/boot
  return to 33.8% (λ = 0.5), 10.2% (λ = 0.1) and 7.7% (λ = 0). After B the 6/9
  neurons keep about 29% of their weight with unchanged receptive fields (cosine
  0.965 to the baseline); when θ decays the original boot detectors fire again
  (cosine 0.998). Synapses B never selects set the 7–8% floor.
- B at λ = 0 and Method D's synapse gate at full leak produce the same weights.

## Tests

`python -m unittest test_method_b -v` (dataset-independent).
