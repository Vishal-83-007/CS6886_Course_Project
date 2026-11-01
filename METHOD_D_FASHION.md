# Fashion-MNIST Method D: request-gated leak with retained replay

Same method as [METHOD_D_MNIST.md](METHOD_D_MNIST.md), on the Fashion-MNIST
baseline trained with weight normalisation 39. The default request `class6_9`
removes **Shirt and Ankle boot** (`gold_weights/fashion_class6_9_wn39`). Results:
[RESULTS_B_D.md](RESULTS_B_D.md).

## Weight normalisation during replay

The baseline was trained with each neuron's input weights rescaled to sum to 39
before every image, so replay normalises too (`--weight-norm 39`, the default for
Fashion), but **never on gated synapses**:

- ungated neurons are normalised to 39, as in training;
- partly gated neurons (synapse gate): their ungated synapses keep the sum they had
  in the baseline (`--norm-gated partial`). Without this, those neurons' ungated
  weights grew unchecked (one reached a sum of 173) and the network collapsed to
  predicting Coat;
- after Method B (`--start-weights` + `--norm-target start`) every neuron keeps its
  dampened sum, so normalisation cannot undo the dampening.

Reported settings: **neuron gate, leak 0.005, 1,000 images** (131 neurons; gated
weight 5,109 → 34); synapse gate α = 2 (92,290 synapses on 175 neurons), leak 0.005,
1,000 images.

## Run locally

```bash
python run_method_d.py --dataset fashion --baseline-dir /path/to/fashion_baseline \
  --gate neuron --leak 0.005 --replay-images 1000 \
  --gold-dir gold_weights/fashion_class6_9_wn39 --out-dir results/fashion_method_d
```

## Aqua

```bash
GATES="neuron synapse" LEAKS="0.0002 0.001 0.005" REPLAYS="1000 5000" WEIGHT_NORM=39 \
  DATASET=fashion BASELINE=$FB bash jobs/leak_sweep.sh
```

## Interpretation

- Neuron gate: 81.3% retained / 0% removed (Gold 81.1 / 0), 0% relapse after
  20,000 adaptive replay images. Synapse gate: 80.8 / 1.3, relapse 6.9% (its
  partly gated neurons keep their unselected synapses).
- Relabelled with all 10 classes, Shirt/boot are 59.6% recoverable against Gold's
  69.4%; Gold is above the original baseline (59.4%) here because its
  Sandal/Sneaker neurons broaden to cover boots once no boot neurons compete
  (they respond to boots at 1.30× their own-class rate, against 0.16× in the
  baseline). Short replay cannot recreate that.
