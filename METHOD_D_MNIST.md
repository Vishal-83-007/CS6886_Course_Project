# MNIST Method D: request-gated leak with retained replay

The default request is **joint deletion of digits 6 and 9** (`class6_9`), matching
`gold_weights/mnist_class6_9`. Method D is the proposal's ASP-inspired idea:
replay retained images with the normal STDP rule while forget-associated weights
leak, so the network keeps its retained knowledge while the forgotten features
fade. For Fashion-MNIST see [METHOD_D_FASHION.md](METHOD_D_FASHION.md). Results:
[RESULTS_B_D.md](RESULTS_B_D.md).

## Method (`src/replay.py`, compiled with Brian2 `cpp_standalone`)

1. **Gate (fixed before replay).**
   - `neuron`: every afferent synapse of the neurons the baseline labelled 6 or 9
     (166 of 800 neurons on MNIST). Needs only the saved labels.
   - `synapse`: the synapses Method B selects, `I(D_f) > α·I(D)` (α = 2:
     140,348 synapses on 224 neurons). Needs Method B's importance file.
2. **Replay.** R images drawn from the retained classes only (no 6 or 9; checked),
   presented exactly as in training (350 ms + 150 ms rest, retry with +32 Hz if
   fewer than 5 spikes) with STDP on at the training learning rate.
3. **Leak.** After every image, `w ← (1 − leak)·w` for gated synapses, so after R
   images a gated weight keeps `(1 − leak)^R` of its value plus whatever STDP adds.
4. Thresholds θ are **frozen** by default (no growth, no decay), so lower firing
   cannot come from changed thresholds; `--theta updated` lets them adapt.
5. Frozen evaluation with the baseline's fixed labels (plus relabelled read-outs).

Reported settings: **neuron gate, leak 0.001, 5,000 images** (gated weight
10,656 → 72); **synapse gate, leak 0.005, 1,000 images** (robust variant; the
leak 0.001 / 1,000-image setting leaves 37% of the gated weight and relapses).
`--gate none` is plain replay without leak (repair/relapse runs).

## Run locally

```bash
python run_method_d.py --dataset mnist --baseline-dir /path/to/mnist_baseline \
  --gate neuron --leak 0.001 --replay-images 5000 \
  --gold-dir gold_weights/mnist_class6_9 --out-dir results/mnist_method_d

# synapse gate (needs the importance pass from METHOD_B_MNIST.md)
python run_method_d.py --dataset mnist --baseline-dir /path/to/mnist_baseline \
  --gate synapse --importance results/mnist/importance/seed1/importance.npz \
  --leak 0.005 --replay-images 1000 --out-dir results/mnist_method_d_synapse

# relapse test: 20,000 retained images, thresholds adapting, no leak
python run_method_d.py --dataset mnist --baseline-dir /path/to/mnist_baseline \
  --gate none --theta updated --replay-images 20000 \
  --start-weights results/mnist_method_d/replay/replayed_weights.npz --out-dir results/mnist_relapse_d
```

The replay folder holds `replayed_weights.npz` and `metrics.json` (gated synapses
and neurons, gated and other weight sums before and after, θ before and after).

## Aqua

```bash
GATES="neuron synapse" LEAKS="0.0002 0.001 0.005" REPLAYS="1000 5000" \
  DATASET=mnist BASELINE=$MB bash jobs/leak_sweep.sh          # resumable sweep
qsub -v DATASET=mnist,BASELINE=$MB,GATE=neuron,LEAK=0.001,REPLAY=5000,EVAL_SPLIT=final jobs/replay_cpp.cmd
DATASET=mnist BASELINE=$MB STARTS="<weights .npz>" WEIGHT_NORM=0 REPLAYS=20000 THETAS=updated \
  bash jobs/repair_sweep.sh                                   # relapse test
```

## Interpretation

- Neuron gate: 91.5% retained / 0% removed with the baseline's labels (Gold 92.1 / 0),
  and it is the only method with **0% relapse**: with 0.7% of the weight left, no
  6/9 neuron fires even when θ falls to ~13 mV.
- Relabelled with all 10 classes, 6/9 are still 79.8% recoverable (Gold 76.5%):
  that information sits in retained-class neurons whose features overlap 6 and 9,
  which no method removes.
- Plain replay without leak (fine-tuning on retained data) forgets nothing; with
  adaptive θ it even makes 6/9 more recognisable (88% → 94.5%), because the 6/9
  neurons lose fewer threshold increments than the retained-class neurons.

## Tests

`python -m unittest test_method_d -v`: per-dataset defaults, gate options, that the
replayed weights are evaluated with the baseline's labels, and that subset
requests are refused (Method D's neuron gate is class-level by construction).
