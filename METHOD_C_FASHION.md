# Fashion-MNIST Method C on Aqua

`run_method_c_fashion.py` implements the same C1/C2 erasure replay as MNIST,
using checked Fashion data, request-specific replay/metrics and Fashion
collateral-effect reporting. It loads an existing baseline; no baseline training
or new dependencies are required. The existing CPU environment and C++ compiler
are reused. No remote job is submitted by packaging.

## Checkpoints and requests

The default baseline is the completed 800-neuron model:
`fashion/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_wn39_cpp_2507283`.
It uses seed 0, three epochs, all 60,000 training images, 500 assignment examples
per class and training normalization 39. Its historical accuracy is 74.28%.

- `class6`: shirt-only; replay all 6,000 training shirts. No Gold is
  selected automatically. Supply `--gold-dir` / PBS `GOLD_DIR` when a matched
  shirt-only Gold is available.
- `class6_9` (default): shirt and ankle boot; replay 12,000 training images. The matching
  `gold_weights/fashion_class6_9_wn39` is selected automatically.
- Other `classK` or increasing multi-class requests are accepted. They also need
  their own matched Gold for retraining comparisons.

The new 400-neuron baseline can be used **after its current-code training run
finishes**: set `BASELINE_DIR` to that completed run directory. Do not select the
older defective 400-neuron run or resume its snapshots. A 400-neuron baseline
needs a 400-neuron Gold with matching training configuration. For joint deletion
without that Gold, set `NO_GOLD=1` to bypass the automatic 800-neuron Gold. A
shirt-only job already defaults to running without Gold. Baseline weights,
labels, config and `manifest.npz` must remain together.

Gold checks include dataset/fingerprint, exact request, neuron count, seeds,
epochs, split, training subset, assignment count, training normalization and
input encoding. Neither a joint Gold for shirt-only deletion nor an 800-neuron
Gold for a 400-neuron baseline is accepted.

## Transfer and run

Locally:

```bash
python3 scripts/package_method_c.py --dataset fashion
scp -P 40826 method_c_fashion_code.tar.gz ep23b048@aqua.iitm.ac.in:~/CS6886/
```

On Aqua, from the project root with existing data/checkpoints/environment:

```bash
cd ~/CS6886
tar -xzf method_c_fashion_code.tar.gz

# Short pilots: one replay image per forgotten class, one pass, ten test images.
qsub -v REQUEST=class6,VARIANT=C1,SCALE=0.1,PASSES=1,REPLAY_PER_CLASS=1,EVAL_LIMIT=10 jobs/fashion_method_c.cmd
qsub -v REQUEST=class6,VARIANT=C2,SCALE=0.1,PASSES=1,REPLAY_PER_CLASS=1,EVAL_LIMIT=10 jobs/fashion_method_c.cmd
qsub -v REQUEST=class6_9,VARIANT=C2,SCALE=0.1,PASSES=1,REPLAY_PER_CLASS=1,EVAL_LIMIT=10 jobs/fashion_method_c.cmd
```

After successful pilots, full shirt-only runs (passes 1, 2 and 5; full test set):

```bash
qsub -v REQUEST=class6,VARIANT=C1,SCALE=0.1 jobs/fashion_method_c.cmd
qsub -v REQUEST=class6,VARIANT=C2,SCALE=0.1 jobs/fashion_method_c.cmd
```

For the same prespecified scale sweep as MNIST, repeat each request with C1/C2
and scales 0.1, 0.5, 1.0. Each job starts from the original baseline and captures
passes 1, 2 and 5 of a single trajectory. Do not chain variants or scales.
For example, the joint sweep is:

```bash
for variant in C1 C2; do
    for scale in 0.1 0.5 1.0; do
        qsub -v REQUEST=class6_9,VARIANT="$variant",SCALE="$scale" jobs/fashion_method_c.cmd
    done
done
```

For the fresh 400-neuron run, replace the example path with its actual completed
directory (the job ID is part of the name):

```bash
qsub -v BASELINE_DIR=results/fashion/n400_baseline_seed0_ep3_trainall_val0_evalall_assign500_wn39_cpp_JOBID,REQUEST=class6,VARIANT=C2,SCALE=0.1 jobs/fashion_method_c.cmd
```

Each job requests one CPU core and 24 hours. A full job performs up to five
replay passes and multiple full-test evaluations. Measure throughput on Aqua
before assuming it fits; compiled replay has no exact resume. Results go to
`results/fashion_method_c/<request>_<variant>_s<scale>_<jobid>/`.

Overrides via `qsub -v`: `BASELINE_DIR`, `GOLD_DIR`, `NO_GOLD=1`, `REQUEST`,
`VARIANT`, `SCALE`, `PASSES`, `REPLAY_PER_CLASS`, `EVAL_LIMIT`, `REPLAY_SEED`,
`INFERENCE_SEED`, `SKIP_EVALUATION=1`, `ADAPT_THRESHOLDS=1`, `FASHION_PYTHON`,
`FASHION_DATA`, `FASHION_RESULTS`, `FASHION_ENV_ARCHIVES`. The default interpreter
is `$HOME/fashion-cpu/bin/python`, data is `data/fashion`, and archive-staged
environments reuse `results/fashion/environments`. `NO_GOLD` and `GOLD_DIR` are
mutually exclusive. For several custom pass values, pass a quoted space-separated
list to `PASSES`; the default already supplies 1, 2, 5.

The archive installs an isolated `method_c_fashion_runtime/`, the new job and this
guide. It includes all matching inference/replay dependencies, tests, pinned CPU
requirements and SHA256 manifest. It does not overwrite training sources, MNIST
runtimes, Method A runtime, `aqua_common.sh`, models or datasets. To include the
existing CPython 3.12 offline wheels:

```bash
python3 scripts/package_method_c.py --dataset fashion --include-wheels --output method_c_fashion_offline_cp312.tar.gz
```

## Protocol and interpretation

At each original postsynaptic STDP event, using the current edited weight:

```text
delta_base = eta * (pretrace - x_tar) * (w_max - w)**mu
C1: delta = -scale * delta_base
C2: delta = -scale * max(delta_base, 0)
w = clip(w + delta, 0, w_max)
```

C1 can increase weights when the original rule would depress them; C2 cannot
increase weights. **Training normalization 39 is not reapplied during erasure**:
it could replenish erased weights and violate C2's monotonic decrease. The
baseline's saved normalized weights are the starting point. Optional saved
`input_total_hz` normalization is preserved for both replay and evaluation.

Replay selects only requested classes from the baseline's saved training IDs.
It never uses test, held-out validation, retain or Gold images for updates.
Each pass permutes exactly that replay set, with shared recorded orders across
variants/scales. Exposure is 350 ms plus 150 ms continuous zero-input rest, at
dt=0.5 ms. There are no replay retries or rate boosts, even for silent images.
Updates stop during rest; traces and simulator state continue. Primary theta is
frozen. `ADAPT_THRESHOLDS=1` is a separately named homeostasis ablation.

Each checkpoint is evaluated with frozen weights/theta and original neuron
labels using the baseline inference retry policy. Baseline, Method C and Gold
start fresh with common evaluation IDs and initial seed; retries consume different
RNG lengths. Fresh baseline accuracy can differ from the historical training-run
evaluation. Compare against the freshly evaluated baseline. Gold is used only
as a reference and does not enter erasure.

`summary.json` reports retain/forget/per-class accuracy, spikes, prediction
disagreement with baseline/Gold, weight changes and update counts. Each trial
has pass checkpoints, immutable original labels, per-presentation diagnostics
and per-image evaluation responses. Each evaluated pass also saves
`evaluation/fashion_comparison.json`: named per-class accuracy changes against
the fresh baseline, correct-to-wrong/wrong-to-correct counts, retained-class
shirt-prediction rates and the distribution of true-shirt predictions. Inspect
classes 0 (T-shirt/top), 2 (pullover), 3 (dress), 4 (coat) individually.

Every snapshot is checked for finite legal weights, frozen primary thresholds
and C2 weight monotonicity/no positive updates. Source artifacts are hashed and
checked after execution. Full source-manifest replay plans can be checked without
Brian simulation:

```bash
python3 method_c_fashion_runtime/run_method_c_fashion.py --prepare-only --out-dir results/fashion_method_c/plan_shirt
```

For the supplied baseline there is no held-out validation set. The sweep is
prespecified test reporting; do not select scale/pass based on test results.
Silence or zero shirt accuracy alone does not establish erasure, especially
given the baseline's low shirt accuracy and shared clothing responses. No MIA
or relearning audit is added. Pilot subsets only check execution.

Run tests in the CPU environment:

```bash
cd method_c_fashion_runtime
python3 -m unittest test_method_c test_method_c_fashion -v
```

## Local validation

All nine scientific/Fashion tests passed using Python 3.12, Brian2 2.9.0 and
NumPy 1.26.4 from the existing offline CPU wheels. Tests cover C1/C2 events,
clipping, rest, frozen/adaptive theta, compiled/runtime agreement, request-specific
metrics, replay training membership, validation overlap and Gold matching.

Real 800-neuron pilots completed for shirt-only C1 and C2 (scale 0.1, one pass,
one replay image, ten test images) and joint class6/9 C2 (two replay images, ten
test images, matched Gold). The shirt pilots ran from the extracted transfer
bundle. Checkpoints, frozen evaluation, class comparison files and C2 monotonicity
checks passed. Full preparation verified 6,000 shirt IDs and 12,000 joint IDs;
the MNIST default still prepared its 11,867 original joint IDs. Archive hashes
and Python/PBS syntax were checked. Full unlearning effectiveness and Aqua
throughput have not been measured; no Aqua job was submitted here.
