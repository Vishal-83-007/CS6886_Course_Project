# MNIST Method A: label-guided neuron silencing

The default request is **joint deletion of digits 6 and 9**, matching the existing
`gold_weights/mnist_class6_9` reference. `class6` and `class9` are supported as
independent interventions starting from the original baseline, not sequential
deletions. Each needs its own matched Gold for a retraining comparison. The script
rejects a Gold whose request differs; it does not use joint Gold for single-class
requests. For Fashion-MNIST use the separate entry point documented in
[METHOD_A_FASHION.md](METHOD_A_FASHION.md).

No training scripts or source models are changed. Every run requires a new output
directory. The checkpoint copies baseline weights and original labels unchanged
and adds `intervention.npz`, containing the forgotten classes, neuron mask and
separate threshold override. **The weights file alone does not implement Method A:**
the inference adapter must also apply the intervention. Source artifact SHA256
hashes are recorded and checked again after execution.

The existing network builder has no native spike-enable mask. Following the
specification's fallback, the adapter adds **1 volt** to the effective theta of
selected neurons during simulation. The persistent saved theta is unchanged.
This places their firing threshold above the model's maximum reversal potential
(0 volts). Every selected neuron's cumulative spike count is checked after every
attempt including rest; any spike aborts evaluation. Weights, baseline theta and
the fixed inhibitory circuit are preserved. Their paired inhibitory neurons lose
the normal excitatory drive as a consequence of silencing, rather than a separate
inhibitory edit.

## Run locally

Use the project's CPU Python environment with Brian2 2.9, NumPy 1.26 and a C++
compiler, from the repository root. Defaults select the supplied 800-neuron
baseline and joint Gold folders. Checkpoint preparation needs only NumPy:

```bash
python run_method_a.py --prepare-only --out-dir results/method_a_mnist_prepared
```

Preparation copies files and constructs the intervention; it does not claim a
verified simulation or accuracy. A fresh evaluation also prepares its checkpoint.
Start with a balanced 100-image test smoke run (accuracy is only diagnostic):

```bash
python run_method_a.py --eval-limit 100 --out-dir results/method_a_mnist_smoke
```

Then evaluate the full official test set:

```bash
python run_method_a.py --out-dir results/method_a_mnist_full
```

Official MNIST IDX files must be available in `data/mnist`, or pass `--data-dir`.
The loader checks official file checksums and the baseline's dataset fingerprint.
It never downloads data implicitly. The default full run re-evaluates baseline,
Method A and Gold on all 10,000 test images with the same initial inference seed
and image order. Gold receives no baseline weights or label assignments.

For an independent single-class intervention:

```bash
python run_method_a.py --request class9 --out-dir results/method_a_mnist_class9
```

No Gold is selected automatically for individual requests. Supply a matching
`--gold-dir` if available. Training Gold is outside this implementation.

## Aqua

The existing CPU environment and `jobs/aqua_common.sh` are reused. Older Aqua
checkouts may have a Fashion-only `src/fashion_data.py`; copying only the new
Method A entry point/module is insufficient. Build the complete isolated runtime:

```bash
python scripts/package_method_a.py
scp -P 40826 method_a_mnist_code.tar.gz ep23b048@aqua.iitm.ac.in:~/CS6886/
```

On Aqua, extract from the project root:

```bash
cd ~/CS6886
tar -xzf method_a_mnist_code.tar.gz
```

This installs matching data, encoding, network and Method A modules inside
`method_a_runtime/`, plus the Method A job/docs. It does not replace the existing
training `src/` or `aqua_common.sh`, and includes no model, dataset or environment
archives. The job automatically prefers this isolated runtime if present; the
working directory stays at the project root, so default model/data paths still
apply. Regenerate and transfer the bundle after future local changes. Provide
the source model folders and official MNIST files separately. No training or job
submission is automatic.

```bash
qsub -v EVAL_LIMIT=100 jobs/mnist_method_a.cmd
qsub jobs/mnist_method_a.cmd
```

The job uses one CPU core/thread and requests 12 hours. Baseline and Gold test
phases previously took about 47 minutes each; Method A can change spike activity
and retries, so this is not a completion guarantee. `FASHION_PYTHON`,
`FASHION_DATA`, `FASHION_RESULTS`, `BASELINE_DIR`, `GOLD_DIR`, `REQUEST`,
`EVAL_LIMIT` and `INFERENCE_SEED` can be supplied using `qsub -v`.

For Aqua checkouts using the archive-staging version of `aqua_common.sh`, the
job explicitly reuses `results/fashion/environments`, independently of the new
Method A results directory. It expects the existing `fashion-cpu-cp312.json` and
the environment archive named in that metadata, not the model ZIP files. If your
verified environment archives live elsewhere, pass their directory using
`qsub -v FASHION_ENV_ARCHIVES=/absolute/path/to/environments,EVAL_LIMIT=100 jobs/mnist_method_a.cmd`.

## Inference and interpretation

Python builds the unchanged network and loads the checkpoint, then Brian2's
`cpp_standalone` compiles the **whole image/retry loop**. This avoids repeated
Python simulator calls. The baseline encoding is preserved, with dt=0.5 ms,
350 ms presentation, 150 ms zero-input rest and at most six attempts. No weight
normalization or plasticity runs. Frozen theta never decays or increments.

The state protocol matches the existing baseline evaluator: fresh builder state
at the start, followed by continuous inter-image rest; no new hard reset between
images is introduced. This does not restore unavailable terminal training state.
Consequently, regenerated baseline accuracy can differ from its historical
training-run evaluation. Compare the freshly evaluated models with one another.
Common initial seed/order is recorded; model-dependent retries consume different
amounts of RNG, so this is not exact per-image common Poisson realizations.

Original baseline neuron labels and the original mean-per-neuron class voting
rule remain in place for Method A. Silenced labels have zero scores rather than
being removed from the head. Ties choose the lowest class ID, as in existing code.
A separate **readout-suppression control** uses unchanged baseline responses and
sets scores for deleted labels to negative infinity. It requires no extra
simulation and is explicitly separate from physical silencing. This is a head
suppression control, not retain-only neuron reassignment.

`summary.json` reports overall/per-class/retain/forget test accuracy, firing
summaries, silence/tie rates, retries and runtime, and prediction disagreement
with Gold and the controls. Each simulated model saves confusion and per-image
counts, scores, truth and predictions. Gold is compared in class coordinates,
not neuron coordinates; independently trained neuron identities are not aligned.
The specification's MIA, training-member evaluation, probe/relearning audit and
retain-only reassignment ablation are not implemented here. No parameters are
selected using test results. Zero forgotten-class accuracy is not sufficient
evidence of forgetting, and the silenced receptive fields remain stored.

Tests:

```bash
python -m unittest test_method_a -v
```

The bounded integration test compiles a tiny network, checks silence even when
all six retries are exhausted, verifies frozen parameters, and compares unmasked
compiled responses with the existing runtime simulator under the same seed.
