# Fashion-MNIST Method A on Aqua

Method A is post-training label-guided neuron silencing. No Fashion baseline
retraining, new learning rule or new dependencies are required. The new
`run_method_a_fashion.py` entry point reuses the MNIST Method A implementation
with Fashion dataset checks, checkpoint defaults and collateral-effect reporting.
MNIST entry points keep their existing defaults. Method C is unchanged.

The default source is
`fashion/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_wn39_cpp_2507283`:
800 excitatory neurons, seed 0, three epochs over all 60,000 training examples,
500 assignment examples per class, compiled CPU, training weight normalization
39 and input-total-Hz normalization off. Its historical full-test accuracy was
74.28%; shirt 29.8%, T-shirt/top 75.5%, pullover 62.6%, dress 89.0%, coat 55.2%.
Of 1,000 true shirts, its saved confusion matrix predicts 226 as T-shirt/top,
211 as pullover and 132 as coat. This makes per-class collateral effects useful
to measure. Shirt-only silences 75/800 neurons; class6/9 silences 131/800.
The older `fashion_normal_cpu_aqua/` artifacts use a different 400-neuron setup;
do not mix their Gold models with this baseline.

Weight normalization was applied during learning only. Frozen Method A inference
loads the saved normalized weights, disables STDP and theta adaptation, and does
not normalize again. The original pixel/4 Poisson rates, 0.5 ms timestep,
350 ms presentation, 150 ms rest and six-attempt retry protocol are preserved.
If an alternate checkpoint uses `input_total_hz`, inference reads that setting.
Gold must match the dataset, training counts, seeds, neuron count, input encoding,
training normalization and exact deletion request.

Each request starts independently from the original baseline:

| Request | Intervention | Gold |
| --- | --- | --- |
| `class6` | Shirt only | No automatic Gold; supply a matching shirt-only model |
| `class6_9` (default) | Shirt and ankle boot | Existing `gold_weights/fashion_class6_9_wn39` selected automatically |
| `class0`, `class9`, etc. | Other independent class deletions | Explicit matching Gold required for a retraining comparison |

Gold is comparison-only. Pass `--no-gold` locally or `NO_GOLD=1` to the PBS job
to run joint deletion without the automatic Gold checkpoint.

The joint class6/9 Gold cannot measure shirt-only deletion. A shirt-only run still
compares baseline, Method A and the readout-suppression control without Gold.

## Transfer and run

Build locally, then transfer the small code archive:

```bash
python3 scripts/package_method_a.py --dataset fashion
scp -P 40826 method_a_fashion_code.tar.gz ep23b048@aqua.iitm.ac.in:~/CS6886/
```

On Aqua, from the project root containing `jobs/aqua_common.sh`, the existing
CPU environment, data and source model folders:

```bash
cd ~/CS6886
tar -xzf method_a_fashion_code.tar.gz
qsub -v REQUEST=class6,EVAL_LIMIT=100 jobs/fashion_method_a.cmd
qsub -v REQUEST=class6_9,EVAL_LIMIT=100 jobs/fashion_method_a.cmd
```

After successful smoke jobs, run the full 10,000-image test set:

```bash
qsub -v REQUEST=class6 jobs/fashion_method_a.cmd
qsub -v REQUEST=class6_9 jobs/fashion_method_a.cmd
```

The archive installs `method_a_fashion_runtime/`, the new job and this guide.
It includes matching network/encoding/data code and tests; it does not replace
Aqua's training sources, MNIST runtimes or environment helper. No data, weights
or environments are bundled. No remote submission happens during packaging.
Regenerate the bundle after edits.

Job defaults: one CPU core, 12 hours, `$HOME/fashion-cpu/bin/python`,
`data/fashion/`, and outputs under `results/fashion_method_a/<request>_<jobid>/`.
Check timing in the smoke logs before assuming a full job fits the allocation.
Override `FASHION_PYTHON`, `FASHION_DATA`, `FASHION_RESULTS`, `BASELINE_DIR`,
`GOLD_DIR`, `INFERENCE_SEED` or `EVAL_LIMIT` using `qsub -v` if Aqua paths differ.
For archive-staged environments the job reuses `results/fashion/environments`;
override `FASHION_ENV_ARCHIVES` if needed. Keep the four official Fashion gzip
IDX files together; the loader verifies checksums and the checkpoint fingerprint.

Local preparation needs NumPy only:

```bash
python3 run_method_a_fashion.py --prepare-only --out-dir results/fashion_a_prepared
python3 run_method_a_fashion.py --request class6_9 --prepare-only --out-dir results/fashion_a_joint_prepared
```

For a new matched shirt-only Gold, the existing Fashion trainer already supports
the required configuration. Run this inside a suitably allocated Aqua CPU job,
using the current training code, not the Method A runtime:

```bash
"$FASHION_PYTHON" run_fashion.py --dataset fashion --data-dir data/fashion \
  --request class6 --seed 0 --split-seed 42 --backend cpp --threads 1 \
  --n-excitatory 800 --epochs 3 --validation-size 0 --train-limit 0 \
  --assign-per-class 500 --eval-limit 0 --weight-norm 39 --input-total-hz 0 \
  --out-dir results/fashion_gold_class6_wn39
```

Then pass `GOLD_DIR=results/fashion_gold_class6_wn39` for a new shirt-only Method A
job. The existing `jobs/aqua_cpp_train.cmd` has 400-neuron defaults and does not
pass weight normalization; its defaults do not reproduce this 800-neuron Gold.
Gold starts with fresh initialization and excludes shirt from learning and
assignment. This retraining is optional for Method A execution.

## Interpret the output

`summary.json` reports accuracy, retain/forget accuracy, each class's accuracy,
silence/ties, retries, firing counts, verified frozen parameters and prediction
disagreement. Each model saves confusion matrices and per-image responses.
`fashion_comparison.json` adds named per-class accuracy differences against the
freshly evaluated baseline, counts of correct-to-wrong and wrong-to-correct
changes, each class's rate of shirt predictions, and where true shirts are
predicted. Inspect retained classes 0 (T-shirt/top), 2 (pullover), 3 (dress) and
4 (coat) separately, alongside aggregate retain accuracy. Accuracy differences
are fractions; multiply by 100 for percentage points.

The checkpoint copies original weights/labels and saves the mask and +1 volt
threshold override separately in `intervention.npz`. Inference must apply that
override; loading the weights alone restores ordinary baseline behavior. Silence
is checked after every attempt including rest. Original labels remain in the
readout. A separate readout-suppression control suppresses forgotten class scores
without changing baseline spikes.

All models use the same test IDs/order and initial inference seed, starting from
fresh builder state with continuous inter-image rest. Model-dependent retries
consume different RNG lengths. Fresh baseline accuracy can differ from historical
training-run evaluation because terminal simulator state was not saved. Use the
fresh baseline for paired comparisons. Low shirt accuracy after silencing alone
does not establish forgetting: shirt starts weak, retained representations may be
shared, and silenced weights remain stored. No MIA or relearning audit is added.
The 100-image smoke jobs validate execution, not deletion quality.

Validation command (CPU environment with Brian2 and a C++ compiler):

```bash
python3 -m unittest test_method_a test_method_a_fashion -v
```

Local validation used the existing offline CPU wheels (Python 3.12, Brian2 2.9,
NumPy 1.26.4). All 11 Method A, Fashion Method A and Method C regression tests
passed. Balanced 10-image shirt-only and joint class6/9 runs with the real
800-neuron checkpoint verified frozen parameters, zero silenced spikes and
output generation. The joint run executed from the extracted transfer bundle
and included the matching Fashion Gold. These small execution checks do not
estimate full-test performance. No Aqua job was submitted from this workspace.
