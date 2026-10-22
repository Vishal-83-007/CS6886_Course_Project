# Fashion-MNIST baseline, Gold and unlearning on Aqua

Updated after uploaded diagnostics, 18 October 2026. Fashion-MNIST only; MNIST
is handled by another member. The Fashion-MNIST runner, runtime checkpoints,
compiled controller and PBS pilot/training scripts are now implemented. See the
runbook for tested paths and remaining CUDA validation. No new Aqua jobs have
been submitted from this workspace.

## Decisions

Keep the conductance-based Diehl–Cook power-law STDP model, N=400. Start with
three shuffled epochs and save one-/two-/three-epoch checkpoints. Tune using
validation, not test data. No Fashion-MNIST accuracy promise can be inferred
from the earlier MNIST experiment.

Use the live two-running-GPU-job limit. GPU count and job count are separate.
For the first GPU pilot, request both V100s on one node and mask one worker to
each, because the one-GPU diagnostic did not expose device isolation. Keep CPU
work in separate allocations. Baseline and Gold can train concurrently after the
configuration is locked: Gold starts fresh and never needs baseline weights.

## Confirmed Aqua diagnostics (18 October 2026)

The uploaded `aqua_diagnostics/2406453.hn1/probe.log` and
`2406456.hn1/probe.log` supersede public-policy assumptions:

- Live `gpuq`: `max_run = [u:PBS_GENERIC=2]`, i.e. the generic per-user limit
  is TWO running GPU jobs, not the public page's four. No per-user GPU-count
  limit is shown. `resources_max.ngpus=24` is a queue job resource maximum,
  not an entitlement to 24 GPUs. Maximum GPU walltime is 48h.
- GPU node: two Tesla V100-PCIE-32GB GPUs; driver 470.57.02, driver-reported
  CUDA 11.4. GCC 9.2.0; default toolkit 10.1. CUDA 11.4 is available: explicitly
  unload cuda10.1/load cuda11.4. Newer listed toolkit modules do not by themselves
  establish runtime compatibility with this driver.
- One requested GPU exposed BOTH devices with `CUDA_VISIBLE_DEVICES` unset.
  Do not guess a free GPU from utilization. The initial GPU pilot requests both
  GPUs on one node and launches one explicitly masked worker per allocated GPU.
  That consumes one running GPU job. A second job may be possible but is not a
  guaranteed GPU allocation. Do not expand beyond admitted resources.
- CPU node: Xeon Gold 6248, 40 physical cores; GPU host: Gold 6142, 32 cores.
  Server small-resource limits: `smallc=80`, `smallq=5` running, `smallq=20`
  queued-resource limit per generic user. These are custom tracked resources;
  with the observed `smallq=1` per CPU job this suggests up to five such jobs,
  subject to routing/account rules. The short CPU probe actually ran in small8,
  despite the example's small20 submission; use the actual allocated queue.
- Default Python failed on `importlib.metadata`; the user subsequently confirmed
  login Python 2.6.6 and module anaconda3_2024.10. Use the new CPU setup job to
  create dedicated Python 3.10 CPU/GPU environments. The probe now prints Python
  version before optional package checks.

Ready jobs and commands are in `FASHION_AQUA_RUNBOOK.md`. Training can begin
on CPUs while CUDA is validated. Use the resumable Cython runner for long jobs,
compiled CPU when the measured run fits a single allocation, and CUDA only for
bounded pilots until the controller passes on actual Aqua hardware. CUDA source
is implemented but has not been compiled/run locally: no nvcc/GPU here, and
GPU execution remains untested. Its dependencies now install/import offline
under Python 3.12, but this is not GPU-controller validation.

## What counts as unlearning, and how many Gold models

The [SSD paper](https://arxiv.org/pdf/2308.07707) tests full-class, subclass and
random-example forgetting. The [NeurIPS challenge](https://unlearning-challenge.github.io/)
defines the target relative to retraining without the deleted data. There is
no universal required number of classes, seeds or deletion fractions.

A single class is sufficient for debugging and a narrow single-class claim.
For this project, use this preregistered main suite; it is our design choice,
not a universal benchmark prescription:

| Request | Deleted training examples | Purpose |
| --- | --- | --- |
| class6 | All 6,000 Shirt examples | First pilot; shared clothing features |
| class0 | All 6,000 T-shirt/top examples | Second clothing-class request |
| class9 | All 6,000 Ankle boot examples | Different visual category |
| random10 | Fixed stratified random 6,000, 600 per class | Example deletion with all classes retained |

Class IDs and splits follow [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
Different feature overlap is a hypothesis to inspect through confusion/firing
results, not an established difficulty ranking. Do not select requests based
on favorable final results. These are independent requests from the original
baseline, not cumulative deletions.

Each seed needs one shared baseline and four fresh retain-only Gold models:
five training runs. Target three paired model seeds = 15 runs. Keep deletion
IDs fixed across seeds; use a separate recorded split seed for random10. If
budget allows, repeat its mask to measure deletion-set variability separately.
A Gold is reused across methods and hyperparameters for the same request,
configuration and seed, but cannot be reused for another retain set.

Priority: baseline + class6 Gold at seed 0; complete four requests at seed 0;
then complete seeds 1 and 2. Single-seed results are preliminary. Optional
extensions are 1%/5% random deletion, partial-class deletion, all-ten-class
leave-one-out, and sequential requests. Each new retain set adds Gold cost.
Do not call an arbitrary half-class split a semantic subclass.

## Data, readout and evaluation rules

- Develop on stratified 55k training / 5k validation, with immutable IDs,
  checksums, orders, assignment IDs and forget/retain masks. Assignment is a
  recorded subset of training, never validation/test. Training/assignment
  overlap is allowed but must be included in membership/exposure accounting.
- Final locked runs use all official 60k training images for baseline and 54k
  retain images for each Gold, all 10k test images for final evaluation.
  Three epochs means 180k baseline or 162k Gold training presentations.
- Gold excludes deleted IDs from STDP, assignment, replay and fitted readout
  or calibration. Use fresh initialization, paired seeds, the same configuration
  and retained order where applicable. No baseline fine-tuning disguised as Gold.
- Baseline and random10 Gold use 5k allowed assignment examples, balanced by
  available class. Class Gold uses 4.5k, 500 per remaining class. Assignment
  must not contain any forgotten ID; record assignment exposure for MIA.
- Keep class IDs stable. With class removal, retained-only neuron assignment
  has no neuron labeled as the deleted class. Zero forget-class accuracy can
  therefore be an output/readout effect. Include unchanged-baseline/readout
  suppression as a control and inspect representation/firing behavior.
- For unlearning, report original fixed labels AND retain-only reassigned
  labels, including reassignment cost. Gold has its own labels: baseline and
  Gold neuron identities are not aligned. Represent inactive neurons explicitly
  rather than assigning every dead neuron to class 0; define silence/ties.
- For random10, all ten classes remain. Gold may correctly classify deleted
  images through generalization. Do not aim for zero accuracy: match Gold.

Compare baseline, Gold, unlearned and no-op controls on identical IDs and
Poisson evaluation seeds. Report training-forget and training-retain metrics
separately from held-out test metrics. Class requests split test into 1k
removed-class and 9k retained-class examples. Random deletion has no deleted
test members; report overall/per-class test accuracy and deleted-training
behavior separately. Method A is a class-level method, not a random-example
method; label inapplicable comparisons accordingly.

Save per-class confusion, per-image spike counts/class scores/predictions,
silence/tie rates, neuron coverage, behavioral gaps to Gold and runtime.
Compare class-coordinate outputs, not raw Gold neuron indices or weight
distances. Include three fixed Poisson realizations on an audit subset plus
one complete test pass per final model; budget additional passes separately.

MIA: class-matched nonmembers, balanced member/nonmember evaluation, disjoint
attack fitting/calibration and reporting. Nonmembers must never train or assign
neurons. Compare the same attack on baseline, Gold and unlearned models. Save
ROC-AUC, clearly defined attack advantage and low-FPR TPR only where sample
counts support it. Raw spike scores are not probabilities; any fitted probability
calibration must use allowed retain data only. Begin with a modest score-based
attack and disclose its power; shadow-model audits are additional compute.

[Improved benchmarks](https://arxiv.org/html/2405.19211v1) shows that passing a
weak MIA need not survive stronger attacks and highlights repeated requests.
Neither finite MIA results, low accuracy nor relearning tests certify exact
unlearning. Include representation and relearning diagnostics as empirical
probes. Compare behavior with the distribution of Gold results across seeds.

## Implementation sequence and current state

1. Fashion-MNIST cached IDX/NumPy loader, offline execution and dataset checks;
   deterministic manifests for class0/class6/class9/random10 and development
   validation. Assert exclusion, counts and train/test separation. No new MNIST work.
2. Import-safe runner; true epochs, actual unique/presentation counts; baseline/
   Gold mode; seed/config/forget IDs; separate output paths and phase timing.
   Current sampler caps counts and cannot implement repeated epochs.
3. Atomic periodic checkpoints: weights/theta, membrane/conductance/pretrace,
   timers/refractory/time state, RNG, order/cursor. Verify exact continuation
   on the same backend. Existing weight/theta saves are inference only. If CUDA
   resume is approximate, say so and do not rely on it for controlled primary runs.
4. Start CPU pilots after these foundations. Existing Cython/NumPy execute on
   CPU even inside GPU jobs; preserve Cython as the reference.
5. Refactor presentation/retry control into compiled C++ standalone/Brian2CUDA.
   Preserve 350 ms stimulation, 150 ms rest, dt=0.5 ms, six-attempt cap, STDP
   during failed attempts and frozen assignment/evaluation. No Python callback
   as a standalone solution, per-image compilation/restart or full weight copy.
   Chunk inputs with complete state/time continuity and bounded artifacts.
6. CPU/GPU semantics checks with deterministic inputs, then stochastic pipeline
   checks. RNG streams need not be bitwise equal. Benchmark 100 warmup + 1,000
   measured images with compilation separate, and repeat at a later checkpoint.
   Record retries/failures, spikes, utilization, weight/theta distributions and
   all toolchain versions. Benchmark compiled CPU with 1/2/4 threads too.
7. Lock dependencies against actual Aqua driver/toolkit/compiler. CUDA 11.4 in
   the example is not proof it is the only toolkit. Time-box the initial CUDA
   feasibility spike to about 90 minutes after setup; keep CPU work running.
   This is a decision checkpoint, not a promise of completing a port in 90 minutes.
8. Implement evaluation above, then real CPU/GPU training job scripts with
   isolated caches/builds and recoverable outputs. Use scratch for execution;
   source and durable artifacts remain outside temporary scratch as site policy
   requires. Save checkpoints during execution, not only at EXIT. Plan graceful
   checkpoint/exit before walltime and resume in a new allocation.

Keep duration, precision, rate gain and retries unchanged during backend checks.
Validate intentional tuning separately. Validation must not mutate continuing
training state or RNG; use frozen copies. Lock configuration before expensive
Gold runs and regenerate matching Gold models if it changes. CPU execution and compiled-loop tests now pass locally. CUDA remains a smoke-test
path until actual Aqua compilation and semantic checks succeed.

## CPU + GPU schedule

First submit the CPU pilot, comparing Cython and compiled C++ on identical
1,000-image Fashion-MNIST pools. In parallel, a two-GPU pilot runs baseline and
class6 with explicit per-worker masks. The GPU runner is intentionally restricted
to <=1,000 unique training images and one epoch pending Aqua validation.

After validation/config lock, use two independent workers (baseline + class6,
then random10 + class9, then class0 + next-seed baseline). This may be one
admitted two-GPU job or separate correctly bound allocations. The generic limit
is TWO running GPU jobs. If a second dual-GPU job is admitted, four workers can
advance the suite; do not equate a job limit with guaranteed GPU capacity.
Gold has no dependency on baseline weights and can start at config lock.

CPU production: one core per Cython model (exact resume supported). Several
independent CPU jobs may run within the observed small-resource limits. Compiled
CPU is the preferred speed experiment; `aqua_cpp_train.cmd` is available once the
pilot predicts completion within 48h with headroom. Its snapshots are inference
only, not exact training continuation. More requested CPUs do not automatically
parallelize Cython. Benchmark explicit OpenMP 1/2/4 threads before reserving them.

CPU/GPU comparisons need validated equivalent dynamics; until then keep complete
matched baseline/Gold pairs on the same backend. Do not merge independently
trained STDP weights to simulate parallel training. CPU preparation, plotting,
evaluation and later importance work can free GPU capacity.

## Conditional runtime and budget

No Fashion-MNIST Aqua rate is measured. The old 9h/30k from another dataset is
not a Fashion-MNIST estimate. These counts include a single final test pass:

- Baseline: 180k train + 5k assign + 10k test = 195k presentations.
- Class Gold: 162k + 4.5k + 10k = 176.5k each.
- random10 Gold: 162k + 5k + 10k = 177k.
- One-seed suite = 901.5k; three-seed suite = 2,704.5k presentations.

| Assumed effective s/presentation | Baseline | Class Gold | One-seed total device-hours | Three-seed total device-hours |
| --- | ---: | ---: | ---: | ---: |
| 0.10 | 5.42 h | 4.90 h | 25.04 h | 75.13 h |
| 0.25 | 13.54 h | 12.26 h | 62.60 h | 187.81 h |
| 0.50 | 27.08 h | 24.51 h | 125.21 h | 375.63 h |

One-seed five-model suite with four GPUs: approximately 9.8/24.5/49.0h at
those rates; with two GPUs: 14.7/36.8/73.5h. Baseline + class6 concurrently:
5.4/13.5/27.1h. Add at least 20% headroom plus setup/build/queue wait. These
are arithmetic scenarios, not V100 or A100 predictions. Different phases,
requests and checkpoints can have different rates. Device-hours mean GPU-hours
only on GPUs; CPU jobs need independently measured rates.

Use `Tbuild + Ptrain*strain + Passign*sassign + Ptest*stest + Tcheckpoint`.
Rates include retries; do not multiply them twice. Add validation, stochastic
audits, importance accumulation, privacy-model training and unlearning separately.
Slow CPU runs may exceed 48h and require validated checkpoint continuation.
One-epoch baseline with assign/test is 75k: ~2.1h at 0.1 s or ~5.2h at 0.25 s,
before headroom. Today target probes, usable code, pilots and first recoverable
runs; a complete validated CUDA refactor tonight is not promised.

A100 access can be requested in parallel if approval is slow. IITM lists A100s
under [PARAM Shakti](https://cc.iitm.ac.in/hpce/paramshakti.html); confirm its
account, scheduler, charging and queue limits separately. N=400 can be dominated
by control/kernel-launch overhead. Request measured compute, not an assumed
V100-to-A100 speed ratio. Compare time-to-result including queue/setup.

## Two-week delivery

| Dates | Deliverable |
| --- | --- |
| Oct 18 | Submit probes, manifests/checkpoint work, CPU pilot, CUDA feasibility |
| Oct 19–20 | Backend correctness, measured throughput, validation/config lock |
| Oct 21–22 | Seed-0 baseline + four Gold models; remaining seeds started |
| Oct 23–25 | A/B, hypothetical-STDP I(D), limited validation sweeps |
| Oct 26–28 | Anti-STDP/LTD-only/gated leak; remaining paired runs |
| Oct 29–30 | Held-out MIA, behavioral/representation/relearning and cost results |
| Oct 31–Nov 1 | Reruns, figures, reproducible report and buffer |

First gate is reliable baseline + class6 Gold. Final target is four requests
and three paired seeds, with all limitations explicit. Defer N=1600, all-ten-
class Gold, fraction sweeps and stretch goals first if time is tight. Importance
is separately benchmarked frozen observation, not free training output; report
one-time I(D), per-request forgetting, reassignment, inference and retraining costs.
N=1600 requires its own Gold and budget (4x input synapses, ~16x inhibitory
connections), so it is optional after the N=400 experiment is secured.

Additional implementation references:

- [Brian standalone execution](https://brian2.readthedocs.io/en/stable/user/computation.html)
- [Brian2CUDA GPU selection](https://brian2cuda.readthedocs.io/en/latest/introduction/cuda_configuration.html)
- [Brian2CUDA installation](https://brian2cuda.readthedocs.io/en/latest/introduction/install.html)
- [Aqua operational documents](https://cc.iitm.ac.in/hpce/aqua.html)

## Setup correction after job 2406770

The compute node reports glibc 2.17 but failed DNS resolution for
conda.anaconda.org before creating environments. The user confirmed the installed
anaconda3_2024.10 module already provides Python 3.12.7. The setup now reuses
that interpreter in isolated venvs and installs transferred glibc-2.17-compatible
CPython 3.12 wheels with --no-index; no online Conda environment creation.
See the runbook and scripts/prepare_aqua_wheels.py. Earlier Python 3.10 setup
recommendations are superseded for Aqua. Keep exact versions fixed for resume.
