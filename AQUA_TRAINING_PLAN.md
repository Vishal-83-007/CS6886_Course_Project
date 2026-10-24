# Fashion-MNIST CPU training and unlearning plan

Updated 19 October 2026. Production is compiled C++ on CPU. Accelerator training
has been retired at the user's request after Aqua pilot 2422956 completed but
showed substantially worse behavior and throughput than the CPU reference.
Historical diagnostics/results are preserved. No remote jobs have been submitted
by this assistant.

## Locked first production suite

Use the existing conductance-based Diehl–Cook power-law STDP model, N=400,
dt=0.5 ms, 350 ms stimulation + 150 ms rest, and at most six attempts per
presentation. Keep model/rate/precision settings fixed across baseline and Gold.
Use three shuffled epochs, seed 0 first, split seed 42 and 500 allowed assignment
examples per available class. This is the selected initial production configuration,
not a claim that accuracy has converged or that hyperparameters were tuned.

The production job explicitly uses all 60,000 official training images for
baseline, 54,000 retain images for each Gold, and all 10,000 official test images
for final evaluation. Validation size is 0 in the production job; CLI development
runs still default to a held-out 5,000-image validation split. Do not change model
settings based on final test results. For intentional tuning, use separate
VALIDATION_SIZE=5000 runs, then regenerate matching final baseline/Gold models.

One independent model per one-core PBS allocation. Baseline and Gold start fresh
and can run concurrently. Gold has no dependency on trained baseline weights.
The launcher submits baseline, class6, class0, class9 and random10 at seed 0;
SEEDS="1 2" extends the paired suite. Queue admission controls concurrency;
recorded small-resource diagnostics suggested five such running jobs per user,
subject to current routing and other jobs. Do not request extra cores unless a
separate threading benchmark justifies them.

## Measured evidence and walltime

Aqua CPU job 2417846: identical image schedule, exact assignment/evaluation spike
responses and predictions for C++ and Cython; final weights differ by at most
2.78e-16. C++ phase times were 190.895s training (1,000 images), 23.8031s
assignment (100), and 22.7457s validation (100). Both scored 43/100 validation
accuracy. This is a short pilot, not final model accuracy.

Linear projection for the final baseline (180k training, 5k assignment, 10k
evaluation presentations) is 10.51h before compilation/I/O and headroom; 20%
headroom gives about 12.6h. Later-epoch spike activity and retries can change
rates substantially. Jobs request 48h; inspect progress and actual queue limits.

C++ saves inference weights/thresholds every 5,000 presentations and each epoch,
not full training state. A killed run cannot resume exactly and must restart
fresh. Keep the Cython reference/exact-resume fallback, but production jobs use
C++ only. No exact-resume capability is claimed for the compiled controller.

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

## Execution and outputs

See FASHION_AQUA_RUNBOOK.md for the CPU-only transfer archive, cleanup and
submission commands. Reuse the already validated CPU environment; no rebuild
is required. Each PBS job expands that archive locally and isolates its builds.
Durable result directories include request, seed, data split, backend and job ID.

Production trains all epochs, assigns neurons on allowed training examples with
learning frozen, then evaluates every test image with learning frozen. Saves
include config/data/source/order hashes, environment, manifests, inference
snapshots, final weights/theta, neuron labels, response counts, predictions,
class scores, confusion and per-class metrics, retry statistics and phase times.
The current runner does not yet implement the complete unlearning/MIA/audit
suite described above. Training the baseline/Gold models establishes its inputs.

Priority: inspect seed-0 baseline + class6, complete all seed-0 requests, then
seeds 1 and 2. Run the same selected configuration and backend for every paired
comparison. Defer N=1600 and fraction/request extensions until this suite is secure.
