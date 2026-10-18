# Local Fashion-MNIST validation, 18 October 2026

This is local CPU evidence, not Aqua/V100 performance or a final accuracy result.
The actual Aqua pilot remains required. Source and model settings accompany each
run in `results/fashion/` (ignored generated artifacts).

Official Fashion-MNIST gzip checksums passed. Combined array SHA256:
`4ebb3c784723f3893b570544d19021412b2d875f8d6ab717fbe94827b682d34d`.
The checked gzip data is staged under ignored `data/fashion/` for offline transfer.

N=400 compiled CPU, seed 0, split seed 42, 1,000 balanced training examples,
one epoch, 100 assignment examples, 100 held-out validation examples:

| Phase | Presentations | Seconds |
| --- | ---: | ---: |
| STDP training | 1,000 | 77.6025 |
| Frozen label assignment | 100 | 10.4226 |
| Frozen validation | 100 | 10.1509 |

Build + execution: 114.04 seconds; simulation phases totaled 98.18 seconds.
Training throughput: 0.0776 s/presentation INCLUDING retries. Of 1,200 total
examples, attempt counts were [907, 167, 81, 24, 15, 6]; four exhausted all retries.
Validation accuracy: 43%, silence: 0%, tied class scores: 4%, unassigned neurons: 0.
This tiny validation set and short training do not establish converged accuracy.

A purely linear projection at these local phase rates is roughly 3.9h baseline
training plus ~0.4h final assignment/test (about 4.3h total), excluding later-epoch activity changes,
headroom and compilation. It is not an Aqua estimate. Use target CPU/GPU pilots.

Regression coverage includes deletion/assignment exclusion, common retained
example order, distinct epochs, no silent sample caps, exact runtime resume,
compiled-clock continuity, finite weights, and frozen weights/thresholds. An
initial clock-rewind bug in the generated-loop prototype was detected and fixed;
its regression test checks elapsed simulated time and frozen observation.

The runtime and compiled CPU implementations preserve the existing model.
CUDA code is restricted to bounded pilots and remains uncompiled/unexecuted here.
GPU dependencies were subsequently installed/import-checked offline under Python
3.12; no GPU/nvcc execution is available here.

Retain-only class6 pilot also completed with 900 training examples (the same
1,000-example pool minus Shirt), 90 retained assignment examples and 100
validation examples. Training/assignment/validation took 93.72/14.71/14.68s.
Validation accuracy was 19% overall, 21.1% on retained classes; Shirt accuracy
was zero under retained-only assignment. Different short-run accuracy is not an
unlearning result or a convergence comparison. The two pilots ran under differing
concurrent local CPU load; do not interpret their timing difference as request cost.
Final and end-of-training weights/thresholds were identical during frozen
observation in this Gold run. Six repository regression checks passed in the final suite (161 seconds).

Cython continuation was additionally verified across fresh processes. Brian 2.9
stores process-local random-buffer pointers in its default RNG checkpoint; the
new runner separately saves buffer contents and restores newly allocated
buffers. Without this handling, cross-process Cython resume failed. After the
fix, checkpoint/resume produced bit-for-bit identical weights and responses.
The repository includes a dedicated regression for this failure.

Offline recovery after Aqua job 2406770: downloaded glibc-2.17 CPython-3.12
wheels; created fresh local Python 3.12.11 CPU/GPU environments with --no-index;
both pip checks and imports passed. Three Python-3.12 data/compiled-loop tests
passed (18.6s). A Python-3.12 Cython resume test could not compile locally because
the local system Python lacks Python.h; this does not diagnose Aqua's Anaconda
headers. Prior Python-3.10 Cython resume checks passed. Aqua pilots must establish
its actual Cython/compiler/GPU behavior.
