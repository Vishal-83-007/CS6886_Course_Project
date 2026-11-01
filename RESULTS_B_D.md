# Results: Methods B and D (classes 6 and 9; random 10% subset)

N = 800 Diehl & Cook STDP network, compiled C++ (Brian2 `cpp_standalone`) on the IITM AQUA cluster; Fashion-MNIST uses weight normalisation 39. How to run each method: [METHOD_B_MNIST.md](METHOD_B_MNIST.md), [METHOD_B_FASHION.md](METHOD_B_FASHION.md), [METHOD_B_SUBSET.md](METHOD_B_SUBSET.md), [METHOD_D_MNIST.md](METHOD_D_MNIST.md), [METHOD_D_FASHION.md](METHOD_D_FASHION.md).

Baselines (all 10 classes, 10,000 test images): MNIST 89.37%, Fashion-MNIST 74.28% (42.84% before weight normalisation). Every number below comes from the `metrics.json` files printed by `aqua/progress.sh`. The Gold measurement on the forgotten subset images was still queued at submission and is marked pending.

## Setup
- **Gold:** retrained from scratch without classes 6 and 9 (MNIST: 6 and 9; Fashion: Shirt and Ankle boot).
- **B (SSD with hypothetical-STDP importance):**
  - select synapses where I(Df) > α·I(D);
  - multiply each selected synapse by β = min(λ·I(D)/I(Df), 1);
  - α = 2 throughout. λ = 0 means the selected synapses are set to zero.
- **D (replay with gated leak):**
  - replay retained images with STDP on and θ frozen;
  - each image, gated weights shrink by w *= (1 − leak);
  - the gate is either the neuron (baseline label 6/9) or the synapse (same selection as B).
- **Readouts:**
  - *fixed*: the baseline's neuron labels;
  - *all-class*: neurons relabelled using all 10 classes (500 images per class), i.e. what is still recoverable.
- **Splits:** the test set is split into tune (500 per class, used for choosing settings) and final (the rest, used for reporting).
- **Gap:** the mean absolute difference from Gold over the four numbers (fixed retained/removed, all-class retained/removed).
- **Statistical noise:** removed-class numbers rest on about 1,000 images, so their standard error is about 1.3–1.5 points.

## Final split (retained / removed, %)
| MNIST | Fixed | All-class | Gap to Gold |
|---|---|---|---|
| Gold | 92.09 / 0 | 84.11 / 76.53 | — |
| Baseline | 89.11 / 87.59 | 89.09 / 88.52 | 26.89 |
| B λ=0.5 | 91.74 / 0 | 82.30 / 73.94 | 1.19 |
| B λ=0.1 | 91.62 / 0 | 82.02 / 71.66 | 1.86 |
| B λ=0.5 + repair (20k, θ updated) | 91.45 / 0.41 | 83.46 / 73.22 | 1.25 |
| D neuron l0.001 r5000 | 91.54 / 0 | 83.51 / 79.83 | 1.11 |
| D synapse l0.001 r1000 | 91.77 / 0 | 82.47 / 72.70 | 1.45 |
| D synapse l0.005 r1000 | 91.45 / 0 | 82.05 / 72.60 | 1.66 |

| Fashion | Fixed | All-class | Gap to Gold |
|---|---|---|---|
| Gold | 81.10 / 0 | 71.05 / 69.40 | — |
| Baseline | 76.98 / 60.00 | 77.15 / 59.40 | 20.05 |
| B λ=0.5 | 80.88 / 2.10 | 70.40 / 62.90 | 2.37 |
| B λ=0.1 | 80.20 / 1.80 | 70.17 / 60.20 | 3.19 |
| D neuron l0.005 r1000 | 81.25 / 0 | 72.00 / 59.60 | 2.73 |
| D synapse l0.005 r1000 (partial normalisation) | 80.75 / 1.30 | 71.23 / 59.50 | 2.93 |

What these tables show:
- On these four numbers, the methods are not distinguishable from each other on either dataset.
- B + repair looked best on the tune split (gap 0.37), but on the final split it is level with the rest. Its tune advantage was selection noise.
- On Fashion, Gold's all-class removed accuracy (69.4) is above the baseline's own (59.4). So the Gold gap partly *rewards* leftover 6/9 information there; don't rank Fashion methods by it.

## Relapse test (tune split)
The test: 20,000 retained images of STDP with θ updated, starting from each unlearned model, then scored with the baseline's fixed labels. Shown as removed % (retained %).

| Start | MNIST | Fashion |
|---|---|---|
| Fine-tune only (baseline) | 94.5 (81.4) | 62.4 (71.6) |
| B λ=0.5 | 0.6 (91.9) | 33.8 (71.8) |
| B λ=0.1 | 0.1 (91.3) | 10.2 (77.4) |
| B λ=0 | 0.1 (92.0) | 7.7 (77.6) |
| D synapse, leak to 37% (l0.001 r1000) | 56.2 (89.0) | — |
| D synapse, leak to 0.7% (l0.005 r1000) | 0.0 (91.9) | 6.9 (78.0) |
| D neuron, leak to 0.7% | **0.0** (92.3) | **0.0** (80.0) |

Before the relapse test, λ = 0, 0.1 and 0.5 give the same results. They only differ once θ is allowed to adapt.

## Findings
1. **Homeostasis undoes gain reduction.**
   - A silent neuron's θ decays as exp(−t/τθ) with τθ = 10⁴ s. The predicted surviving fraction was 0.78 after 5k images and 0.37 after 20k. Observed: 0.78 / 0.75 after 5k and 0.38 / 0.43 after 20k (MNIST / Fashion).
   - After B on Fashion, the 6/9 neurons keep 29% of their weight sum, and their fields are unchanged (cosine 0.965 to the baseline).
   - Once θ decays, the original detectors fire again: 21 boot neurons (cosine 0.998 to their original fields) and 17 Shirt neurons (0.977).
2. **Removing the classes from the input stream shifts the homeostatic balance toward the removed-class neurons.**
   - During replay, retained-class neurons fire more and their θ rises (35.9 → 38.7 mV in the B run).
   - The 6/9 neurons fire less and their θ falls.
   - As a result, plain fine-tuning makes 6/9 *more* recognisable (MNIST: 88.0% → 94.5%).
3. **Relapse depends on gain left and whether the field is intact.** Across the runs:
   - fine-tune (100% weight) > D synapse at 37% (56%, fields intact, cosine 0.999) > B λ=0.5 (Fashion 29%, intact) > B λ=0.1 > B λ=0 ≈ D synapse fully leaked (Fashion floor of 7–8%, from synapses that were never selected) > D neuron (0%; no neuron fires even with θ down to 10–13 mV).
4. **B at λ → 0 and D synapse at full leak converge to the same weights** (identical weight totals: MNIST 41,841, Fashion 26,698). The axis that matters is *what is zeroed*: selected synapses, or every synapse of the removed-class neurons.
5. **Leftover information in shared features.**
   - With D neuron after relapse, no 6/9 neuron fires, yet 65 other MNIST neurons get relabelled 6/9, and all-class removed accuracy is 87.7%.
   - Gold also has this kind of information (77.2% all-class removed on the tune split, without ever seeing 6/9), but the baseline-derived networks carry more of it. None of the methods removes it.
6. **Related work:**
   - [Siddiqui et al. 2025](https://arxiv.org/abs/2505.22310): fine-tuning on the retain set alone restores forget-set accuracy for many unlearning methods; resistance is predicted by how far the weights moved from the original.
   - [Hu et al., ICLR 2025](https://arxiv.org/abs/2406.13356): the same effect ("benign relearning") in LLMs.

## Selection protocol
- **Finalists:** the smallest Gold gap on the tune split (the original rule).
- **Robust variants:** chosen *after* the relapse results were seen, and reported as such. Among settings within noise of the best tune gap, take the one with the least remaining removed-class weight:
  - MNIST D synapse: l0.005 r1000;
  - B: λ = 0.1 (λ = 0 is the limit).

## Random-subset forgetting (`sub6_9_p10`)
- **Forget set D_f:** a fixed random 10% of the training images of classes 6 and 9. MNIST: 1,185 images (591 sixes, 594 nines); Fashion: 1,200 (600 Shirts, 600 Ankle boots). The other 90% of those classes stays in training.
- **Selection rule:** for class 6 and then class 9, shuffle that class's training indices with `numpy.random.default_rng(46)` and keep the first 10%.
- **Gold:** retrained from scratch without D_f (MNIST 58,815 images, Fashion 58,800). D_f is also excluded from neuron labelling in every model's evaluation.
- **Method B (SSD):** I(D_f) comes from a separate importance pass over D_f only. α ∈ {2, 5, 10, 25}, λ = 1, as the SSD paper uses for random forgetting. Method D is not run:
  - its neuron gate is class-level by construction;
  - its synapse gate at full leak equals SSD with λ = 0.
- **Metrics:** computed on the full 10,000-image test set, with no setting chosen from them. Fixed labels = each model's own training labels (Gold) or the baseline's labels (baseline, SSD).
  - accuracy on D_f;
  - accuracy on an equal-size random sample of *kept* 6/9 training images;
  - accuracy on the test images of 6/9 and of the other classes;
  - a membership AUC: feature = true-class score minus the best other class; members = training images, non-members = test 6/9 images; 0.5 = no membership signal.

### What SSD selects
| α | MNIST: SSD selects | MNIST: selected by I(classes 6, 9) | MNIST: overlap | Fashion: SSD selects | Fashion: by I(classes) | Fashion: overlap |
|---|---|---|---|---|---|---|
| 2 | 136,421 (21.8%) | 140,348 | 129,776 | 108,876 (17.4%) | 92,290 | 79,409 |
| 5 | 34,911 (5.6%) | 18,788 | 13,164 | 7,887 (1.3%) | 0 | 0 |
| 10 | 39 | 0 | 0 | 19 | 0 | 0 |
| 25 | 0 | 0 | 0 | 0 | 0 | 0 |

Correlation of I(D_f) with I(classes 6, 9) over all synapses: 0.971 (MNIST), 0.944 (Fashion). The class-level ratio I(classes)/I(D) cannot exceed |D|/|6 ∪ 9|: 5.06 for MNIST, exactly 5 for Fashion.

### Results (fixed labels, %)
| MNIST | D_f (train) | Kept 6/9 (train) | Test 6/9 | Test other | Membership AUC D_f vs test |
|---|---|---|---|---|---|
| Baseline | 87.51 | 89.54 | 88.05 | 89.53 | 0.483 |
| Gold | pending (expected ≈ test 6/9) | — | 86.2* | 88.9* | pending (expected ≈ 0.5) |
| SSD α=25 (0 synapses) | 87.51 | 89.54 | 88.05 | 89.53 | 0.483 |
| SSD α=10 | 87.43 | 89.20 | 88.05 | 89.52 | 0.479 |
| SSD α=5 | 57.38 | 61.27 | 59.48 | 90.54 | 0.478 |
| SSD α=2 | 0.08 | 0.08 | 0.10 | 92.02 | 0.526 |

| Fashion | D_f (train) | Kept 6/9 (train) | Test 6/9 | Test other | Membership AUC D_f vs test |
|---|---|---|---|---|---|
| Baseline | 60.00 | 62.67 | 59.80 | 77.61 | 0.499 |
| Gold | pending (expected ≈ test 6/9) | — | 65.6* | 76.5* | pending (expected ≈ 0.5) |
| SSD α=25 (0 synapses) | 60.00 | 62.67 | 59.80 | 77.61 | 0.499 |
| SSD α=10 | 59.75 | 61.92 | 59.80 | 77.65 | 0.497 |
| SSD α=5 | 50.50 | 53.42 | 50.65 | 79.22 | 0.506 |
| SSD α=2 | 0.33 | 0.25 | 0.20 | 80.06 | 0.495 |

\* From Gold's training-run test evaluation, with Gold's own labels:
- MNIST: 6 = 94.68%, 9 = 78.10%; pooled over 958 + 1,009 test images; overall 88.37%.
- Fashion: Shirt = 40.6%, boot = 90.5%; overall 74.30%.

### Findings
1. **The baseline holds no image-specific memory of D_f.**
   - It classifies D_f as well as unseen test images of the same classes (MNIST 87.5 vs 88.1; Fashion 60.0 vs 59.8).
   - The membership AUC is about 0.5 (0.48–0.51) for D_f and for kept training images alike.
   - D_f and the kept 6/9 training images were *both* trained on, yet differ by 2–3 points. That is the sampling noise between two random sets of about 1,200 images.
2. **Importance cannot separate a random subset from its classes.**
   - I(D_f) correlates 0.94–0.97 with whole-class importance.
   - At α = 2, SSD selects mostly class-level synapses (95% overlap on MNIST) and wipes out 6/9 entirely. That is over-forgetting, the "Streisand effect" of Foster et al.
   - At α = 5 it damages the whole classes (MNIST −30 points, Fashion −9) on D_f, kept and test images equally.
   - At α ≥ 10 it changes (almost) nothing.
3. **No α removes D_f without removing its class.** The D_f, kept and test 6/9 columns move together at every α.
4. **Retraining noise exceeds any forgetting effect.**
   - Gold differs from the baseline on test 6/9 by −1.9 (MNIST) and +5.8 points (Fashion; Shirt is unstable between runs).
   - With one seed, "matching Gold" can only be judged within a few points.
   - The baseline and SSD α ≥ 10 are within that range; α ≤ 5 is far outside it.
5. **Conclusion:** in this prototype-based STDP network, random-subset forgetting needs no intervention. The right unlearning action is to do nothing, and importance-based dampening can only over-forget. This matches SSD's own random-forgetting result (retrained D_f accuracy 94.1%, about equal to its retain accuracy).

## Compute cost (single CPU core on AQUA)
| Step | MNIST | Fashion |
|---|---|---|
| Gold retraining (classes 6, 9 removed) | 13.7 h | 16.6 h |
| Importance pass over 60,000 training images (one-time per baseline; we ran two) | 6.6–6.9 h | 8.6–9.1 h |
| Method B dampening | seconds | seconds |
| Method D neuron-gate replay (reported setting) | 0.39 h | 0.13 h |
| Frozen evaluation (5,000 labelling + 5,000 or 10,000 test images) | 0.6–1.4 h | 0.9–1.9 h |

The importance file stores every class separately, so later requests for other classes need only the seconds-long dampening step.
