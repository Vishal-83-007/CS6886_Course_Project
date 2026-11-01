# Four Local-Plasticity Unlearning Approaches: Implementation Plan

This document specifies only the four methods in the project proposal. Baseline and Gold checkpoints already exist for MNIST and Fashion-MNIST; training those models is outside scope. Equations and parameter sweeps below are implementation choices for this project unless explicitly attributed to a source.

**Request convention:** treat classes **6 and 9 as jointly removed**: `forget_classes = {6, 9}`. For MNIST these are digits 6 and 9; for Fashion-MNIST they are Shirt and Ankle boot. Run each method independently on each dataset's Baseline and compare with its corresponding Gold trained without both classes. A Gold missing both classes is not a matched reference for a request to remove only one of them.

**Shared implementation contract used by all four methods**

- Let `D` be the actual data used to train the Baseline, `Df = {x in D: y in {6,9}}`, and `Dr = D \ Df`. Use original dataset IDs rather than renumbering the retained classes.
- Load a fresh Baseline copy for every method, variant, and hyperparameter trial. The Gold is a comparison model, never an input to the weight-edit rule or replay process.
- Modify the existing input-to-excitatory weights `W[i,j]` of shape `[784, N]`. Infer `N`, weight bounds, encoding, simulator timestep, presentation duration, inhibition, and the exact STDP rule from the supplied checkpoint/configuration. Keep the excitatory-inhibitory circuitry fixed.
- Preserve the existing preprocessing and Poisson rate scale. Reset membrane potentials, conductances, refractory state, and spike traces between independent samples using the Baseline's reset/rest protocol. Restore persistent adaptive thresholds from the checkpoint; do not accidentally reset them to their training initialization.
- Freeze adaptive threshold values during importance estimation, primary unlearning runs, and inference: disable both spike-triggered increments and threshold decay. An explicit threshold-adaptation ablation may enable both during replay, then freeze the resulting values for evaluation.
- Keep the original neuron-to-class assignments and voting rule for the primary comparison. Do not delete classes 6/9 from the voting head to manufacture forgetting. Evaluate any retain-only reassignment separately; apply the same reassignment procedure to an unchanged Baseline as a readout control.
- Disable automatic weight renormalization in the primary unlearning implementations. Renormalization can undo uniform neuron-level shrinkage and redistribute synapse-level edits. If tested, report it as a separate variant with its exact placement.
- For every resulting checkpoint, record retain-class performance and forget-input responses relative to Baseline and Gold, using fixed inference seeds. Zero forget accuracy alone does not demonstrate representation erasure, particularly when Gold cannot output removed labels. Inspect spike responses/receptive fields as well. Select parameters on validation data, not test data.

## A. Label-Guided Neuron Silencing

### Mechanism and inputs

Disable excitatory neurons whose saved assigned labels belong to the forget request. This is a coarse intervention on whole neurons, requiring the Baseline's label assignment and the requested class IDs but no image replay.

Let `a[j]` be the saved label of neuron `j`:

$$
N_f=\{j:a[j]\in\{6,9\}\}.
$$

Each neuron has one assigned label under argmax assignment, even if it responds to several classes. Therefore, a neuron outside `Nf` may still encode forget-class features, and a neuron inside `Nf` may support retained classes.

### Implementation

1. Load the Baseline and its saved assignments; construct `forget_neuron_mask = isin(assignments, [6, 9])`.
2. Prefer a persistent excitatory spike-generation mask that prevents selected neurons from spiking. Keep it active during simulation, not just while counting votes. Their paired inhibitory neurons then receive no excitatory spikes through the usual pathway.
3. If a native spike-generation mask is unavailable, set an explicit per-neuron threshold override high enough to prevent firing under the configured simulator, and verify zero spikes. Store this override separately from the frozen homeostatic threshold.
4. The proposal's weight-edit alternative is `W[:, forget_neuron_mask] = 0`. Use this as a separately identified implementation if desired. Zero afferent weights are equivalent to functional silence only when there is no other drive that can make those neurons fire.
5. Save the mask or threshold override with the checkpoint. Keep the original class assignments, including the assignments of silenced neurons.

```python
model = load_fresh_baseline(dataset)
mask = np.isin(model.assignments, [6, 9])
model.set_excitatory_spike_enabled(~mask)  # simulator adapter
save_checkpoint(model, method="A", forget_neuron_mask=mask)
```

### Required checks and interpretation

- Verify selected neurons emit no spikes on both forget and retain inputs, and the simulator does not silently restore their thresholds.
- Record the number and fraction of neurons silenced and the retain-class losses. If no neuron is assigned to a requested class, report an empty intervention rather than assigning neurons using test data.
- In the mask implementation, the underlying receptive fields remain stored. In the zero-weight implementation, selected receptive fields are removed, but residual forget information may remain elsewhere.

This method tests the usefulness of label-based localization. Functional silence or disappearance of a class output is not evidence that the complete network matches retraining. No relearning or capacity recycling is included in Method A.

## B. Selective Synaptic Dampening with Hypothetical STDP Importance

### Mechanism and inputs

Retain the selection-and-shrinkage structure of SSD, replacing its gradient/Fisher importance with a **local STDP-update proxy**. This proxy is not Fisher information and has no automatic equivalence to it.

Inputs are the Baseline, forget images `Df`, and a cached whole-training-set importance tensor `I_D`. If that cache does not exist, compute it once from the already available Baseline; no Baseline retraining is needed. This preparation requires access to `D` and must be counted separately from request-time work.

### Compute the importance tensors without changing the model

For each image `x` and Poisson realization `k`, simulate the frozen Baseline while accumulating the updates its original STDP rule **would** produce:

$$
U_{ij}(x,k)=\sum_{e\in\mathcal E(x,k)}\delta w^{\mathrm{STDP}}_{ij,e}(W_0),
\qquad
I_{ij}(S)=\frac{1}{K|S|}\sum_{x\in S}\sum_{k=1}^{K}U_{ij}(x,k)^2.
$$

Here `E(x,k)` is the set of plasticity events in that presentation; evaluate every event using unchanged Baseline weights `W0` and evolving spike traces. This is the squared **net hypothetical update per presentation**, matching the proposal's definition, not a sum of squared event updates.

1. Implement the Baseline STDP expression once as an event-update function. Reuse it in a recording mode that updates an accumulator instead of `W`.
2. Allow synaptic traces to evolve normally while freezing weights and thresholds. Do not turn off the trace machinery when disabling learning.
3. Reset the presentation accumulator and transient neuron state for each image/realization. Square the accumulated update before averaging; do not square the dataset-average signed update.
4. Use the same encoding, update convention, durations, and `K` for `I_D` and `I_f`. Start with `K=1`; compare against `K=3` for noise sensitivity.
5. Cache `I_D` with checkpoint hash, data/sampling IDs, STDP configuration, seeds, `K`, and normalization convention. Never reuse a cache across datasets or different Baseline weights.

Positive and negative events can cancel in `U`, and saturated synapses may have tiny updates despite strong selectivity. An optional importance ablation can use per-presentation event energy `sum_e(delta_w_e**2)` instead; identify it separately and use that convention for both tensors.

### Apply the SSD-style edit

For numerically reliable entries, select and dampen:

$$
M_{ij}=\mathbf 1\{I_{ij}(D_f)>\alpha I_{ij}(D)\},
\qquad
\beta_{ij}=\min\left(1,\frac{\lambda I_{ij}(D)}{I_{ij}(D_f)}\right),
$$

$$
W'_{ij}=\begin{cases}\beta_{ij}W_{ij},&M_{ij}=1,\\W_{ij},&M_{ij}=0.\end{cases}
$$

Mask entries whose `I_f` is below a documented numerical floor before division. This avoids dividing by zero without giving an arbitrary additive epsilon disproportionate influence on small importance values. An example floor is `max(float32_tiny, 1e-8 * I_D.max())`, subject to inspection of the actual scale.

```python
I_D = load_or_compute_importance(baseline, D, config)
I_f = compute_importance(baseline, Df, config)
floor = max(np.finfo(np.float32).tiny, 1e-8 * I_D.max())
selected = (I_f > alpha * I_D) & (I_f > floor)
W_new = baseline.W.copy()
beta = np.minimum(1.0, lam * I_D[selected] / I_f[selected])
W_new[selected] *= beta
```

For this shrinking rule, larger `alpha` selects fewer weights, while smaller positive `lambda` produces stronger dampening of selected weights. Do not renormalize afterwards. Clip only to the Baseline's legal weight range.

### Parameter choices and checks

- Proposed initial grid: `alpha in {1, 2, 3, 4}`, `lambda in {0.1, 0.5, 1.0}`. Include the unchanged Baseline as the no-edit control.
- For nonnegative sample-mean importance, with `rho = |Df|/|D|`, the exact decomposition is `I_D = rho*I_f + (1-rho)*I_r`. Thus `I_f/I_D <= 1/rho` wherever defined. With approximately balanced classes and two classes removed, `rho ≈ 0.2`, so the ratio is bounded by approximately 5. A strict selection threshold at or above that bound selects nothing. This bound holds exactly for consistently computed full-data tensors; independent Monte Carlo estimates can deviate.
- Two-class deletion removes about 20% of the data, so the original SSD approximation `I_D ≈ I_r` should not be assumed exact. Keep `I_D` for the main proposal method; optionally derive `I_r = (I_D-rho*I_f)/(1-rho)` using matched per-sample statistics as a separately labeled sensitivity variant.
- Verify importance collection leaves weights and persistent thresholds unchanged, all edit factors lie in `[0,1]`, and unselected weights are unchanged. Save selection masks, importance tensors, and shrinkage factors.

## C. Plasticity-Based Erasure: Anti-STDP and LTD-Only

### Mechanism and inputs

Replay only `Df` while changing the Baseline's local plasticity update. C1 and C2 are two variants of this one approach, each starting independently from the Baseline. Neither variant uses retain replay in its primary implementation.

Let `delta_w_base` denote one event's signed update from the **actual Baseline STDP rule**, evaluated using the current weights and traces. Use `s_u > 0` as a dimensionless multiplier of that rule's existing learning rate.

### C1: Full anti-STDP

Reverse the signed local update at every original plasticity event:

$$
\delta w^{C1}_{ij}=-s_u\,\delta w^{\mathrm{base}}_{ij}.
$$

Ordinary potentiation becomes depression, but ordinary depression becomes potentiation. Consequently, C1 may strengthen background or otherwise uncorrelated inputs. This is an empirical risk to measure, not an inverse of the full training trajectory.

### C2: LTD-only erasure

Apply only the negative of the Baseline's potentiating component:

$$
\delta w^{C2}_{ij}=-s_u\max(\delta w^{\mathrm{base}}_{ij},0).
$$

For a Baseline with separately implemented LTP and LTD events, negate the LTP contribution and suppress ordinary LTD. For a Diehl-Cook-style post-event trace rule

$$
\delta w^{\mathrm{base}}_{ij}=\eta(x_i^{\mathrm{pre}}-x_{\mathrm{tar}})(w_{\max}-w_{ij})^{\mu},
$$

take the positive part of this expression at the original post-spike event. Do not introduce an additional pair-based timing rule unless the Baseline actually uses one. The proposal's causal-pair description motivates this variant; this definition adapts it to the implemented trace rule.

### Replay procedure

1. Load a fresh Baseline; freeze thresholds and disable weight normalization.
2. Shuffle `Df` using a recorded seed. Encode each presentation with the original preprocessing and Poisson scale.
3. Let the network generate spikes with its current edited weights. Compute C1 or C2 updates online at the Baseline's plasticity event times, using the same trace/event ordering.
4. Apply each event update and clip to `[w_min, w_max]`. Preserve inhibition and neuron labels.
5. Reset transient state between samples. Save checkpoints after 1, 2, and 5 passes; evaluate with all learning disabled.

```python
def erasure_update(delta_w_base, variant, scale):
    if variant == "C1":
        return -scale * delta_w_base
    if variant == "C2":
        return -scale * np.maximum(delta_w_base, 0.0)
    raise ValueError(variant)
```

### Parameter choices and checks

- Proposed sweep: `s_u in {0.1, 0.5, 1.0}`, passes in `{1, 2, 5}`. Save intermediate checkpoints from the same trajectory rather than rerunning every prefix.
- Use fixed presentation durations by default. Do not silently increase input rates or repeat silent images until they spike: such retries can counteract erasure and alter cost. If the Baseline relies on retries, document the changed unlearning policy and keep ordinary inference behavior consistent across comparisons.
- Count positive/negative applied updates, clipped weights, and spikes per presentation. C2 must never increase an individual weight in the absence of normalization; C1 may increase weights.
- If forget-input firing collapses, event-driven erasure can stall. Report the stall; stronger suppression does not automatically imply more complete erasure.
- Compare frozen thresholds against one explicitly labeled adaptation ablation. Separate synaptic effects from homeostatic suppression.

## D. Request-Gated ASP-Inspired Leak with Retain Replay

### Mechanism and inputs

Apply normal STDP on retained examples while selectively shrinking forget-associated weights. This is the proposal's request-gated adaptation of ASP's forgetting/recovery idea, not a reproduction of ASP and not the dopamine-based CFN algorithm.

Inputs are the Baseline, a fixed forget gate, and retained images `Dr` or a class-balanced subset `R` drawn from its eight retained classes. The neuron-gated variant can construct its gate from saved class assignments alone. The synapse-gated variant needs `I_D` and `I_f` from Method B; once computed, it does not replay forget images during this method's weight-edit phase.

### Construct and freeze the gate

**D1: Neuron gate**

$$
g_j=\mathbf 1\{a[j]\in\{6,9\}\},\qquad g_{ij}=g_j.
$$

Broadcast across the input dimension. All afferent weights of selected neurons leak equally; other neurons receive only retain STDP updates.

**D2: Synapse gate**

Use the same reliability floor and selection convention as Method B:

$$
g_{ij}=\mathbf 1\{I_{ij}(D_f)>\alpha I_{ij}(D)\}\,
\mathbf 1\{I_{ij}(D_f)>\varepsilon_{\mathrm{floor}}\}.
$$

A soft gate is optional; binary gating is the primary implementation. Compute all gates once from the unchanged Baseline and keep them fixed through retain replay. Gate updates would define a different adaptive procedure.

### Apply decay and retained-data recovery

The intended continuous-time combination is

$$
\frac{dw_{ij}}{dt}=\text{retain-STDP event contributions}-\kappa g_{ij}w_{ij},
$$

where `kappa` has units of inverse simulated time. Implement normal STDP events and multiplicative exponential leak:

$$
w_{ij}\leftarrow w_{ij}\exp(-\kappa g_{ij}\Delta t).
$$

Use continuous per-synapse decay between events when available, or apply the multiplier at every simulation timestep with a documented ordering. This guarantees nonnegative decay factors; an Euler factor `1-kappa*g*dt` can become negative for large steps.

1. Load a fresh Baseline, construct D1 or D2, freeze thresholds, and disable fixed-sum/fixed-norm weight normalization.
2. Choose and save a class-balanced retained replay subset `R`. Begin with 100 images per retained class, or all available examples if fewer exist. No forget samples enter the replay stream.
3. Replay `R` in shuffled order with normal Baseline STDP, scaled by an optional recovery multiplier `s_r`.
4. Apply gated leak during the active presentation window. Turn it off during inter-image rest for the primary protocol so its total exposure is unambiguous. Reset transient traces/state between images.
5. Clip to legal bounds. After each pass, freeze learning and leak for validation and checkpointing. Resume from the same trajectory for later passes.
6. Save weights, gate, gate provenance, replay IDs/order, `s_r`, and total active leak duration.

Parameterize decay by the intended **total log-decay budget** `b = kappa*T_active`, where `T_active = passes*|R|*T_present`. Choose `b in {0.1, 0.5, 1, 2}` and set `kappa = b/T_active`. In the absence of STDP and clipping, a gated weight then ends at `exp(-b)` times its original value. If the desired experiment instead holds the physical leak rate fixed across replay lengths, fix `kappa` and report the resulting different budgets explicitly.

### Parameter choices and checks

- Compare D1 and D2 independently. Suggested passes: `{1, 2, 5}`; recovery multiplier `s_r in {0.1, 1.0}`; D2 selection thresholds from Method B. Expand the replay subset only if retained-class recovery remains inadequate.
- Run retain-STDP-only (`g=0`) and leak-only (`s_r=0`) controls to identify which component causes the observed changes.
- Check that zero-gated synapses receive no leak, leak alone never increases weights, and no forget image enters replay. Normal retain STDP may still modify zero-gated weights and may increase gated weights.
- Do not renormalize each incoming weight column after D1's uniform shrinkage: it would exactly cancel pure uniform shrinkage if no other change intervenes. Any normalization ablation must therefore be interpreted separately.
- Keep the original voting assignments for the primary result. Retain-only reassignment may reveal that a formerly forget-labeled neuron has been repurposed, but its effect must be separated from the synaptic intervention using the unchanged-Baseline readout control.

**Source distinction:** the proposal defines Method D using ASP-inspired leak and retain replay. The attached CFN paper (Allred and Roy, 2020, Sections 2.2 and 2.4) instead uses novelty-triggered dopaminergic stimulation and enhanced local learning to recycle less-used representations. Adding that machinery would change the model architecture and method; it is not required for any of these four implementations.

**Source basis:** `proposal_final_report.pdf`, approach slides 7-11, defines Methods A-D. `Selective-Synaptic-Dampening.pdf`, Equation 4 and Algorithm 1, supplies Method B's selection/dampening structure; the hypothetical-STDP importance is this project's substitution. `CFN-SNN.pdf` supplies the related-work distinction above, not an implementation of ASP. The original ASP paper is referenced by the proposal but is not among the attached sources.
