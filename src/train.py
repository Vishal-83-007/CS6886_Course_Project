"""
Training loop for the Diehl & Cook baseline SNN.

Procedure (Diehl & Cook Sec. 2.6):
    1. Present training examples one at a time with STDP active.
    2. After training, freeze weights (STDP off).
    3. Assign each excitatory neuron the class it responded to most
       strongly on average, over one pass of (a subset of) the training
       set -- this is the ONLY place labels are used.
"""

import numpy as np
from tqdm import tqdm

from .encoding import present_example
from .network import normalize_weights


def train(components, X_train, y_train, verbose=True, weight_norm_target=78.0):
    """
    Run unsupervised STDP training over X_train. Labels are NOT used here.

    weight_norm_target: each excitatory neuron's total incoming weight is
    rescaled to this value after every presentation (see
    network.normalize_weights for why this is required, not optional --
    without it, individual neurons saturate at their refractory-limited
    max firing rate regardless of input, and the network never learns
    discriminative representations). Set to None to disable (not
    recommended -- kept only for debugging/ablation).
    """
    n = len(X_train)
    n_e = components['n_e']
    max_possible = n_e * 70  # refractory-limited max spikes per 350ms presentation
    consecutive_saturated = 0

    iterator = tqdm(range(n), desc='training') if verbose else range(n)
    if weight_norm_target is not None:
        normalize_weights(components, target_sum=weight_norm_target)
    for idx in iterator:
        counts = present_example(components, X_train[idx], learning_on=True)
        if weight_norm_target is not None:
            normalize_weights(components, target_sum=weight_norm_target)

        # --- Runaway detection -------------------------------------
        # KNOWN OPEN ISSUE (see README "Known issues"): the network can
        # occasionally fall into an input-independent oscillatory state
        # where the excitatory layer fires near its absolute maximum
        # rate regardless of what's presented. This is a recurrent E/I
        # dynamical instability, not a learning bug -- once entered, all
        # further training on that run is uninformative. We can't fully
        # prevent it yet, so at minimum we detect and report it loudly
        # rather than silently continuing to "train" on garbage.
        frac_saturated = counts.sum() / max_possible
        if frac_saturated > 0.5:
            consecutive_saturated += 1
        else:
            consecutive_saturated = 0
        if consecutive_saturated == 3:
            print(f"\n[WARNING] Excitatory layer has been >50% of max firing "
                  f"rate for 3 consecutive presentations (example {idx}). "
                  f"This usually means the network has entered the known "
                  f"input-independent oscillatory instability -- see "
                  f"README 'Known issues'. Training beyond this point is "
                  f"likely not learning anything useful. Consider "
                  f"stopping and re-running with a different seed, or "
                  f"see README for mitigation options.\n")
    return components


def assign_labels(components, X_assign, y_assign, verbose=True):
    """
    Run examples through the frozen network and record, for each
    excitatory neuron, its average firing rate per digit class.
    Returns: neuron_labels (n_e,) -- the class each neuron is assigned to.
    """
    n_e = components['n_e']
    rate_sums = np.zeros((10, n_e))
    class_counts = np.zeros(10)

    iterator = tqdm(range(len(X_assign)), desc='assigning labels') if verbose else range(len(X_assign))
    for idx in iterator:
        counts = present_example(components, X_assign[idx], learning_on=False)
        digit = int(y_assign[idx])
        rate_sums[digit] += counts
        class_counts[digit] += 1

    class_counts[class_counts == 0] = 1  # avoid div by zero
    avg_rates = rate_sums / class_counts[:, None]     # (10, n_e)
    neuron_labels = np.argmax(avg_rates, axis=0)       # (n_e,)
    return neuron_labels, avg_rates
