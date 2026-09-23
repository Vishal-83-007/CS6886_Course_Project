"""
Training loop -- faithful to the original main loop's structure:
    normalize_weights() is called once per presentation, BEFORE setting
    that presentation's input rates (source: the `else: normalize_weights()`
    branch right before `rates = ...` in the main while-loop, active
    whenever not in test_mode).

Neuron -> class label assignment matches get_new_assignments(): for each
neuron, across the examples shown, take the class with the highest
average firing rate (source uses argmax-by-comparison; same result as
np.argmax over the per-class average rate matrix).
"""

import numpy as np
from tqdm import tqdm

from .encoding import present_example
from .network import normalize_weights


def train(components, X_train, y_train, verbose=True, weight_norm_target=78.0):
    n = len(X_train)
    n_e = components['n_e']
    max_possible = n_e * 70
    consecutive_saturated = 0

    iterator = tqdm(range(n), desc='training') if verbose else range(n)
    for idx in iterator:
        if weight_norm_target is not None:
            normalize_weights(components, target_sum=weight_norm_target)
        counts = present_example(components, X_train[idx], learning_on=True)

        frac_saturated = counts.sum() / max_possible
        if frac_saturated > 0.5:
            consecutive_saturated += 1
        else:
            consecutive_saturated = 0
        if consecutive_saturated == 3:
            print(f"\n[WARNING] Excitatory layer >50% of max firing rate for "
                  f"3 consecutive presentations (example {idx}). See README "
                  f"'Known issues' -- training beyond this point may not be "
                  f"learning anything useful.\n")
    return components


def assign_labels(components, X_assign, y_assign, verbose=True):
    n_e = components['n_e']
    rate_sums = np.zeros((10, n_e))
    class_counts = np.zeros(10)

    iterator = tqdm(range(len(X_assign)), desc='assigning labels') if verbose else range(len(X_assign))
    for idx in iterator:
        counts = present_example(components, X_assign[idx], learning_on=False)
        digit = int(y_assign[idx])
        rate_sums[digit] += counts
        class_counts[digit] += 1

    class_counts[class_counts == 0] = 1
    avg_rates = rate_sums / class_counts[:, None]
    neuron_labels = np.argmax(avg_rates, axis=0)
    return neuron_labels, avg_rates
