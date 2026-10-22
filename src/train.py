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


def train(components, X_train, y_train, verbose=True):
    """Run unsupervised STDP training over X_train. Labels are NOT used here."""
    n = len(X_train)
    iterator = tqdm(range(n), desc='training') if verbose else range(n)
    for idx in iterator:
        present_example(components, X_train[idx], learning_on=True)
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
