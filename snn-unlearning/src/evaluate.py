"""
Test-set evaluation: present each test image with plasticity OFF, get the
excitatory spike-count vector, average by assigned neuron label, and
predict the class with the highest average firing rate.
"""

import numpy as np
from tqdm import tqdm

from .encoding import present_example


def predict(components, neuron_labels, image, n_e):
    counts = present_example(components, image, learning_on=False)
    class_rates = np.zeros(10)
    class_neuron_totals = np.zeros(10)
    for label in range(10):
        mask = (neuron_labels == label)
        if mask.sum() > 0:
            class_rates[label] = counts[mask].mean()
    return int(np.argmax(class_rates)), counts


def evaluate(components, neuron_labels, X_test, y_test, verbose=True):
    n = len(X_test)
    correct = 0
    preds = np.zeros(n, dtype=int)
    iterator = tqdm(range(n), desc='testing') if verbose else range(n)
    for idx in iterator:
        pred, _ = predict(components, neuron_labels, X_test[idx], components['n_e'])
        preds[idx] = pred
        if pred == int(y_test[idx]):
            correct += 1
    accuracy = correct / n
    return accuracy, preds
