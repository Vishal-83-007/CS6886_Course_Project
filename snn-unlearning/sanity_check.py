"""
Internal sanity check (not part of the deliverable pipeline): uses
synthetic 28x28 patterns instead of real MNIST (no internet access in
this dev sandbox) to confirm the full STDP + homeostasis + label
assignment + evaluation pipeline actually learns to discriminate classes,
before handing the code off.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np

from src.network import build_network
from src.train import train, assign_labels
from src.evaluate import evaluate

rng = np.random.default_rng(0)


def make_pattern(kind):
    img = np.zeros((28, 28))
    if kind == 0:
        img[2:10, 2:10] = 255       # top-left bright block
    elif kind == 1:
        img[10:18, 10:18] = 255     # center bright block
    elif kind == 2:
        img[18:26, 18:26] = 255     # bottom-right bright block
    return img.flatten()


def make_dataset(n_per_class, noise_std=25):
    X, y = [], []
    for k in range(3):
        base = make_pattern(k)
        for _ in range(n_per_class):
            noisy = base + rng.normal(0, noise_std, size=784)
            X.append(np.clip(noisy, 0, 255))
            y.append(k)
    X, y = np.array(X), np.array(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


X_train, y_train = make_dataset(n_per_class=30)
X_test, y_test = make_dataset(n_per_class=15)

t0 = time.time()
comp = build_network(n_excitatory=15, seed=0)
print('network built')

train(comp, X_train, y_train, verbose=False)
print('training done in', time.time() - t0, 's')

t0 = time.time()
neuron_labels, avg_rates = assign_labels(comp, X_train, y_train, verbose=False)
print('label assignment done in', time.time() - t0, 's')
print('neuron labels:', neuron_labels)

t0 = time.time()
acc, preds = evaluate(comp, neuron_labels, X_test, y_test, verbose=False)
print('evaluation done in', time.time() - t0, 's')
print('TEST ACCURACY:', acc, '(chance = 0.333)')
print('predictions:', preds)
print('true labels:', y_test)
