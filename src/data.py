"""
MNIST loading with a fallback chain so this works on Kaggle (with or
without internet toggled on) and on a plain local/CI machine.

Order tried:
    1. tensorflow.keras.datasets.mnist   (pre-cached on most Kaggle images)
    2. torchvision.datasets.MNIST         (downloads if needed)
    3. sklearn.datasets.fetch_openml('mnist_784')  (downloads if needed)

Returns numpy arrays: X_train, y_train, X_test, y_test
    X_*: (N, 784) float32, values in [0, 255]
    y_*: (N,) int
"""

import numpy as np


def load_mnist():
    # ---- Try 1: keras ----
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype(np.float32)
        X_test = X_test.reshape(-1, 784).astype(np.float32)
        return X_train, y_train.astype(int), X_test, y_test.astype(int)
    except Exception:
        pass

    # ---- Try 2: torchvision ----
    try:
        from torchvision import datasets
        train_ds = datasets.MNIST(root='/tmp/mnist', train=True, download=True)
        test_ds = datasets.MNIST(root='/tmp/mnist', train=False, download=True)
        X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32)
        y_train = train_ds.targets.numpy().astype(int)
        X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32)
        y_test = test_ds.targets.numpy().astype(int)
        return X_train, y_train, X_test, y_test
    except Exception:
        pass

    # ---- Try 3: sklearn / openml ----
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    return X_train, y_train, X_test, y_test


def subset_per_class(X, y, n_per_class, seed=0):
    """Balanced subset: n_per_class examples of each digit 0-9, shuffled."""
    rng = np.random.default_rng(seed)
    idx_all = []
    for digit in range(10):
        idx = np.where(y == digit)[0]
        chosen = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        idx_all.append(chosen)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]
