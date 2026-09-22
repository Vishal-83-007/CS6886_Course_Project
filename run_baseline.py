"""
Baseline training script: Diehl & Cook STDP-trained SNN on MNIST.

Run this directly on Kaggle (as a script, or paste cells into a notebook).
It will:
    1. Load MNIST (with fallback chain, see src/data.py)
    2. Build the Diehl & Cook 3-layer SNN (src/network.py)
    3. Train with STDP, unsupervised, on a configurable subset
    4. Assign class labels to neurons post-hoc
    5. Evaluate test accuracy
    6. Save: trained weights (.npz), receptive-field plot (.png),
       neuron_labels (.npy) -- all needed as the starting point for the
       unlearning stage (Phase 2 of the project).

Tune N_EXCITATORY / N_TRAIN / N_TEST at the top based on your time budget:
    - Quick proposal demo:  N_EXCITATORY=100, N_TRAIN=1000  (~20-30 min)
    - Stronger baseline:    N_EXCITATORY=400, N_TRAIN=3000  (~70-90 min)
    - Paper-scale:          N_EXCITATORY=6400, N_TRAIN=60000 (many hours,
                             do this only once you have the full pipeline
                             validated at smaller scale)
"""

import os
import time
import numpy as np

from src.network import build_network
from src.data import load_mnist, subset_per_class
from src.train import train, assign_labels
from src.evaluate import evaluate
from src.utils import plot_receptive_fields, save_weights

# ---------------------------------------------------------------------
# Config -- edit these
# ---------------------------------------------------------------------
N_EXCITATORY = 400
N_TRAIN = 1000          # balanced across 10 classes (N_TRAIN/10 per class)
N_ASSIGN = 500           # subset used for post-hoc neuron->label assignment
N_TEST = 500             # subset of the MNIST test set used for accuracy
SEED = 0
OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./results"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------
print("Loading MNIST...")
X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()
X_train, y_train = subset_per_class(X_train_full, y_train_full, N_TRAIN // 10, seed=SEED)
X_assign, y_assign = subset_per_class(X_train_full, y_train_full, N_ASSIGN // 10, seed=SEED + 1)
X_test, y_test = subset_per_class(X_test_full, y_test_full, N_TEST // 10, seed=SEED + 2)
print(f"train={len(X_train)}  assign={len(X_assign)}  test={len(X_test)}")

# ---------------------------------------------------------------------
# 2. Network
# ---------------------------------------------------------------------
print(f"Building network with {N_EXCITATORY} excitatory neurons...")
components = build_network(n_excitatory=N_EXCITATORY, seed=SEED)

# ---------------------------------------------------------------------
# 3. Train (unsupervised STDP)
# ---------------------------------------------------------------------
print("Training...")
t0 = time.time()
train(components, X_train, y_train, verbose=True)
print(f"Training took {time.time() - t0:.1f}s")

# Save trained weights immediately -- this is the checkpoint the
# unlearning stage (Phase 2) will load and operate on.
save_weights(components, os.path.join(OUT_DIR, "baseline_weights.npz"))
print("Saved baseline_weights.npz")

# ---------------------------------------------------------------------
# 4. Assign neuron -> class labels (post-hoc, frozen weights)
# ---------------------------------------------------------------------
print("Assigning labels...")
neuron_labels, avg_rates = assign_labels(components, X_assign, y_assign, verbose=True)
np.save(os.path.join(OUT_DIR, "neuron_labels.npy"), neuron_labels)
print("Neuron label distribution:", np.bincount(neuron_labels, minlength=10))

# ---------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------
print("Evaluating on test set...")
t0 = time.time()
accuracy, preds = evaluate(components, neuron_labels, X_test, y_test, verbose=True)
print(f"Evaluation took {time.time() - t0:.1f}s")
print(f"\n=== BASELINE TEST ACCURACY: {accuracy * 100:.2f}% ===\n")

with open(os.path.join(OUT_DIR, "baseline_accuracy.txt"), "w") as f:
    f.write(f"accuracy={accuracy}\nn_excitatory={N_EXCITATORY}\nn_train={N_TRAIN}\nn_test={N_TEST}\n")

# ---------------------------------------------------------------------
# 6. Visualize learned receptive fields
# ---------------------------------------------------------------------
print("Plotting receptive fields...")
plot_receptive_fields(components, save_path=os.path.join(OUT_DIR, "receptive_fields.png"))
print(f"All outputs saved to {OUT_DIR}")
