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

Use --help to set neuron and presentation counts. Run a short pilot on the
target CPU to estimate time before committing to a longer training run.
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from brian2 import prefs

from src.network import build_network
from src.data import load_mnist, subset_per_class
from src.train import train, assign_labels
from src.evaluate import evaluate
from src.utils import plot_receptive_fields, save_weights

# ---------------------------------------------------------------------
# Defaults. Use --help to override without editing the script.
# ---------------------------------------------------------------------
N_EXCITATORY = 400
N_TRAIN = 30000         # exploratory budget; paper-scale 400-neuron run used 180000
N_ASSIGN = 2000
N_TEST = 1000
SEED = 0
DEFAULT_OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./results"

parser = argparse.ArgumentParser(description='Train and evaluate the CPU Diehl-Cook baseline')
parser.add_argument('--n-excitatory', type=int, default=N_EXCITATORY)
parser.add_argument('--n-train', type=int, default=N_TRAIN)
parser.add_argument('--n-assign', type=int, default=N_ASSIGN)
parser.add_argument('--n-test', type=int, default=N_TEST)
parser.add_argument('--seed', type=int, default=SEED)
parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
parser.add_argument('--backend', choices=('cython', 'numpy'), default='cython',
                    help='Brian2 CPU code generation target (default: compiled Cython)')
args = parser.parse_args()
for field in ('n_train', 'n_assign', 'n_test'):
    value = getattr(args, field)
    if value < 10 or value % 10:
        parser.error(f'--{field.replace("_", "-")} must be a positive multiple of 10')
if args.n_excitatory < 1:
    parser.error('--n-excitatory must be positive')

N_EXCITATORY, N_TRAIN = args.n_excitatory, args.n_train
N_ASSIGN, N_TEST, SEED = args.n_assign, args.n_test, args.seed
OUT_DIR = args.out_dir
prefs.codegen.target = args.backend

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
train_seconds = time.time() - t0
print(f"Training took {train_seconds:.1f}s ({train_seconds / N_TRAIN:.3f}s/image)")

# Save trained weights immediately -- this is the checkpoint the
# unlearning stage (Phase 2) will load and operate on.
save_weights(components, os.path.join(OUT_DIR, "baseline_weights.npz"))
print("Saved baseline_weights.npz")

# ---------------------------------------------------------------------
# 4. Assign neuron -> class labels (post-hoc, frozen weights)
# ---------------------------------------------------------------------
print("Assigning labels...")
t0 = time.time()
neuron_labels, avg_rates = assign_labels(components, X_assign, y_assign, verbose=True)
assign_seconds = time.time() - t0
np.save(os.path.join(OUT_DIR, "neuron_labels.npy"), neuron_labels)
np.save(os.path.join(OUT_DIR, "per_class_spike_rates.npy"), avg_rates)
print("Neuron label distribution:", np.bincount(neuron_labels, minlength=10))

# ---------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------
print("Evaluating on test set...")
t0 = time.time()
accuracy, preds = evaluate(components, neuron_labels, X_test, y_test, verbose=True)
test_seconds = time.time() - t0
print(f"Evaluation took {test_seconds:.1f}s")
print(f"\n=== BASELINE TEST ACCURACY: {accuracy * 100:.2f}% ===\n")

confusion = np.zeros((10, 10), dtype=int)
np.add.at(confusion, (y_test.astype(int), preds), 1)
np.save(os.path.join(OUT_DIR, "confusion_matrix.npy"), confusion)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(confusion, cmap='Blues')
ax.set(xlabel='Predicted digit', ylabel='True digit',
       xticks=range(10), yticks=range(10), title='MNIST confusion matrix')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

with open(os.path.join(OUT_DIR, "baseline_accuracy.txt"), "w") as f:
    for key, value in dict(accuracy=accuracy, n_excitatory=N_EXCITATORY,
                           n_train=N_TRAIN, n_assign=N_ASSIGN, n_test=N_TEST,
                           seed=SEED, backend=args.backend, train_seconds=train_seconds,
                           assign_seconds=assign_seconds,
                           test_seconds=test_seconds).items():
        f.write(f'{key}={value}\n')

# ---------------------------------------------------------------------
# 6. Visualize learned receptive fields
# ---------------------------------------------------------------------
print("Plotting receptive fields...")
fig = plot_receptive_fields(components, save_path=os.path.join(OUT_DIR, "receptive_fields.png"))
plt.close(fig)
print(f"All outputs saved to {OUT_DIR}")
