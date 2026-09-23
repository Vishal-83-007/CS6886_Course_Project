"""Visualization helpers: plot the learned input->excitatory receptive
fields as a grid of 28x28 images, matching Fig. 2A of Diehl & Cook / Fig. 4
of the ASP paper."""

import numpy as np
import matplotlib.pyplot as plt


def get_weight_matrix(components):
    """Returns (n_e, 784) matrix of input->excitatory weights."""
    syn = components['syn_input_exc']
    n_e = components['n_e']
    n_input = components['n_input']
    w = np.zeros((n_input, n_e))
    w[syn.i[:], syn.j[:]] = syn.w[:]
    return w.T  # (n_e, 784)


def plot_receptive_fields(components, save_path=None, max_neurons=100):
    w = get_weight_matrix(components)
    n_e = min(w.shape[0], max_neurons)
    grid_size = int(np.ceil(np.sqrt(n_e)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for idx in range(grid_size * grid_size):
        ax = axes.flat[idx]
        ax.axis('off')
        if idx < n_e:
            img = w[idx].reshape(28, 28)
            ax.imshow(img, cmap='hot_r')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def save_weights(components, path):
    """Save raw synapse weights + connectivity indices for later reload
    (needed so the unlearning stage can start from the trained baseline
    without retraining)."""
    syn = components['syn_input_exc']
    np.savez(path, w=np.array(syn.w[:]), i=np.array(syn.i[:]), j=np.array(syn.j[:]),
              n_e=components['n_e'], n_input=components['n_input'])


def load_weights(components, path):
    data = np.load(path)
    syn = components['syn_input_exc']
    # assumes same connectivity ordering (all-to-all, built the same way)
    syn.w = data['w']
    return components
