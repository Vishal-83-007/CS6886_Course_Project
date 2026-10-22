"""Visualization helpers: plot the learned input->excitatory receptive
fields as a grid of 28x28 images, matching Fig. 2A of Diehl & Cook / Fig. 4
of the ASP paper."""

import numpy as np
import matplotlib.pyplot as plt
from brian2 import volt


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
    """Save the learned weights and thresholds needed for frozen inference."""
    syn = components['syn_input_exc']
    if (not np.isfinite(syn.w[:]).all() or
            not np.isfinite(components['exc'].theta[:] / volt).all()):
        raise FloatingPointError('Nonfinite weights/thresholds: refusing invalid checkpoint')
    np.savez(path, w=np.array(syn.w[:]), i=np.array(syn.i[:]), j=np.array(syn.j[:]),
              theta_volts=np.array(components['exc'].theta[:] / volt),
              n_e=components['n_e'], n_input=components['n_input'])


def load_weights(components, path):
    with np.load(path) as data:
        syn = components['syn_input_exc']
        if 'theta_volts' not in data:
            raise ValueError('Checkpoint predates threshold saving; retrain the baseline')
        if (int(data['n_e']) != components['n_e'] or
                int(data['n_input']) != components['n_input'] or
                not np.array_equal(data['i'], syn.i[:]) or
                not np.array_equal(data['j'], syn.j[:])):
            raise ValueError('Checkpoint connectivity does not match this network')
        syn.w = data['w']
        components['exc'].theta = data['theta_volts'] * volt
    return components
