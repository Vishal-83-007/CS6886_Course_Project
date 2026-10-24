"""Selective Synaptic Dampening with hypothetical-STDP importance (Method B).

Offline, numpy only (seconds). From a frozen importance pass on the trained
baseline (importance.npz: per-class sums of squared hypothetical updates):

    I(S)   = sum_{c in S} sumsq_c / sum_{c in S} n_c
    D      = all ten classes, D_f = the removed classes (e.g. class6_9)
    select (i, j)        if  I_ij(D_f) > alpha * I_ij(D)
    w'_ij = beta_ij w_ij,    beta_ij = min(lambda * I_ij(D) / I_ij(D_f), 1)   (selected only)

Foster et al. (SSD) use the training-set importance I(D), computed once, and
alpha in [5, 50], lambda in [0.1, 1], with D_f one class out of 20 or 100.
Because D contains D_f, I_ij(D_f) / I_ij(D) can never exceed |D| / |D_f|:
10 for one MNIST class, about 5 for two (class6_9). So here alpha must lie
between 1 and that bound (e.g. 1.5 - 4 for class6_9); larger alpha selects
nothing. Thresholds theta are not changed.
Writes an npz in the baseline_weights.npz format (for run_frozen.py evaluate)
and a JSON with what was changed.

    python dampen.py --importance <importance run>/importance.npz \\
        --weights <baseline run>/baseline_weights.npz --forget class6_9 \\
        --alpha 10 --lam 1 --out dampened_a10_l1.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.fashion_data import forget_classes


def importance(sumsq, class_count, classes):
    classes = list(classes)
    return sumsq[classes].sum(axis=0) / class_count[classes].sum()


def dampen(w, i_d, i_f, alpha, lam):
    selected = i_f > alpha * i_d
    beta = np.ones_like(w)
    beta[selected] = np.minimum(lam * i_d[selected] / i_f[selected], 1.0)
    return w * beta, selected, beta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--importance', nargs='+', required=True,
                   help='importance.npz of one or more passes (averaged)')
    p.add_argument('--weights', required=True, help='baseline_weights.npz of the trained baseline')
    p.add_argument('--forget', required=True, help='removed classes, e.g. class6_9')
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--lam', type=float, required=True)
    p.add_argument('--out', required=True, help='output .npz')
    a = p.parse_args()
    removed = forget_classes(a.forget)
    if not removed:
        p.error('--forget must look like class6 or class6_9')
    sumsq, count = 0, 0
    for path in a.importance:
        with np.load(path) as data:
            sumsq = sumsq + data['sumsq']
            count = count + data['class_count']
            n_e = int(data['n_e'])
    with np.load(a.weights) as data:
        weights = {k: data[k] for k in data.files}
    if int(weights['n_e']) != n_e or weights['w'].shape != (784 * n_e,):
        raise ValueError('Importance and weights come from different network sizes')
    i_d = importance(sumsq, count, range(10))
    i_f = importance(sumsq, count, removed)
    alpha_bound = float(count.sum() / count[list(removed)].sum())
    if a.alpha >= alpha_bound:
        print(f'WARNING: alpha={a.alpha} >= |D|/|D_f| = {alpha_bound:.2f}: no synapse can be selected')
    new_w, selected, beta = dampen(weights['w'], i_d, i_f, a.alpha, a.lam)
    out = Path(a.out)
    np.savez(out, **{**weights, 'w': new_w})
    w_matrix = lambda v: v.reshape(784, n_e)
    stats = dict(alpha=a.alpha, lam=a.lam, forget=a.forget, importance_files=a.importance,
                 alpha_bound=alpha_bound,
                 weights=a.weights, selected_synapses=int(selected.sum()),
                 selected_fraction=float(selected.mean()),
                 mean_beta_selected=float(beta[selected].mean()) if selected.any() else None,
                 weight_sum_before=float(weights['w'].sum()), weight_sum_after=float(new_w.sum()),
                 neurons_touched=int(np.any(w_matrix(selected), axis=0).sum()),
                 selected_per_neuron_max=int(w_matrix(selected).sum(axis=0).max()))
    out.with_suffix('.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
