"""Method D: request-gated leak during retained-data replay.

Entry point that chains the tools used for every reported Method D result:

1. ``run_replay.py``: replay R retained training images (no image of the
   forgotten classes) with normal STDP while the gated weights leak,
   w <- (1 - leak) * w after every image. Gate ``neuron``: every synapse of the
   neurons the baseline labelled with a forgotten class. Gate ``synapse``: the
   synapses Method B selects, I(D_f) > alpha * I(D). Thresholds theta are frozen
   by default. Gate ``none`` (no leak) is the repair/relapse replay.
2. ``run_frozen.py evaluate``: frozen evaluation with the baseline's fixed labels
   (plus the relabelled read-outs), and optionally the matched Gold.

Defaults are the settings reported for the neuron gate: MNIST leak 0.001 for
5,000 images; Fashion-MNIST leak 0.005 for 1,000 images with the training
weight normalisation (39), applied to ungated synapses only.
"""
import argparse
import json
from pathlib import Path
import sys

from .fashion_data import forget_classes
from .method_b import add_eval_arguments, check_model_dir, evaluation_steps, new_out_dir, run, summarize

DEFAULTS = {'mnist': dict(leak=0.001, replay_images=5000, weight_norm=0),
            'fashion': dict(leak=0.005, replay_images=1000, weight_norm=39)}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('mnist', 'fashion'), required=True)
    p.add_argument('--baseline-dir', required=True, help='Trained baseline run folder')
    p.add_argument('--request', default='class6_9', help='Removed classes: class6_9 (default), class6, ...')
    p.add_argument('--gate', choices=('neuron', 'synapse', 'none'), default='neuron')
    p.add_argument('--importance', nargs='*', default=[],
                   help='Synapse gate: importance.npz of the full-training-set pass(es)')
    p.add_argument('--alpha', type=float, default=2.0, help='Synapse gate: I(Df) > alpha * I(D)')
    p.add_argument('--leak', type=float, default=None,
                   help='Fraction of every gated weight removed after each replay image '
                        '(default 0.001 MNIST, 0.005 Fashion)')
    p.add_argument('--replay-images', type=int, default=None,
                   help='Retained images replayed (default 5000 MNIST, 1000 Fashion)')
    p.add_argument('--theta', choices=('frozen', 'updated'), default='frozen',
                   help="'updated' lets thresholds adapt during replay (used by the relapse test)")
    p.add_argument('--weight-norm', type=float, default=None,
                   help='Training weight normalisation, never applied to gated synapses '
                        '(default 0 MNIST, 39 Fashion)')
    p.add_argument('--norm-target', choices=('weight-norm', 'start'), default='weight-norm',
                   help="'start' keeps every neuron's weight sum from --start-weights (use after Method B)")
    p.add_argument('--start-weights', default=None,
                   help='Start from these weights instead of the baseline (e.g. a Method B output for '
                        'repair, or an unlearned model for the relapse test)')
    p.add_argument('--out-dir', required=True, help='New, empty folder outside the model folders')
    add_eval_arguments(p)
    a = p.parse_args(argv)
    if not forget_classes(a.request):
        p.error('Method D supports whole-class requests only, e.g. class6_9')
    if a.gate == 'synapse' and not a.importance:
        p.error('--gate synapse needs --importance')
    for key, value in DEFAULTS[a.dataset].items():
        if getattr(a, key) is None:
            setattr(a, key, value)
    if a.gate == 'none':
        a.leak = 0.0
    return a


def main(argv=None):
    a = arguments(argv)
    base = check_model_dir(a.baseline_dir, 'baseline')
    gold = check_model_dir(a.gold_dir, 'Gold') if a.gold_dir else None
    out = new_out_dir(a.out_dir, (base, gold))
    start = Path(a.start_weights).resolve() if a.start_weights else base / 'baseline_weights.npz'
    cmd = [sys.executable, 'run_replay.py', '--dataset', a.dataset, '--weights', start,
           '--labels', base / 'neuron_labels.npy', '--forget', a.request, '--gate', a.gate,
           '--alpha', a.alpha, '--leak', a.leak, '--replay-images', a.replay_images, '--theta', a.theta,
           '--weight-norm', a.weight_norm, '--norm-target', a.norm_target,
           '--input-total-hz', a.input_total_hz, '--out-dir', out / 'replay']
    if a.data_dir:
        cmd += ['--data-dir', a.data_dir]
    if a.importance:
        cmd += ['--importance', *a.importance]
    run(cmd, a.dry_run)
    weights = out / 'replay' / 'replayed_weights.npz'
    names = []
    if not a.skip_evaluation:
        run(evaluation_steps(a, weights, base / 'neuron_labels.npy', out, 'eval_method_d'), a.dry_run)
        names.append('eval_method_d')
        if gold is not None:
            run(evaluation_steps(a, gold / 'baseline_weights.npz', gold / 'neuron_labels.npy', out, 'eval_gold'),
                a.dry_run)
            names.append('eval_gold')
    if not a.dry_run:
        replay = json.loads((out / 'replay' / 'metrics.json').read_text())
        summarize(out, names, dict(method='D', request=a.request, gate=a.gate, leak=a.leak,
                                   replay_images=a.replay_images, theta=a.theta,
                                   gated_synapses=replay.get('gated_synapses'),
                                   gated_weight_sum_before=replay.get('gated_weight_sum_before'),
                                   gated_weight_sum_after=replay.get('gated_weight_sum_after')))
    return 0
