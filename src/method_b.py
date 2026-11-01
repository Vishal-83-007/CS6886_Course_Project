"""Method B: Selective Synaptic Dampening with hypothetical-STDP importance.

Entry point that chains the tools used for every reported Method B result:

1. ``dampen.py``: select synapses with I(D_f) > alpha * I(D) and multiply them
   by beta = min(lambda * I(D) / I(D_f), 1). Thresholds are unchanged.
2. ``run_frozen.py evaluate``: frozen evaluation of the dampened weights with
   the baseline's fixed neuron labels (plus the relabelled read-outs).
3. Optionally the same evaluation of a matched Gold model with its own labels.

The importance files come from ``run_frozen.py importance`` (or
``jobs/importance_cpp.cmd``): one or more passes over all training images for
I(D), and, for random-subset requests such as ``sub6_9_p10``, a pass over the
forgotten images only (``--forget-importance``). Computing I(D) takes hours, so
it is an input here rather than a step.

The baseline directory must contain ``baseline_weights.npz`` and
``neuron_labels.npy`` (the format written by ``run_fashion.py``).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .fashion_data import forget_classes, subset_request

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, dry_run):
    """Run one step (or print it with --dry-run); stop on failure."""
    print('+ ' + ' '.join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    result = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if result.returncode:
        raise SystemExit(f'step failed with exit code {result.returncode}: {cmd[1]}')


def check_model_dir(path, what):
    path = Path(path).resolve()
    for name in ('baseline_weights.npz', 'neuron_labels.npy'):
        if not (path / name).is_file():
            raise SystemExit(f'{what} {path} has no {name}')
    return path


def new_out_dir(path, protected):
    out = Path(path).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty: choose a new --out-dir')
    for p in protected:
        if p is not None and (out == p or p in out.parents):
            raise SystemExit('--out-dir must be outside the baseline and Gold directories')
    out.mkdir(parents=True, exist_ok=True)
    return out


def add_eval_arguments(p):
    p.add_argument('--data-dir', default=None, help='Official IDX files; defaults to data/<dataset>')
    p.add_argument('--eval-split', choices=('tune', 'final', 'all'), default=None,
                   help="Test images: 'tune' (500/class, for choosing settings), 'final' (the other "
                        "half, for reporting) or 'all'. Default final; all for subset requests")
    p.add_argument('--eval-limit', type=int, default=0,
                   help='Balanced subset of the split for a quick smoke run (multiple of 10)')
    p.add_argument('--assign-per-class', type=int, default=500,
                   help='Labelling images per class for the relabelled read-outs')
    p.add_argument('--input-total-hz', type=float, default=0, help='Must match training (0 for all our runs)')
    p.add_argument('--gold-dir', default=None,
                   help='Matched Gold model folder (trained with the same --request); evaluated '
                        'with its own labels for comparison. Never an input to the edit')
    p.add_argument('--skip-evaluation', action='store_true',
                   help='Only produce the edited weights')
    p.add_argument('--dry-run', action='store_true', help='Print the steps without running them')


def evaluation_steps(a, weights, labels, out_dir, name):
    split = a.eval_split or ('all' if subset_request(a.request) else 'final')
    cmd = [sys.executable, 'run_frozen.py', 'evaluate', '--dataset', a.dataset,
           '--weights', weights, '--fixed-labels', labels, '--forget', a.request,
           '--eval-split', split, '--eval-limit', a.eval_limit,
           '--assign-per-class', a.assign_per_class, '--input-total-hz', a.input_total_hz,
           '--out-dir', out_dir / name]
    if a.data_dir:
        cmd += ['--data-dir', a.data_dir]
    if subset_request(a.request):
        cmd.append('--forget-set')
    return cmd


def summarize(out_dir, names, extra):
    summary = dict(extra)
    for name in names:
        m = out_dir / name / 'metrics.json'
        if m.is_file():
            metrics = json.loads(m.read_text())
            summary[name] = {k: {key: r.get(key) for key in ('accuracy', 'retain_accuracy', 'forget_accuracy')}
                             for k, r in metrics['readouts'].items()}
            if 'forget_set' in metrics:
                summary[name + '_forget_set'] = metrics['forget_set']
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('mnist', 'fashion'), required=True)
    p.add_argument('--baseline-dir', required=True, help='Trained baseline run folder')
    p.add_argument('--request', default='class6_9', help='class6_9 (default), class6, ... or sub6_9_p10')
    p.add_argument('--importance', nargs='+', required=True,
                   help='importance.npz of the full-training-set pass(es); several are averaged')
    p.add_argument('--forget-importance', nargs='*', default=[],
                   help='Subset requests: importance.npz of the pass over the forgotten images')
    p.add_argument('--alpha', type=float, default=2.0, help='Selection: I(Df) > alpha * I(D) (default 2)')
    p.add_argument('--lam', type=float, default=0.5, help='Dampening strength lambda (default 0.5)')
    p.add_argument('--out-dir', required=True, help='New, empty folder outside the model folders')
    add_eval_arguments(p)
    a = p.parse_args(argv)
    if not (forget_classes(a.request) or subset_request(a.request)):
        p.error('--request must look like class6, class6_9 or sub6_9_p10')
    if subset_request(a.request) and not a.forget_importance:
        p.error('a subset request needs --forget-importance')
    return a


def main(argv=None):
    a = arguments(argv)
    base = check_model_dir(a.baseline_dir, 'baseline')
    gold = check_model_dir(a.gold_dir, 'Gold') if a.gold_dir else None
    out = new_out_dir(a.out_dir, (base, gold))
    weights = out / 'method_b_weights.npz'
    cmd = [sys.executable, 'dampen.py', '--importance', *a.importance, '--weights', base / 'baseline_weights.npz',
           '--forget', a.request, '--alpha', a.alpha, '--lam', a.lam, '--out', weights]
    if a.forget_importance:
        cmd[cmd.index('--weights'):cmd.index('--weights')] = ['--forget-importance', *a.forget_importance]
    run(cmd, a.dry_run)
    names = []
    if not a.skip_evaluation:
        run(evaluation_steps(a, weights, base / 'neuron_labels.npy', out, 'eval_method_b'), a.dry_run)
        names.append('eval_method_b')
        if gold is not None:
            run(evaluation_steps(a, gold / 'baseline_weights.npz', gold / 'neuron_labels.npy', out, 'eval_gold'),
                a.dry_run)
            names.append('eval_gold')
    if not a.dry_run:
        stats = json.loads(weights.with_suffix('.json').read_text())
        summarize(out, names, dict(method='B', request=a.request, alpha=a.alpha, lam=a.lam,
                                   selected_synapses=stats['selected_synapses'],
                                   neurons_touched=stats['neurons_touched']))
    return 0
