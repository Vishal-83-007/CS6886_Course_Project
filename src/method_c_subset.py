"""CSV-defined Method C forgetting, reusing the existing C1/C2 compiled rules."""
import argparse
import csv
import os
from pathlib import Path
import shutil

import numpy as np

from .fashion_data import balanced_ids, fingerprint, load_dataset, save_json
from .method_a import class_scores, compiled_counts, load_model, report, sha256
from .method_c import BASELINE, compiled_replay


def checked_ids(ids, size, name):
    ids = np.asarray(ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= size))):
        raise ValueError(f'Invalid {name}: require unique in-range integer IDs')
    return ids.astype(np.int64)


def read_forget_csv(path, labels):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['train_index', 'label']:
            raise ValueError('CSV header must be train_index,label')
        rows = list(reader)
    if not rows:
        raise ValueError('Forget CSV is empty')
    if any(set(row) != {'train_index', 'label'} or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Every CSV row must have exactly train_index,label')
    ids = checked_ids(np.array([int(r['train_index']) for r in rows]), len(labels), 'CSV')
    truth = np.array([int(r['label']) for r in rows])
    if not np.array_equal(truth, labels[ids]):
        raise ValueError('CSV labels differ from official training labels')
    return np.sort(ids)


def make_plan(arrays, train, validation, forget, per_class, retain_per_class, eval_limit, seed):
    y = arrays[1]
    train = checked_ids(train, len(y), 'baseline train')
    validation = checked_ids(validation, len(y), 'baseline validation')
    if np.intersect1d(train, validation).size or not np.isin(forget, train).all():
        raise ValueError('Training/validation overlap or requested image was not in baseline training')
    retain = np.setdiff1d(train, forget)
    assignment = balanced_ids(y, retain, per_class, seed)
    if not len(assignment) or set(y[assignment]) != set(y[retain]):
        raise ValueError('Assignment must represent every retained class')
    # Retain evaluation is disjoint from assignment to avoid readout fitting bias.
    retain_pool = np.setdiff1d(retain, assignment)
    retain_eval = (balanced_ids(y, retain_pool, retain_per_class, seed + 1)
                   if retain_per_class else retain_pool)
    test = np.arange(len(arrays[3]))
    if eval_limit:
        test = balanced_ids(arrays[3], test, eval_limit // 10, seed + 2)
    return dict(train=train, forget=forget, retain=retain, validation=validation,
                assignment=assignment, retain_evaluation=retain_eval, test_evaluation=test)


def assigned_labels(counts, truth):
    sums = np.zeros((10, counts.shape[1]))
    np.add.at(sums, truth, counts)
    rates = sums / np.maximum(np.bincount(truth, minlength=10), 1)[:, None]
    labels = rates.argmax(axis=0)
    labels[rates.max(axis=0) == 0] = -1
    return labels


def check_gold(directory, config, plan, dataset, request):
    gc, weights, labels = load_model(directory, dataset)
    if gc.get('request') != request:
        raise ValueError('Gold request differs; whole-class Gold is not a subset reference')
    for key in ('dataset', 'dataset_sha256', 'n_excitatory', 'seed', 'split_seed',
                'epochs', 'validation_size', 'train_limit', 'assign_per_class'):
        if gc.get(key) != config.get(key):
            raise ValueError(f'Gold mismatch: {key}')
    for key in ('input_total_hz', 'weight_norm'):
        if float(gc.get(key, 0)) != float(config.get(key, 0)):
            raise ValueError(f'Gold mismatch: {key}')
    with np.load(Path(directory) / 'manifest.npz', allow_pickle=False) as manifest:
        for key, expected in (('train', plan['retain']), ('forget', plan['forget']),
                              ('validation', plan['validation'])):
            actual = checked_ids(manifest[key], 60000, f'Gold {key}')
            if not np.array_equal(np.sort(actual), np.sort(expected)):
                raise ValueError(f'Gold {key} IDs do not match the exact requested split')
        assignment = checked_ids(manifest['assignment'], 60000,
                                 'Gold assignment')
        if not len(assignment) or not np.isin(assignment, plan['retain']).all():
            raise ValueError('Gold assignment must use retained images only')
    # Classes 6 and 9 remain valid outputs for subset Gold.
    return gc, weights, labels


def evaluation_data(arrays, plan):
    f, r, t = (plan[k] for k in ('forget', 'retain_evaluation', 'test_evaluation'))
    ids = np.concatenate([f, r, t])
    split = np.concatenate([np.full(len(f), 'forget_train'), np.full(len(r), 'retain_train'),
                            np.full(len(t), 'test')])
    images = np.concatenate([arrays[0][f], arrays[0][r], arrays[2][t]])
    truth = np.concatenate([arrays[1][f], arrays[1][r], arrays[3][t]])
    classes = np.unique(arrays[1][f])
    masks = {name: split == name for name in ('forget_train', 'retain_train', 'test')}
    masks['retain_train_requested_classes'] = masks['retain_train'] & np.isin(truth, classes)
    masks['retain_train_other_classes'] = masks['retain_train'] & ~np.isin(truth, classes)
    masks['test_requested_classes'] = masks['test'] & np.isin(truth, classes)
    masks['test_other_classes'] = masks['test'] & ~np.isin(truth, classes)
    return images, truth, ids, split, masks


def evaluate_subset(weights, labels, data, seed, target, build, input_total_hz):
    images, truth, ids, split, masks = data
    counts, timing = compiled_counts(weights, images, seed, np.zeros(len(labels)), build, input_total_hz)
    scores, predictions = class_scores(counts, labels)
    metrics = dict(timing=timing, groups={})
    target.mkdir(parents=True)
    for name, mask in masks.items():
        if not mask.any():
            metrics['groups'][name] = dict(n_evaluation=0, accuracy=None)
            continue
        row, _, _, confusion = report(counts[mask], labels, truth[mask], ())
        # These old metrics partition by class deletion and do not describe subset membership.
        for key in ('retain_accuracy', 'forget_class_accuracy', 'mean_spikes_retain', 'mean_spikes_forget'):
            row.pop(key)
        row['mean_total_spikes'] = float(counts[mask].sum(axis=1).mean())
        metrics['groups'][name] = row
        np.save(target / f'{name}_confusion.npy', confusion)
    save_json(target / 'metrics.json', metrics)
    np.savez_compressed(target / 'responses.npz', dataset_ids=ids, dataset_split=split,
                        truth=truth, predictions=predictions, scores=scores, counts=counts)
    return metrics, predictions, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('mnist', 'fashion'), required=True)
    p.add_argument('--forget-csv', type=Path, required=True)
    p.add_argument('--request', default='sub6_9_p10', help='Experiment name, not a class selector')
    p.add_argument('--baseline-dir', type=Path)
    p.add_argument('--gold-dir', type=Path, help='Optional completed subset Gold with manifest.npz')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--variants', nargs='+', choices=('C1', 'C2'), default=['C1', 'C2'])
    p.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.5, 1.0])
    p.add_argument('--passes', nargs='+', type=int, default=[1, 2, 5])
    p.add_argument('--replay-seed', type=int, default=2026)
    p.add_argument('--inference-seed', type=int, default=12345)
    p.add_argument('--assignment-seed', type=int, default=45)
    p.add_argument('--assign-per-class', type=int, default=500)
    p.add_argument('--retain-per-class', type=int, default=100, help='Retain training evaluation per class; 0 = all except assignment')
    p.add_argument('--eval-limit', type=int, default=0, help='Test evaluation only: 0 = all; otherwise balanced multiple of 10')
    p.add_argument('--prepare-only', action='store_true', help='Validate and write exact plan; no simulation')
    p.add_argument('--skip-evaluation', action='store_true', help='Replay/save weights without fitting a readout or evaluating')
    a = p.parse_args()
    if (min(a.passes) < 1 or any(not np.isfinite(s) or s <= 0 for s in a.scales)
            or any(len(v) != len(set(v)) for v in (a.passes, a.scales, a.variants))
            or a.assign_per_class < 1 or a.retain_per_class < 0 or a.eval_limit < 0
            or a.eval_limit > 10000 or a.eval_limit % 10
            or any(not 0 <= s < 2**32 - 2 for s in (a.replay_seed, a.inference_seed, a.assignment_seed))):
        p.error('Invalid pass/scale/seed/limit or duplicate trial')
    if a.out_dir.exists():
        p.error('Output already exists; choose a new directory')
    a.baseline_dir = a.baseline_dir or (BASELINE if a.dataset == 'mnist' else
        Path('fashion/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_wn39_cpp_2507283'))
    for directory in (a.baseline_dir, a.gold_dir):
        if directory and directory.resolve() in a.out_dir.resolve().parents:
            p.error('Output must be outside source checkpoint directories')
    config, weights, original_labels = load_model(a.baseline_dir, a.dataset)
    if config.get('request') != 'baseline':
        p.error('Require original baseline')
    arrays = load_dataset(a.data_dir or Path(f'data/{a.dataset}'), dataset=a.dataset)
    if fingerprint(arrays) != config['dataset_sha256']:
        p.error('Dataset fingerprint differs from baseline')
    forget = read_forget_csv(a.forget_csv, arrays[1])
    with np.load(a.baseline_dir / 'manifest.npz', allow_pickle=False) as manifest:
        plan = make_plan(arrays, manifest['train'], manifest['validation'], forget,
                         a.assign_per_class, a.retain_per_class, a.eval_limit, a.assignment_seed)
    if len(plan['validation']) != config['validation_size']:
        p.error('Baseline validation size mismatch')
    gold = check_gold(a.gold_dir, config, plan, a.dataset, a.request) if a.gold_dir else None
    sources = {str(path.resolve()): sha256(path) for directory in (a.baseline_dir, a.gold_dir)
               if directory for path in directory.iterdir() if path.is_file()}
    sources[str(a.forget_csv.resolve())] = sha256(a.forget_csv)
    rng = np.random.default_rng(a.replay_seed)
    orders = np.array([rng.permutation(len(forget)) for _ in range(max(a.passes))])
    a.out_dir.mkdir(parents=True)
    shutil.copyfile(a.forget_csv, a.out_dir / 'forget_set.csv')
    np.savez_compressed(a.out_dir / 'replay_manifest.npz', **plan, replay_order_ids=forget[orders])
    metadata = dict(method='C_subset', dataset=a.dataset, request=a.request,
        baseline_config=config, gold_config=gold[0] if gold else None,
        forget_csv_sha256=sha256(a.forget_csv), n_forget=len(forget), n_retain=len(plan['retain']),
        forget_label_counts={str(k): int(np.sum(arrays[1][forget] == k)) for k in np.unique(arrays[1][forget])},
        variants=a.variants, scales=a.scales, passes=sorted(a.passes),
        replay_seed=a.replay_seed, inference_seed=a.inference_seed, assignment_seed=a.assignment_seed,
        assign_per_class=a.assign_per_class, retain_per_class=a.retain_per_class, eval_limit=a.eval_limit,
        voting='retain-only reassignment of frozen baseline; shared unchanged by C1/C2; Gold assigned independently on same IDs',
        threshold_adaptation=False, weight_normalization=False,
        replay='exact CSV images each pass; original C1/C2 rules, 350ms exposure, 150ms rest, no retries',
        evaluation='all forgotten train; retain train excluding assignment; official test; IDs qualified by dataset_split',
        parameter_selection='prespecified sweep; do not select using forgotten-image or test metrics',
        source_artifact_sha256=sources,
        code_sha256={f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')})
    save_json(a.out_dir / 'config.json', metadata)
    save_json(a.out_dir / 'status.json', dict(status='prepared', evaluated=False))
    print(f'{a.dataset}: {len(forget)} forget, {len(plan["retain"])} retain; Gold={gold is not None}', flush=True)
    if a.prepare_only:
        return
    build = Path(os.environ.get('FASHION_BUILD_ROOT', str(a.out_dir / 'compiled'))).resolve() / a.out_dir.name
    summary = dict(gold_comparison_available=gold is not None)
    references = {}
    labels = None
    data = None
    hz = float(config.get('input_total_hz', 0))
    try:
        if not a.skip_evaluation:
            data = evaluation_data(arrays, plan)
            for name, mw in [('baseline', weights)] + ([('gold', gold[1])] if gold else []):
                save_json(a.out_dir / 'status.json', dict(status='assigning', model=name))
                counts, timing = compiled_counts(mw, arrays[0][plan['assignment']], a.inference_seed,
                    np.zeros(int(mw['n_e'])), build / f'{name}_assignment', hz)
                ml = assigned_labels(counts, arrays[1][plan['assignment']])
                np.savez_compressed(a.out_dir / f'{name}_assignment.npz', ids=plan['assignment'],
                                    counts=counts, labels=ml)
                save_json(a.out_dir / f'{name}_assignment_timing.json', timing)
                if name == 'baseline':
                    labels = ml
                    save_json(a.out_dir / 'readout_change.json', dict(
                        n_labels_changed=int(np.sum(labels != original_labels)),
                        reason='exclude CSV images from neuron labeling'))
                save_json(a.out_dir / 'status.json', dict(status='evaluating', model=name))
                metrics, pred, counts = evaluate_subset(mw, ml, data, a.inference_seed,
                    a.out_dir / name, build / f'{name}_eval', hz)
                summary[name] = metrics
                references[name] = (pred, counts)
            save_json(a.out_dir / 'summary.json', summary)
        for variant in a.variants:
            for scale in a.scales:
                name = f'{variant}_s{scale:g}'
                target = a.out_dir / name
                target.mkdir()
                save_json(a.out_dir / 'status.json', dict(status='replaying', trial=name))
                snapshots, diagnostics, timing = compiled_replay(weights, arrays[0][forget], orders,
                    a.replay_seed, variant, scale, build / name, sorted(a.passes), hz, False)
                np.savez_compressed(target / 'presentations.npz', diagnostics=diagnostics,
                                    dataset_ids=forget[orders].ravel())
                trial = dict(timing=timing, checkpoints={})
                for passes, edited in snapshots.items():
                    checkpoint = target / f'pass_{passes}'
                    checkpoint.mkdir()
                    np.savez_compressed(checkpoint / 'baseline_weights.npz', **edited)
                    if labels is not None:
                        np.save(checkpoint / 'neuron_labels.npy', labels)
                    save_json(checkpoint / 'config.json', dict(config, request=a.request, method=variant,
                        scale=scale, passes=passes, readout_available=labels is not None,
                        forget_csv_sha256=metadata['forget_csv_sha256'], threshold_adaptation=False,
                        weight_normalization=False))
                    dw = edited['w'] - weights['w']
                    checks = dict(weight_l1_change=float(np.abs(dw).sum()),
                        weights_increased=int(np.sum(dw > 0)), weights_decreased=int(np.sum(dw < 0)),
                        frozen_thresholds_verified=bool(np.array_equal(edited['theta_volts'], weights['theta_volts'])))
                    save_json(checkpoint / 'intervention.json', checks)
                    result = dict(intervention=checks)
                    if not a.skip_evaluation:
                        save_json(a.out_dir / 'status.json', dict(status='evaluating', trial=name, passes=passes))
                        metrics, pred, counts = evaluate_subset(edited, labels, data, a.inference_seed,
                            checkpoint / 'evaluation', build / f'{name}_p{passes}_eval', hz)
                        result['metrics'] = metrics
                        result['comparison'] = {ref: {group: dict(
                            prediction_disagreement=float(np.mean(pred[mask] != rp[mask])),
                            mean_absolute_total_spike_difference=float(np.abs(
                                counts[mask].sum(axis=1).astype(float) - rc[mask].sum(axis=1)).mean()))
                            for group, mask in data[4].items() if mask.any()}
                            for ref, (rp, rc) in references.items()}
                    trial['checkpoints'][str(passes)] = result
                    summary[name] = trial
                    save_json(a.out_dir / 'summary.json', summary)
        save_json(a.out_dir / 'status.json', dict(status='complete', evaluated=not a.skip_evaluation))
    except Exception as error:
        save_json(a.out_dir / 'status.json', dict(status='failed', error=str(error)))
        raise
    finally:
        if any(sha256(path) != digest for path, digest in sources.items()):
            save_json(a.out_dir / 'status.json', dict(status='failed', error='Source artifact changed'))
            raise RuntimeError('Source artifact changed during execution')


if __name__ == '__main__':
    main()
