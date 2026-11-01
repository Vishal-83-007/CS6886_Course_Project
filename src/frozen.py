"""Frozen-network passes for unlearning (compiled CPU backend).

A trained model (``baseline_weights.npz``: weights and adaptive thresholds) is
loaded into a fresh network with learning switched OFF: STDP does not change
any weight and the thresholds theta neither grow nor decay.

Two modes:

importance
    Present training images and record, per synapse, the update STDP WOULD
    have made (same rule and learning rate as training):
        dw_ij(x) = sum over postsynaptic spikes during image x (all retry
                   attempts) of eta * (pretrace_ij - x_tar) * (w_max - w_ij)^mu
    and accumulate its square per class c:
        S_c,ij = sum_{x in class c} dw_ij(x)^2,   n_c = number of images.
    Then I_ij(S) = sum_{c} S_c,ij / sum_{c} n_c for any union S of classes,
    e.g. I(D) (all ten) and I(D_f) (removed classes); this is the
    hypothetical-STDP importance of the proposal (Method B).

evaluate
    Present the labelling images (the baseline's 500 per class, all ten
    classes) and test images, store all spike counts, and score several
    read-outs: the model's own labels from all ten classes, its own labels
    from retained classes only, and optionally fixed labels from a file
    (e.g. the baseline's neuron_labels.npy).
    With a subset request (--forget sub6_9_p10) the labelling images come from
    that request's manifest, so the forgotten images are never used for labels,
    and --forget-set also presents the forgotten training images D_f and an
    equally large random sample of the RETAINED training images of the same
    classes, and scores accuracy and a membership-inference AUC on each
    (members: training images; non-members: test images of the same classes).

Input encoding, presentation (350 ms + 150 ms rest) and the retry rule
(< 5 spikes: boost rates by 32 Hz, up to 6 attempts) are the same as in
training. ``--input-total-hz`` must match the value used to train the model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .fashion_data import (balanced_ids, fingerprint, forget_classes, load_dataset,
                           make_manifest, save_json, subset_request, valid_request)

# STAGE_TRAIN_EVAL: a TRAINING image that is scored (D_f and its retained
# same-class reference); the compiled loop shows test images only for STAGE_EVAL.
STAGE_IMPORTANCE, STAGE_ASSIGN, STAGE_EVAL, STAGE_TRAIN_EVAL = 0, 1, 2, 3
GROUP_FORGET, GROUP_RETAINED_SAME = 1, 2


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('importance', 'evaluate'))
    p.add_argument('--dataset', choices=('fashion', 'mnist'), required=True)
    p.add_argument('--data-dir', default=None, help='Defaults to data/<dataset>')
    p.add_argument('--weights', required=True,
                   help='npz with w, i, j, theta_volts, n_e, n_input (baseline_weights.npz '
                        'of a run folder, or a dampened copy)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--poisson-seed', type=int, default=0,
                   help='Seed of the Poisson input spikes (and of the image order)')
    p.add_argument('--split-seed', type=int, default=42,
                   help='Must match training: selects the labelling images')
    p.add_argument('--input-total-hz', type=float, default=0,
                   help='Same value as used in training (0 = rate is pixel / 4)')
    p.add_argument('--threads', type=int, default=1, help='Compiler jobs')
    # importance
    p.add_argument('--train-limit', type=int, default=0,
                   help='importance: balanced subset of training images (multiple of 10); '
                        '0 = all 60,000')
    p.add_argument('--importance-set', choices=('train', 'forget'), default='train',
                   help="importance: 'train' = all training images (I(D) and per-class); "
                        "'forget' = only the forgotten images of the subset request --forget "
                        "(I(D_f) for random-subset forgetting)")
    # evaluate
    p.add_argument('--eval-split', choices=('tune', 'final', 'all'), default='tune',
                   help="evaluate: 'tune' = fixed balanced 5,000 test images (500 per class) "
                        "for choosing hyper-parameters, 'final' = the other 5,000, 'all' = 10,000")
    p.add_argument('--eval-limit', type=int, default=0,
                   help='evaluate: balanced subset of the chosen split for quick checks')
    p.add_argument('--assign-per-class', type=int, default=500,
                   help='evaluate: labelling images per class (same IDs as training when 500)')
    p.add_argument('--forget', default='',
                   help='evaluate: removed classes, e.g. class6_9 (adds retained/removed '
                        'accuracies and the retained-only read-out), or a random subset, e.g. '
                        'sub6_9_p10 (labelling images exclude the forgotten ones)')
    p.add_argument('--forget-set', action='store_true',
                   help='evaluate, subset request only: also score the forgotten training '
                        'images and a same-size sample of retained images of the same classes')
    p.add_argument('--fixed-labels', default='',
                   help="evaluate: neuron_labels.npy to score as a fixed read-out "
                        "(e.g. the baseline's labels for an unlearned model)")
    a = p.parse_args(argv)
    if a.forget and not valid_request(a.forget) or a.forget in ('baseline', 'random10'):
        p.error('--forget must look like class6, class6_9 or sub6_9_p10')
    if (a.forget_set or a.importance_set == 'forget') and not subset_request(a.forget):
        p.error('--forget-set and --importance-set forget need a subset --forget, e.g. sub6_9_p10')
    if a.train_limit % 10 or a.eval_limit % 10:
        p.error('--train-limit and --eval-limit must be multiples of ten')
    a.data_dir = a.data_dir or f'data/{a.dataset}'
    return a


def test_split(test_labels, split, split_seed):
    """Fixed tuning / final halves of the official test set."""
    all_ids = np.arange(len(test_labels))
    tune = np.sort(balanced_ids(test_labels, all_ids, 500, split_seed + 5))
    if split == 'tune':
        return tune
    if split == 'final':
        return np.setdiff1d(all_ids, tune)
    return all_ids


def class_labels(counts, labels, classes):
    """Neuron label = class with the highest mean response (-1: never fired)."""
    rates = np.zeros((10, counts.shape[1]))
    for c in classes:
        mask = labels == c
        if np.any(mask):
            rates[c] = counts[mask].mean(axis=0)
    neuron_labels = np.argmax(rates, axis=0)
    neuron_labels[np.max(rates, axis=0) == 0] = -1
    return neuron_labels


def predict(counts, neuron_labels):
    """Class with the highest mean count over its neurons; -1 if nothing is labelled."""
    scores = np.full((len(counts), 10), -np.inf)
    for c in range(10):
        mask = neuron_labels == c
        if np.any(mask):
            scores[:, c] = counts[:, mask].mean(axis=1)
    predictions = scores.argmax(axis=1)
    if not np.any(neuron_labels >= 0):
        predictions[:] = -1
    ties = np.sum(scores == scores.max(axis=1)[:, None], axis=1) > 1
    return predictions, ties


def score(predictions, ties, truth, counts, removed):
    removed_mask = np.isin(truth, removed)
    result = dict(
        accuracy=float(np.mean(predictions == truth)),
        per_class_accuracy={str(c): float(np.mean(predictions[truth == c] == c))
                            for c in np.unique(truth)},
        silent_fraction=float(np.mean(counts.sum(axis=1) == 0)),
        tie_fraction=float(np.mean(ties)),
        predicted_per_class=np.bincount(predictions[predictions >= 0], minlength=10).tolist())
    if len(removed):
        result['retain_accuracy'] = float(np.mean(predictions[~removed_mask] == truth[~removed_mask]))
        result['forget_accuracy'] = float(np.mean(predictions[removed_mask] == truth[removed_mask]))
        result['forget_predicted_as'] = np.bincount(
            np.where(predictions[removed_mask] < 0, 10, predictions[removed_mask]),
            minlength=11).tolist()
    return result


def run(a):
    from brian2 import defaultclock, get_device, prefs, set_device, volt
    from .encoding import PRESENT_TIME, REST_TIME
    from .network import build_network

    train_x, train_y, test_x, test_y = load_dataset(a.data_dir, dataset=a.dataset)
    with np.load(a.weights) as data:
        weights = {k: data[k] for k in data.files}
    n_e, n_input = int(weights['n_e']), int(weights['n_input'])
    # connect() makes all-to-all synapses in this order (checked against every
    # saved baseline); refuse anything else rather than mis-assign weights.
    if (n_input != 784 or not np.array_equal(weights['i'], np.repeat(np.arange(784), n_e))
            or not np.array_equal(weights['j'], np.tile(np.arange(n_e), 784))):
        raise ValueError(f'{a.weights}: unexpected synapse order or size')
    if not (np.isfinite(weights['w']).all() and np.isfinite(weights['theta_volts']).all()):
        raise ValueError(f'{a.weights}: nonfinite weights or thresholds')
    ns = 784 * n_e

    rng = np.random.default_rng(np.random.SeedSequence([a.poisson_seed, 911]))
    group = None
    if a.mode == 'importance':
        if a.importance_set == 'forget':
            ids = rng.permutation(make_manifest(train_y, a.forget, a.split_seed, 0, 1)['forget'])
        else:
            manifest = make_manifest(train_y, 'baseline', a.split_seed, 0, 1, a.train_limit)
            ids = rng.permutation(manifest['train'])
        stage = np.full(len(ids), STAGE_IMPORTANCE, np.uint8)
    else:
        # A subset request labels neurons without its forgotten images.
        label_request = a.forget if a.forget and subset_request(a.forget) else 'baseline'
        subset_manifest = make_manifest(train_y, label_request, a.split_seed, 0, a.assign_per_class)
        assignment = subset_manifest['assignment']
        evaluation = test_split(test_y, a.eval_split, a.split_seed)
        if a.eval_limit:
            evaluation = np.sort(balanced_ids(test_y, evaluation, a.eval_limit // 10, a.split_seed + 6))
        extra = np.empty(0, np.int64)
        group = np.zeros(0, np.uint8)
        if a.forget_set:
            forget = np.sort(subset_manifest['forget'])
            same = np.setdiff1d(np.flatnonzero(np.isin(train_y, subset_request(a.forget)[0])), forget)
            same = np.sort(np.random.default_rng(a.split_seed + 7).permutation(same)[:len(forget)])
            extra = np.concatenate([forget, same])
            group = np.concatenate([np.full(len(forget), GROUP_FORGET, np.uint8),
                                    np.full(len(same), GROUP_RETAINED_SAME, np.uint8)])
        ids = np.concatenate([assignment, evaluation, extra])
        stage = np.concatenate([np.full(len(assignment), STAGE_ASSIGN, np.uint8),
                                np.full(len(evaluation), STAGE_EVAL, np.uint8),
                                np.full(len(extra), STAGE_TRAIN_EVAL, np.uint8)])
        group = np.concatenate([np.zeros(len(assignment) + len(evaluation), np.uint8), group])
    labels_of = np.where(stage == STAGE_EVAL, test_y[np.minimum(ids, len(test_y) - 1)], train_y[ids])

    root = Path(a.out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'{root} is nonempty: choose a new --out-dir')
    root.mkdir(parents=True, exist_ok=True)
    code = b''.join((Path(__file__).parent / name).read_bytes()
                    for name in ('frozen.py', 'network.py', 'encoding.py', 'fashion_data.py'))
    config = {k: v for k, v in vars(a).items() if k not in ('out_dir', 'data_dir', 'threads')}
    config.update(weights_sha256=hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),
                  dataset_sha256=fingerprint((train_x, train_y, test_x, test_y)),
                  schedule_sha256=fingerprint((ids, stage)), n_excitatory=n_e,
                  source_sha256=hashlib.sha256(code).hexdigest(), backend='cpp', schema=1)
    save_json(root / 'config.json', config)
    np.savez(root / 'schedule.npz', ids=ids, stage=stage, labels=labels_of,
             **({} if group is None else dict(group=group)))
    save_json(root / 'status.json', dict(status='building', backend='cpp'))

    set_device('cpp_standalone', build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 0
    prefs.devices.cpp_standalone.extra_make_args_unix = [f'-j{a.threads}']
    prefs.codegen.cpp.headers += ['<fstream>', '<vector>', '<chrono>', '<stdexcept>',
                                  '<algorithm>', '<cmath>']
    components = build_network(n_e, a.poisson_seed, importance=(a.mode == 'importance'))
    syn, exc = components['syn_input_exc'], components['exc']
    syn.w = weights['w']
    exc.theta = weights['theta_volts'] * volt
    syn.eta = 0.0                    # no weight changes
    exc.plasticity_enabled = 0       # theta frozen: no growth, no decay
    if a.mode == 'importance':
        syn.dw_hyp = 0.0
    device = get_device()

    def queue_run(duration):
        # Same clock handling as src/fashion_standalone.py: keep compiled time
        # running across the repeated runs instead of rewinding every image.
        offset = len(device.main_queue)
        components['net'].run(duration)
        names = {device.get_array_name(defaultclock.variables['t']): 'fashion_network.t',
                 device.get_array_name(defaultclock.variables['timestep']):
                 'static_cast<int64_t>(std::llround(fashion_network.t / '
                 + device.get_array_name(defaultclock.variables['dt']) + '[0]))'}
        changed = set()
        for index in range(offset, len(device.main_queue)):
            kind, values = device.main_queue[index]
            if kind == 'set_by_single_value' and values[0] in names:
                device.main_queue[index] = (kind, (values[0], values[1], names[values[0]]))
                changed.add(values[0])
        if changed != set(names):
            raise RuntimeError('Unsupported Brian clock queue layout; refusing unsafe compiled loop')

    def host(obj, key):
        var = obj.variables[key]
        if var.dynamic:
            return device.get_array_name(var, access_data=False) + '.data()'
        return device.get_array_name(var)

    work_root = (Path(os.environ['FASHION_BUILD_ROOT']) / root.parent.name / root.name
                 if 'FASHION_BUILD_ROOT' in os.environ else root / 'build_work')
    binary = work_root / 'inputs'
    binary.mkdir(parents=True)
    train_x.tofile(binary / 'train.bin')
    test_x.tofile(binary / 'test.bin')
    ids.astype('<i8').tofile(binary / 'ids.bin')
    stage.tofile(binary / 'stage.bin')
    labels_of.astype(np.uint8).tofile(binary / 'labels.bin')
    quoted = lambda path: json.dumps(str(path))
    rates = host(components['input'], 'rates')
    mon = host(components['exc_monitor'], 'count')
    dw = host(syn, 'dw_hyp') if a.mode == 'importance' else None
    n = len(ids)
    importance_mode = int(a.mode == 'importance')

    device.insert_code('main', f'''
    auto load_bytes = [](const char* path, size_t count) {{
        std::vector<unsigned char> values(count);
        std::ifstream file(path, std::ios::binary);
        if(!file.read(reinterpret_cast<char*>(values.data()), count))
            throw std::runtime_error("Cannot read input bytes");
        return values;
    }};
    auto frozen_train = load_bytes({quoted(binary / 'train.bin')}, {train_x.size});
    auto frozen_test = load_bytes({quoted(binary / 'test.bin')}, {test_x.size});
    auto frozen_stage = load_bytes({quoted(binary / 'stage.bin')}, {n});
    auto frozen_label = load_bytes({quoted(binary / 'labels.bin')}, {n});
    std::vector<int64_t> frozen_ids({n});
    std::ifstream frozen_id_file({quoted(binary / 'ids.bin')}, std::ios::binary);
    if(!frozen_id_file.read(reinterpret_cast<char*>(frozen_ids.data()), {n} * sizeof(int64_t)))
        throw std::runtime_error("Cannot read schedule");
    std::ofstream frozen_responses({quoted(root / 'counts.bin')}, std::ios::binary);
    if(!frozen_responses) throw std::runtime_error("Cannot write response file");
    // Importance: per-class sums of squared hypothetical updates (10 x synapses).
    std::vector<double> frozen_sumsq({importance_mode} ? (size_t)10 * {ns} : 0, 0.0);
    std::vector<double> frozen_sum({importance_mode} ? (size_t)10 * {ns} : 0, 0.0);
    long long frozen_class_count[10] = {{0}};
    long long frozen_attempts[6] = {{0, 0, 0, 0, 0, 0}};
    long long frozen_exhausted = 0;
    double frozen_seconds = 0;
    std::vector<int> frozen_before({n_e}), frozen_counts({n_e});
    for(size_t frozen_position = 0; frozen_position < {n}; ++frozen_position) {{
        auto frozen_start = std::chrono::steady_clock::now();
        const int frozen_phase = frozen_stage[frozen_position];
        const unsigned char* frozen_image =
            (frozen_phase == {STAGE_EVAL} ? frozen_test.data() : frozen_train.data())
            + frozen_ids[frozen_position] * 784;
        double frozen_scale = 1.0;
        if({a.input_total_hz!r} > 0) {{
            double frozen_pixel_sum = 0;
            for(int k = 0; k < 784; ++k) frozen_pixel_sum += frozen_image[k] / 4.0;
            if(frozen_pixel_sum > 0) frozen_scale = {a.input_total_hz!r} / frozen_pixel_sum;
        }}
        if({importance_mode}) std::fill({dw or 'frozen_counts.data()'}, {dw or 'frozen_counts.data()'} + {ns if dw else 0}, 0);
        int frozen_total = 0, frozen_attempt = 0;
        for(; frozen_attempt < 6; ++frozen_attempt) {{
            for(int k = 0; k < 784; ++k)
                {rates}[k] = frozen_image[k] / 4.0 * frozen_scale
                              * (1.0 + frozen_attempt * 32.0 / 63.75);
            std::copy({mon}, {mon} + {n_e}, frozen_before.begin());
    ''')
    queue_run(PRESENT_TIME)
    device.insert_code('main', f'''
            frozen_total = 0;
            for(int k = 0; k < {n_e}; ++k) {{
                frozen_counts[k] = {mon}[k] - frozen_before[k];
                frozen_total += frozen_counts[k];
            }}
            std::fill({rates}, {rates} + 784, 0.0);
    ''')
    queue_run(REST_TIME)
    device.insert_code('main', f'''
            if(frozen_total >= 5) break;
        }}
        ++frozen_attempts[std::min(frozen_attempt, 5)];
        if(frozen_total < 5) ++frozen_exhausted;
        if({importance_mode}) {{
            const size_t frozen_offset = (size_t)frozen_label[frozen_position] * {ns};
            ++frozen_class_count[frozen_label[frozen_position]];
            for(size_t k = 0; k < {ns if dw else 0}; ++k) {{
                const double update = {dw or 'frozen_sum.data()'}[k];
                frozen_sumsq[frozen_offset + k] += update * update;
                frozen_sum[frozen_offset + k] += update;
            }}
        }}
        frozen_responses.write(reinterpret_cast<const char*>(frozen_counts.data()), {n_e} * sizeof(int));
        if(!frozen_responses) throw std::runtime_error("Response write failed");
        frozen_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - frozen_start).count();
        if((frozen_position + 1) % 100 == 0)
            std::cout << "progress=" << frozen_position + 1 << "/{n}" << std::endl;
    }}
    frozen_responses.close();
    if({importance_mode}) {{
        std::ofstream frozen_sq_file({quoted(root / 'importance_sumsq.bin')}, std::ios::binary);
        std::ofstream frozen_sum_file({quoted(root / 'importance_sum.bin')}, std::ios::binary);
        frozen_sq_file.write(reinterpret_cast<const char*>(frozen_sumsq.data()), frozen_sumsq.size() * sizeof(double));
        frozen_sum_file.write(reinterpret_cast<const char*>(frozen_sum.data()), frozen_sum.size() * sizeof(double));
        if(!frozen_sq_file || !frozen_sum_file) throw std::runtime_error("Cannot write importance");
    }}
    std::ofstream frozen_timing({quoted(root / 'standalone_timing.json')});
    frozen_timing << "{{\\\"seconds\\\":" << frozen_seconds << ",\\\"attempts\\\":[";
    for(int k = 0; k < 6; ++k) frozen_timing << (k ? "," : "") << frozen_attempts[k];
    frozen_timing << "],\\\"exhausted\\\":" << frozen_exhausted << ",\\\"class_count\\\":[";
    for(int k = 0; k < 10; ++k) frozen_timing << (k ? "," : "") << frozen_class_count[k];
    frozen_timing << "]}}";
    if(!frozen_timing) throw std::runtime_error("Cannot write timing file");
    ''')
    start = time.monotonic()
    device.build(directory=str(work_root / 'build'), compile=True, run=True)
    timing = json.loads((root / 'standalone_timing.json').read_text())
    timing['build_and_run_seconds'] = time.monotonic() - start
    counts = np.fromfile(root / 'counts.bin', dtype=np.int32).reshape(n, n_e)
    if a.mode == 'importance':
        summarize_importance(a, root, counts, stage, labels_of, timing, weights, n_e)
    else:
        summarize_evaluation(a, root, counts, stage, labels_of, ids, timing, n_e, group)
    (root / 'counts.bin').unlink()
    save_json(root / 'status.json', dict(status='complete'))
    return 0


def summarize_importance(a, root, counts, stage, labels_of, timing, weights, n_e):
    ns = 784 * n_e
    sumsq = np.fromfile(root / 'importance_sumsq.bin').reshape(10, ns)
    sums = np.fromfile(root / 'importance_sum.bin').reshape(10, ns)
    class_count = np.array(timing['class_count'], dtype=np.int64)
    if not np.array_equal(class_count, np.bincount(labels_of, minlength=10)):
        raise RuntimeError('Per-class image counts differ from the schedule')
    if not (np.isfinite(sumsq).all() and np.isfinite(sums).all()):
        raise FloatingPointError('Nonfinite importance values')
    np.savez(root / 'importance.npz', sumsq=sumsq, sum=sums, class_count=class_count,
             i=weights['i'], j=weights['j'], n_e=n_e, n_input=784)
    for name in ('importance_sumsq.bin', 'importance_sum.bin'):
        (root / name).unlink()
    i_d = sumsq.sum(axis=0) / class_count.sum()
    result = dict(mode='importance', importance_set=a.importance_set, forget=a.forget,
                  n_images=int(len(stage)),
                  spikes_per_image=float(counts.sum(axis=1).mean()),
                  importance_D_mean=float(i_d.mean()), importance_D_max=float(i_d.max()),
                  importance_D_zero_fraction=float(np.mean(i_d == 0)), **timing)
    save_json(root / 'metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)


def class_scores(counts, neuron_labels):
    """Mean count of each class's neurons (-inf for a class without neurons)."""
    scores = np.full((len(counts), 10), -np.inf)
    for c in range(10):
        mask = neuron_labels == c
        if np.any(mask):
            scores[:, c] = counts[:, mask].mean(axis=1)
    return scores


def true_class_margin(counts, neuron_labels, truth):
    """Score of the true class minus the best other class (membership-attack feature)."""
    scores = class_scores(counts, neuron_labels)
    scores = np.where(np.isfinite(scores), scores, -1e9)
    true = scores[np.arange(len(truth)), truth]
    scores[np.arange(len(truth)), truth] = -np.inf
    return true - scores.max(axis=1)


def auc(positive, negative):
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind='mergesort')
    _, inverse, count = np.unique(values[order], return_inverse=True, return_counts=True)
    ranks = np.empty(len(values))
    ranks[order] = (np.cumsum(count) - (count - 1) / 2)[inverse]
    n_p, n_n = len(positive), len(negative)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def summarize_evaluation(a, root, counts, stage, labels_of, ids, timing, n_e, group=None):
    assign_counts, eval_counts = counts[stage == STAGE_ASSIGN], counts[stage == STAGE_EVAL]
    assign_labels, truth = labels_of[stage == STAGE_ASSIGN], labels_of[stage == STAGE_EVAL]
    removed = list(forget_classes(a.forget)) if a.forget else []
    retained = [c for c in range(10) if c not in removed]
    subset = subset_request(a.forget) if a.forget else None
    readouts = {'own_labels_all_classes': class_labels(assign_counts, assign_labels, range(10))}
    if removed:
        readouts['own_labels_retained_classes'] = class_labels(assign_counts, assign_labels, retained)
    if a.fixed_labels:
        fixed = np.load(a.fixed_labels)
        if fixed.shape != (n_e,):
            raise ValueError('--fixed-labels has the wrong number of neurons')
        readouts['fixed_labels'] = fixed
    result = dict(mode='evaluate', eval_split=a.eval_split, n_assignment=int(len(assign_counts)),
                  n_evaluation=int(len(truth)), removed_classes=removed,
                  spikes_per_eval_image=float(eval_counts.sum(axis=1).mean()), readouts={}, **timing)
    saved = dict(assignment_ids=ids[stage == STAGE_ASSIGN], assignment_counts=assign_counts,
                 assignment_labels=assign_labels, evaluation_ids=ids[stage == STAGE_EVAL],
                 evaluation_counts=eval_counts, truth=truth)
    if group is not None and np.any(stage == STAGE_TRAIN_EVAL):
        train_counts = counts[stage == STAGE_TRAIN_EVAL]
        train_truth = labels_of[stage == STAGE_TRAIN_EVAL]
        train_group = group[stage == STAGE_TRAIN_EVAL]
        same_test = np.isin(truth, subset[0])
        saved.update(train_eval_ids=ids[stage == STAGE_TRAIN_EVAL], train_eval_counts=train_counts,
                     train_eval_truth=train_truth, train_eval_group=train_group)
        result['forget_set'] = {}
        for name, neuron_labels in readouts.items():
            train_pred, _ = predict(train_counts, neuron_labels)
            test_pred, _ = predict(eval_counts[same_test], neuron_labels)
            margin_train = true_class_margin(train_counts, neuron_labels, train_truth)
            margin_test = true_class_margin(eval_counts[same_test], neuron_labels, truth[same_test])
            f, r_ = train_group == GROUP_FORGET, train_group == GROUP_RETAINED_SAME
            result['forget_set'][name] = dict(
                n_forget=int(f.sum()), n_retained_same=int(r_.sum()), n_test_same=int(same_test.sum()),
                forget_accuracy=float(np.mean(train_pred[f] == train_truth[f])),
                retained_same_class_accuracy=float(np.mean(train_pred[r_] == train_truth[r_])),
                test_same_class_accuracy=float(np.mean(test_pred == truth[same_test])),
                test_other_class_accuracy=float(np.mean(
                    predict(eval_counts[~same_test], neuron_labels)[0] == truth[~same_test])),
                membership_auc_forget_vs_test=auc(margin_train[f], margin_test),
                membership_auc_retained_vs_test=auc(margin_train[r_], margin_test),
                forget_mean_spikes=float(train_counts[f].sum(axis=1).mean()),
                test_same_mean_spikes=float(eval_counts[same_test].sum(axis=1).mean()))
    for name, neuron_labels in readouts.items():
        predictions, ties = predict(eval_counts, neuron_labels)
        r = score(predictions, ties, truth, eval_counts, removed)
        r['neurons_per_class'] = np.bincount(neuron_labels[neuron_labels >= 0], minlength=10).tolist()
        r['unlabelled_neurons'] = int(np.sum(neuron_labels < 0))
        result['readouts'][name] = r
        saved[f'labels_{name}'] = neuron_labels
        saved[f'predictions_{name}'] = predictions
    np.savez(root / 'responses.npz', **saved)
    save_json(root / 'metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main(argv=None):
    return run(arguments(argv))
