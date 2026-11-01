"""Label-guided silencing with frozen compiled-CPU inference.

The existing builder has no spike-enable variable. A separate +1 V threshold
override implements the specification's fallback. Frozen baseline theta and
weights remain unchanged in the saved checkpoint; the override is an adapter
input, not an edited homeostatic threshold. In this conductance model voltage
cannot exceed the 0 V excitatory reversal potential. We also check every
observed silenced-neuron count, including rest and failed retry attempts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .fashion_data import (CLASS_NAMES, balanced_ids, fingerprint, forget_classes,
                          load_dataset, save_json)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(directory, dataset='mnist'):
    directory = Path(directory).resolve()
    config = json.loads((directory / 'config.json').read_text())
    if config.get('dataset') != dataset:
        raise ValueError(f'Expected a {dataset} checkpoint')
    with np.load(directory / 'baseline_weights.npz', allow_pickle=False) as z:
        weights = {k: z[k].copy() for k in z.files}
    labels = np.load(directory / 'neuron_labels.npy', allow_pickle=False)
    n = int(weights['n_e'])
    if (int(weights['n_input']) != 784 or config['n_excitatory'] != n
            or labels.shape != (n,) or weights['theta_volts'].shape != (n,)):
        raise ValueError('Checkpoint dimensions do not match config/labels')
    if (not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < -1) | (labels > 9))):
        raise ValueError('Invalid saved neuron labels')
    if (not np.array_equal(weights['i'], np.repeat(np.arange(784), n))
            or not np.array_equal(weights['j'], np.tile(np.arange(n), 784))):
        raise ValueError('Expected the existing all-to-all connectivity/order')
    if (weights['w'].shape != (784 * n,)
            or not np.isfinite(weights['w']).all()
            or not np.isfinite(weights['theta_volts']).all()
            or np.any((weights['w'] < 0) | (weights['w'] > 1))):
        raise ValueError('Invalid weights/thresholds')
    return config, weights, labels


def class_scores(counts, labels, suppress=()):
    """Same mean-per-assigned-neuron vote and lowest-label tie rule as training."""
    scores = np.full((len(counts), 10), -np.inf)
    for label in range(10):
        selected = labels == label
        if np.any(selected) and label not in suppress:
            scores[:, label] = counts[:, selected].mean(axis=1)
    predictions = scores.argmax(axis=1)
    if not np.isfinite(scores).any():
        predictions[:] = -1
    return scores, predictions


def report(counts, labels, truth, forget_classes, suppress=()):
    scores, predictions = class_scores(counts, labels, suppress)
    forget = np.isin(truth, forget_classes)
    accuracy = lambda mask: float(np.mean(predictions[mask] == truth[mask])) if np.any(mask) else None
    confusion = np.zeros((10, 11), dtype=np.int64)
    np.add.at(confusion, (truth, np.where(predictions < 0, 10, predictions)), 1)
    result = dict(accuracy=accuracy(np.ones(len(truth), dtype=bool)),
                  retain_accuracy=accuracy(~forget), forget_class_accuracy=accuracy(forget),
                  per_class_accuracy={str(k): accuracy(truth == k) for k in np.unique(truth)},
                  n_evaluation=len(truth), silent_fraction=float(np.mean(counts.sum(axis=1) == 0)),
                  tie_fraction=float(np.mean((scores == scores.max(axis=1)[:, None]).sum(axis=1) > 1)),
                  mean_spikes_retain=float(counts[~forget].sum(axis=1).mean()) if (~forget).any() else None,
                  mean_spikes_forget=float(counts[forget].sum(axis=1).mean()) if forget.any() else None,
                  unassigned_neurons=int(np.sum(labels < 0)))
    return result, scores, predictions, confusion


def compiled_counts(weights, images, seed, override, directory, input_total_hz=0.0):
    """One compiled image/retry loop; no Python per-image simulation calls.

    Start from the builder's resting state and use the original continuous
    150 ms inter-image rest protocol (no new hard resets). All models start
    fresh with the same Poisson seed and image order. Retry counts may differ,
    so the continuing RNG stream is not identical per image across models.
    """
    from brian2 import get_device, prefs, set_device, start_scope, defaultclock, volt
    from .network import build_network, set_plasticity
    from .encoding import PRESENT_TIME, REST_TIME

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    n = int(weights['n_e'])
    if override.shape != (n,) or np.any(override < 0) or not np.isfinite(override).all():
        raise ValueError('Invalid threshold override')
    masked = override > 0
    if np.any(weights['theta_volts'][masked] + override[masked] <= 0.072):
        raise ValueError('Override must put effective firing threshold above 0 V')
    set_device('cpp_standalone', build_on_run=False)
    device = get_device()
    device.reinit()
    device.activate(build_on_run=False)
    start_scope()
    prefs.devices.cpp_standalone.openmp_threads = 0
    prefs.devices.cpp_standalone.extra_make_args_unix = ['-j1']
    prefs.codegen.cpp.headers += ['<fstream>', '<vector>', '<algorithm>', '<stdexcept>', '<cmath>']
    c = build_network(n, seed)
    c['syn_input_exc'].w = weights['w']
    c['exc'].theta = (weights['theta_volts'] + override) * volt
    set_plasticity(c, False)
    images = np.asarray(images, dtype=np.uint8, order='C')
    if images.ndim != 2 or images.shape[1] != 784 or not len(images):
        raise ValueError('Expected nonempty 28x28 image matrix')
    images.tofile(directory / 'images.bin')
    masked.astype(np.uint8).tofile(directory / 'mask.bin')

    def array(obj, key):
        var = obj.variables[key]
        return (device.get_array_name(var, access_data=False) + '.data()'
                if var.dynamic else device.get_array_name(var))

    rates = array(c['input'], 'rates')
    monitor = array(c['exc_monitor'], 'count')
    quote = lambda name: json.dumps(str(directory / name))
    device.insert_code('main', f'''
    std::vector<unsigned char> method_a_images({images.size}), method_a_mask({n});
    std::ifstream method_a_input({quote('images.bin')}, std::ios::binary);
    std::ifstream method_a_mask_input({quote('mask.bin')}, std::ios::binary);
    if (!method_a_input.read(reinterpret_cast<char*>(method_a_images.data()), {images.size}) ||
        !method_a_mask_input.read(reinterpret_cast<char*>(method_a_mask.data()), {n}))
        throw std::runtime_error("Cannot read inference inputs");
    std::ofstream method_a_output({quote('counts.bin')}, std::ios::binary);
    if (!method_a_output) throw std::runtime_error("Cannot open inference output");
    std::vector<int> method_a_before({n}), method_a_counts({n});
    long long method_a_attempts[6] = {{0,0,0,0,0,0}};
    long long method_a_exhausted = 0;
    for (size_t image = 0; image < {len(images)}; ++image) {{
        int total = 0, attempt = 0;
        const unsigned char* pixels = method_a_images.data() + image * 784;
        double scale = 1.0, sum = 0.0;
        if ({float(input_total_hz)!r} > 0) {{
            for(int k=0; k<784; ++k) sum += pixels[k] / 4.0;
            if(sum > 0) scale = {float(input_total_hz)!r} / sum;
        }}
        for (; attempt < 6; ++attempt) {{
            for(int k=0; k<784; ++k)
                {rates}[k] = pixels[k] / 4.0 * scale * (1.0 + attempt * 32.0 / 63.75);
            std::copy({monitor}, {monitor}+{n}, method_a_before.begin());
    ''')

    def queue_run(duration):
        # Same repeated-run clock fix as the existing compiled training path.
        offset = len(device.main_queue)
        c['net'].run(duration)
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
            raise RuntimeError('Unsupported Brian clock queue layout')

    queue_run(PRESENT_TIME)
    device.insert_code('main', f'''
            total = 0;
            for(int k=0; k<{n}; ++k) {{
                method_a_counts[k] = {monitor}[k] - method_a_before[k];
                total += method_a_counts[k];
            }}
            std::fill({rates}, {rates}+784, 0.0);
    ''')
    queue_run(REST_TIME)
    device.insert_code('main', f'''
            for(int k=0; k<{n}; ++k)
                if(method_a_mask[k] && {monitor}[k] != 0)
                    throw std::runtime_error("Silenced neuron emitted a spike");
            if(total >= 5) break;
        }}
        ++method_a_attempts[std::min(attempt,5)];
        if(total < 5) ++method_a_exhausted;
        method_a_output.write(reinterpret_cast<char*>(method_a_counts.data()), {n}*sizeof(int));
        if(!method_a_output) throw std::runtime_error("Inference write failed");
        if((image+1)%100 == 0)
            std::cout << "evaluation=" << image+1 << "/{len(images)}" << std::endl;
    }}
    method_a_output.close();
    std::ofstream method_a_stats({quote('retry_stats.txt')});
    for(int k=0; k<6; ++k) method_a_stats << method_a_attempts[k] << " ";
    method_a_stats << method_a_exhausted;
    if(!method_a_stats) throw std::runtime_error("Cannot save retry statistics");
    ''')
    start = time.monotonic()
    device.build(directory=str(directory / 'build'), compile=True, run=True)
    elapsed = time.monotonic() - start
    if (not np.array_equal(np.array(c['syn_input_exc'].w[:]), weights['w'])
            or not np.array_equal(np.array(c['exc'].theta[:] / volt), weights['theta_volts'] + override)):
        raise RuntimeError('Frozen inference changed weights or thresholds')
    counts = np.fromfile(directory / 'counts.bin', dtype=np.int32).reshape(len(images), n)
    if np.any(counts[:, masked]):
        raise RuntimeError('Silencing failed')
    stats = np.loadtxt(directory / 'retry_stats.txt', dtype=np.int64)
    return counts, dict(build_and_run_seconds=elapsed, attempts=stats[:6].tolist(),
                        exhausted=int(stats[6]), frozen_parameters_verified=True,
                        zero_silenced_spikes_verified=True)


def deletion_request(value):
    try:
        if forget_classes(value):
            return value
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('Use classK or increasing classes such as class6_9')


def fashion_comparison(truth, outputs, summaries, classes):
    """Paired per-class collateral effects, including shirt/upper-body confusions."""
    baseline = outputs['baseline']
    comparisons = {}
    for model, predictions in outputs.items():
        rows = {}
        for k, name in enumerate(CLASS_NAMES):
            selected = truth == k
            if not selected.any():
                continue
            before = baseline[selected] == k
            after = predictions[selected] == k
            rows[str(k)] = dict(
                name=name, forgotten=k in classes, n=int(selected.sum()),
                accuracy=summaries[model]['per_class_accuracy'][str(k)],
                accuracy_delta_vs_baseline=float(after.mean() - before.mean()),
                baseline_correct_now_wrong=int(np.sum(before & ~after)),
                baseline_wrong_now_correct=int(np.sum(~before & after)),
                predicted_as_shirt_fraction=float(np.mean(predictions[selected] == 6)))
        shirt = truth == 6
        comparisons[model] = dict(
            per_class=rows,
            shirt_prediction_distribution={str(k): float(np.mean(predictions[shirt] == k))
                                           for k in range(-1, 10)} if shirt.any() else {},
            upper_body_retain_classes=[k for k in (0, 2, 3, 4) if k not in classes])
    return dict(class_names=dict(enumerate(CLASS_NAMES)), comparisons=comparisons)


def main(dataset='mnist'):
    if dataset not in ('mnist', 'fashion'):
        raise ValueError('Unsupported dataset')
    fashion = dataset == 'fashion'
    baseline = ('fashion/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_wn39_cpp_2507283'
                if fashion else 'mnist/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_cpp_2461985')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, default=Path(baseline))
    gold_options = p.add_mutually_exclusive_group()
    gold_options.add_argument('--gold-dir', type=Path, help='Matched retain-only Gold for comparison; defaults to existing joint Gold for class6_9')
    gold_options.add_argument('--no-gold', action='store_true', help='Run without a Gold comparison checkpoint')
    p.add_argument('--request', type=deletion_request, default='class6_9')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=Path(f'data/{dataset}'))
    p.add_argument('--eval-limit', type=int, default=0, help='0 = full test; otherwise balanced multiple of 10')
    p.add_argument('--inference-seed', type=int, default=12345)
    p.add_argument('--prepare-only', action='store_true', help='Save the independent intervention checkpoint without simulation/data/Brian2')
    a = p.parse_args()
    if a.eval_limit < 0 or a.eval_limit > 10000 or a.eval_limit % 10:
        p.error('--eval-limit must be 0 or a multiple of ten up to 10000')
    if not 0 <= a.inference_seed < 2**32:
        p.error('--inference-seed must be an unsigned 32-bit integer')
    classes = [int(k) for k in a.request.removeprefix('class').split('_')]
    config, weights, labels = load_model(a.baseline_dir, dataset)
    if config.get('request') != 'baseline':
        p.error('--baseline-dir must identify the original baseline')
    if a.gold_dir is None and not a.no_gold and a.request == 'class6_9':
        a.gold_dir = Path('gold_weights/fashion_class6_9_wn39' if fashion else 'gold_weights/mnist_class6_9')
    gold = load_model(a.gold_dir, dataset) if a.gold_dir else None
    if gold:
        gc = gold[0]
        for key in ('dataset', 'dataset_sha256', 'n_excitatory', 'seed', 'split_seed', 'epochs', 'validation_size', 'train_limit', 'assign_per_class'):
            if gc[key] != config[key]:
                p.error(f'Gold mismatch in {key}')
        if gc.get('request') != a.request:
            p.error('Gold request must match exactly; joint Gold cannot reference individual deletion')
        if np.isin(gold[2], classes).any():
            p.error('Gold assignments include a forgotten class')
        if float(gc.get('input_total_hz', 0)) != float(config.get('input_total_hz', 0)):
            p.error('Gold input encoding differs')
        if float(gc.get('weight_norm', 0)) != float(config.get('weight_norm', 0)):
            p.error('Gold training weight normalization differs')
    if a.out_dir.exists():
        p.error('Output directory already exists; choose a new directory')
    # Refuse output paths nested inside either source artifact directory.
    for source in (a.baseline_dir, a.gold_dir):
        if source and (a.out_dir.resolve() == source.resolve() or source.resolve() in a.out_dir.resolve().parents):
            p.error('Output must be outside source model directories')
    sources = {str(path.resolve()): sha256(path) for directory in (a.baseline_dir, a.gold_dir)
               if directory for path in directory.iterdir() if path.is_file()}
    intervention_start = time.perf_counter()
    mask = np.isin(labels, classes)
    override = mask.astype(np.float64)  # +1 volt, stored separately from frozen theta.
    intervention_seconds = time.perf_counter() - intervention_start
    a.out_dir.mkdir(parents=True)
    save_json(a.out_dir / 'status.json', dict(status='preparing'))
    checkpoint = a.out_dir / 'checkpoint'
    checkpoint.mkdir()
    shutil.copyfile(a.baseline_dir / 'baseline_weights.npz', checkpoint / 'baseline_weights.npz')
    shutil.copyfile(a.baseline_dir / 'neuron_labels.npy', checkpoint / 'neuron_labels.npy')
    np.savez(checkpoint / 'intervention.npz', forget_neuron_mask=mask, threshold_override_volts=override,
             forget_classes=np.array(classes, dtype=np.int64))
    metadata = dict(method='A', implementation='persistent_threshold_override', dataset=dataset,
                    request=a.request, baseline_config=config, gold_config=gold[0] if gold else None,
                    source_artifact_sha256=sources, n_silenced=int(mask.sum()),
                    fraction_silenced=float(mask.mean()), empty_intervention=not bool(mask.any()),
                    silenced_per_class={str(k): int(np.sum(labels == k)) for k in classes},
                    intervention_seconds_excluding_io=intervention_seconds,
                    inference_seed=a.inference_seed, eval_limit=a.eval_limit,
                    voting='original saved labels; mean spikes per assigned neuron; lowest-label ties',
                    state_protocol='fresh builder state, checkpoint theta; original continuous 150ms inter-image rest',
                    rng_protocol='same initial seed/order; retries can consume different RNG stream lengths',
                    weight_normalization=False, threshold_adaptation=False,
                    code_sha256=sha256(Path(__file__)))
    save_json(a.out_dir / 'config.json', metadata)
    print(f'{a.request}: silencing {mask.sum()}/{len(mask)} neurons ({mask.mean():.2%})', flush=True)
    try:
        if a.prepare_only:
            save_json(a.out_dir / 'status.json', dict(status='prepared', evaluated=False))
            return
        arrays = load_dataset(a.data_dir, dataset=dataset)
        if fingerprint(arrays) != config['dataset_sha256']:
            raise ValueError(f'{dataset} data fingerprint differs from baseline')
        ids = np.arange(len(arrays[3]))
        if a.eval_limit:
            ids = balanced_ids(arrays[3], ids, a.eval_limit // 10, a.inference_seed)
        truth = arrays[3][ids]
        np.save(a.out_dir / 'evaluation_ids.npy', ids)
        build_root = Path(os.environ.get('FASHION_BUILD_ROOT', str(a.out_dir / 'compiled'))).resolve() / a.out_dir.resolve().name
        summary = {}
        outputs = {}
        models = [('baseline', weights, labels, np.zeros_like(override), config),
                  ('method_a', weights, labels, override, config)]
        if gold:
            models.append(('gold', gold[1], gold[2], np.zeros_like(override), gold[0]))
        for name, model_weights, model_labels, threshold_override, model_config in models:
            save_json(a.out_dir / 'status.json', dict(status='evaluating', model=name))
            counts, timing = compiled_counts(model_weights, arrays[2][ids], a.inference_seed,
                                              threshold_override, build_root / name,
                                              float(model_config.get('input_total_hz', 0)))
            metrics, scores, predictions, confusion = report(counts, model_labels, truth, classes)
            metrics.update(timing)
            target = a.out_dir / name
            target.mkdir()
            save_json(target / 'metrics.json', metrics)
            np.save(target / 'confusion_matrix.npy', confusion)
            np.savez(target / 'responses.npz', evaluation_ids=ids, truth=truth, predictions=predictions,
                     counts=counts, scores=scores)
            summary[name] = metrics
            outputs[name] = predictions
            if name == 'baseline':
                control, cs, cp, cc = report(counts, model_labels, truth, classes, suppress=classes)
                target = a.out_dir / 'readout_suppression_control'
                target.mkdir()
                save_json(target / 'metrics.json', control)
                np.save(target / 'confusion_matrix.npy', cc)
                np.savez(target / 'predictions.npz', evaluation_ids=ids, truth=truth, predictions=cp, scores=cs)
                summary['readout_suppression_control'] = control
                outputs['readout_suppression_control'] = cp
        for name in ('baseline', 'readout_suppression_control', 'gold'):
            if name in outputs:
                summary[f'method_a_disagreement_with_{name}'] = {
                    split: float(np.mean(outputs['method_a'][select] != outputs[name][select]))
                    for split, select in [('all', np.ones(len(truth), dtype=bool)),
                                          ('retain', ~np.isin(truth, classes)), ('forget', np.isin(truth, classes))]}
        summary['gold_comparison_available'] = gold is not None
        if fashion:
            save_json(a.out_dir / 'fashion_comparison.json',
                      fashion_comparison(truth, outputs, summary, classes))
        save_json(a.out_dir / 'summary.json', summary)
        save_json(a.out_dir / 'status.json', dict(status='complete'))
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        save_json(a.out_dir / 'status.json', dict(status='failed', error=str(error)))
        raise
    finally:
        if any(sha256(path) != expected for path, expected in sources.items()):
            save_json(a.out_dir / 'status.json', dict(status='failed', error='Source artifact changed'))
            raise RuntimeError('Source artifact changed during execution')


if __name__ == '__main__':
    main()
