"""Independent MNIST/Fashion C1/C2 replay using the Brian compiled CPU model."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .fashion_data import balanced_ids, fingerprint, load_dataset, save_json
from .method_a import (compiled_counts, deletion_request, fashion_comparison,
                       load_model, report, sha256)

BASELINE = Path('mnist/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_cpp_2461985')
GOLD = Path('gold_weights/mnist_class6_9')


def build_erasure_network(n, seed, variant, scale, adapt_thresholds=False):
    """Replace only the input synapse, retaining its trace and event scheduling.

    Extract the raw expression from the original clipped assignment so the
    transformation uses the *unclipped* event increment at the current weight.
    Per-synapse counters avoid shared write races and count effective changes.
    """
    from brian2 import Synapses
    from .network import (build_network, STDP_EQS, STDP_ON_PRE, STDP_ON_POST,
                          ETA, MU, X_TAR, TC_PRE, W_MAX_EE)
    if variant not in ('C1', 'C2') or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Expected C1/C2 and finite positive scale')
    prefix, suffix = 'w = clip(w + ', ', 0, w_max)'
    rule = STDP_ON_POST.strip()
    if not rule.startswith(prefix) or not rule.endswith(suffix):
        raise ValueError('Unsupported baseline STDP expression')
    expression = rule[len(prefix):-len(suffix)]
    transform = '-erasure_scale * delta_base' if variant == 'C1' else '-erasure_scale * clip(delta_base, 0, inf)'
    c = build_network(n, seed)
    old = c['syn_input_exc']
    c['net'].remove(old)
    # Standalone device registers weak references to the original arrays.
    # Keep the removed object alive until code generation has completed.
    c['_original_input_synapse'] = old
    syn = Synapses(c['input'], c['exc'], model=STDP_EQS + '''
    erasure_scale : 1 (shared)
    erasure_enabled : 1 (shared)
    positive_updates : 1
    negative_updates : 1
    clipped_updates : 1
    ''', on_pre=STDP_ON_PRE, on_post=f'''
    delta_base = {expression}
    delta_edit = erasure_enabled * ({transform})
    old_weight = w
    candidate = w + delta_edit
    w = clip(candidate, 0, w_max)
    positive_updates += int(w > old_weight)
    negative_updates += int(w < old_weight)
    clipped_updates += int(candidate < 0 or candidate > w_max)
    ''', method='euler', name='method_c_input_to_exc')
    syn.connect()
    syn.w = 0  # Loaded from the supplied checkpoint before replay.
    syn.tau_tr = TC_PRE
    syn.x_tar = X_TAR
    syn.mu = MU
    syn.eta = ETA
    syn.w_max = W_MAX_EE
    syn.erasure_scale = scale
    syn.erasure_enabled = 1
    c['net'].add(syn)
    c['syn_input_exc'] = syn
    c['exc'].plasticity_enabled = int(adapt_thresholds)
    return c


def compiled_replay(weights, images, orders, seed, variant, scale, directory,
                    save_passes=(1, 2, 5), input_total_hz=0.0, adapt_thresholds=False):
    """One compilation/trajectory, fixed exposure, snapshots at pass boundaries."""
    from brian2 import get_device, prefs, set_device, start_scope, defaultclock, volt
    from .encoding import PRESENT_TIME, REST_TIME
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    images = np.asarray(images, dtype=np.uint8, order='C')
    orders = np.asarray(orders, dtype=np.int64, order='C')
    if (images.ndim != 2 or images.shape[1] != 784 or not len(images)
            or orders.ndim != 2 or orders.shape[1] != len(images)
            or any(not np.array_equal(np.sort(row), np.arange(len(images))) for row in orders)):
        raise ValueError('Each pass must permute the nonempty replay set exactly once')
    if not save_passes or min(save_passes) < 1 or max(save_passes) > len(orders):
        raise ValueError('Snapshot passes outside replay trajectory')
    n = int(weights['n_e'])
    set_device('cpp_standalone', build_on_run=False)
    device = get_device()
    device.reinit()
    device.activate(build_on_run=False)
    start_scope()
    prefs.devices.cpp_standalone.openmp_threads = 0
    prefs.devices.cpp_standalone.extra_make_args_unix = ['-j1']
    prefs.codegen.cpp.headers += ['<fstream>', '<vector>', '<algorithm>', '<stdexcept>', '<cmath>']
    c = build_erasure_network(n, seed, variant, scale, adapt_thresholds)
    c['syn_input_exc'].w = weights['w']
    c['exc'].theta = weights['theta_volts'] * volt
    images.tofile(directory / 'images.bin')
    orders.tofile(directory / 'orders.bin')

    def array(obj, key):
        var = obj.variables[key]
        return (device.get_array_name(var, access_data=False) + '.data()'
                if var.dynamic else device.get_array_name(var))

    def queue_run(duration):
        # Same clock repair as fashion_standalone and Method A: C++ owns time.
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

    syn = c['syn_input_exc']
    rates, monitor = array(c['input'], 'rates'), array(c['exc_monitor'], 'count')
    w, theta = array(syn, 'w'), array(c['exc'], 'theta')
    enabled = array(syn, 'erasure_enabled')
    counters = [array(syn, key) for key in ('positive_updates', 'negative_updates', 'clipped_updates')]
    quote = lambda name: json.dumps(str(directory / name))
    device.insert_code('main', f'''
    std::vector<unsigned char> replay_images({images.size});
    std::vector<int64_t> replay_orders({orders.size});
    std::ifstream replay_in({quote('images.bin')}, std::ios::binary);
    std::ifstream order_in({quote('orders.bin')}, std::ios::binary);
    if(!replay_in.read(reinterpret_cast<char*>(replay_images.data()), {images.size}) ||
       !order_in.read(reinterpret_cast<char*>(replay_orders.data()), {orders.nbytes}))
        throw std::runtime_error("Cannot read replay inputs");
    std::ofstream diagnostics({quote('presentations.csv')});
    diagnostics << "pass,position,replay_index,active_spikes,rest_spikes,positive_updates,negative_updates,clipped_updates\\n";
    std::vector<int> replay_before({n});
    double counter_before[3] = {{0,0,0}};
    for(int pass=1; pass<={len(orders)}; ++pass) {{
        for(int position=0; position<{len(images)}; ++position) {{
            int64_t image = replay_orders[(pass-1)*{len(images)}+position];
            const unsigned char* pixels = replay_images.data()+image*784;
            double scale=1.0, sum=0.0;
            if({float(input_total_hz)!r} > 0) {{
                for(int k=0;k<784;++k) sum += pixels[k]/4.0;
                if(sum>0) scale={float(input_total_hz)!r}/sum;
            }}
            for(int k=0;k<784;++k) {rates}[k]=pixels[k]/4.0*scale;
            std::copy({monitor}, {monitor}+{n}, replay_before.begin());
            {enabled}[0]=1;
    ''')
    queue_run(PRESENT_TIME)
    device.insert_code('main', f'''
            long long active_spikes=0;
            for(int k=0;k<{n};++k) active_spikes += {monitor}[k]-replay_before[k];
            std::copy({monitor}, {monitor}+{n}, replay_before.begin());
            std::fill({rates}, {rates}+784, 0.0);
            {enabled}[0]=0;
    ''')
    queue_run(REST_TIME)
    device.insert_code('main', f'''
            long long rest_spikes=0;
            for(int k=0;k<{n};++k) rest_spikes += {monitor}[k]-replay_before[k];
            double totals[3]={{0,0,0}};
            for(int k=0;k<{784*n};++k) {{
                totals[0]+={counters[0]}[k]; totals[1]+={counters[1]}[k]; totals[2]+={counters[2]}[k];
            }}
            diagnostics << pass << "," << position << "," << image << "," << active_spikes << "," << rest_spikes;
            for(int k=0;k<3;++k) {{ diagnostics << "," << totals[k]-counter_before[k]; counter_before[k]=totals[k]; }}
            diagnostics << "\\n";
            if((position+1)%100==0) std::cout << "pass=" << pass << " replay=" << position+1 << "/{len(images)}" << std::endl;
        }}
        if({' || '.join(f'pass=={int(p)}' for p in save_passes)}) {{
            std::string stem={json.dumps(str(directory / 'pass_'))}+std::to_string(pass);
            std::ofstream weights_out(stem+"_w.bin", std::ios::binary);
            std::ofstream theta_out(stem+"_theta.bin", std::ios::binary);
            weights_out.write(reinterpret_cast<char*>({w}), {784*n}*sizeof(double));
            theta_out.write(reinterpret_cast<char*>({theta}), {n}*sizeof(double));
            if(!weights_out || !theta_out) throw std::runtime_error("Cannot save replay snapshot");
        }}
    }}
    diagnostics.close();
    if(!diagnostics) throw std::runtime_error("Cannot save replay diagnostics");
    ''')
    started = time.monotonic()
    device.build(directory=str(directory / 'build'), compile=True, run=True)
    elapsed = time.monotonic() - started
    snapshots = {}
    previous = weights['w']
    for p in sorted(save_passes):
        current = {key: np.array(value, copy=True) for key, value in weights.items()}
        current['w'] = np.fromfile(directory / f'pass_{p}_w.bin', dtype=np.float64)
        current['theta_volts'] = np.fromfile(directory / f'pass_{p}_theta.bin', dtype=np.float64)
        if (current['w'].shape != previous.shape or not np.isfinite(current['w']).all()
                or np.any((current['w'] < 0) | (current['w'] > 1))):
            raise RuntimeError('Invalid replay weights')
        if current['theta_volts'].shape != weights['theta_volts'].shape or not np.isfinite(current['theta_volts']).all():
            raise RuntimeError('Invalid replay thresholds')
        if not adapt_thresholds and not np.array_equal(current['theta_volts'], weights['theta_volts']):
            raise RuntimeError('Frozen replay thresholds changed')
        if variant == 'C2' and np.any(current['w'] > previous):
            raise RuntimeError('C2 increased a weight')
        previous = current['w']
        snapshots[p] = current
    diagnostics = np.genfromtxt(directory / 'presentations.csv', delimiter=',', names=True, ndmin=1)
    if len(diagnostics) != orders.size or (variant == 'C2' and np.any(diagnostics['positive_updates'])):
        raise RuntimeError('Invalid replay diagnostics')
    return snapshots, diagnostics, dict(build_and_run_seconds=elapsed,
                                       frozen_thresholds_verified=not adapt_thresholds)


def evaluate(weights, labels, images, truth, ids, seed, target, build_dir, input_total_hz,
             classes=(6, 9)):
    counts, timing = compiled_counts(weights, images, seed, np.zeros(len(labels)), build_dir, input_total_hz)
    metrics, scores, predictions, confusion = report(counts, labels, truth, classes)
    metrics.update(timing)
    target.mkdir(parents=True)
    save_json(target / 'metrics.json', metrics)
    np.save(target / 'confusion_matrix.npy', confusion)
    np.savez_compressed(target / 'responses.npz', evaluation_ids=ids, truth=truth,
                        predictions=predictions, scores=scores, counts=counts)
    return metrics, predictions, counts


def main(dataset='mnist'):
    if dataset not in ('mnist', 'fashion'):
        raise ValueError('Unsupported dataset')
    fashion = dataset == 'fashion'
    baseline = (Path('fashion/n800_baseline_seed0_ep3_trainall_val0_evalall_assign500_wn39_cpp_2507283')
                if fashion else BASELINE)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, default=baseline)
    gold_options = p.add_mutually_exclusive_group()
    gold_options.add_argument('--gold-dir', type=Path, help='Matched Gold; defaults to joint Gold for class6_9')
    gold_options.add_argument('--no-gold', action='store_true', help='Evaluate without a Gold reference')
    p.add_argument('--request', type=deletion_request, default='class6_9')
    p.add_argument('--data-dir', type=Path, default=Path(f'data/{dataset}'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--variants', nargs='+', choices=('C1', 'C2'), default=['C1', 'C2'])
    p.add_argument('--scales', nargs='+', type=float, default=[0.1, 0.5, 1.0])
    p.add_argument('--passes', nargs='+', type=int, default=[1, 2, 5])
    p.add_argument('--replay-seed', type=int, default=2026)
    p.add_argument('--inference-seed', type=int, default=12345)
    p.add_argument('--replay-per-class', type=int, default=0, help='0 = all Df; positive = balanced PILOT subset')
    p.add_argument('--eval-limit', type=int, default=0, help='0 = full split; otherwise balanced multiple of ten')
    p.add_argument('--skip-evaluation', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Validate sources and save replay plan, without compiling')
    p.add_argument('--adapt-thresholds', action='store_true', help='Explicit homeostasis ablation; inference remains frozen')
    a = p.parse_args()
    classes = [int(k) for k in a.request.removeprefix('class').split('_')]
    if a.gold_dir is None and not a.no_gold and a.request == 'class6_9':
        a.gold_dir = Path('gold_weights/fashion_class6_9_wn39') if fashion else GOLD
    if (any(not np.isfinite(s) or s <= 0 for s in a.scales) or min(a.passes) < 1
            or len(set(a.scales)) != len(a.scales) or len(set(a.variants)) != len(a.variants)
            or len(set(a.passes)) != len(a.passes) or a.replay_per_class < 0
            or a.eval_limit < 0 or a.eval_limit % 10
            or any(not 0 <= s < 2**32 for s in (a.replay_seed, a.inference_seed))):
        p.error('Invalid scale/pass/seed/limit or duplicate trials')
    if a.out_dir.exists():
        p.error('Output already exists; choose a fresh directory')
    for source in (a.baseline_dir, a.gold_dir):
        if source and (a.out_dir.resolve() == source.resolve() or source.resolve() in a.out_dir.resolve().parents):
            p.error('Output must be outside source model directories')
    config, weights, labels = load_model(a.baseline_dir, dataset)
    gold = load_model(a.gold_dir, dataset) if a.gold_dir else None
    if config.get('request') != 'baseline':
        p.error('Require the original baseline')
    if gold:
        gc, gw, gl = gold
        if gc.get('request') != a.request or np.isin(gl, classes).any():
            p.error('Gold must match the exact deletion request and exclude its labels')
        for key in ('dataset', 'dataset_sha256', 'n_excitatory', 'seed', 'split_seed', 'epochs',
                    'validation_size', 'train_limit', 'assign_per_class'):
            if config.get(key) != gc.get(key):
                p.error(f'Gold mismatch: {key}')
        for key in ('input_total_hz', 'weight_norm'):
            if float(config.get(key, 0)) != float(gc.get(key, 0)):
                p.error(f'Gold mismatch: {key}')
    sources = {str(path.resolve()): sha256(path)
               for directory in (a.baseline_dir, a.gold_dir) if directory
               for path in directory.iterdir() if path.is_file()}
    arrays = load_dataset(a.data_dir, dataset=dataset)
    if fingerprint(arrays) != config['dataset_sha256']:
        p.error(f'{dataset} fingerprint differs from baseline')
    with np.load(a.baseline_dir / 'manifest.npz', allow_pickle=False) as manifest:
        train_ids = manifest['train'].copy()
        validation = manifest['validation'].copy()
    if (train_ids.ndim != 1 or not np.issubdtype(train_ids.dtype, np.integer)
            or len(np.unique(train_ids)) != len(train_ids) or np.any((train_ids < 0) | (train_ids >= len(arrays[1])))):
        p.error('Invalid source training IDs')
    if (validation.ndim != 1 or not np.issubdtype(validation.dtype, np.integer)
            or len(np.unique(validation)) != len(validation)
            or np.any((validation < 0) | (validation >= len(arrays[1])))
            or np.intersect1d(train_ids, validation).size
            or len(validation) != config['validation_size']):
        p.error('Invalid or overlapping source validation IDs')
    forget_ids = train_ids[np.isin(arrays[1][train_ids], classes)]
    if a.replay_per_class:
        forget_ids = balanced_ids(arrays[1], forget_ids, a.replay_per_class, a.replay_seed)
    if set(arrays[1][forget_ids].tolist()) != set(classes):
        p.error('Replay requires every requested forget class')
    # Separate order RNG from the Brian Poisson seed; reuse orders for all trials.
    rng = np.random.default_rng(a.replay_seed)
    orders = np.array([rng.permutation(len(forget_ids)) for _ in range(max(a.passes))])
    split = 'validation' if len(validation) else 'test'
    eval_x, eval_y = (arrays[0], arrays[1]) if len(validation) else (arrays[2], arrays[3])
    ids = validation if len(validation) else np.arange(len(eval_y))
    if a.eval_limit:
        ids = balanced_ids(eval_y, ids, a.eval_limit // 10, a.inference_seed)
    a.out_dir.mkdir(parents=True)
    np.savez_compressed(a.out_dir / 'replay_manifest.npz', forget_ids=forget_ids,
                        forget_labels=arrays[1][forget_ids], replay_order_ids=forget_ids[orders], evaluation_ids=ids)
    metadata = dict(method='C', dataset=dataset, request=a.request,
                    baseline_config=config, gold_config=gold[0] if gold else None,
                    variants=a.variants, scales=a.scales, passes=sorted(a.passes),
                    replay_seed=a.replay_seed, inference_seed=a.inference_seed,
                    n_forget=len(forget_ids), replay_per_class=a.replay_per_class,
                    pilot=bool(a.replay_per_class or a.eval_limit), evaluation_split=split,
                    parameter_selection='none; prespecified sweep; test results must not select parameters',
                    state_protocol='original continuous 150ms rest; no hard reset; traces evolve during rest',
                    exposure='350ms at pixel/4 rates with saved input_total_hz scaling; no retries or boosts; updates disabled during 150ms rest',
                    threshold_adaptation=a.adapt_thresholds, weight_normalization=False,
                    voting='original saved labels; mean spikes; lowest-label ties',
                    source_artifact_sha256=sources,
                    code_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                                 ('method_c.py', 'method_a.py', 'network.py', 'encoding.py', 'fashion_data.py')})
    save_json(a.out_dir / 'config.json', metadata)
    save_json(a.out_dir / 'status.json', dict(status='prepared'))
    if a.prepare_only:
        print(f'Prepared {len(forget_ids)} forget images, {len(orders)} passes; evaluation split={split}', flush=True)
        return
    build_root = Path(os.environ.get('FASHION_BUILD_ROOT', str(a.out_dir / 'compiled'))).resolve() / a.out_dir.resolve().name
    summary, predictions, reference_counts = {'gold_comparison_available': gold is not None}, {}, {}
    try:
        if not a.skip_evaluation:
            references = [('baseline', weights, labels, config)]
            if gold:
                references.append(('gold', gw, gl, gc))
            for name, mw, ml, mc in references:
                save_json(a.out_dir / 'status.json', dict(status='evaluating', model=name))
                metrics, pred, counts = evaluate(mw, ml, eval_x[ids], eval_y[ids], ids, a.inference_seed,
                                                a.out_dir / name, build_root / name,
                                                float(mc.get('input_total_hz', 0)), classes)
                summary[name], predictions[name], reference_counts[name] = metrics, pred, counts
        for variant in a.variants:
            for scale in a.scales:
                name = f'{variant}_s{scale:g}' + ('_adaptive' if a.adapt_thresholds else '')
                target = a.out_dir / name
                target.mkdir()
                save_json(a.out_dir / 'status.json', dict(status='replaying', trial=name))
                snapshots, diagnostics, timing = compiled_replay(
                    weights, arrays[0][forget_ids], orders, a.replay_seed, variant, scale,
                    build_root / name, sorted(a.passes), float(config.get('input_total_hz', 0)), a.adapt_thresholds)
                # Persist diagnostics out of scratch before job completion.
                np.savez_compressed(target / 'presentations.npz', diagnostics=diagnostics,
                                    dataset_ids=forget_ids[orders].ravel())
                trial = dict(timing=timing, checkpoints={})
                for passes, edited in snapshots.items():
                    checkpoint = target / f'pass_{passes}'
                    checkpoint.mkdir()
                    np.savez_compressed(checkpoint / 'baseline_weights.npz', **edited)
                    np.save(checkpoint / 'neuron_labels.npy', labels)
                    save_json(checkpoint / 'config.json', dict(config, request=a.request, method=variant,
                              scale=scale, passes=passes, threshold_adaptation=a.adapt_thresholds,
                              weight_normalization=False, original_request='baseline'))
                    dw = edited['w'] - weights['w']
                    selected = diagnostics[diagnostics['pass'] <= passes]
                    checks = dict(weights_increased=int(np.sum(dw > 0)), weights_decreased=int(np.sum(dw < 0)),
                                  weights_at_zero=int(np.sum(edited['w'] == 0)),
                                  weight_l1_change=float(np.abs(dw).sum()),
                                  positive_updates=int(selected['positive_updates'].sum()),
                                  negative_updates=int(selected['negative_updates'].sum()),
                                  clipped_updates=int(selected['clipped_updates'].sum()),
                                  silent_presentations=int(np.sum(selected['active_spikes'] == 0)),
                                  final_pass_no_update_fraction=float(np.mean(
                                      (diagnostics[diagnostics['pass'] == passes]['negative_updates'] +
                                       diagnostics[diagnostics['pass'] == passes]['positive_updates']) == 0)))
                    save_json(checkpoint / 'intervention.json', checks)
                    result = dict(intervention=checks)
                    if not a.skip_evaluation:
                        save_json(a.out_dir / 'status.json', dict(status='evaluating', trial=name, passes=passes))
                        metrics, pred, counts = evaluate(edited, labels, eval_x[ids], eval_y[ids], ids, a.inference_seed,
                                                        checkpoint / 'evaluation', build_root / f'{name}_p{passes}_eval',
                                                        float(config.get('input_total_hz', 0)), classes)
                        result['metrics'] = metrics
                        result['comparison'] = {}
                        for ref in predictions:
                            result['comparison'][ref] = {
                                part: dict(prediction_disagreement=float(np.mean(pred[mask] != predictions[ref][mask])),
                                           mean_absolute_total_spike_difference=float(np.abs(
                                               counts[mask].sum(axis=1).astype(float) -
                                               reference_counts[ref][mask].sum(axis=1)).mean()))
                                for part, mask in [('retain', ~np.isin(eval_y[ids], classes)),
                                                   ('forget', np.isin(eval_y[ids], classes))]}
                        if fashion:
                            save_json(checkpoint / 'evaluation/fashion_comparison.json', fashion_comparison(
                                eval_y[ids], dict(predictions, method_c=pred),
                                dict({key: summary[key] for key in predictions}, method_c=metrics), classes))
                    trial['checkpoints'][str(passes)] = result
                summary[name] = trial
                save_json(a.out_dir / 'summary.json', summary)
        save_json(a.out_dir / 'status.json', dict(status='complete', evaluated=not a.skip_evaluation))
    except Exception as error:
        save_json(a.out_dir / 'status.json', dict(status='failed', error=str(error)))
        raise
    finally:
        if any(sha256(path) != expected for path, expected in sources.items()):
            save_json(a.out_dir / 'status.json', dict(status='failed', error='Source artifact changed'))
            raise RuntimeError('Source artifact changed during execution')


if __name__ == '__main__':
    main()
