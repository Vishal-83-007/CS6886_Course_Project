"""Method D: request-gated leak during replay of the retained data (compiled CPU).

Starting from the trained baseline, R retained training images (D_r, a
seeded random subset) are shown with STDP ON (training rule and learning
rate), and after every image the gated weights leak:

    dw_ij = dw_ij^STDP(x_r) - lambda_u * g_ij * w_ij,     x_r ~ D_r

so after R images an untouched gated weight keeps (1 - lambda_u)^R of its
value, while STDP can re-potentiate whatever D_r still uses.

Gate g_ij (binary):
    neuron   g_ij = 1 for every input synapse of a neuron whose baseline
             label is a removed class (neuron_labels.npy of the baseline)
    synapse  g_ij = 1 where I_ij(D_f) > alpha * I_ij(D), from the
             hypothetical-STDP importance passes (same selection as SSD)

Thresholds theta: frozen by default (no growth, no decay), so that lower
firing cannot come from changed thresholds; ``--theta updated`` lets them
evolve as in training.

Weight normalisation (Fashion baseline was trained with WEIGHT_NORM=39):
with ``--weight-norm`` > 0 it is applied before every presentation, as in
training, but ONLY to neurons without any gated synapse; normalising a
gated neuron would rescale its leaked weights back up.

Writes replayed_weights.npz in the baseline_weights.npz format, ready for
``run_frozen.py evaluate``.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .fashion_data import fingerprint, forget_classes, load_dataset, make_manifest, save_json


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=('fashion', 'mnist'), required=True)
    p.add_argument('--data-dir', default=None, help='Defaults to data/<dataset>')
    p.add_argument('--weights', required=True, help='baseline_weights.npz of the trained baseline')
    p.add_argument('--labels', required=True, help='neuron_labels.npy of the trained baseline')
    p.add_argument('--forget', required=True, help='removed classes, e.g. class6_9')
    p.add_argument('--gate', choices=('neuron', 'synapse'), required=True)
    p.add_argument('--importance', nargs='*', default=[],
                   help='synapse gate: importance.npz of one or more passes (averaged)')
    p.add_argument('--alpha', type=float, default=2.0, help='synapse gate: I(Df) > alpha * I(D)')
    p.add_argument('--leak', type=float, required=True,
                   help='lambda_u: fraction of each gated weight removed after every replay image')
    p.add_argument('--replay-images', type=int, required=True, help='R: number of D_r images')
    p.add_argument('--theta', choices=('frozen', 'updated'), default='frozen')
    p.add_argument('--weight-norm', type=float, default=0,
                   help='0 = off; else as in training, for ungated neurons only')
    p.add_argument('--input-total-hz', type=float, default=0, help='same as training')
    p.add_argument('--poisson-seed', type=int, default=3, help='input spikes and replay order')
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args(argv)
    if not forget_classes(a.forget):
        p.error('--forget must look like class6 or class6_9')
    if not 0 <= a.leak < 1:
        p.error('--leak must be in [0, 1)')
    if a.replay_images < 1:
        p.error('--replay-images must be at least 1')
    if a.gate == 'synapse' and not a.importance:
        p.error('--gate synapse needs --importance files')
    a.data_dir = a.data_dir or f'data/{a.dataset}'
    return a


def make_gate(a, n_e):
    removed = list(forget_classes(a.forget))
    if a.gate == 'neuron':
        labels = np.load(a.labels)
        if labels.shape != (n_e,):
            raise ValueError('--labels has the wrong number of neurons')
        gated_neurons = np.isin(labels, removed)
        return np.tile(gated_neurons, 784)          # synapse k -> neuron k % n_e
    sumsq, count = 0, 0
    for path in a.importance:
        with np.load(path) as data:
            if int(data['n_e']) != n_e:
                raise ValueError(f'{path}: importance from another network size')
            sumsq = sumsq + data['sumsq']
            count = count + data['class_count']
    i_d = sumsq.sum(axis=0) / count.sum()
    i_f = sumsq[removed].sum(axis=0) / count[removed].sum()
    return i_f > a.alpha * i_d


def run(a):
    from brian2 import defaultclock, get_device, prefs, set_device, volt
    from .encoding import PRESENT_TIME, REST_TIME
    from .network import build_network, ETA

    train_x, train_y, test_x, test_y = load_dataset(a.data_dir, dataset=a.dataset)
    with np.load(a.weights) as data:
        weights = {k: data[k] for k in data.files}
    n_e = int(weights['n_e'])
    ns = 784 * n_e
    if (int(weights['n_input']) != 784
            or not np.array_equal(weights['i'], np.repeat(np.arange(784), n_e))
            or not np.array_equal(weights['j'], np.tile(np.arange(n_e), 784))):
        raise ValueError(f'{a.weights}: unexpected synapse order or size')
    gate = make_gate(a, n_e)
    gated_neuron = gate.reshape(784, n_e).any(axis=0)
    normalise = (~gated_neuron).astype(np.uint8)

    retain = make_manifest(train_y, a.forget, a.split_seed, 0, 1)['train']
    if a.replay_images > len(retain):
        raise ValueError(f'--replay-images exceeds the {len(retain)} retained training images')
    rng = np.random.default_rng(np.random.SeedSequence([a.poisson_seed, 577]))
    ids = rng.permutation(retain)[:a.replay_images]
    if np.any(np.isin(train_y[ids], list(forget_classes(a.forget)))):
        raise RuntimeError('Replay set contains a removed class')

    root = Path(a.out_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f'{root} is nonempty: choose a new --out-dir')
    root.mkdir(parents=True, exist_ok=True)
    code = b''.join((Path(__file__).parent / name).read_bytes()
                    for name in ('replay.py', 'network.py', 'encoding.py', 'fashion_data.py'))
    config = {k: v for k, v in vars(a).items() if k not in ('out_dir', 'data_dir', 'threads')}
    config.update(weights_sha256=hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),
                  dataset_sha256=fingerprint((train_x, train_y, test_x, test_y)),
                  replay_ids_sha256=fingerprint((ids,)), n_excitatory=n_e, backend='cpp',
                  source_sha256=hashlib.sha256(code).hexdigest(), mode='replay', schema=1)
    save_json(root / 'config.json', config)
    np.savez(root / 'schedule.npz', ids=ids, gate=gate, normalised_neurons=normalise)
    save_json(root / 'status.json', dict(status='building', backend='cpp'))

    set_device('cpp_standalone', build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 0
    prefs.devices.cpp_standalone.extra_make_args_unix = [f'-j{a.threads}']
    prefs.codegen.cpp.headers += ['<fstream>', '<vector>', '<chrono>', '<stdexcept>',
                                  '<algorithm>', '<cmath>']
    components = build_network(n_e, a.poisson_seed)
    syn, exc = components['syn_input_exc'], components['exc']
    syn.w = weights['w']
    exc.theta = weights['theta_volts'] * volt
    syn.eta = ETA                                         # STDP on, training rate
    exc.plasticity_enabled = int(a.theta == 'updated')    # theta frozen by default
    device = get_device()

    def queue_run(duration):
        # Same clock handling as src/fashion_standalone.py.
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
    ids.astype('<i8').tofile(binary / 'ids.bin')
    gate.astype(np.uint8).tofile(binary / 'gate.bin')
    normalise.tofile(binary / 'normalise.bin')
    quoted = lambda path: json.dumps(str(path))
    rates = host(components['input'], 'rates')
    mon = host(components['exc_monitor'], 'count')
    w = host(syn, 'w')
    w_max = host(syn, 'w_max')
    post = host(syn, '_synaptic_post')
    theta = host(exc, 'theta')
    n = len(ids)
    keep = 1.0 - a.leak

    device.insert_code('main', f'''
    auto load_bytes = [](const char* path, size_t count) {{
        std::vector<unsigned char> values(count);
        std::ifstream file(path, std::ios::binary);
        if(!file.read(reinterpret_cast<char*>(values.data()), count))
            throw std::runtime_error("Cannot read input bytes");
        return values;
    }};
    auto replay_train = load_bytes({quoted(binary / 'train.bin')}, {train_x.size});
    auto replay_gate = load_bytes({quoted(binary / 'gate.bin')}, {ns});
    auto replay_normalise = load_bytes({quoted(binary / 'normalise.bin')}, {n_e});
    std::vector<int64_t> replay_ids({n});
    std::ifstream replay_id_file({quoted(binary / 'ids.bin')}, std::ios::binary);
    if(!replay_id_file.read(reinterpret_cast<char*>(replay_ids.data()), {n} * sizeof(int64_t)))
        throw std::runtime_error("Cannot read replay ids");
    std::vector<double> replay_colsum({n_e});
    auto replay_normalize = [&]() {{   // training normalisation, ungated neurons only
        std::fill(replay_colsum.begin(), replay_colsum.end(), 0.0);
        for(size_t k = 0; k < {ns}; ++k) replay_colsum[{post}[k]] += {w}[k];
        for(size_t k = 0; k < {ns}; ++k) {{
            const int j = {post}[k];
            if(replay_normalise[j] && replay_colsum[j] > 0)
                {w}[k] = std::min({w}[k] * ({a.weight_norm!r} / replay_colsum[j]), {w_max}[k]);
        }}
    }};
    long long replay_attempts[6] = {{0, 0, 0, 0, 0, 0}};
    long long replay_exhausted = 0, replay_spikes = 0;
    double replay_seconds = 0;
    std::vector<int> replay_before({n_e});
    for(size_t replay_position = 0; replay_position < {n}; ++replay_position) {{
        auto replay_start = std::chrono::steady_clock::now();
        const unsigned char* replay_image = replay_train.data() + replay_ids[replay_position] * 784;
        double replay_scale = 1.0;
        if({a.input_total_hz!r} > 0) {{
            double replay_pixel_sum = 0;
            for(int k = 0; k < 784; ++k) replay_pixel_sum += replay_image[k] / 4.0;
            if(replay_pixel_sum > 0) replay_scale = {a.input_total_hz!r} / replay_pixel_sum;
        }}
        int replay_total = 0, replay_attempt = 0;
        for(; replay_attempt < 6; ++replay_attempt) {{
            if({int(a.weight_norm > 0)}) replay_normalize();
            for(int k = 0; k < 784; ++k)
                {rates}[k] = replay_image[k] / 4.0 * replay_scale
                              * (1.0 + replay_attempt * 32.0 / 63.75);
            std::copy({mon}, {mon} + {n_e}, replay_before.begin());
    ''')
    queue_run(PRESENT_TIME)
    device.insert_code('main', f'''
            replay_total = 0;
            for(int k = 0; k < {n_e}; ++k) replay_total += {mon}[k] - replay_before[k];
            std::fill({rates}, {rates} + 784, 0.0);
    ''')
    queue_run(REST_TIME)
    device.insert_code('main', f'''
            if(replay_total >= 5) break;
        }}
        ++replay_attempts[std::min(replay_attempt, 5)];
        if(replay_total < 5) ++replay_exhausted;
        replay_spikes += replay_total;
        for(size_t k = 0; k < {ns}; ++k)            // request-gated leak
            if(replay_gate[k]) {w}[k] *= {keep!r};
        replay_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - replay_start).count();
        if((replay_position + 1) % 100 == 0)
            std::cout << "progress=" << replay_position + 1 << "/{n}" << std::endl;
    }}
    std::ofstream replay_w_file({quoted(root / 'replayed_w.bin')}, std::ios::binary);
    std::ofstream replay_t_file({quoted(root / 'replayed_theta.bin')}, std::ios::binary);
    replay_w_file.write(reinterpret_cast<const char*>({w}), {ns} * sizeof(double));
    replay_t_file.write(reinterpret_cast<const char*>({theta}), {n_e} * sizeof(double));
    if(!replay_w_file || !replay_t_file) throw std::runtime_error("Cannot write replayed weights");
    std::ofstream replay_timing({quoted(root / 'standalone_timing.json')});
    replay_timing << "{{\\\"seconds\\\":" << replay_seconds << ",\\\"spikes_per_image\\\":"
        << (double)replay_spikes / {n} << ",\\\"attempts\\\":[";
    for(int k = 0; k < 6; ++k) replay_timing << (k ? "," : "") << replay_attempts[k];
    replay_timing << "],\\\"exhausted\\\":" << replay_exhausted << "}}";
    if(!replay_timing) throw std::runtime_error("Cannot write timing file");
    ''')
    start = time.monotonic()
    device.build(directory=str(work_root / 'build'), compile=True, run=True)
    timing = json.loads((root / 'standalone_timing.json').read_text())
    timing['build_and_run_seconds'] = time.monotonic() - start
    new_w = np.fromfile(root / 'replayed_w.bin')
    new_theta = np.fromfile(root / 'replayed_theta.bin')
    if not (np.isfinite(new_w).all() and np.isfinite(new_theta).all()):
        raise FloatingPointError('Nonfinite replayed weights/thresholds')
    np.savez(root / 'replayed_weights.npz', **{**weights, 'w': new_w, 'theta_volts': new_theta})
    for name in ('replayed_w.bin', 'replayed_theta.bin'):
        (root / name).unlink()
    old_w = weights['w']
    col = lambda v: v.reshape(784, n_e).sum(axis=0)
    result = dict(mode='replay', gate=a.gate, leak=a.leak, replay_images=n, theta=a.theta,
                  gated_synapses=int(gate.sum()), gated_fraction=float(gate.mean()),
                  gated_neurons=int(gated_neuron.sum()),
                  leak_factor_if_untouched=keep ** n,
                  gated_weight_sum_before=float(old_w[gate].sum()),
                  gated_weight_sum_after=float(new_w[gate].sum()),
                  ungated_weight_sum_before=float(old_w[~gate].sum()),
                  ungated_weight_sum_after=float(new_w[~gate].sum()),
                  weight_sum_per_gated_neuron_after=(float(col(new_w)[gated_neuron].mean())
                                                     if gated_neuron.any() else None),
                  weight_sum_per_other_neuron_after=float(col(new_w)[~gated_neuron].mean()),
                  theta_mV_mean_before=float(weights['theta_volts'].mean() * 1e3),
                  theta_mV_mean_after=float(new_theta.mean() * 1e3), **timing)
    save_json(root / 'metrics.json', result)
    save_json(root / 'status.json', dict(status='complete'))
    print(json.dumps(result, indent=2), flush=True)
    return 0


def main(argv=None):
    return run(arguments(argv))
