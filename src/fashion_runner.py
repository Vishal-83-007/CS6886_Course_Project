"""MNIST / Fashion-MNIST runner with offline data, explicit epochs and runtime checkpoints."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pickle
import shutil
import signal
import subprocess
import time

import numpy as np

from .fashion_data import (balanced_ids, epoch_order, fingerprint, forget_classes,
                           load_dataset, make_manifest, save_json, valid_request)


def request_name(value):
    if not valid_request(value):
        raise argparse.ArgumentTypeError(
            f"'{value}' is not baseline, random10, classK (K = 0..9) or several classes "
            "in increasing order such as class6_9")
    return value


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('fashion', 'mnist'), default='fashion',
                   help='Which official IDX dataset to train on')
    p.add_argument('--data-dir', default=None, help='Defaults to data/<dataset>')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--request', type=request_name, default='baseline',
                   help='baseline | classK | classK_L_... (several classes) | random10')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--n-excitatory', type=int, default=400)
    p.add_argument('--validation-size', type=int, default=5000,
                   help='0 locks a final full-training/full-test experiment')
    p.add_argument('--assign-per-class', type=int, default=500)
    p.add_argument('--train-limit', type=int, default=0,
                   help='Common balanced pilot pool before deletion; 0 means all training IDs')
    p.add_argument('--eval-limit', type=int, default=0,
                   help='Balanced validation/test subset for timing pilots only')
    p.add_argument('--backend', choices=('cython', 'numpy', 'cpp', 'cuda'), default='cython')
    p.add_argument('--resume', action='store_true', help='Exact same-environment CPU runtime resume')
    p.add_argument('--checkpoint-seconds', type=float, default=900)
    p.add_argument('--max-seconds', type=float, default=0,
                   help='Gracefully checkpoint after this runtime budget; exit 75 if incomplete')
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='Also checkpoint every N presentations (useful for smoke tests)')
    p.add_argument('--threads', type=int, default=1, help='Compiled CPU threads / compiler jobs')
    p.add_argument('--weight-norm', type=float, default=0,
                   help='Diehl & Cook released-code weight normalisation: before every '
                        'training presentation, scale each excitatory neuron\'s input '
                        'weights to sum to this value (78 in Diehl & Cook), then clip to '
                        'w_max. 0 = off. Compiled CPU backend only')
    p.add_argument('--input-total-hz', type=float, default=0,
                   help='Brightness normalisation: scale every image (training, labelling '
                        'and test) so its 784 input rates sum to this many Hz before the '
                        'retry boost. 0 = off (rate = pixel / 4). Compiled CPU backend only')
    p.add_argument('--gpu-id', type=int, default=None,
                   help='Allocated GPU index within CUDA_VISIBLE_DEVICES; required for CUDA')
    p.add_argument('--compile-only', action='store_true', help='Build standalone without executing')
    p.add_argument('--allow-long-cuda', action='store_true',
                   help='Permit CUDA runs beyond the 1,000-image/1-epoch pilot limit. Use only '
                        'after a CUDA pilot on the target GPU has completed successfully.')
    a = p.parse_args()
    if a.data_dir is None:
        a.data_dir = f'data/{a.dataset}'
    for name in ('epochs', 'n_excitatory', 'assign_per_class', 'threads'):
        if getattr(a, name) < 1:
            p.error(f'{name} must be positive')
    if any(getattr(a, n) < 0 for n in ('validation_size', 'train_limit', 'eval_limit',
                                      'checkpoint_seconds', 'checkpoint_every', 'max_seconds')):
        p.error('Counts and time budgets must be nonnegative')
    if a.backend in ('cpp', 'cuda') and (a.resume or a.max_seconds):
        p.error('Standalone resume/time-budget interruption is not yet supported; use cython')
    if (a.weight_norm or a.input_total_hz) and a.backend != 'cpp':
        p.error('--weight-norm and --input-total-hz are implemented for --backend cpp only')
    if a.weight_norm < 0 or a.input_total_hz < 0:
        p.error('--weight-norm and --input-total-hz must be nonnegative')
    if a.backend == 'cuda' and a.gpu_id is None:
        p.error('CUDA requires --gpu-id identifying an allocated device; visibility is not entitlement')
    return a


def setup_run(a):
    arrays = load_dataset(a.data_dir, dataset=a.dataset)
    manifest = make_manifest(arrays[1], a.request, a.split_seed, a.validation_size,
                             a.assign_per_class, a.train_limit)
    eval_ids = manifest['validation'] if a.validation_size else np.arange(len(arrays[3]))
    eval_y = arrays[1] if a.validation_size else arrays[3]
    if a.eval_limit:
        if a.eval_limit % 10:
            raise ValueError('eval-limit must be a multiple of ten')
        eval_ids = balanced_ids(eval_y, eval_ids, a.eval_limit // 10, a.split_seed + 4)
    manifest['evaluation'] = eval_ids
    stages, ids = [], []
    for epoch in range(a.epochs):
        order = epoch_order(manifest, a.seed, epoch)
        stages.append(np.zeros(len(order), dtype=np.uint8))
        ids.append(order)
    for stage, values in ((1, manifest['assignment']), (2, eval_ids)):
        stages.append(np.full(len(values), stage, dtype=np.uint8))
        ids.append(values)
    schedule = dict(stage=np.concatenate(stages), ids=np.concatenate(ids))
    config = {k: v for k, v in vars(a).items() if k not in
              ('resume', 'max_seconds', 'checkpoint_seconds', 'checkpoint_every', 'compile_only',
               'out_dir', 'data_dir', 'allow_long_cuda')}
    config['dataset_sha256'] = fingerprint(arrays)
    config['schedule_sha256'] = fingerprint(tuple(schedule.values()))
    config['schema'] = 1
    code_files = ('network.py', 'encoding.py', 'fashion_data.py', 'fashion_runner.py',
                  'fashion_standalone.py', 'utils.py')
    config['source_sha256'] = hashlib.sha256(b''.join(
        (Path(__file__).parent / name).read_bytes() for name in code_files)).hexdigest()
    config['evaluation_split'] = 'validation' if a.validation_size else 'test'
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    root = Path(a.out_dir).resolve()
    if root.exists() and any(root.iterdir()) and not a.resume:
        raise FileExistsError(f'{root} is nonempty: use --resume or a new output directory')
    root.mkdir(parents=True, exist_ok=True)
    if a.resume:
        previous = json.loads((root / 'config.json').read_text())
        if previous != config:
            raise ValueError('Resume configuration/dataset/schedule differs from saved run')
        environment = json.loads((root / 'environment.json').read_text())
        if environment['python'] != platform.python_version():
            raise ValueError('Exact resume requires the original Python version')
        for package in ('brian2', 'numpy', 'Cython'):
            if environment['packages'][package] != importlib.metadata.version(package):
                raise ValueError(f'Exact resume requires original {package} version')
    else:
        save_json(root / 'config.json', config)
        np.savez(root / 'manifest.npz', **manifest)
        np.savez(root / 'schedule.npz', **schedule)
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            revision, dirty = 'unknown', 'unknown'
        packages = {}
        for package in ('brian2', 'brian2cuda', 'numpy', 'Cython'):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        save_json(root / 'environment.json', dict(python=platform.python_version(),
                  platform=platform.platform(), revision=revision, dirty=dirty, packages=packages))
    return arrays, manifest, schedule, root, signature


def phase_image(arrays, schedule, position, validation):
    stage, idx = int(schedule['stage'][position]), int(schedule['ids'][position])
    return arrays[2 if stage == 2 and not validation else 0][idx].astype(np.float64)


def runtime_checkpoint(root, components, progress, counts):
    """Publish a complete generation only after all state and metadata are durable."""
    directory = root / 'checkpoints'
    directory.mkdir(exist_ok=True)
    generation = directory / f'checkpoint_{time.time_ns()}'
    generation.mkdir()
    components['net'].store('resume', filename=str(generation / 'network.pkl'))
    save_runtime_rng(generation / 'rng.pkl')
    np.save(generation / 'counts.npy', counts)
    progress = dict(progress, learning_on=components['_learning_on'])
    save_json(generation / 'progress.json', progress)
    # An interrupted write leaves CURRENT pointing at the previous complete generation.
    save_json(directory / 'CURRENT.json', dict(generation=generation.name))
    old = sorted(directory.glob('checkpoint_*'))[:-2]
    for path in old:
        if path.is_dir():
            shutil.rmtree(path)


def save_runtime_rng(path):
    """Brian 2.9 stores Cython RNG buffer addresses, which cannot cross processes.

    Save their CONTENTS as well as NumPy's generator/index state. Only our own
    trusted checkpoints are supported; pickle is not an interchange format.
    """
    from brian2 import get_device
    from brian2.codegen.generators.cython_generator import _BUFFER_SIZE
    state = get_device().get_random_state()
    for name in ('rand_buffer', 'randn_buffer'):
        pointer = int(state[name][0])
        state[name] = (np.ctypeslib.as_array((ctypes.c_double * _BUFFER_SIZE).from_address(pointer)).copy()
                       if pointer else np.empty(0, dtype=np.float64))
    with open(path, 'wb') as stream:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)


def restore_runtime_rng(path):
    from brian2 import get_device
    from brian2.codegen.generators.cython_generator import _BUFFER_SIZE
    with open(path, 'rb') as stream:
        state = pickle.load(stream)
    device = get_device()
    libc = ctypes.CDLL(None)
    libc.malloc.argtypes = [ctypes.c_size_t]
    libc.malloc.restype = ctypes.c_void_p
    libc.free.argtypes = [ctypes.c_void_p]
    for name in ('rand_buffer', 'randn_buffer'):
        values = state[name]
        if len(values) not in (0, _BUFFER_SIZE):
            raise ValueError('Incompatible Brian RNG buffer size')
        pointer_array = getattr(device, name)
        if int(pointer_array[0]):
            libc.free(ctypes.c_void_p(int(pointer_array[0])))
        pointer_array[0] = 0
        if len(values):
            # Brian's Cython generator owns/frees this malloc buffer thereafter.
            address = libc.malloc(values.nbytes)
            if not address:
                raise MemoryError('Cannot restore Brian RNG buffer')
            ctypes.memmove(address, values.ctypes.data, values.nbytes)
            pointer_array[0] = address
        getattr(device, name + '_index')[:] = state[name + '_index']
    np.random.set_state(state['numpy_state'])


def run_runtime(a, arrays, manifest, schedule, root, signature):
    from brian2 import prefs
    from .encoding import present_example
    from .network import build_network
    from .utils import save_weights
    prefs.codegen.target = a.backend
    if a.backend == 'cython':
        cache_root = Path(os.environ.get('FASHION_BUILD_ROOT', str(root)))
        prefs.codegen.runtime.cython.cache_dir = str(cache_root / root.parent.name / root.name / 'cython_cache')
    components = build_network(a.n_excitatory, a.seed)
    n_training = int(np.count_nonzero(schedule['stage'] == 0))
    counts = np.zeros((len(schedule['ids']) - n_training, a.n_excitatory), dtype=np.int32)
    progress = dict(position=0, signature=signature, phase_seconds=[0., 0., 0.],
                    attempts=[0] * 6, exhausted=0)
    if a.resume:
        directory = root / 'checkpoints'
        generation = directory / json.loads((directory / 'CURRENT.json').read_text())['generation']
        progress = json.loads((generation / 'progress.json').read_text())
        if progress['signature'] != signature:
            raise ValueError('Checkpoint signature differs')
        # Ignore process-local pointers in Brian's saved random state.
        components['net'].restore('resume', filename=str(generation / 'network.pkl'),
                                  restore_random_state=False)
        restore_runtime_rng(generation / 'rng.pkl')
        components['_learning_on'] = progress['learning_on']
        counts = np.load(generation / 'counts.npy', allow_pickle=False)
    interrupted = [False]
    def request_stop(signum, frame):
        interrupted[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = last_checkpoint = time.monotonic()
    try:
        for position in range(progress['position'], len(schedule['ids'])):
            stage = int(schedule['stage'][position])
            step_start = time.monotonic()
            result = present_example(components, phase_image(arrays, schedule, position,
                                     a.validation_size), learning_on=(stage == 0))
            progress['phase_seconds'][stage] += time.monotonic() - step_start
            if position == 0:
                progress['first_presentation_seconds'] = time.monotonic() - step_start
            progress['attempts'][components['_last_attempts'] - 1] += 1
            progress['exhausted'] += int(components['_last_retry_exhausted'])
            if stage != 0:
                counts[position - n_training] = result
            progress['position'] = position + 1
            if stage == 0 and (position + 1) % len(manifest['train']) == 0:
                epoch = (position + 1) // len(manifest['train'])
                save_weights(components, root / f'epoch_{epoch}_weights.npz')
            now = time.monotonic()
            stop = interrupted[0] or (a.max_seconds and now - started >= a.max_seconds)
            checkpoint_due = ((a.checkpoint_seconds and now - last_checkpoint >= a.checkpoint_seconds)
                              or (a.checkpoint_every and (position + 1) % a.checkpoint_every == 0))
            if stop or checkpoint_due:
                runtime_checkpoint(root, components, progress, counts)
                last_checkpoint = time.monotonic()
            if (position + 1) % 100 == 0 or stop:
                print(f"progress={position + 1}/{len(schedule['ids'])} "
                      f"phase_seconds={progress['phase_seconds']} attempts={progress['attempts']}", flush=True)
            if stop and position + 1 < len(schedule['ids']):
                save_json(root / 'status.json', dict(status='checkpointed', **progress))
                return 75
        save_weights(components, root / 'baseline_weights.npz')
        runtime_checkpoint(root, components, progress, counts)
        summarize(a, arrays, manifest, counts, root, progress)
        return 0
    except Exception:
        # Never publish an inconsistent position if simulation fails mid-example.
        # The last complete generation remains resumable.
        save_json(root / 'status.json', dict(status='failed',
                  note='Resume from the last published complete checkpoint, if present'))
        raise


def summarize(a, arrays, manifest, counts, root, progress):
    assignment = counts[:len(manifest['assignment'])]
    observed = arrays[1][manifest['assignment']]
    rate_sums = np.zeros((10, a.n_excitatory))
    class_counts = np.bincount(observed, minlength=10)
    np.add.at(rate_sums, observed, assignment)
    rates = rate_sums / np.maximum(class_counts, 1)[:, None]
    labels = np.argmax(rates, axis=0)
    labels[np.max(rates, axis=0) == 0] = -1
    test_counts = counts[len(assignment):]
    scores = np.full((len(test_counts), 10), -np.inf)
    for label in range(10):
        selected = labels == label
        if np.any(selected):
            scores[:, label] = test_counts[:, selected].mean(axis=1)
    predictions = scores.argmax(axis=1)
    if not np.any(labels >= 0):
        predictions[:] = -1
    truth = (arrays[1] if a.validation_size else arrays[3])[manifest['evaluation']]
    per_class = {str(label): float(np.mean(predictions[truth == label] == label))
                 for label in np.unique(truth)}
    forget_mask = np.isin(truth, forget_classes(a.request))  # all False unless classes removed
    confusion = np.zeros((10, 11), dtype=np.int64)
    np.add.at(confusion, (truth, np.where(predictions < 0, 10, predictions)), 1)
    np.save(root / 'neuron_labels.npy', labels)
    np.save(root / 'per_class_spike_rates.npy', rates)
    np.save(root / 'confusion_matrix.npy', confusion)
    np.savez(root / 'responses.npz', assignment_ids=manifest['assignment'],
             assignment_counts=assignment, evaluation_ids=manifest['evaluation'],
             evaluation_counts=test_counts, scores=scores, predictions=predictions, truth=truth)
    result = dict(accuracy=float(np.mean(predictions == truth)), per_class_accuracy=per_class,
                  evaluation_split='validation' if a.validation_size else 'test',
                  n_train_unique=len(manifest['train']), n_presentations=a.epochs * len(manifest['train']),
                  n_assignment=len(assignment), n_evaluation=len(truth),
                  silent_fraction=float(np.mean(test_counts.sum(axis=1) == 0)),
                  tie_fraction=float(np.mean(np.sum(scores == scores.max(axis=1)[:, None], axis=1) > 1)),
                  unassigned_neurons=int(np.sum(labels < 0)), request=a.request,
                  retain_accuracy=float(np.mean(predictions[~forget_mask] == truth[~forget_mask])),
                  forget_class_accuracy=(float(np.mean(predictions[forget_mask] == truth[forget_mask]))
                                        if np.any(forget_mask) else None),
                  **progress)
    save_json(root / 'metrics.json', result)
    save_json(root / 'status.json', dict(status='complete'))
    print(json.dumps(result, indent=2), flush=True)


def main():
    a = arguments()
    arrays, manifest, schedule, root, signature = setup_run(a)
    if a.backend in ('cpp', 'cuda'):
        from .fashion_standalone import run_standalone
        return run_standalone(a, arrays, manifest, schedule, root, signature)
    return run_runtime(a, arrays, manifest, schedule, root, signature)
