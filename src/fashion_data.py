"""Offline MNIST / Fashion-MNIST data and reproducible deletion manifests.

Both datasets use the same official gzip IDX layout (60k train, 10k test,
28x28 uint8, labels 0-9), so everything downstream is dataset-neutral.
"""
import gzip
import hashlib
import json
import re
import struct
import urllib.request
from pathlib import Path

import numpy as np

# Official MD5 checksums. Every load verifies them, offline included.
DATASETS = {
    'fashion': dict(
        name='fashion-mnist',
        files={
            'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
            'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
            't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
            't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
        },
        urls=('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',)),
    'mnist': dict(
        name='mnist',
        files={
            'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
            'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
            't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
            't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
        },
        # Mirrors tried in order; yann.lecun.com often refuses downloads.
        urls=('https://ossci-datasets.s3.amazonaws.com/mnist/',
              'https://storage.googleapis.com/cvdf-datasets/mnist/',
              'https://raw.githubusercontent.com/fgnt/mnist/master/')),
}
# Backwards-compatible names for existing Fashion-MNIST code.
FILES = DATASETS['fashion']['files']
BASE_URL = DATASETS['fashion']['urls'][0]
# baseline = all training data; classK = Gold model retrained without every
# training image of class K (K = 0..9); classK_L_... = without every training
# image of several classes, listed in increasing order (e.g. class6_9);
# random10 = without a fixed stratified random 10% of the training images.
# REQUESTS lists the single-request names (used for prepared manifests);
# valid_request() also accepts the multi-class names.
REQUESTS = ('baseline',) + tuple(f'class{k}' for k in range(10)) + ('random10',)


def forget_classes(request):
    """Classes a request removes: 'class6' -> (6,), 'class6_9' -> (6, 9), else ()."""
    match = re.fullmatch(r'class(\d(?:_\d)*)', request)
    if not match:
        return ()
    classes = tuple(int(c) for c in match.group(1).split('_'))
    if list(classes) != sorted(set(classes)) or len(classes) > 9:
        raise ValueError(f'{request}: list each class once, in increasing order, '
                         'and keep at least one class (e.g. class6_9)')
    return classes


def valid_request(request):
    if request in ('baseline', 'random10'):
        return True
    try:
        return bool(forget_classes(request))
    except ValueError:
        return False
CLASS_NAMES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


def fingerprint(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _download(urls, name, expected, path):
    temporary = path.with_suffix('.download')
    errors = []
    for base in urls:
        try:
            with urllib.request.urlopen(base + name, timeout=120) as response:
                temporary.write_bytes(response.read())
        except OSError as error:
            errors.append(f'{base}: {error}')
            continue
        if hashlib.md5(temporary.read_bytes()).hexdigest() == expected:
            temporary.replace(path)
            return
        errors.append(f'{base}: checksum mismatch')
    temporary.unlink(missing_ok=True)
    raise OSError(f'Could not download {name}:\n  ' + '\n  '.join(errors))


def load_dataset(directory, download=False, dataset='fashion'):
    """Load checked official gzip IDX files; download only when explicitly asked."""
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset {dataset!r}; choose from {sorted(DATASETS)}')
    spec = DATASETS[dataset]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = []
    for name, expected in spec['files'].items():
        path = directory / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path}: run prepare_fashion.py --dataset {dataset} '
                                        f'--download first, or copy the four .gz files there')
            _download(spec['urls'], name, expected, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Dataset checksum mismatch: {path} is not official {spec["name"]} '
                             f'(wrong dataset in this folder?)')
        payload = gzip.decompress(path.read_bytes())
        if 'images' in name:
            magic, count, rows, cols = struct.unpack('>IIII', payload[:16])
            if magic != 2051 or (rows, cols) != (28, 28):
                raise ValueError(f'Invalid image IDX header: {name}')
            array = np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(count, 784).copy()
        else:
            magic, count = struct.unpack('>II', payload[:8])
            if magic != 2049:
                raise ValueError(f'Invalid label IDX header: {name}')
            array = np.frombuffer(payload, dtype=np.uint8, offset=8).copy()
            if len(array) != count or not np.all(array < 10):
                raise ValueError(f'Invalid labels: {name}')
        expected_count = 60000 if name.startswith('train') else 10000
        if len(array) != expected_count:
            raise ValueError(f'Unexpected dataset size: {name}')
        arrays.append(array)
    return tuple(arrays)


def balanced_ids(labels, pool, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        ids = np.asarray(pool)[labels[pool] == label]
        if len(ids):
            if len(ids) < per_class:
                raise ValueError(f'Class {label} has {len(ids)} examples, needs {per_class}')
            selected.extend(rng.permutation(ids)[:per_class])
    return rng.permutation(np.asarray(selected, dtype=np.int64))


def make_manifest(labels, request='baseline', split_seed=42, validation_size=5000,
                  assign_per_class=500, train_limit=0):
    if not valid_request(request) or validation_size < 0 or validation_size % 10:
        raise ValueError('Invalid request or validation size (must be a multiple of ten)')
    all_ids = np.arange(len(labels))
    validation = (balanced_ids(labels, all_ids, validation_size // 10, split_seed)
                  if validation_size else np.empty(0, dtype=np.int64))
    pool = np.setdiff1d(all_ids, validation)
    # The pilot subset is common to all requests, before deleting any examples.
    if train_limit:
        if train_limit % 10:
            raise ValueError('train-limit must be a multiple of ten')
        pool = balanced_ids(labels, pool, train_limit // 10, split_seed + 1)
    if forget_classes(request):
        forget = pool[np.isin(labels[pool], forget_classes(request))]
    elif request == 'random10':
        rng = np.random.default_rng(split_seed + 2)
        forget = np.concatenate([rng.permutation(pool[labels[pool] == c])[
            :len(pool[labels[pool] == c]) // 10] for c in range(10)])
    else:
        forget = np.empty(0, dtype=np.int64)
    retain = np.setdiff1d(pool, forget)
    assignment = balanced_ids(labels, retain, assign_per_class, split_seed + 3)
    return {key: np.asarray(value, dtype=np.int64) for key, value in
            dict(pool=pool, train=retain, forget=forget, validation=validation,
                 assignment=assignment).items()}


def epoch_order(manifest, model_seed, epoch):
    """Filter a shared baseline permutation so retained examples keep their order."""
    order = np.random.default_rng(np.random.SeedSequence([model_seed, epoch, 173])).permutation(manifest['pool'])
    return order[np.isin(order, manifest['train'])]


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
