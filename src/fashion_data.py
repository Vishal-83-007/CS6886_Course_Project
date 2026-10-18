"""Offline Fashion-MNIST data and reproducible deletion manifests."""
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
BASE_URL = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
REQUESTS = ('baseline', 'class6', 'class0', 'class9', 'random10')
CLASS_NAMES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


def fingerprint(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_dataset(directory, download=False):
    """Load checked official gzip IDX files; download only when explicitly asked."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = []
    for name, expected in FILES.items():
        path = directory / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path}: run prepare_fashion.py --download first')
            temporary = path.with_suffix('.download')
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
                temporary.write_bytes(response.read())
            if hashlib.md5(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Download checksum mismatch: {name}')
            temporary.replace(path)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Dataset checksum mismatch: {path}')
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
    if request not in REQUESTS or validation_size < 0 or validation_size % 10:
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
    if request.startswith('class'):
        forget = pool[labels[pool] == int(request[5:])]
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
