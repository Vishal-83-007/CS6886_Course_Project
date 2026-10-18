"""Download/check Fashion-MNIST before submitting offline training jobs."""
import argparse
from pathlib import Path

from src.fashion_data import REQUESTS, fingerprint, load_dataset, make_manifest, save_json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/fashion')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--validation-size', type=int, default=5000)
    parser.add_argument('--split-seed', type=int, default=42)
    args = parser.parse_args()
    arrays = load_dataset(args.data_dir, args.download)
    dataset_hash = fingerprint(arrays)
    root = Path(args.data_dir)
    for request in REQUESTS:
        manifest = make_manifest(arrays[1], request, args.split_seed, args.validation_size)
        np.savez(root / f'manifest_{request}_val{args.validation_size}.npz', **manifest)
    save_json(root / 'dataset.json', dict(dataset='fashion-mnist', sha256=dataset_hash,
              train_size=len(arrays[0]), test_size=len(arrays[2])))
    print(f'Verified Fashion-MNIST: {dataset_hash}; offline data ready in {root}')


if __name__ == '__main__':
    main()
