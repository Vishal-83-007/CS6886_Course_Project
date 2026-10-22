"""Download/check MNIST or Fashion-MNIST before submitting offline training jobs.

    python prepare_fashion.py --dataset mnist      # check data/mnist (offline)
    python prepare_fashion.py --dataset mnist --download
"""
import argparse
from pathlib import Path

from src.fashion_data import (DATASETS, REQUESTS, fingerprint, load_dataset,
                              make_manifest, save_json)
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='fashion')
    parser.add_argument('--data-dir', default=None, help='Defaults to data/<dataset>')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--validation-size', type=int, default=5000)
    parser.add_argument('--split-seed', type=int, default=42)
    args = parser.parse_args()
    data_dir = args.data_dir or f'data/{args.dataset}'
    arrays = load_dataset(data_dir, args.download, dataset=args.dataset)
    dataset_hash = fingerprint(arrays)
    root = Path(data_dir)
    for request in REQUESTS:
        manifest = make_manifest(arrays[1], request, args.split_seed, args.validation_size)
        np.savez(root / f'manifest_{request}_val{args.validation_size}.npz', **manifest)
    name = DATASETS[args.dataset]['name']
    save_json(root / 'dataset.json', dict(dataset=name, sha256=dataset_hash,
              train_size=len(arrays[0]), test_size=len(arrays[2])))
    print(f'Verified {name}: {dataset_hash}; offline data ready in {root}')


if __name__ == '__main__':
    main()
