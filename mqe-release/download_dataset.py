"""
download_dataset.py
Downloads the visual-cube-triple-play-v0 dataset (train + val) from OGBench.

Usage (run on a Great Lakes login node after 'pip install -e .[train]'):
    python download_dataset.py --dataset_dir /path/to/data

The dataset files saved:
    <dataset_dir>/visual-cube-triple-play-v0.npz       (~2-3 GB)
    <dataset_dir>/visual-cube-triple-play-v0-val.npz   (smaller)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'impls'))

import ogbench

DATASETS = [
    'visual-cube-triple-play-v0',
    'visual-scene-play-v0',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        default=os.path.expanduser('~/ogbench_data'),
        help='Directory to save the downloaded dataset files.',
    )
    args = parser.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)
    for name in DATASETS:
        train_path = os.path.join(args.dataset_dir, f'{name}.npz')
        if os.path.exists(train_path):
            print(f'Already exists, skipping: {train_path}')
            continue
        print(f'Downloading {name} to {args.dataset_dir} ...')
        ogbench.download_datasets([name], dataset_dir=args.dataset_dir)
        print(f'  Done: {train_path}')

    print('\nAll datasets ready. Dataset paths:')
    for name in DATASETS:
        print(f'  {args.dataset_dir}/{name}.npz')
