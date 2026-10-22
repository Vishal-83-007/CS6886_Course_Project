"""Data exclusion, exact runtime continuation and compiled-loop regression tests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.fashion_data import epoch_order, make_manifest


def synthetic_run(backend, destination, resume=False, stop=False):
    from src.fashion_runner import run_runtime
    from src.fashion_standalone import run_standalone
    rng = np.random.default_rng(123)
    y = np.repeat(np.arange(10, dtype=np.uint8), 2)
    x = rng.integers(0, 256, (20, 784), dtype=np.uint8)
    arrays = (x, y, x.copy(), y.copy())
    manifest = make_manifest(y, validation_size=0, assign_per_class=1, train_limit=10)
    manifest['evaluation'] = np.arange(10, dtype=np.int64)
    schedule = dict(stage=np.repeat(np.arange(3, dtype=np.uint8), 10),
                    ids=np.concatenate([epoch_order(manifest, 0, 0), manifest['assignment'],
                                        manifest['evaluation']]))
    args = argparse.Namespace(backend=backend, n_excitatory=5, seed=0, epochs=1,
        validation_size=0, request='baseline', resume=resume,
        max_seconds=0.000001 if stop else 0, checkpoint_seconds=0, checkpoint_every=0,
        threads=1, gpu_id=None, compile_only=False, train_limit=10)
    root = Path(destination)
    root.mkdir(exist_ok=True)
    fn = run_standalone if backend == 'cpp' else run_runtime
    return fn(args, arrays, manifest, schedule, root, 'synthetic-regression')


class FashionDataChecks(unittest.TestCase):
    def test_deletion_assignment_exclusion_and_paired_order(self):
        y = np.repeat(np.arange(10), 6000)
        baseline = make_manifest(y, validation_size=0)
        for request in ('class6', 'class0', 'class9', 'random10'):
            gold = make_manifest(y, request, validation_size=0)
            self.assertEqual(len(gold['forget']), 6000)
            self.assertEqual(len(gold['train']), 54000)
            self.assertFalse(np.intersect1d(gold['forget'], gold['assignment']).size)
            order = epoch_order(baseline, 2, 1)
            np.testing.assert_array_equal(epoch_order(gold, 2, 1),
                                           order[np.isin(order, gold['train'])])
            if request.startswith('class'):
                self.assertNotIn(int(request[5:]), y[gold['assignment']])
        self.assertFalse(np.array_equal(epoch_order(baseline, 2, 0), epoch_order(baseline, 2, 1)))

    def test_no_silent_subset_cap_or_validation_overlap(self):
        y = np.repeat(np.arange(10), 100)
        manifest = make_manifest(y, validation_size=100, assign_per_class=5)
        self.assertFalse(np.intersect1d(manifest['train'], manifest['validation']).size)
        with self.assertRaises(ValueError):
            make_manifest(y, validation_size=0, assign_per_class=101)


MNIST_DIR = Path(__file__).parent / 'data' / 'mnist'


@unittest.skipUnless((MNIST_DIR / 'train-images-idx3-ubyte.gz').exists(), 'data/mnist not staged')
class MnistDataChecks(unittest.TestCase):
    def test_official_mnist_loads_offline(self):
        from src.fashion_data import load_dataset
        x_train, y_train, x_test, y_test = load_dataset(MNIST_DIR, dataset='mnist')
        self.assertEqual((x_train.shape, x_test.shape), ((60000, 784), (10000, 784)))
        self.assertEqual(x_train.dtype, np.uint8)
        np.testing.assert_array_equal(np.unique(y_train), np.arange(10))
        # Official MNIST class counts (digit 1 is the most frequent).
        self.assertEqual(int(np.bincount(y_train)[1]), 6742)
        self.assertEqual(int(np.bincount(y_test)[1]), 1135)

    def test_mnist_files_are_rejected_as_fashion(self):
        from src.fashion_data import load_dataset
        with self.assertRaises(ValueError):
            load_dataset(MNIST_DIR, dataset='fashion')


class FashionExecutionChecks(unittest.TestCase):
    def launch(self, backend, path, resume=False, stop=False):
        code = (f'from test_fashion import synthetic_run; '
                f'raise SystemExit(synthetic_run({backend!r}, {str(path)!r}, {resume!r}, {stop!r}))')
        return subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                              capture_output=True, text=True, timeout=180)

    def check_exact_runtime_resume(self, backend):
        with tempfile.TemporaryDirectory() as directory:
            full, resumed = Path(directory) / 'full', Path(directory) / 'resumed'
            result = self.launch(backend, full)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = self.launch(backend, resumed, stop=True)
            self.assertEqual(result.returncode, 75, result.stderr)
            result = self.launch(backend, resumed, resume=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for filename in ('baseline_weights.npz', 'responses.npz'):
                with np.load(full / filename) as left, np.load(resumed / filename) as right:
                    for key in left.files:
                        np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def test_exact_numpy_resume(self):
        self.check_exact_runtime_resume('numpy')

    def test_exact_cython_resume_across_processes(self):
        self.check_exact_runtime_resume('cython')

    def test_compiled_clock_continuity_and_frozen_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'compiled'
            result = self.launch('cpp', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(output / 'baseline_weights.npz') as final:
                self.assertTrue(np.isfinite(final['w']).all())
                before = np.fromfile(output / 'inference_snapshots/position_10_w.bin')
                np.testing.assert_array_equal(before, final['w'])
                theta = np.fromfile(output / 'inference_snapshots/position_10_theta.bin')
                np.testing.assert_array_equal(theta, final['theta_volts'])
            metrics = json.loads((output / 'metrics.json').read_text())
            attempts = sum((i + 1) * n for i, n in enumerate(metrics['attempts']))
            times = list((output / 'build/results').glob('_array_defaultclock_t_*'))
            self.assertEqual(len(times), 1)
            self.assertAlmostEqual(float(np.fromfile(times[0])[0]), 0.5 * attempts)


if __name__ == '__main__':
    unittest.main()
