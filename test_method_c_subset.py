"""Exact membership, retain-only readout, subset metrics, and runner integration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.fashion_data import fingerprint
from src.method_c_subset import (assigned_labels, check_gold, evaluation_data, evaluate_subset,
                                 main, make_plan, read_forget_csv)


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        y = np.repeat(np.arange(10, dtype=np.uint8), 6)
        x = np.full((len(y), 784), 100, dtype=np.uint8)
        self.arrays = (x, y, x.copy(), y.copy())
        self.csv = self.root / 'forget.csv'
        self.csv.write_text('train_index,label\n36,6\n54,9\n')
        self.forget = np.array([36, 54])
        self.plan = make_plan(self.arrays, np.arange(60), np.array([], dtype=np.int64),
                              self.forget, 1, 1, 10, 45)
        self.baseline = self.model('baseline')

    def model(self, name, request='baseline'):
        path = self.root / name
        path.mkdir()
        config = dict(dataset='mnist', request=request, n_excitatory=3,
                      dataset_sha256=fingerprint(self.arrays), seed=0, split_seed=42,
                      epochs=3, validation_size=0, train_limit=0, assign_per_class=1)
        (path / 'config.json').write_text(json.dumps(config))
        np.savez(path / 'baseline_weights.npz', n_e=3, n_input=784,
                 i=np.repeat(np.arange(784), 3), j=np.tile(np.arange(3), 784),
                 w=np.full(784 * 3, 0.1), theta_volts=np.full(3, 0.02))
        np.save(path / 'neuron_labels.npy', np.array([0, 6, 9]))
        np.savez(path / 'manifest.npz', train=np.arange(60), validation=np.array([], dtype=np.int64))
        return path

    def invoke(self, out, *args):
        with (patch('sys.argv', ['subset', '--dataset', 'mnist', '--baseline-dir', str(self.baseline),
                                '--forget-csv', str(self.csv), '--out-dir', str(out),
                                '--assign-per-class', '1', '--retain-per-class', '1', '--eval-limit', '10',
                                '--variants', 'C2', '--scales', '0.1', '--passes', '1', *args]),
              patch('src.method_c_subset.load_dataset', return_value=self.arrays),
              contextlib.redirect_stdout(io.StringIO())):
            main()

    def test_csv_validation(self):
        np.testing.assert_array_equal(read_forget_csv(self.csv, self.arrays[1]), self.forget)
        for content in ('train_index,label\n36,9\n', 'train_index,label\n36,6\n36,6\n',
                        'train_index,label\n60,6\n', 'train_index,label\n-1,9\n',
                        'train_index,label\n', 'index,label\n36,6\n'):
            self.csv.write_text(content)
            with self.assertRaises(ValueError):
                read_forget_csv(self.csv, self.arrays[1])

    def test_membership_and_order_are_exact(self):
        out = self.root / 'prepared'
        self.invoke(out, '--prepare-only')
        with np.load(out / 'replay_manifest.npz') as manifest:
            np.testing.assert_array_equal(manifest['forget'], self.forget)
            np.testing.assert_array_equal(np.sort(manifest['replay_order_ids'][0]), self.forget)
            self.assertEqual(len(manifest['retain']), 58)
            self.assertEqual(np.sum(self.arrays[1][manifest['retain']] == 6), 5)
            self.assertEqual(np.sum(self.arrays[1][manifest['retain']] == 9), 5)
            self.assertFalse(np.intersect1d(manifest['assignment'], self.forget).size)
            self.assertFalse(np.intersect1d(manifest['assignment'], manifest['retain_evaluation']).size)
        self.assertFalse((out / 'baseline_assignment.npz').exists())

    def test_nonmember_forget_image_rejected(self):
        with self.assertRaises(ValueError):
            make_plan(self.arrays, np.delete(np.arange(60), 36), np.array([], dtype=np.int64),
                      self.forget, 1, 1, 10, 45)

    def test_gold_membership_and_assignment_and_class_outputs(self):
        gold = self.model('gold', request='sub6_9_p10')
        config = json.loads((self.baseline / 'config.json').read_text())
        def manifest(train, assignment):
            np.savez(gold / 'manifest.npz', train=train, forget=self.forget,
                     assignment=assignment, validation=np.array([], dtype=np.int64))
        manifest(self.plan['retain'], self.plan['assignment'])
        # A subset Gold may still label neurons 6 and 9.
        self.assertEqual(check_gold(gold, config, self.plan, 'mnist', 'sub6_9_p10')[2].tolist(), [0, 6, 9])
        manifest(np.arange(60), self.plan['assignment'])
        with self.assertRaises(ValueError):
            check_gold(gold, config, self.plan, 'mnist', 'sub6_9_p10')
        manifest(self.plan['retain'], self.forget)
        with self.assertRaises(ValueError):
            check_gold(gold, config, self.plan, 'mnist', 'sub6_9_p10')
        with self.assertRaises(ValueError):
            check_gold(gold, config, self.plan, 'mnist', 'class6_9')

    def test_assignment_mean_ties_and_unassigned(self):
        counts = np.array([[2, 3, 0], [4, 1, 0], [4, 2, 0]])
        self.assertEqual(assigned_labels(counts, np.array([6, 6, 9])).tolist(), [9, 6, -1])

    def test_metrics_distinguish_retained_sixes_from_forget_sixes(self):
        data = evaluation_data(self.arrays, self.plan)
        counts = np.zeros((len(data[0]), 3), dtype=np.int32)
        counts[:, 0] = 5  # Predict class 6 everywhere.
        with patch('src.method_c_subset.compiled_counts', return_value=(counts, {})):
            metrics, _, _ = evaluate_subset({}, np.array([6, 9, 0]), data, 123,
                                             self.root / 'metrics', self.root / 'build', 0)
        groups = metrics['groups']
        self.assertEqual(groups['forget_train']['n_evaluation'], 2)
        self.assertEqual(groups['forget_train']['accuracy'], 0.5)
        self.assertEqual(groups['retain_train_requested_classes']['n_evaluation'], 2)
        self.assertEqual(groups['retain_train_requested_classes']['accuracy'], 0.5)
        self.assertNotIn('forget_class_accuracy', groups['test'])
        with np.load(self.root / 'metrics/responses.npz') as responses:
            self.assertEqual(set(responses['dataset_split']), {'forget_train', 'retain_train', 'test'})

    def test_full_orchestration_uses_only_csv_replay_and_fixed_retain_readout(self):
        def inference(weights, images, seed, override, directory, hz):
            return np.tile([5, 3, 0], (len(images), 1)), {}
        def replay(weights, images, orders, seed, variant, scale, directory, passes, hz, adapt):
            np.testing.assert_array_equal(images, self.arrays[0][self.forget])
            self.assertEqual(orders.shape, (1, 2))
            self.assertFalse(adapt)
            edited = {k: np.array(v, copy=True) for k, v in weights.items()}
            edited['w'] *= 0.9
            return {1: edited}, np.zeros(2, dtype=[('negative_updates', int)]), {}
        out = self.root / 'full'
        with (patch('src.method_c_subset.compiled_counts', side_effect=inference) as infer,
              patch('src.method_c_subset.compiled_replay', side_effect=replay)):
            self.invoke(out)
        self.assertEqual(infer.call_count, 3)  # One assignment; baseline and edited evaluations.
        summary = json.loads((out / 'summary.json').read_text())
        self.assertFalse(summary['gold_comparison_available'])
        self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'complete')
        with np.load(out / 'baseline_assignment.npz') as assignment:
            self.assertFalse(np.intersect1d(assignment['ids'], self.forget).size)
            np.testing.assert_array_equal(np.load(out / 'C2_s0.1/pass_1/neuron_labels.npy'), assignment['labels'])
        self.assertTrue(summary['C2_s0.1']['checkpoints']['1']['intervention']['frozen_thresholds_verified'])


class SuppliedManifestTests(unittest.TestCase):
    def test_official_csvs_match_seed46_selection(self):
        from src.fashion_data import load_dataset
        root = Path(__file__).resolve().parent
        for dataset, expected in [('mnist', (591, 594)), ('fashion', (600, 600))]:
            if not (root / 'data' / dataset / 'train-labels-idx1-ubyte.gz').exists():
                continue  # Source data intentionally omitted from transfer archive.
            arrays = load_dataset(root / 'data' / dataset, dataset=dataset)
            ids = read_forget_csv(root / f'forget_set_{dataset}_sub6_9_p10.csv', arrays[1])
            rng = np.random.default_rng(46)
            selected = []
            for k, n in zip((6, 9), expected):
                pool = np.flatnonzero(arrays[1] == k)
                rng.shuffle(pool)
                selected.extend(pool[:len(pool) // 10])
                self.assertEqual(np.sum(arrays[1][ids] == k), n)
            np.testing.assert_array_equal(ids, np.sort(selected))


if __name__ == '__main__':
    unittest.main()
