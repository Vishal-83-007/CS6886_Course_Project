"""Fashion replay membership, Gold matching, split safety and request metrics."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.fashion_data import fingerprint
from src.method_c import evaluate, main


class FashionMethodCTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        y = np.repeat(np.arange(10, dtype=np.uint8), 2)
        x = np.full((20, 784), 100, dtype=np.uint8)
        self.arrays = (x, y, x.copy(), y.copy())
        self.baseline = self.model('baseline')

    def model(self, name, request='baseline', **updates):
        path = self.root / name
        path.mkdir()
        config = dict(dataset='fashion', request=request, n_excitatory=3,
                      dataset_sha256=fingerprint(self.arrays), seed=0, split_seed=42,
                      epochs=3, validation_size=0, train_limit=0, assign_per_class=1,
                      weight_norm=39.0, input_total_hz=0.0)
        config.update(updates)
        (path / 'config.json').write_text(json.dumps(config))
        np.savez(path / 'baseline_weights.npz', n_e=3, n_input=784,
                 i=np.repeat(np.arange(784), 3), j=np.tile(np.arange(3), 784),
                 w=np.full(784 * 3, 0.1), theta_volts=np.full(3, 0.02))
        labels = [0, 6, 9] if request == 'baseline' else [0, 0, 9] if request == 'class6' else [0, 0, 0]
        np.save(path / 'neuron_labels.npy', np.array(labels))
        np.savez(path / 'manifest.npz', train=np.arange(20), validation=np.array([], dtype=np.int64))
        return path

    def invoke(self, out, *args):
        with (patch('sys.argv', ['fashion', '--baseline-dir', str(self.baseline),
                                '--out-dir', str(out), '--prepare-only', *args]),
              patch('src.method_c.load_dataset', return_value=self.arrays),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            main(dataset='fashion')

    def test_shirt_replay_uses_training_ids_and_records_no_gold(self):
        # Exclude one shirt from the baseline's actual training membership.
        np.savez(self.baseline / 'manifest.npz', train=np.delete(np.arange(20), 13),
                 validation=np.array([], dtype=np.int64))
        out = self.root / 'shirt'
        self.invoke(out, '--request', 'class6')
        config = json.loads((out / 'config.json').read_text())
        self.assertEqual(config['request'], 'class6')
        self.assertEqual(config['dataset'], 'fashion')
        self.assertIsNone(config['gold_config'])
        self.assertFalse(config['weight_normalization'])
        with np.load(out / 'replay_manifest.npz') as manifest:
            self.assertEqual(manifest['forget_ids'].tolist(), [12])
            self.assertEqual(manifest['forget_labels'].tolist(), [6])
            self.assertEqual(manifest['replay_order_ids'].shape, (5, 1))

    def test_joint_replay_and_optional_matching_gold(self):
        gold = self.model('gold_joint', request='class6_9')
        out = self.root / 'joint'
        self.invoke(out, '--request', 'class6_9', '--gold-dir', str(gold))
        with np.load(out / 'replay_manifest.npz') as manifest:
            self.assertEqual(manifest['forget_ids'].tolist(), [12, 13, 18, 19])
        self.invoke(self.root / 'joint_no_gold', '--no-gold')
        config = json.loads((self.root / 'joint_no_gold/config.json').read_text())
        self.assertEqual(config['request'], 'class6_9')
        self.assertIsNone(config['gold_config'])
        with np.load(self.root / 'joint_no_gold/replay_manifest.npz') as manifest:
            self.assertEqual(manifest['forget_ids'].tolist(), [12, 13, 18, 19])

    def test_wrong_gold_request_and_normalization_rejected(self):
        gold = self.model('wrong_request', request='class6_9')
        with self.assertRaises(SystemExit):
            self.invoke(self.root / 'bad1', '--request', 'class6', '--gold-dir', str(gold))
        gold = self.model('wrong_norm', request='class6', weight_norm=0)
        with self.assertRaises(SystemExit):
            self.invoke(self.root / 'bad2', '--request', 'class6', '--gold-dir', str(gold))
        gold = self.model('matched', request='class6')
        self.invoke(self.root / 'good', '--request', 'class6', '--gold-dir', str(gold))

    def test_validation_overlap_rejected(self):
        config_path = self.baseline / 'config.json'
        config = json.loads(config_path.read_text())
        config['validation_size'] = 1
        config_path.write_text(json.dumps(config))
        np.savez(self.baseline / 'manifest.npz', train=np.arange(20), validation=np.array([12]))
        with self.assertRaises(SystemExit):
            self.invoke(self.root / 'overlap', '--no-gold')

    def test_evaluation_splits_shirt_from_retained_ankle_boot(self):
        counts = np.array([[0, 5, 0], [7, 0, 0]])
        with patch('src.method_c.compiled_counts', return_value=(counts, {})):
            metrics, _, _ = evaluate({}, np.array([0, 6, 9]), np.zeros((2, 784)),
                                      np.array([6, 9]), np.arange(2), 123,
                                      self.root / 'eval', self.root / 'build', 0, classes=[6])
        self.assertEqual(metrics['forget_class_accuracy'], 1)
        self.assertEqual(metrics['retain_accuracy'], 0)


if __name__ == '__main__':
    unittest.main()
