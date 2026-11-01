"""Fashion checkpoint isolation, Gold matching and paired collateral metrics."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.method_a import deletion_request, fashion_comparison, load_model, main


class FashionMethodATests(unittest.TestCase):
    def model(self, path, request='baseline', **updates):
        path.mkdir()
        config = dict(dataset='fashion', request=request, n_excitatory=4,
                      dataset_sha256='fixture', seed=0, split_seed=42, epochs=3,
                      validation_size=0, train_limit=0, assign_per_class=500,
                      weight_norm=39.0, input_total_hz=0.0)
        config.update(updates)
        (path / 'config.json').write_text(json.dumps(config))
        np.savez(path / 'baseline_weights.npz', n_e=4, n_input=784,
                 i=np.repeat(np.arange(784), 4), j=np.tile(np.arange(4), 784),
                 w=np.full(784 * 4, 0.1), theta_volts=np.full(4, 0.02))
        labels = np.array([0, 4, 6, 9]) if request == 'baseline' else np.array([0, 4, 0, 9])
        np.save(path / 'neuron_labels.npy', labels)

    def invoke(self, baseline, out, *args):
        with patch('sys.argv', ['fashion', '--baseline-dir', str(baseline),
                               '--out-dir', str(out), '--prepare-only', *args]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main(dataset='fashion')

    def test_shirt_preparation_preserves_weights_labels_and_theta(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline, out = root / 'baseline', root / 'out'
            self.model(baseline)
            self.invoke(baseline, out, '--request', 'class6')
            config = json.loads((out / 'config.json').read_text())
            self.assertEqual(config['dataset'], 'fashion')
            self.assertEqual(config['request'], 'class6')
            self.assertIsNone(config['gold_config'])
            self.assertEqual(config['n_silenced'], 1)
            for name in ('baseline_weights.npz', 'neuron_labels.npy'):
                self.assertEqual((baseline / name).read_bytes(), (out / 'checkpoint' / name).read_bytes())
            with np.load(out / 'checkpoint/intervention.npz') as saved:
                self.assertEqual(saved['forget_neuron_mask'].tolist(), [False, False, True, False])
                self.assertEqual(saved['forget_classes'].tolist(), [6])
            with self.assertRaises(ValueError):
                load_model(baseline)  # MNIST/Method C callers remain MNIST-only.

    def test_default_joint_request_without_gold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline, out = root / 'baseline', root / 'out'
            self.model(baseline)
            self.invoke(baseline, out, '--no-gold')
            config = json.loads((out / 'config.json').read_text())
            self.assertEqual(config['request'], 'class6_9')
            self.assertIsNone(config['gold_config'])
            self.assertEqual(config['n_silenced'], 2)
            with np.load(out / 'checkpoint/intervention.npz') as saved:
                self.assertEqual(saved['forget_classes'].tolist(), [6, 9])
            with self.assertRaises(SystemExit):
                self.invoke(baseline, root / 'invalid', '--no-gold',
                            '--gold-dir', str(root / 'absent_gold'))

    def test_gold_request_and_training_normalization_must_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline, gold = root / 'baseline', root / 'gold'
            self.model(baseline)
            self.model(gold, request='class6_9')
            with self.assertRaises(SystemExit):
                self.invoke(baseline, root / 'wrong_request', '--request', 'class6', '--gold-dir', str(gold))
            config = json.loads((gold / 'config.json').read_text())
            config.update(request='class6', weight_norm=0)
            (gold / 'config.json').write_text(json.dumps(config))
            with self.assertRaises(SystemExit):
                self.invoke(baseline, root / 'wrong_norm', '--request', 'class6', '--gold-dir', str(gold))
            config['weight_norm'] = 39
            (gold / 'config.json').write_text(json.dumps(config))
            self.invoke(baseline, root / 'matched', '--request', 'class6', '--gold-dir', str(gold))

    def test_paired_collateral_and_shirt_confusion(self):
        truth = np.array([0, 4, 6, 6])
        outputs = dict(baseline=np.array([0, 4, 6, 4]), method_a=np.array([4, 4, 0, 4]))
        summaries = {name: dict(per_class_accuracy={str(k): float(np.mean(pred[truth == k] == k))
                                                  for k in np.unique(truth)})
                     for name, pred in outputs.items()}
        result = fashion_comparison(truth, outputs, summaries, [6])['comparisons']['method_a']
        self.assertEqual(result['per_class']['0']['accuracy_delta_vs_baseline'], -1)
        self.assertEqual(result['per_class']['0']['baseline_correct_now_wrong'], 1)
        self.assertEqual(result['per_class']['4']['accuracy_delta_vs_baseline'], 0)
        self.assertEqual(result['shirt_prediction_distribution']['4'], 0.5)
        self.assertEqual(result['upper_body_retain_classes'], [0, 2, 3, 4])

    def test_request_validation(self):
        self.assertEqual(deletion_request('class6_9'), 'class6_9')
        for value in ('random10', 'baseline', 'class6_6', 'class9_6'):
            with self.assertRaises(argparse.ArgumentTypeError):
                deletion_request(value)


if __name__ == '__main__':
    unittest.main()
