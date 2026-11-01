"""Method A scientific invariants and a bounded compiled-vs-runtime smoke test."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.method_a import class_scores, compiled_counts, report


class ReadoutTests(unittest.TestCase):
    def test_silencing_preserves_class_head_and_control_is_distinct(self):
        labels = np.array([0, 6, 6, 9])
        counts = np.array([[3, 8, 2, 0], [0, 0, 0, 0]])
        scores, predictions = class_scores(counts, labels)
        self.assertEqual(predictions.tolist(), [6, 0])
        self.assertEqual(scores[0, 6], 5)  # Mean, not sum.
        silenced = counts.copy()
        silenced[:, np.isin(labels, [6, 9])] = 0
        scores, _ = class_scores(silenced, labels)
        self.assertEqual(scores[0, 6], 0)  # Original label remains eligible.
        control_scores, _ = class_scores(counts, labels, suppress=[6, 9])
        self.assertTrue(np.isneginf(control_scores[0, 6]))

    def test_metrics_split_and_absent_classes(self):
        counts = np.array([[5, 0], [0, 4], [3, 0]])
        metrics, _, _, cm = report(counts, np.array([0, 9]), np.array([0, 9, 6]), [6, 9])
        self.assertEqual(metrics['retain_accuracy'], 1)
        self.assertEqual(metrics['forget_class_accuracy'], 0.5)
        self.assertEqual(cm[6, 0], 1)
        _, predictions = class_scores(counts, np.array([-1, -1]))
        self.assertEqual(predictions.tolist(), [-1, -1, -1])


@unittest.skipUnless(importlib.util.find_spec('brian2'), 'Brian2 required for compiled smoke test')
class CompiledTests(unittest.TestCase):
    def test_compiled_matches_runtime_and_silences_all_attempts(self):
        from brian2 import prefs, set_device, start_scope, get_device, volt
        from src.network import build_network, set_plasticity
        from src.encoding import present_example

        n = 4
        seed = 17
        weights = dict(n_e=np.array(n), n_input=np.array(784),
                       w=np.random.default_rng(1).uniform(0.1, 0.3, 784*n),
                       theta_volts=np.full(n, 0.02))
        images = np.array([np.full(784, 100), np.full(784, 240)], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            baseline, stats = compiled_counts(weights, images, seed, np.zeros(n), Path(td)/'baseline')
            self.assertTrue(stats['frozen_parameters_verified'])
            self.assertTrue(np.any(baseline))
            masked, stats = compiled_counts(weights, images, seed, np.ones(n), Path(td)/'masked')
            self.assertFalse(np.any(masked))
            self.assertEqual(stats['exhausted'], len(images))
            self.assertEqual(stats['attempts'][5], len(images))

        get_device().reinit()
        set_device('runtime')
        start_scope()
        prefs.codegen.target = 'numpy'
        c = build_network(n, seed)
        c['syn_input_exc'].w = weights['w']
        c['exc'].theta = weights['theta_volts'] * volt
        set_plasticity(c, False)
        reference = np.array([present_example(c, image, learning_on=False) for image in images])
        np.testing.assert_array_equal(baseline, reference)
        np.testing.assert_array_equal(c['syn_input_exc'].w[:], weights['w'])
        np.testing.assert_array_equal(c['exc'].theta[:]/volt, weights['theta_volts'])


if __name__ == '__main__':
    unittest.main()
