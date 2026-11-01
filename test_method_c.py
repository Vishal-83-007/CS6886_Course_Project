"""Method C event-rule and compiled trajectory checks against runtime replay."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.method_c import build_erasure_network, compiled_replay


class MethodCTests(unittest.TestCase):
    def reset_runtime(self):
        from brian2 import get_device, set_device, start_scope, prefs
        get_device().reinit()
        set_device('runtime')
        start_scope()
        prefs.codegen.target = 'numpy'

    def test_actual_post_event_rule_positive_and_negative_trace(self):
        from brian2 import ms, volt, Hz
        from src.network import ETA, MU, X_TAR
        for variant in ('C1', 'C2'):
            self.reset_runtime()
            c = build_erasure_network(1, 13, variant, 0.5)
            c['syn_input_exc'].w = 0.3
            trace = np.tile([0.0, 2.0], 392)
            c['syn_input_exc'].pretrace = trace
            c['input'].rates = 0 * Hz
            c['exc'].theta = 0.02 * volt
            c['exc'].v = 0 * volt  # Exactly one forced postsynaptic event.
            c['net'].run(0.5 * ms)
            # First event is at t=0: no preceding trace decay.
            base = ETA * (trace - X_TAR) * (1 - 0.3)**MU
            delta = -0.5 * (base if variant == 'C1' else np.maximum(base, 0))
            np.testing.assert_allclose(c['syn_input_exc'].w[:], 0.3 + delta, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(c['exc'].theta[:]/volt, [0.02])
            self.assertEqual(int(np.sum(c['syn_input_exc'].positive_updates[:])), 392 if variant == 'C1' else 0)
            self.assertEqual(int(np.sum(c['syn_input_exc'].negative_updates[:])), 392)

    def test_compiled_matches_runtime_across_passes_and_freezes_in_rest(self):
        from brian2 import volt, Hz, get_device
        from src.encoding import PRESENT_TIME, REST_TIME
        n, seed = 3, 19
        weights = dict(n_e=np.array(n), n_input=np.array(784),
                       w=np.random.default_rng(7).uniform(0.1, 0.3, 784*n),
                       theta_volts=np.full(n, 0.02))
        images = np.array([np.full(784, 100), np.full(784, 240)], dtype=np.uint8)
        orders = np.array([[1, 0], [0, 1]])
        for variant in ('C1', 'C2'):
            with tempfile.TemporaryDirectory() as td:
                snapshots, stats, timing = compiled_replay(weights, images, orders, seed, variant, 1.0,
                                                          Path(td)/variant, save_passes=(1, 2))
                self.assertTrue(timing['frozen_thresholds_verified'])
                self.assertTrue(np.any(stats['negative_updates'] > 0))
                self.assertTrue(np.any(stats['active_spikes'] > 0))
            self.reset_runtime()
            c = build_erasure_network(n, seed, variant, 1.0)
            c['syn_input_exc'].w = weights['w']
            c['exc'].theta = weights['theta_volts'] * volt
            active_counts = []
            for passes, order in enumerate(orders, 1):
                for idx in order:
                    c['input'].rates = images[idx].astype(float) / 4 * Hz
                    c['syn_input_exc'].erasure_enabled = 1
                    before = np.array(c['exc_monitor'].count[:])
                    c['net'].run(PRESENT_TIME)
                    active_counts.append(int(np.sum(c['exc_monitor'].count[:] - before)))
                    c['input'].rates = 0 * Hz
                    c['syn_input_exc'].erasure_enabled = 0
                    before_rest = np.array(c['syn_input_exc'].w[:])
                    c['net'].run(REST_TIME)
                    np.testing.assert_array_equal(c['syn_input_exc'].w[:], before_rest)
                np.testing.assert_allclose(snapshots[passes]['w'], c['syn_input_exc'].w[:], rtol=0, atol=2e-13)
                np.testing.assert_array_equal(snapshots[passes]['theta_volts'], c['exc'].theta[:]/volt)
            np.testing.assert_array_equal(stats['active_spikes'], active_counts)
        get_device().reinit()

    def test_threshold_adaptation_ablation_enables_increment_and_decay(self):
        from brian2 import ms, volt
        from src.network import THETA_PLUS_E, TAU_THETA
        self.reset_runtime()
        c = build_erasure_network(1, 31, 'C2', 0.1, adapt_thresholds=True)
        c['syn_input_exc'].w = 0
        c['exc'].theta = 0.02 * volt
        c['exc'].v = 0 * volt
        c['net'].run(0.5 * ms)
        expected = 0.02 * np.exp(-float((0.5 * ms) / TAU_THETA)) + float(THETA_PLUS_E / volt)
        np.testing.assert_allclose(c['exc'].theta[:] / volt, [expected], rtol=0, atol=1e-14)
        before = np.array(c['exc'].theta[:] / volt)
        c['net'].run(0.5 * ms)
        self.assertTrue(np.all(c['exc'].theta[:] / volt < before))

    def test_clipping_and_disabled_updates(self):
        from brian2 import ms, volt
        for variant in ('C1', 'C2'):
            self.reset_runtime()
            c = build_erasure_network(1, 4, variant, 1e8)
            c['syn_input_exc'].w = 0.3
            c['syn_input_exc'].pretrace = np.tile([0.0, 2.0], 392)
            c['exc'].theta = 0.02 * volt
            c['exc'].v = 0 * volt
            c['net'].run(0.5 * ms)
            expected = np.tile([1.0 if variant == 'C1' else 0.3, 0.0], 392)
            np.testing.assert_array_equal(c['syn_input_exc'].w[:], expected)
            self.assertEqual(int(np.sum(c['syn_input_exc'].clipped_updates[:])), 784 if variant == 'C1' else 392)


if __name__ == '__main__':
    unittest.main()
