"""Small regression checks for the baseline's frozen evaluation path."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from brian2 import mV

from src.encoding import present_example
from src.network import build_network
from src.utils import load_weights, save_weights


class BaselineChecks(unittest.TestCase):
    def test_frozen_inference_and_checkpoint(self):
        components = build_network(n_excitatory=5, seed=7)
        self.assertTrue(np.allclose(components['exc'].theta[:] / mV, 20))

        # A blank image should not produce spontaneous excitatory spikes.
        blank = np.zeros(784)
        counts = present_example(components, blank, learning_on=False,
                                 max_rate_scale_tries=1)
        self.assertEqual(int(counts.sum()), 0)

        weights = np.array(components['syn_input_exc'].w[:])
        theta = np.array(components['exc'].theta[:] / mV)
        bright = np.full(784, 255)
        for _ in range(3):
            present_example(components, bright, learning_on=False,
                            max_rate_scale_tries=1)
        np.testing.assert_array_equal(components['syn_input_exc'].w[:], weights)
        np.testing.assert_array_equal(components['exc'].theta[:] / mV, theta)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.npz'
            save_weights(components, path)
            restored = build_network(n_excitatory=5, seed=8)
            load_weights(restored, path)
            np.testing.assert_array_equal(restored['syn_input_exc'].w[:], weights)
            np.testing.assert_array_equal(restored['exc'].theta[:] / mV, theta)


if __name__ == '__main__':
    unittest.main()
