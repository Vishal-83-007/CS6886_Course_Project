"""Method D entry point: argument checks, per-dataset defaults and step construction.

    python -m unittest test_method_d -v
"""
import io
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from src import method_d
from test_method_b import write_model


class MethodDTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_model(self.tmp / 'baseline')

    def dry(self, dataset, *args):
        with redirect_stdout(io.StringIO()) as out:
            method_d.main(['--dataset', dataset, '--baseline-dir', str(self.tmp / 'baseline'),
                           '--out-dir', str(self.tmp / f'out_{dataset}_{len(args)}'), '--dry-run', *args])
        return [l for l in out.getvalue().splitlines() if l.startswith('+ ')]

    def test_reported_defaults(self):
        mnist = self.dry('mnist')[0]
        self.assertIn('--gate neuron', mnist)
        self.assertIn('--leak 0.001', mnist)
        self.assertIn('--replay-images 5000', mnist)
        self.assertIn('--weight-norm 0', mnist)
        fashion = self.dry('fashion')[0]
        self.assertIn('--leak 0.005', fashion)
        self.assertIn('--replay-images 1000', fashion)
        self.assertIn('--weight-norm 39', fashion)

    def test_evaluates_replayed_weights_with_baseline_labels(self):
        steps = self.dry('mnist')
        self.assertEqual(len(steps), 2)
        self.assertIn('replayed_weights.npz', steps[1])
        self.assertIn(str(self.tmp / 'baseline' / 'neuron_labels.npy'), steps[1])

    def test_gate_none_has_no_leak_and_synapse_needs_importance(self):
        self.assertIn('--leak 0.0', self.dry('mnist', '--gate', 'none')[0])
        with self.assertRaises(SystemExit), unittest.mock.patch('sys.stderr', io.StringIO()):
            self.dry('mnist', '--gate', 'synapse')

    def test_subset_request_refused(self):
        with self.assertRaises(SystemExit), unittest.mock.patch('sys.stderr', io.StringIO()):
            self.dry('mnist', '--request', 'sub6_9_p10')


if __name__ == '__main__':
    unittest.main()
