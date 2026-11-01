"""Method B entry point: argument checks, step construction and the numpy-only edit.

    python -m unittest test_method_b -v
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from src import method_b


def write_model(folder, n_e=6, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    np.savez(folder / 'baseline_weights.npz', w=rng.random(784 * n_e),
             i=np.repeat(np.arange(784), n_e).astype(np.int32), j=np.tile(np.arange(n_e), 784).astype(np.int32),
             theta_volts=rng.random(n_e) * 0.02, n_e=n_e, n_input=784)
    np.save(folder / 'neuron_labels.npy', np.arange(n_e) % 10)


def write_importance(path, classes, n_e=6, seed=1):
    rng = np.random.default_rng(seed)
    count = np.zeros(10, np.int64)
    count[list(classes)] = 50
    sumsq = rng.random((10, 784 * n_e)) * count[:, None]
    np.savez(path, sumsq=sumsq, sum=np.zeros_like(sumsq), class_count=count, n_e=n_e, n_input=784)


class MethodBTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_model(self.tmp / 'baseline')
        write_importance(self.tmp / 'imp.npz', range(10))

    def run_main(self, *args):
        with redirect_stdout(io.StringIO()) as out:
            code = method_b.main(['--dataset', 'mnist', '--baseline-dir', str(self.tmp / 'baseline'),
                                  '--importance', str(self.tmp / 'imp.npz'), *args])
        return code, out.getvalue()

    def test_class_edit_only_shrinks_selected_weights(self):
        code, _ = self.run_main('--out-dir', str(self.tmp / 'out'), '--skip-evaluation', '--alpha', '1.1')
        self.assertEqual(code, 0)
        before = np.load(self.tmp / 'baseline' / 'baseline_weights.npz')
        after = np.load(self.tmp / 'out' / 'method_b_weights.npz')
        self.assertTrue(np.all(after['w'] <= before['w'] + 1e-12))
        self.assertTrue(np.array_equal(after['theta_volts'], before['theta_volts']))
        summary = json.loads((self.tmp / 'out' / 'summary.json').read_text())
        self.assertEqual(summary['method'], 'B')
        self.assertGreater(summary['selected_synapses'], 0)

    def test_dry_run_builds_evaluation_with_baseline_labels(self):
        write_model(self.tmp / 'gold', seed=5)
        _, out = self.run_main('--out-dir', str(self.tmp / 'dry'), '--dry-run', '--gold-dir', str(self.tmp / 'gold'))
        lines = [l for l in out.splitlines() if l.startswith('+ ')]
        self.assertEqual(len(lines), 3)
        self.assertIn('dampen.py', lines[0])
        self.assertIn('--eval-split final', lines[1])
        self.assertIn(str(self.tmp / 'baseline' / 'neuron_labels.npy'), lines[1])
        self.assertIn(str(self.tmp / 'gold' / 'neuron_labels.npy'), lines[2])

    def test_subset_request_needs_forget_importance_and_scores_forget_set(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                unittest.mock.patch('sys.stderr', io.StringIO()):
            self.run_main('--out-dir', str(self.tmp / 'x'), '--request', 'sub6_9_p10')
        write_importance(self.tmp / 'df.npz', (6, 9))
        _, out = self.run_main('--out-dir', str(self.tmp / 'sub'), '--dry-run', '--request', 'sub6_9_p10',
                               '--forget-importance', str(self.tmp / 'df.npz'))
        self.assertIn('--forget-importance', out)
        self.assertIn('--forget-set', out)
        self.assertIn('--eval-split all', out)

    def test_refuses_output_inside_baseline_or_nonempty(self):
        with self.assertRaises(SystemExit):
            self.run_main('--out-dir', str(self.tmp / 'baseline' / 'sub'), '--skip-evaluation')
        (self.tmp / 'full').mkdir()
        (self.tmp / 'full' / 'f').write_text('x')
        with self.assertRaises(SystemExit):
            self.run_main('--out-dir', str(self.tmp / 'full'), '--skip-evaluation')


import unittest.mock  # noqa: E402  (used above)

if __name__ == '__main__':
    unittest.main()
