"""Check production submission without contacting a scheduler."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent


class CpuSubmissionChecks(unittest.TestCase):
    def test_suite_preview_submission_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            for name in ('bin', 'archives', 'results'):
                (root / name).mkdir()
            for suffix in ('json', 'tar.gz'):
                (root / 'archives' / f'fashion-cpu-cp312.{suffix}').write_text('fixture')
            qsub = root / 'bin/qsub'
            qsub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$QSUB_CALLS"\n'
                            'printf "12345.hn1\\n"\n')
            qsub.chmod(0o755)
            env = dict(os.environ, PATH=str(root / 'bin') + ':' + os.environ['PATH'],
                       FASHION_ENV_ARCHIVES=str(root / 'archives'),
                       FASHION_RESULTS=str(root / 'results'), QSUB_CALLS=str(root / 'calls'),
                       REQUESTS='baseline class6 class0 class9 random10', SEEDS='0',
                       VALIDATION_SIZE='0', EPOCHS='3')

            def launch(mode, **overrides):
                return subprocess.run(['bash', 'scripts/submit_aqua_cpu_suite.sh', mode],
                                      cwd=ROOT, env=dict(env, **overrides),
                                      capture_output=True, text=True, timeout=10)

            preview = launch('--dry-run')
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertFalse((root / 'calls').exists())
            submitted = launch('--submit')
            self.assertEqual(submitted.returncode, 0, submitted.stderr)
            calls = (root / 'calls').read_text().splitlines()
            self.assertEqual(len(calls), 5)
            for request, call in zip(env['REQUESTS'].split(), calls):
                self.assertIn(f'REQUEST={request},SEED=0,VALIDATION_SIZE=0,EPOCHS=3', call)
                self.assertIn(f'FASHION_ENV_ARCHIVES={root}/archives', call)
                self.assertTrue(call.endswith('jobs/aqua_cpp_train.cmd'))
            records = list((root / 'results').glob('cpu_submissions_*.tsv'))
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0].read_text().splitlines()), 6)
            for invalid in ({'REQUESTS': 'unknown'}, {'SEEDS': '-1'},
                            {'VALIDATION_SIZE': '7'}, {'EPOCHS': '0'}):
                with self.subTest(invalid=invalid):
                    result = launch('--submit', **invalid)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual((root / 'calls').read_text().splitlines(), calls)

    def test_transfer_cleanup_preserves_results(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / 'scripts').mkdir()
            (root / 'jobs').mkdir()
            script = root / 'scripts/apply_cpu_only_cleanup.sh'
            script.write_bytes((ROOT / 'scripts/apply_cpu_only_cleanup.sh').read_bytes())
            retired = root / 'requirements-aqua-gpu.txt'
            retired.write_text('obsolete dependency')
            results = root / 'results/fashion'
            results.mkdir(parents=True)
            historic = results / 'historic-model'
            historic.write_text('preserve')
            subprocess.run(['bash', str(script)], check=True, capture_output=True, timeout=10)
            self.assertFalse(retired.exists())
            self.assertEqual(historic.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
