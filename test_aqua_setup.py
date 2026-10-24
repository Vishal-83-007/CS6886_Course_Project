"""Offline environment relocation, integrity and bounded process checks."""
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import venv

from scripts.install_aqua_offline import pack_environment, run_step
from scripts.stage_aqua_env import stage_environment


class AquaEnvironmentChecks(unittest.TestCase):
    def create_archive(self, root):
        requirement = root / 'requirements.txt'
        requirement.write_text('example==1\n')
        prefix = root / 'source/fashion-cpu'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
        archives = root / 'archives'
        version = f'{sys.version_info.major}.{sys.version_info.minor}'
        archive = pack_environment(prefix, archives, 'cpu', version, requirement)
        return archives, archive, requirement

    def test_relocated_python_and_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            archives, _, requirement = self.create_archive(root)
            python = stage_environment(archives, root / 'destination', 'cpu', requirement)
            prefix = subprocess.check_output([str(python), '-c', 'import sys; print(sys.prefix)'],
                                             text=True).strip()
            self.assertEqual(prefix, str(root / 'destination/fashion-cpu'))
            self.assertEqual(python, stage_environment(archives, root / 'destination', 'cpu', requirement))

    def test_corrupt_archive_and_changed_requirements_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            archives, archive, requirement = self.create_archive(root)
            original = archive.read_bytes()
            archive.write_bytes(b'incomplete archive')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                stage_environment(archives, root / 'bad', 'cpu', requirement)
            archive.write_bytes(original)
            requirement.write_text('example==2\n')
            with self.assertRaisesRegex(RuntimeError, 'Requirements changed'):
                stage_environment(archives, root / 'changed', 'cpu', requirement)

    def test_install_timeout(self):
        with self.assertRaises(TimeoutError):
            run_step([sys.executable, '-c', 'import time; time.sleep(5)'], 'bounded test', 0.1)

    def test_interpreter_symlinks_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            archives, archive, requirement = self.create_archive(root)
            with tarfile.open(archive) as bundle:
                self.assertTrue(bundle.getmember('fashion-cpu/bin/python').issym())
            python = stage_environment(archives, root / 'destination', 'cpu', requirement)
            self.assertTrue(python.is_symlink())
            subprocess.run([str(python), '-c', 'import sys; assert sys.prefix != sys.base_prefix'],
                           check=True)

    def test_failed_publication_preserves_previous_archive(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            archives, archive, requirement = self.create_archive(root)
            original = archive.read_bytes()
            metadata = archive.with_name('fashion-cpu-cp' +
                                         f'{sys.version_info.major}{sys.version_info.minor}.json')
            previous_metadata = metadata.read_bytes()
            from scripts.install_aqua_offline import sha256
            def bad_destination_hash(path):
                return 'corrupted' if str(path).endswith('.tmp') else sha256(path)
            with patch('scripts.install_aqua_offline.sha256', side_effect=bad_destination_hash):
                with self.assertRaisesRegex(RuntimeError, 'checksum mismatch'):
                    pack_environment(root / 'source/fashion-cpu', archives, 'cpu',
                                     f'{sys.version_info.major}.{sys.version_info.minor}', requirement)
            self.assertEqual(archive.read_bytes(), original)
            self.assertEqual(metadata.read_bytes(), previous_metadata)


if __name__ == '__main__':
    unittest.main()
