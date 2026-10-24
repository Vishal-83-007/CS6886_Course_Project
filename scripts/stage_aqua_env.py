"""Unpack our checked environment archive on the compute node; print Python path."""
import argparse
import json
from pathlib import Path
import platform
import sys
import tarfile

try:
    from .install_aqua_offline import sha256
except ImportError:  # Direct invocation as scripts/stage_aqua_env.py
    from install_aqua_offline import sha256


def stage_environment(archive_dir, destination, kind, requirement):
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    stem = f'fashion-{kind}-cp{version.replace(".", "")}'
    metadata = json.loads((archive_dir / (stem + '.json')).read_text())
    if (metadata['kind'] != kind or metadata['python'] != version or
            metadata['python_exact'] != platform.python_version() or
            metadata['base_executable'] != str(Path(sys._base_executable).resolve())):
        raise RuntimeError('Packed environment differs from the current module Python; rerun setup')
    if metadata['requirement_sha256'] != sha256(requirement):
        raise RuntimeError('Requirements changed since environment packing; rerun setup')
    if (Path(metadata['archive']).name != metadata['archive'] or
            metadata['prefix'] != f'fashion-{kind}'):
        raise RuntimeError('Invalid environment archive metadata')
    archive = archive_dir / metadata['archive']
    prefix = destination / metadata['prefix']
    stamp = prefix / '.aqua_archive_sha256'
    if stamp.exists() and stamp.read_text().strip() == metadata['archive_sha256']:
        return prefix / 'bin/python'
    if prefix.exists():
        raise RuntimeError(f'{prefix} is incomplete/different; use a fresh node-local destination')
    print(f'Staging {archive} into {destination}', file=sys.stderr, flush=True)
    if sha256(archive) != metadata['archive_sha256']:
        raise RuntimeError('Environment archive checksum mismatch; rerun setup')
    destination.mkdir(parents=True, exist_ok=True)
    pending = destination / (metadata['prefix'] + '.extracting')
    if pending.exists():
        raise RuntimeError(f'Interrupted extraction in {pending}; use a fresh destination')
    pending.mkdir()
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts
            if not parts or parts[0] != metadata['prefix'] or '..' in parts or Path(member.name).is_absolute():
                raise RuntimeError('Unexpected environment archive member')
        # These are our own SHA-checked venv archives. Their Python symlinks
        # intentionally point to the verified, shared Anaconda base interpreter.
        bundle.extractall(pending, filter='fully_trusted')
    extracted = pending / metadata['prefix']
    if not (extracted / 'pyvenv.cfg').exists() or not (extracted / 'bin/python').exists():
        raise RuntimeError('Archive does not contain a usable virtual environment')
    extracted.rename(prefix)
    pending.rmdir()
    stamp.write_text(metadata['archive_sha256'] + '\n')
    return prefix / 'bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--kind', choices=('cpu',), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = stage_environment(args.archive_dir, args.destination, args.kind,
                               root / f'requirements-aqua-{args.kind}.txt')
    print(python)


if __name__ == '__main__':
    main()
