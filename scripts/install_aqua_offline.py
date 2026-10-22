"""Run in an Aqua CPU job using the module Python; never contacts package indexes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_wheelhouse(directory, requirement, version):
    marker = directory / 'READY.json'
    if not marker.exists():
        raise RuntimeError(f'Missing {marker}. Run scripts/prepare_aqua_wheels.py '
                           f'--python-version {version} locally and transfer wheelhouse/.')
    manifest = json.loads(marker.read_text())
    if manifest['python'] != version:
        raise RuntimeError('Wheelhouse Python version does not match the module interpreter')
    if manifest['requirement_sha256'] != sha256(requirement):
        raise RuntimeError(f'{requirement.name} changed: regenerate and transfer the wheelhouse')
    if not manifest['wheels']:
        raise RuntimeError('Empty wheelhouse')
    actual = {p.name for p in directory.glob('*.whl')}
    if actual != set(manifest['wheels']):
        raise RuntimeError(f'Wheelhouse inventory differs from READY.json: {directory}')
    for name, expected in manifest['wheels'].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise RuntimeError(f'Wheel checksum mismatch: {name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', type=Path, default=ROOT / 'wheelhouse')
    parser.add_argument('--env-root', type=Path, default=Path.home())
    parser.add_argument('--kind', choices=('cpu', 'gpu', 'both'), default='both')
    args = parser.parse_args()
    # Do not let login-shell package paths leak into the isolated environments.
    os.environ.pop('PYTHONHOME', None)
    os.environ.pop('PYTHONPATH', None)
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise SystemExit('Load anaconda3_2024.10 (Python 3.12), not the system Python')
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise SystemExit('This bundle targets Linux x86_64 Aqua nodes')
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    tag = 'cp' + version.replace('.', '')
    kinds = ('cpu', 'gpu') if args.kind == 'both' else (args.kind,)
    # Preflight everything before creating/installing either environment.
    for kind in kinds:
        verify_wheelhouse(args.wheelhouse / tag / kind,
                         ROOT / f'requirements-aqua-{kind}.txt', version)
    for kind in kinds:
        directory = (args.wheelhouse / tag / kind).resolve()
        prefix = args.env_root / f'fashion-{kind}'
        python = prefix / 'bin/python'
        if python.exists():
            existing = subprocess.check_output([str(python), '-c',
                'import sys; print("%s.%s" % sys.version_info[:2])'], text=True).strip()
            if existing != version:
                raise SystemExit(f'{prefix} uses Python {existing}, need {version}. '
                                 'Use --env-root for a fresh location; no environment was deleted.')
        else:
            # No ensurepip/Conda solve/download; the module's pip bootstraps this venv.
            venv.EnvBuilder(with_pip=False).create(prefix)
        subprocess.run([sys.executable, '-m', 'pip', '--isolated', '--python', str(python),
                        'install', '--no-index', '--find-links', str(directory),
                        '--only-binary=:all:', 'pip==24.3.1', 'wheel==0.45.1',
                        '-r', str(ROOT / f'requirements-aqua-{kind}.txt')], check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
        subprocess.run([str(python), '-c', 'import sys, brian2, numpy; '
                        'print(sys.version); print("Brian", brian2.__version__, "NumPy", numpy.__version__)'],
                       check=True)
        print(f'{kind.upper()} Python: {python}', flush=True)


if __name__ == '__main__':
    main()
