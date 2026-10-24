"""Run in an Aqua CPU job using the module Python; never contacts package indexes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import venv

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message, flush=True)


def run_step(command, name, timeout=1200):
    log(f'BEGIN {name}')
    start = time.monotonic()
    with subprocess.Popen(command) as process:
        while True:
            try:
                status = process.wait(timeout=min(30, max(0.01, timeout - (time.monotonic() - start))))
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError(f'{name} exceeded {timeout}s; inspect the last timestamped stage')
                log(f'{name}: still running ({elapsed:.0f}s)')
        if status:
            raise subprocess.CalledProcessError(status, command)
    log(f'END {name} ({time.monotonic() - start:.1f}s)')


def pack_environment(prefix, destination, kind, version, requirement):
    """Publish one archive to shared storage; environments are expanded only locally."""
    destination.mkdir(parents=True, exist_ok=True)
    name = f'fashion-{kind}-cp{version.replace(".", "")}'
    # Compress locally before copying the single file to the shared filesystem.
    local = prefix.parent / (name + '.tar.gz')
    start = time.monotonic()
    log(f'BEGIN compress {kind}: {prefix} -> {local}')
    def keep(info):
        return None if '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc') else info
    with tarfile.open(local, 'w:gz', compresslevel=1) as bundle:
        bundle.add(prefix, arcname=prefix.name, filter=keep)
    log(f'END compress {kind} ({time.monotonic() - start:.1f}s; {local.stat().st_size} bytes)')
    log(f'BEGIN checksum {kind}')
    digest = sha256(local)
    log(f'END checksum {kind}: {digest}')
    pending = destination / (name + '.tar.gz.tmp')
    log(f'BEGIN copy {kind}: {local} -> {pending}')
    copied = 0
    started = last_report = time.monotonic()
    # Use bounded userspace buffers rather than shutil's platform fast-copy
    # path (e.g. sendfile across node-local storage and Lustre).
    with local.open('rb') as source, pending.open('wb') as target:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            target.write(block)
            copied += len(block)
            if time.monotonic() - last_report >= 30:
                log(f'copy {kind}: {copied}/{local.stat().st_size} bytes')
                last_report = time.monotonic()
    log(f'END copy {kind} ({time.monotonic() - started:.1f}s; {copied} bytes)')
    log(f'BEGIN verify published bytes {kind}')
    if sha256(pending) != digest:
        raise RuntimeError(f'Published {kind} archive checksum mismatch; leaving .tmp unpublished')
    log(f'END verify published bytes {kind}')
    archive = destination / (name + '.tar.gz')
    pending.replace(archive)
    metadata = dict(python=version, python_exact=platform.python_version(), kind=kind,
                    base_executable=str(Path(sys._base_executable).resolve()),
                    archive=archive.name, archive_sha256=digest, prefix=prefix.name,
                    requirement_sha256=sha256(requirement))
    temporary = destination / (name + '.json.tmp')
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(destination / (name + '.json'))
    log(f'END packing {kind}: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)')
    return archive


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
    parser.add_argument('--kind', choices=('cpu',), default='cpu')
    parser.add_argument('--stage-dir', type=Path, help='Node-local directory for wheel copies')
    parser.add_argument('--pack-dir', type=Path, help='Publish packed environments here')
    parser.add_argument('--step-timeout', type=int, default=1200)
    parser.add_argument('--pack-timeout', type=int, default=1800,
                        help='Maximum seconds per compression/copy/checksum subprocess')
    parser.add_argument('--pack-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Do not let login-shell package paths leak into the isolated environments.
    os.environ.pop('PYTHONHOME', None)
    os.environ.pop('PYTHONPATH', None)
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    if args.step_timeout < 1 or args.pack_timeout < 1:
        parser.error('step-timeout and pack-timeout must be positive')
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        raise SystemExit('Load anaconda3_2024.10 (Python 3.12), not the system Python')
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise SystemExit('This bundle targets Linux x86_64 Aqua nodes')
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    tag = 'cp' + version.replace('.', '')
    kinds = ('cpu',)
    if args.pack_only:
        if args.pack_dir is None:
            parser.error('--pack-only requires one --kind and --pack-dir')
        pack_environment(args.env_root / f'fashion-{args.kind}', args.pack_dir,
                         args.kind, version, ROOT / f'requirements-aqua-{args.kind}.txt')
        return
    # Preflight everything before creating/installing either environment.
    for kind in kinds:
        log(f'Checking {kind} wheelhouse')
        verify_wheelhouse(args.wheelhouse / tag / kind,
                         ROOT / f'requirements-aqua-{kind}.txt', version)
    for kind in kinds:
        directory = (args.wheelhouse / tag / kind).resolve()
        if args.stage_dir:
            staged = args.stage_dir / kind
            log(f'BEGIN staging {kind} wheels to {staged}')
            shutil.copytree(directory, staged, dirs_exist_ok=True)
            directory = staged.resolve()
            verify_wheelhouse(directory, ROOT / f'requirements-aqua-{kind}.txt', version)
            log(f'END staging {kind} wheels')
        prefix = args.env_root / f'fashion-{kind}'
        log(f'{kind} install destination: {prefix}')
        python = prefix / 'bin/python'
        if python.exists():
            existing = subprocess.check_output([str(python), '-c',
                'import sys; print("%s.%s" % sys.version_info[:2])'], text=True, timeout=60).strip()
            if existing != version:
                raise SystemExit(f'{prefix} uses Python {existing}, need {version}. '
                                 'Use --env-root for a fresh location; no environment was deleted.')
        else:
            # No ensurepip/Conda solve/download; the module's pip bootstraps this venv.
            # The exact shared base interpreter is checked during staging.
            # Avoid archiving three redundant copies of that interpreter.
            venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix)
        run_step([sys.executable, '-m', 'pip', '--isolated', '--disable-pip-version-check',
                        '--python', str(python),
                        'install', '--no-index', '--find-links', str(directory),
                        '--only-binary=:all:', '--no-compile', '--no-cache-dir',
                        'pip==24.3.1', 'wheel==0.45.1',
                        '-r', str(ROOT / f'requirements-aqua-{kind}.txt')],
                 f'install {kind}', args.step_timeout)
        run_step([str(python), '-m', 'pip', 'check'], f'check {kind}', 180)
        run_step([str(python), '-c', 'import sys, brian2, numpy; '
                        'print(sys.version); print("Brian", brian2.__version__, "NumPy", numpy.__version__)'],
                 f'import {kind}', 180)
        if args.pack_dir:
            run_step([sys.executable, '-u', str(Path(__file__).resolve()),
                      '--pack-only', '--kind', kind, '--env-root', str(args.env_root),
                      '--pack-dir', str(args.pack_dir)],
                     f'pack/publish {kind}', args.pack_timeout)
        print(f'{kind.upper()} Python: {python}', flush=True)


if __name__ == '__main__':
    main()
