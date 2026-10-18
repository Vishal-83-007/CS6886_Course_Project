"""Run locally with internet; download Linux/glibc-2.17 wheels for offline Aqua."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ['pip==24.3.1', 'wheel==0.45.1']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-version', default='3.12', choices=('3.10', '3.11', '3.12'))
    parser.add_argument('--output', default=str(ROOT / 'wheelhouse'))
    args = parser.parse_args()
    tag = 'cp' + args.python_version.replace('.', '')
    destination = Path(args.output) / tag
    for kind in ('cpu', 'gpu'):
        directory = destination / kind
        directory.mkdir(parents=True, exist_ok=True)
        # Remove only the readiness marker before a new download. Old wheel files
        # can be reused; a failed download must not leave a stale ready marker.
        marker = directory / 'READY.json'
        if marker.exists():
            marker.unlink()
        requirement = ROOT / f'requirements-aqua-{kind}.txt'
        command = [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                   '--python-version', args.python_version, '--implementation', 'cp',
                   '--abi', tag, '--dest', str(directory), '-r', str(requirement), *BOOTSTRAP]
        for platform in ('manylinux2014_x86_64', 'manylinux_2_17_x86_64',
                         'manylinux2010_x86_64', 'manylinux_2_12_x86_64',
                         'manylinux1_x86_64', 'manylinux_2_5_x86_64'):
            command.extend(['--platform', platform])
        subprocess.run(command, check=True)
        manifest = dict(python=args.python_version, platform='linux-x86_64-glibc>=2.17',
                        requirement_sha256=sha256(requirement),
                        wheels={p.name: sha256(p) for p in sorted(directory.glob('*.whl'))})
        temporary = directory / 'READY.tmp'
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        temporary.replace(marker)
        print(f'Ready: {directory}', flush=True)
    print(f'Transfer {destination} and the updated scripts/requirements to Aqua.')


if __name__ == '__main__':
    main()
