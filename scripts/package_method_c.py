"""Package isolated Method C sources, pinned dependency list and integrity manifest.

Aqua already has the training environment, data and checkpoints. Optionally
include the existing cp312 CPU wheelhouse for an entirely offline installation.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('mnist', 'fashion'), default='mnist')
    p.add_argument('--output', type=Path)
    p.add_argument('--include-wheels', action='store_true')
    a = p.parse_args()
    if a.output is None:
        a.output = root / f'method_c_{a.dataset}_code.tar.gz'
    runtime = 'method_c_fashion_runtime' if a.dataset == 'fashion' else 'method_c_runtime'
    entry = 'run_method_c_fashion.py' if a.dataset == 'fashion' else 'run_method_c.py'
    members = {
        entry: f'{runtime}/{entry}',
        'test_method_c.py': f'{runtime}/test_method_c.py',
        **{f'src/{name}': f'{runtime}/src/{name}' for name in
           ('__init__.py', 'method_c.py', 'method_a.py', 'fashion_data.py', 'network.py', 'encoding.py')},
        'requirements-aqua-cpu.txt': f'{runtime}/requirements-aqua-cpu.txt',
        f'jobs/{a.dataset}_method_c.cmd': f'jobs/{a.dataset}_method_c.cmd',
        f'METHOD_C_{a.dataset.upper()}.md': f'METHOD_C_{a.dataset.upper()}.md',
    }
    if a.dataset == 'fashion':
        members['test_method_c_fashion.py'] = f'{runtime}/test_method_c_fashion.py'
    if a.include_wheels:
        wheels = sorted((root / 'wheelhouse/cp312/cpu').glob('*.whl'))
        if not wheels:
            p.error('No cp312 CPU wheels found')
        members.update({str(path.relative_to(root)): runtime + '/wheelhouse/' + path.name for path in wheels})
    hashes = {}
    with tarfile.open(a.output, 'w:gz') as bundle:
        for source, target in members.items():
            content = (root / source).read_bytes()
            hashes[target] = hashlib.sha256(content).hexdigest()
            info = tarfile.TarInfo(target)
            info.size, info.mode = len(content), 0o644
            bundle.addfile(info, io.BytesIO(content))
        content = (json.dumps(hashes, indent=2, sort_keys=True) + '\n').encode()
        info = tarfile.TarInfo(f'{runtime}/bundle_sha256.json')
        info.size = len(content)
        bundle.addfile(info, io.BytesIO(content))
    print(f'Created {a.output} ({a.output.stat().st_size:,} bytes)')
    print('SHA256:', hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
