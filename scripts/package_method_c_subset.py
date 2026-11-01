"""Package both CSV-defined Method C experiments in an isolated AQUA runtime."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=root / 'method_c_subset_aqua.tar.gz')
    p.add_argument('--include-wheels', action='store_true')
    a = p.parse_args()
    runtime = 'method_c_subset_runtime'
    names = ['run_method_c_subset.py', 'test_method_c_subset.py', 'test_method_c.py', 'test_method_a.py',
             'test_method_c_fashion.py', 'requirements-aqua-cpu.txt']
    names += [f'src/{n}' for n in ('__init__.py', 'method_c_subset.py', 'method_c.py',
                                  'method_a.py', 'fashion_data.py', 'network.py', 'encoding.py')]
    members = {name: f'{runtime}/{name}' for name in names}
    for dataset in ('mnist', 'fashion'):
        for name in (f'forget_set_{dataset}_sub6_9_p10.csv', f'jobs/{dataset}_method_c_subset.cmd'):
            members[name] = f'{runtime}/{name}'
    for name in ('METHOD_C_SUBSET.md', 'jobs/aqua_common.sh'):
        members[name] = f'{runtime}/{name}'
    if a.include_wheels:
        wheels = sorted((root / 'wheelhouse/cp312/cpu').glob('*.whl'))
        if not wheels:
            p.error('No cp312 CPU wheels found')
        members.update({str(f.relative_to(root)): f'{runtime}/wheelhouse/{f.name}' for f in wheels})
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
        info.size, info.mode = len(content), 0o644
        bundle.addfile(info, io.BytesIO(content))
    print(f'Created {a.output} ({a.output.stat().st_size:,} bytes)')
    print('SHA256:', hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
