"""Bundle an isolated Method A runtime without replacing training code.

Run from any directory. Transfer the archive, then extract from Aqua's project
root. Only the selected Method A job/docs and isolated runtime are installed. Models,
datasets, existing src/ modules and the environment helper are not included.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('mnist', 'fashion'), default='mnist')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = root / f'method_a_{args.dataset}_code.tar.gz'
    runtime = 'method_a_fashion_runtime' if args.dataset == 'fashion' else 'method_a_runtime'
    entry = 'run_method_a_fashion.py' if args.dataset == 'fashion' else 'run_method_a.py'
    members = {
        entry: f'{runtime}/{entry}',
        'test_method_a.py': f'{runtime}/test_method_a.py',
        **{f'src/{name}': f'{runtime}/src/{name}' for name in
           ('__init__.py', 'method_a.py', 'fashion_data.py', 'network.py', 'encoding.py')},
        f'jobs/{args.dataset}_method_a.cmd': f'jobs/{args.dataset}_method_a.cmd',
        f'METHOD_A_{args.dataset.upper()}.md': f'METHOD_A_{args.dataset.upper()}.md',
    }
    if args.dataset == 'fashion':
        members['test_method_a_fashion.py'] = f'{runtime}/test_method_a_fashion.py'
    hashes = {}
    with tarfile.open(args.output, 'w:gz') as bundle:
        for source, target in members.items():
            content = (root / source).read_bytes()
            hashes[target] = hashlib.sha256(content).hexdigest()
            member = tarfile.TarInfo(target)
            member.size = len(content)
            member.mode = 0o644
            bundle.addfile(member, io.BytesIO(content))
        content = (json.dumps(hashes, indent=2, sort_keys=True) + '\n').encode()
        member = tarfile.TarInfo(f'{runtime}/bundle_sha256.json')
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    print(f'Created {args.output} ({args.output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
