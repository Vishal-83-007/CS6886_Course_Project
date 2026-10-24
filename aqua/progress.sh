#!/bin/bash
# Show how far each run (MNIST and Fashion-MNIST) has got. Run from the repository root:
#   bash aqua/progress.sh
# Cython runs: read from the checkpoint saved every 15 min.
# Compiled CPU / GPU runs: read from the run's log (updated every 100 images).
cd "$(dirname "$0")/.." || exit 1
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh 2>/dev/null; fi
module load anaconda3_2024.10 >/dev/null 2>&1
python - <<'EOF'
import json, glob, os, re
import numpy as np
runs = sorted(d for d in glob.glob('results/*/*') + glob.glob('results/*/*/*')
            if os.path.isfile(os.path.join(d, 'config.json')))
if not runs:
    print('No runs yet under results/')
for run in runs:
    name = os.path.relpath(run, 'results')
    backend = json.load(open(f'{run}/config.json')).get('backend', '?')
    status = None
    if os.path.exists(f'{run}/status.json'):
        status = json.load(open(f'{run}/status.json')).get('status')
    done, total = 0, None
    try:
        total = len(np.load(f'{run}/schedule.npz')['ids'])
    except (OSError, KeyError, ValueError):
        pass
    try:  # Cython runs: last published checkpoint
        current = json.load(open(f'{run}/checkpoints/CURRENT.json'))['generation']
        done = json.load(open(f'{run}/checkpoints/{current}/progress.json'))['position']
    except (OSError, KeyError, ValueError):
        pass
    if os.path.exists(f'{run}.log'):  # compiled CPU / GPU runs: last progress line
        hits = re.findall(r'progress=(\d+)/', open(f'{run}.log', errors='replace').read())
        if hits:
            done = max(done, int(hits[-1]))
    if status == 'complete' and total:
        done = total
    if status in (None, 'building'):
        status = 'compiling / starting' if done == 0 else 'running'
    elif (status == 'checkpointed' and os.path.exists(f'{run}/checkpoints/CURRENT.json')
          and os.path.getmtime(f'{run}/checkpoints/CURRENT.json') > os.path.getmtime(f'{run}/status.json')):
        status = 'running (resumed)'
    line = f'{name}   [{backend}]\n    status: {status}'
    if total:
        line += f'   images done: {done}/{total} ({100 * done / total:.1f}%)'
    m = json.load(open(f'{run}/metrics.json')) if os.path.exists(f'{run}/metrics.json') else None
    if m is not None and m.get('mode') == 'importance':
        line += f"\n    importance from {m['n_images']} training images, {m['spikes_per_image']:.1f} spikes/image"
    elif m is not None and m.get('mode') == 'evaluate':
        for name, r in m['readouts'].items():
            line += f"\n    {name}: {100 * r['accuracy']:.2f}% on {m['n_evaluation']} {m['eval_split']} images"
            if 'retain_accuracy' in r:
                line += f"   retained {100 * r['retain_accuracy']:.2f}%   removed {100 * r['forget_accuracy']:.2f}%"
    elif m is not None:
        line += f"\n    ACCURACY: {100 * m['accuracy']:.2f}% on {m['n_evaluation']} {m['evaluation_split']} images"
        if m.get('forget_class_accuracy') is not None:
            line += (f"\n    retained classes: {100 * m['retain_accuracy']:.2f}%   "
                     f"removed class(es): {100 * m['forget_class_accuracy']:.2f}%")
    elif status == 'failed' or (os.path.exists(f'{run}.log') and 'Traceback' in open(f'{run}.log', errors='replace').read()):
        line += f'\n    ERROR: see {run}.log'
    print(line)
EOF
