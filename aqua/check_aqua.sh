#!/bin/bash
# Reports EVERYTHING missing in one pass instead of stopping at the first error.
#   Login node  :  bash aqua/check_aqua.sh 2>&1 | tee aqua_check_login.log
#   Compute node:  qsub aqua/check_aqua.pbs   (also tries a tiny MNIST run)
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -n "${PBS_JOBID:-}" ]; then WHERE=compute; else WHERE=login; fi
section() { echo; echo "===== $*"; }

section "Host ($WHERE node)"
hostname; date
echo "glibc: $(getconf GNU_LIBC_VERSION 2>&1)"
echo "gcc:   $(gcc --version 2>&1 | head -1)"
echo "bash:  $BASH_VERSION"
echo "PBS queue: ${PBS_QUEUE:-n/a}"

section "Default Python (before loading any module)"
python --version 2>&1
python3 --version 2>&1

section "Load anaconda3_2024.10 module"
if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || echo "module command not available"
fi
module load anaconda3_2024.10 2>&1
unset PYTHONHOME PYTHONPATH
echo "python now: $(command -v python)"
python --version 2>&1

# Check the module Python, and the project venv too if it already exists.
pythons=(python)
[ -x "$HOME/fashion-cpu/bin/python" ] && pythons+=("$HOME/fashion-cpu/bin/python")
for py in "${pythons[@]}"; do
section "Python packages in: $py"
"$py" - <<'EOF'
import importlib
pkgs = [  # (import name, needed for)
    ('brian2', 'all training'), ('numpy', 'all'), ('Cython', 'brian2 compiled backend'),
    ('scipy', 'brian2'), ('sympy', 'brian2'), ('jinja2', 'brian2'),
    ('pyparsing', 'brian2'), ('setuptools', 'brian2 compile step'),
    ('matplotlib', 'src/utils.py imports it'), ('tqdm', 'older run_baseline.py path'),
]
missing = []
for name, why in pkgs:
    try:
        m = importlib.import_module(name)
        print(f"  OK       {name:<11} {getattr(m, '__version__', '?'):<12} ({why})")
    except Exception as e:
        print(f"  MISSING  {name:<11} {type(e).__name__}: {e}  ({why})")
        missing.append(name)
print("SUMMARY missing:", ", ".join(missing) if missing else "none")
EOF
"$py" -m pip --version 2>&1
done

section "Internet access from this node (10 s timeout each)"
for url in https://pypi.org/simple/brian2/ https://files.pythonhosted.org https://conda.anaconda.org; do
  code=$(curl -sS -m 10 -o /dev/null -w '%{http_code}' "$url" 2>&1)
  echo "  $url -> $code"
done
if [ "$WHERE" = login ]; then
  tmp=$(mktemp -d)
  echo "  pip download test:"
  timeout 120 python -m pip download brian2==2.9.0 --no-deps -d "$tmp" -q 2>&1 | tail -3
  ls "$tmp"/*.whl >/dev/null 2>&1 && echo "  pip CAN download packages here" || echo "  pip CANNOT download packages here"
  rm -rf "$tmp"
fi

section "MNIST data (bundled in data/mnist; checks the official checksums)"
python prepare_fashion.py --dataset mnist 2>&1 | tail -3
echo "wheelhouse/:"; ls wheelhouse 2>&1
echo "Existing envs:"; ls -d ~/fashion-cpu ~/fashion-gpu 2>&1

if [ "$WHERE" = compute ]; then
  py=""
  for candidate in "${pythons[@]}"; do
    "$candidate" -c 'import brian2' 2>/dev/null && py="$candidate"
  done
  section "Try the code: tiny MNIST run (400 neurons, 100 train images)"
  if [ -n "$py" ]; then
    out=$(mktemp -d)/run
    export MPLBACKEND=Agg OMP_NUM_THREADS=1
    timeout 1500 "$py" -u run_fashion.py --dataset mnist --n-excitatory 400 --epochs 1 \
        --train-limit 100 --assign-per-class 2 --eval-limit 20 --out-dir "$out" 2>&1 | tail -40
    rm -rf "$(dirname "$out")"
  else
    echo "Skipped: no Python here has brian2 installed, so the code cannot start yet."
  fi
fi
section "Done ($WHERE node)"
