#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --cpu-wheels ) ]]; then
    echo "Usage: bash scripts/setup_cpu.sh [--cpu-wheels]" >&2
    exit 2
fi
# Reuse the platform's dependencies by default.
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements/cpu.txt
if [[ ${1:-} == --cpu-wheels ]]; then
    .venv/bin/python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.8.0+cpu' 'torchvision==0.23.0+cpu'
fi
.venv/bin/python -c 'import torch; print("Torch:", torch.__version__)'
.venv/bin/python -m pip freeze > .venv/installed.txt
printf 'Activate with: source .venv/bin/activate\n'
