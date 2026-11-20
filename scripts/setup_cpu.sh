#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse the platform's torch without downloading CUDA wheels again.
python -m venv --system-site-packages .venv
.venv/bin/python -c 'import torch; print("Platform torch:", torch.__version__)'
.venv/bin/python -m pip install -r requirements/cpu.txt
.venv/bin/python -m pip freeze > .venv/installed.txt
printf 'Activate with: source .venv/bin/activate\n'
