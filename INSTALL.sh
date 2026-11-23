#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.3.1
python -m pip install --only-binary=mmcv-full -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e vendor/Co-DETR
python tools/patch_mmcv.py
