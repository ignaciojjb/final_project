#!/usr/bin/env bash
set -euo pipefail
cd /workspace/final_project
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements-ml.txt
.venv/bin/python -m unittest discover -s tests -v
