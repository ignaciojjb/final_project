#!/usr/bin/env bash
set -euo pipefail
# Cloud-only checkout location; local users should follow docs/SETUP.md.
cd /workspace/final_project
# Reuse packages provided by the cloud image; no data download or model fit occurs here.
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements-ml.txt
# Offline synthetic checks validate installation independently of SDSS/CDS connectivity.
.venv/bin/python -m unittest discover -s tests -v
