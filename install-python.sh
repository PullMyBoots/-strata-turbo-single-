#!/usr/bin/env bash
set -Eeuo pipefail
strata_root="$(cd -- "$(dirname -- "$0")" && pwd)"
python3 -m venv "$strata_root/.venv"
"$strata_root/.venv/bin/python" -m pip install -r "$strata_root/requirements.txt"
