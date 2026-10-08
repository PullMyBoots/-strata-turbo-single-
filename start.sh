#!/usr/bin/env bash
set -Eeuo pipefail
strata_root="$(cd -- "$(dirname -- "$0")" && pwd)"
strata_python="${STRATA_PYTHON:-$strata_root/.venv/bin/python}"
if [[ ! -x "$strata_python" ]]; then
  printf '%s\n' 'Python environment missing. Run ./install-python.sh or set STRATA_PYTHON.' >&2
  exit 1
fi
exec "$strata_python" "$strata_root/tools/single_gpu.py" "$@"
