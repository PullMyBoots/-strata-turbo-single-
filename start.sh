#!/usr/bin/env bash
set -Eeuo pipefail
strata_root="$(cd -- "$(dirname -- "$0")" && pwd)"
strata_python="${STRATA_PYTHON:-$strata_root/.venv/bin/python}"
if [[ "${1:-}" == --legacy ]]; then
  shift
  exec "$strata_python" "$strata_root/tools/single_gpu.py" "$@"
fi
if [[ -f "$strata_root/runtime/install.json" ]]; then
  exec "$strata_python" "$strata_root/setup.py" --start "$@"
fi
if [[ -f "$strata_root/distribution/build.json" ]]; then
  exec "$strata_root/setup.sh" "$@"
fi
if [[ ! -x "$strata_python" ]]; then
  printf '%s\n' 'Python environment missing. Run ./setup.sh or set STRATA_PYTHON.' >&2
  exit 1
fi
exec "$strata_python" "$strata_root/tools/single_gpu.py" "$@"
