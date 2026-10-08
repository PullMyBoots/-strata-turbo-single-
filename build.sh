#!/usr/bin/env bash
set -Eeuo pipefail
strata_root="$(cd -- "$(dirname -- "$0")" && pwd)"
strata_build="${STRATA_BUILD_DIR:-$strata_root/build-single}"
cmake -S "$strata_root" -B "$strata_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_CUDA_COMPILER="${CUDACXX:-nvcc}" \
  -DCMAKE_CUDA_ARCHITECTURES="${STRATA_CUDA_ARCH:-89}" \
  -DSTRATA_GGML_DIR="$strata_root/third_party/llama.cpp" \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON \
  -DSTRATA_PORTABLE="${STRATA_PORTABLE:-OFF}"
cmake --build "$strata_build" --target strata conversation_cache_test conversation_memory_test \
  -j "${STRATA_BUILD_JOBS:-12}"
install -m 755 "$strata_build/strata" "$strata_root/engine/strata-single-built.next"
mv -f "$strata_root/engine/strata-single-built.next" "$strata_root/engine/strata-single-built"
python3 "$strata_root/tools/release_manifest.py" --built
