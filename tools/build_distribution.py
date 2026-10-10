"""Build portable CPU binaries for the selected NVIDIA architecture.

Builds the optimized engine, an unmodified pinned upstream comparator, the
upstream engine with our native media reader, and the image/video encoder.
No installed historical binary is overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from distribution import ROOT, digest, save, run


def configure(source, build, arch, nvcc, extra=()):
    run(['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
         '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_ENABLE_HIP=OFF',
         '-DCMAKE_CUDA_COMPILER=' + nvcc, '-DCMAKE_CUDA_ARCHITECTURES=' + arch,
         '-DSTRATA_GGML_DIR=' + str(ROOT / 'third_party/llama.cpp'),
         '-DSTRATA_PORTABLE=ON', '-DSTRATA_BUILD_TESTS=OFF', *extra])


def copy_binary(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    next_file = target.with_suffix('.next')
    shutil.copy2(source, next_file)
    next_file.chmod(0o755)
    next_file.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arch', default='89', help='Semicolon-separated compute capabilities, e.g. 80;86;89.')
    p.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument('--nvcc', default=os.environ.get('CUDACXX') or shutil.which('nvcc'))
    a = p.parse_args()
    if not a.nvcc:
        p.error('CUDA Toolkit is required for this architecture. Install it and rerun; the NVIDIA driver alone cannot compile source.')
    architectures = [int(x) for x in a.arch.split(';')]
    if not architectures or min(architectures) < 75 or a.jobs < 1:
        p.error('Use CUDA compute capability 7.5+ and a positive job count.')
    root = ROOT / 'build-distribution'
    root.mkdir(exist_ok=True)
    binary_dir = ROOT / 'engine/distribution'
    results = {}
    custom = root / 'turbo'
    configure(ROOT, custom, a.arch, a.nvcc)
    run(['cmake', '--build', custom, '--target', 'strata', '-j', a.jobs])
    copy_binary(custom / 'strata', binary_dir / 'strata-turbo')
    results['turbo'] = binary_dir / 'strata-turbo'

    source = ROOT / 'distribution/upstream'
    baseline = root / 'upstream'
    configure(source, baseline, a.arch, a.nvcc)
    run(['cmake', '--build', baseline, '--target', 'strata', '-j', a.jobs])
    copy_binary(baseline / 'strata', binary_dir / 'strata-official')
    results['official'] = binary_dir / 'strata-official'

    # A generated CMake wrapper selects the existing parser port. Numerical
    # source files and compile flags remain those of the pinned upstream tree.
    media_source = root / 'upstream-media-source'
    media_source.mkdir(exist_ok=True)
    for name in ('src', 'include', 'cmake', 'third_party', 'tests'):
        target = media_source / name
        if not target.exists():
            target.symlink_to(source / name, target_is_directory=True)
    entry = ROOT / 'experimental/native-batch/source/generate.cpp'
    include = ROOT / 'experimental/native-batch/source/include'
    cmake = (source / 'CMakeLists.txt').read_text(encoding='utf-8-sig')
    cmake += ('\nset_source_files_properties(src/program/generate.cpp PROPERTIES HEADER_FILE_ONLY TRUE)\n'
              'target_sources(strata PRIVATE "' + str(entry) + '")\n'
              'target_include_directories(strata PRIVATE "' + str(include) + '")\n')
    (media_source / 'CMakeLists.txt').write_text(cmake)
    media_build = root / 'upstream-media'
    configure(media_source, media_build, a.arch, a.nvcc)
    run(['cmake', '--build', media_build, '--target', 'strata', '-j', a.jobs])
    copy_binary(media_build / 'strata', binary_dir / 'strata-upstream-media')
    results['official_media'] = binary_dir / 'strata-upstream-media'

    vision = root / 'vision'
    run(['cmake', '-S', ROOT / 'tools/vision', '-B', vision, '-DCMAKE_BUILD_TYPE=Release',
         '-DSTRATA_VISION_CUDA=ON', '-DSTRATA_PORTABLE=ON',
         '-DCMAKE_CUDA_COMPILER=' + a.nvcc, '-DCMAKE_CUDA_ARCHITECTURES=' + a.arch,
         '-DLLAMA_DIR=' + str(ROOT / 'third_party/llama.cpp')])
    run(['cmake', '--build', vision, '--target', 'strata-vision', '-j', a.jobs])
    copy_binary(vision / 'bin/strata-vision', binary_dir / 'strata-vision')
    results['vision'] = binary_dir / 'strata-vision'
    version = subprocess.check_output([a.nvcc, '--version'], text=True)
    required_glibc = []
    for binary in results.values():
        symbols = subprocess.check_output(['objdump', '-T', str(binary)], text=True)
        required_glibc += [tuple(map(int, v.split('.'))) for v in re.findall(r'\(GLIBC_([0-9.]+)\)', symbols)]
    manifest = {'schema': 1, 'portable_cpu': 'AVX2; AVX512 kernels dispatched at runtime',
                'minimum_glibc': '.'.join(map(str, max(required_glibc))),
                'cuda_architectures': architectures, 'cuda_compiler': version,
                'upstream_commit': 'fb58e0dbc8399662c0e47c76578c6e878b14f6cf',
                'media_adapter': 'SVE2 reader only; upstream numerical kernels unchanged',
                'media_verification': 'Optimized single-session GENI requests use one-token target verification; Swift restores its original expert profile for media and retains adaptive text caching.',
                'optimized_source_files_sha256': {str(p.relative_to(ROOT)): digest(p)
                    for directory in ('src', 'include', 'cmake') for p in sorted((ROOT / directory).rglob('*')) if p.is_file()},
                'binaries': {name: {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                                    'sha256': digest(path)} for name, path in results.items()},
                'performance': 'unqualified until tested on the deployment hardware'}
    manifest['optimized_source_files_sha256']['CMakeLists.txt'] = digest(ROOT / 'CMakeLists.txt')
    save(ROOT / 'distribution/build.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
