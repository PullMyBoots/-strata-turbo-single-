"""Isolated source/object overlay, linking immutable released numerical archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from materialize import materialize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'build'
BUILD = ROOT / 'build-distribution/turbo'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flags(target):
    result = {}
    for line in (BUILD / f'CMakeFiles/{target}.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            result[key] = shlex.split(value)
    return result


def build(parity=False):
    sources = materialize()
    commands = []
    overrides = []
    include = ['-I' + str(OUT / 'include'), '-I' + str(HERE)]
    if parity:
        entry = HERE / 'parity.cpp'
    else:
        entry = OUT / 'generate.cpp'
        vf = flags('strata_engine')
        obj = OUT / 'verify.cpp.o'
        commands.append(['/usr/bin/c++', *include, *vf['CXX_DEFINES'], *vf['CXX_INCLUDES'],
                         *vf['CXX_FLAGS'], '-c', str(OUT / 'verify.cpp'), '-o', str(obj)])
        overrides.append(str(obj))
    ef = flags('strata')
    entry_obj = OUT / ('parity.cpp.o' if parity else 'generate.cpp.o')
    commands.append(['/usr/bin/c++', *include, *ef['CXX_DEFINES'], *ef['CXX_INCLUDES'],
                     *ef['CXX_FLAGS'], '-c', str(entry), '-o', str(entry_obj)])
    cuda = flags('strata_engine')
    obj = OUT / 'resident_dispatch.cu.o'
    commands.append(['/usr/local/cuda/bin/nvcc', *include, *cuda['CUDA_DEFINES'],
                     *cuda['CUDA_FLAGS'], '-c', str(HERE / 'resident_dispatch.cu'), '-o', str(obj)])
    overrides.append(str(obj))
    for command in commands:
        subprocess.run(command, cwd=BUILD, check=True)
    link = shlex.split((BUILD / 'CMakeFiles/strata.dir/link.txt').read_text())
    link = [str(entry_obj) if x.endswith('generate.cpp.o') else x for x in link
            if not x.startswith('-Wl,--dependency-file=')]
    binary = OUT / ('resident-parity' if parity else 'strata-resident-bypass')
    link[link.index('-o') + 1] = str(binary)
    at = link.index('libstrata_engine.a')
    link[at:at] = overrides
    subprocess.run(link, cwd=BUILD, check=True)
    archives = {x: sha(BUILD / x) for x in dict.fromkeys(link)
                if x.endswith('.a') and not Path(x).is_absolute()}
    manifest = dict(sources=sources, commands=commands, link=link, link_cwd=str(BUILD),
                    binary_sha256=sha(binary), unchanged_archive_sha256=archives,
                    conditional_publish_sha256=sha(HERE / 'resident_dispatch.cu'),
                    device_policy='actual routing/residency; no device-name or timing selection',
                    target_math='unchanged archives and resident_plan', screening_only=True)
    (OUT / (binary.name + '-build.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    print(str(binary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity', action='store_true')
    args = parser.parse_args()
    build(args.parity)
