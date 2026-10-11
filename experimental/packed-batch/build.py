"""Rebuild isolated scheduling candidates; never replace installed engines."""
import argparse
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / 'build-distribution/upstream-media'
SOURCE = ROOT / 'experimental/native-batch/source/generate.cpp'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(name):
    identity = json.loads((HERE / 'source-identities.json').read_text())[name]
    base = ROOT / identity['base']
    if sha(base) != identity['base_sha256']:
        raise RuntimeError('Pinned source identity changed: ' + str(base))
    target = HERE / identity.get('generated_file', 'generate.cpp' if name == 'entry' else 'verify.cpp')
    subprocess.run(['patch', '--silent', '--output', str(target), str(base),
                    str(HERE / identity['patch'])], check=True)
    if sha(target) != identity['generated_sha256']:
        raise RuntimeError('Generated source identity mismatch: ' + str(target))
    return target

def flags_for(name):
    flags = {}
    for line in (BUILD / f'CMakeFiles/{name}.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            name, value = line.split(' = ', 1)
            flags[name] = shlex.split(value)
    return flags

def build(gdn=False, parity=False, nvcc='/usr/local/cuda/bin/nvcc', arch='89', adaptive=False):
    if parity and not gdn:
        raise ValueError('--parity requires --gdn')
    if parity and adaptive:
        raise ValueError('--parity cannot be combined with --adaptive')
    entry = HERE / 'gdn_parity.cpp' if parity else materialize('adaptive' if adaptive else 'entry')
    flags = flags_for('strata')
    command = ['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'],
               *flags['CXX_FLAGS'], '-I' + str(HERE), '-c', str(entry),
               '-o', str(entry) + '.o']
    subprocess.run(command, check=True, cwd=BUILD)
    extras = []
    if gdn:
        if not parity:
            verify = materialize('verify')
            vf=flags_for('strata_engine')
            subprocess.run(['/usr/bin/c++',*vf['CXX_DEFINES'],*vf['CXX_INCLUDES'],*vf['CXX_FLAGS'],
                            '-I'+str(HERE),'-c',str(verify),'-o',str(HERE/'verify.cpp.o')],
                           check=True,cwd=BUILD)
        subprocess.run([nvcc,'-O3','-DNDEBUG','-std=c++20',
                        f'--generate-code=arch=compute_{arch},code=[compute_{arch},sm_{arch}]',
                        '-I'+str(HERE),'-c',str(HERE/'packed_gdn.cu'),'-o',str(HERE/'packed_gdn.cu.o')],
                       check=True,cwd=BUILD)
        extras=([] if parity else [str(HERE/'verify.cpp.o')])+[str(HERE/'packed_gdn.cu.o')]
    link = shlex.split((BUILD / 'CMakeFiles/strata.dir/link.txt').read_text())
    link = [str(entry)+'.o' if x.endswith('generate.cpp.o') else x for x in link]
    binary=HERE/('gdn-parity' if parity else 'strata-adaptive-gdn' if adaptive and gdn else
                 'strata-adaptive' if adaptive else 'strata-gdn' if gdn else 'strata-packed')
    link[link.index('-o')+1] = str(binary)
    at=link.index('libstrata_engine.a');link[at:at]=extras
    link = [x for x in link if not x.startswith('-Wl,--dependency-file=')]
    subprocess.run(link, check=True, cwd=BUILD)
    manifest = dict(base_entry_sha256=sha(SOURCE), candidate_entry_sha256=sha(entry),
                    binary_sha256=sha(binary), compile=command, link=link,
                    link_cwd=str(BUILD), screening_only=True,
                    cuda_architecture=arch,
                    numerical_libraries='pinned libraries unchanged; grouped GDN override linked' if gdn and not parity else 'unchanged pinned upstream portable libraries')
    (HERE/(binary.name+'-build.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gdn',action='store_true')
    parser.add_argument('--adaptive',action='store_true')
    parser.add_argument('--parity',action='store_true')
    parser.add_argument('--nvcc',default='/usr/local/cuda/bin/nvcc')
    parser.add_argument('--arch',default='89')
    args=parser.parse_args()
    build(args.gdn,args.parity,args.nvcc,args.arch,args.adaptive)
