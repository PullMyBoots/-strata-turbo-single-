"""Build the rejected operator prototype reproducibly without changing the engine."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE / 'probe.cu')
    p.add_argument('--arch', default='sm_89')
    args = p.parse_args()
    source = args.source.resolve()
    out = HERE / 'build'
    out.mkdir(exist_ok=True)
    baseline = ROOT / 'build-distribution/turbo/CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_bf16.cu.o'
    binary = out / 'probe'
    cmd = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-arch=' + args.arch,
           '-I' + str(ROOT / 'include'), str(source), str(baseline), '-lcublas', '-o', str(binary)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (out / 'build.log').write_text(result.stdout + result.stderr)
    record = dict(command=cmd, exit_code=result.returncode, source_sha256=digest(source),
                  baseline_object_sha256=digest(baseline),
                  limitation='The baseline object must already support the requested architecture; only SM89 was tested.')
    if result.returncode == 0:
        record['binary_sha256'] = digest(binary)
    (out / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(result.stdout + result.stderr, end='')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
