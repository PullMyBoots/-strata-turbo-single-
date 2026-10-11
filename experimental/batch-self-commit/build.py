"""Build an isolated singleton-batch commit candidate from pinned upstream."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / 'build-distribution/upstream-media'
OUT = HERE / 'build'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(s, old, new):
    assert s.count(old) == 1, (old[:120], s.count(old))
    return s.replace(old, new)


def flags(target):
    result = {}
    for line in (BUILD / f'CMakeFiles/{target}.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            result[k] = shlex.split(v)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'distribution/upstream/src/core/verify.cpp'
    original = source.read_text()
    s = replace(original, 'bool mapped(size_t bytes, void** h, void** d) {', '''// Every distinct slot contributes one mandatory token. Do not use this for
// repeated slots (speculative prefixes) or partial-model pipeline stages.
bool batch_singletons(const int* rows, int n) {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_BATCH_SELF_COMMIT");
        return v && v[0] == '1';
    }();
    if (!enabled || !one_token_self_commit() || n < 1) return false;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (rows[i] == rows[j]) return false;
    return true;
}

bool mapped(size_t bytes, void** h, void** d) {''')
    old = '    const bool self_commit = T == 1 && !batch_rec_ && !g_qfuse() && one_token_self_commit();   // see Verifier::commit'
    s = replace(s, old, old + '\n    const bool batch_advance = batch_rec_ && lb_ == 0 && le_ == g.n_layers && batch_singletons(brow_, T);')
    s = replace(s, 'hb + (size_t) first * C, (int) C, (int) (2 * HK), EPS, t - first, cs, 0);',
                   'hb + (size_t) first * C, (int) C, (int) (2 * HK), EPS, t - first, cs, 0, batch_advance);')
    s = replace(s, '(int) HK, (int) HV, t - first, nullptr, cs, 0);',
                   '(int) HK, (int) HV, t - first, batch_advance ? one_ : nullptr, cs, 0);')
    s = replace(s, '                copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);',
                   '                if (!batch_advance) copy_from_mapped(hist_snap_ + (size_t) t * HS, hist, HS, cs);')
    s = replace(s, '                        if (t == tb || brow_[t] != brow_[t - 1])\n                            copy_from_mapped(tail_snap_b_',
                   '                        if (!batch_advance && (t == tb || brow_[t] != brow_[t - 1]))\n                            copy_from_mapped(tail_snap_b_')
    old = '''    const cudaError_t le = cudaGraphLaunch(commit_bm_[bkey(last_rows_, S, row_base_)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch commit launch: ") + cudaGetErrorString(le); return false; }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }'''
    new = '''    // run_slot_rows has already synchronized the forward graph before sampling.
    // For mandatory singleton rows that graph also committed conv/GDN/QSA/PLE.
    const bool advanced = lb_ == 0 && le_ == g_->n_layers && batch_singletons(last_rows_, S);
    if (!advanced) {
        const cudaError_t le = cudaGraphLaunch(commit_bm_[bkey(last_rows_, S, row_base_)], cs_);
        if (le != cudaSuccess) { err = std::string("verify: batch commit launch: ") + cudaGetErrorString(le); return false; }
        const cudaError_t se = cudaStreamSynchronize(cs_);
        if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }
    }'''
    s = replace(s, old, new)
    generated = OUT / 'verify.cpp'
    generated.write_text(s)
    (HERE / 'source.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), s.splitlines(True),
                                                       fromfile='a/src/core/verify.cpp', tofile='b/src/core/verify.cpp')))
    vf = flags('strata_engine')
    obj = OUT / 'verify.cpp.o'
    cmd = ['/usr/bin/c++', *vf['CXX_DEFINES'], *vf['CXX_INCLUDES'], *vf['CXX_FLAGS'], '-c', str(generated), '-o', str(obj)]
    subprocess.run(cmd, cwd=BUILD, check=True)
    link = [x for x in shlex.split((BUILD / 'CMakeFiles/strata.dir/link.txt').read_text()) if not x.startswith('-Wl,--dependency-file=')]
    entry = ROOT / 'experimental/native-batch/source/generate.cpp.o'

    if not entry.is_file():
        # Fresh distribution builds keep this object in CMake's build tree.
        objects = [BUILD / x for x in link if x.endswith('generate.cpp.o')]
        if len(objects) != 1 or not objects[0].is_file():
            raise RuntimeError('Build the upstream media target with tools/build_distribution.py first')
        entry = objects[0]
    link = [str(entry) if x.endswith('generate.cpp.o') else x for x in link]
    binary = OUT / 'strata-batch-self-commit'
    link[link.index('-o') + 1] = str(binary)
    link.insert(link.index('libstrata_engine.a'), str(obj))
    subprocess.run(link, cwd=BUILD, check=True)
    record = dict(source_sha256=sha(source), generated_source_sha256=sha(generated), entry_sha256=sha(entry),
                  binary_sha256=sha(binary), compile=cmd, link=link, cwd=str(BUILD),
                  archives={x: sha(BUILD / x) for x in set(link) if x.endswith('.a') and not Path(x).is_absolute()})
    (OUT / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(binary, flush=True)


if __name__ == '__main__':
    main()
