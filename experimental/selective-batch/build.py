"""Build an isolated scheduler that avoids computing unusable proposals."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'build'
BUILD = ROOT / 'build-distribution/upstream-media'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT / 'experimental/packed-batch/generate.cpp'
    identity = json.loads((base.parent / 'source-identities.json').read_text())['entry']
    assert sha(base) == identity['generated_sha256'], 'Packed parent changed'
    old = s = base.read_text()
    anchor = '    const bool packed_batch_mtp = batch_mtp && packed_env && packed_env[0] && packed_env[0] != \'0\';'
    s = replace(s, anchor, anchor + '\n'
        '    const char* selective_env = std::getenv("STRATA_BATCH_SELECTIVE_MTP");\n'
        '    const bool selective_batch_mtp = packed_batch_mtp && selective_env && selective_env[0] == \'1\';')
    s = replace(s, 'd->set_max_drafts(packed_batch_mtp ? std::clamp(8 / o.batch - 1, 1, o.spec - 1) : 1);',
        'd->set_max_drafts(selective_batch_mtp ? 1 : packed_batch_mtp ? std::clamp(8 / o.batch - 1, 1, o.spec - 1) : 1);')
    anchor = '                if (batch_mtp && sl.active) {\n                    const size_t stride = (size_t) g.hc * (size_t) g.n_embd;'
    s = replace(s, anchor, anchor + '\n'
        '                    if (selective_batch_mtp) {\n'
        '                        int live = 0, rank = 0;\n'
        '                        for (size_t i = 0; i < bs.size(); ++i) if (bs[i].active) {\n'
        '                            ++live; if ((int)i < b) ++rank;\n'
        '                        }\n'
        '                        // Every live session gets a target row. Only spare rows can verify proposals.\n'
        '                        const int proposal_slots = std::min(live, strata::kernels::kVerifyMaxT - live);\n'
        '                        if (rank >= proposal_slots) {\n'
        '                            // Maintain complete draft KV, omitting attention/expert/head proposal work.\n'
        '                            // A later load decrease can resume proposals without replay or truncation.\n'
        '                            if (!slot_mtp[(size_t)b]->prefill(ver.final_R_all() + (size_t)first[t] * stride,\n'
        '                                    outb + first[t], keep[b], pos[first[t]], err)) {\n'
        '                                std::printf("ERR selective batch KV maintenance: %s\\n", err.c_str());\n'
        '                                return false;\n'
        '                            }\n'
        '                            sl.draft_ready = false;\n'
        '                            sl.draft_n = 0;\n'
        '                            continue;\n'
        '                        }\n'
        '                    }')
    s = replace(s, 'slot_mtp[(size_t)b]->set_max_drafts(packed_batch_mtp ? std::clamp(8 / std::max(live_slots,1) - 1, 1, o.spec - 1) : 1);',
        'slot_mtp[(size_t)b]->set_max_drafts(selective_batch_mtp ? 1 : packed_batch_mtp ? std::clamp(8 / std::max(live_slots,1) - 1, 1, o.spec - 1) : 1);')
    assert s.count('packed_batch_mtp ? (float)o.spec_min_p : 0.0f') == 2
    s = s.replace('packed_batch_mtp ? (float)o.spec_min_p : 0.0f',
                  '(packed_batch_mtp && !selective_batch_mtp) ? (float)o.spec_min_p : 0.0f')
    source = OUT / 'generate.cpp'
    source.write_text(s)
    (HERE / 'source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), s.splitlines(True),
        fromfile='a/experimental/packed-batch/generate.cpp', tofile='b/experimental/packed-batch/generate.cpp')))
    flags = {}
    for line in (BUILD / 'CMakeFiles/strata.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            flags[k] = shlex.split(v)
    obj = OUT / 'generate.cpp.o'
    cmd = ['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
           '-I' + str(base.parent), '-c', str(source), '-o', str(obj)]
    subprocess.run(cmd, cwd=BUILD, check=True)
    link = [str(obj) if x.endswith('generate.cpp.o') else x
            for x in shlex.split((BUILD / 'CMakeFiles/strata.dir/link.txt').read_text())
            if not x.startswith('-Wl,--dependency-file=')]
    exe = OUT / 'strata-selective-batch'
    link[link.index('-o') + 1] = str(exe)
    subprocess.run(link, cwd=BUILD, check=True)
    (OUT / 'build.json').write_text(json.dumps(dict(base_sha256=sha(base), source_sha256=sha(source),
        binary_sha256=sha(exe), compile=cmd, link=link, cwd=str(BUILD),
        numerical_archives='Unchanged pinned upstream/media libraries',
        scope='Experimental scheduling; not a release or measured speed claim'), indent=2) + '\n')
    print(exe, flush=True)


if __name__ == '__main__':
    main()
