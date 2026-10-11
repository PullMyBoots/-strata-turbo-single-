"""Build an isolated batched sparse-drafter prefill candidate."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'build'
BUILD = ROOT / 'build-distribution/turbo'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def replace(s, old, new):
    assert s.count(old) == 1, (old[:100], s.count(old))
    return s.replace(old, new)


def main():
    (OUT / 'include/strata/core').mkdir(parents=True, exist_ok=True)
    header = ROOT / 'include/strata/core/mtp.hpp'
    source = ROOT / 'src/prefill/prefill.cpp'
    if 'STRATA_MTP_SPARSE_BATCH' in source.read_text():
        if 'finish_batched_index' not in header.read_text():
            raise RuntimeError('Sparse draft source/header integration is incomplete')
        (OUT / 'include/strata/core/mtp.hpp').write_text(header.read_text())
        (OUT / 'prefill.cpp').write_text(source.read_text())
    else:
        h = header.read_text()
        h = replace(h, '    bool idle(std::string& err) {',
            '    bool finish_batched_index(cudaStream_t stream, std::string& err) {\n'
            '        return index_checkpoint(true, stream, err);\n'
            '    }\n    bool idle(std::string& err) {')
        (OUT / 'include/strata/core/mtp.hpp').write_text(h)
        s = source.read_text()
        s = replace(s, '    if (off || n <= 0 || m.g == nullptr || m.region == nullptr || (st.kv_mode != 0 && !(st.kv_mode == 2 && ring_ok)) ||',
            '    static const bool sparse_batch = [] { const char* v = std::getenv("STRATA_MTP_SPARSE_BATCH"); return v && v[0] == \'1\'; }();\n'
            '    const bool sparse = st.auxiliary_indexer;\n'
            '    if ((sparse && !sparse_batch) || off || n <= 0 || m.g == nullptr || m.region == nullptr ||\n'
            '        (st.kv_mode != 0 && !(st.kv_mode == 2 && ring_ok) && !(sparse && st.kv_mode == 1)) ||')
        old = '    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn) return false;'
        s = replace(s, old, old + '\n'
            '    const void* w_idx = sparse ? mtp.tensor_q8("self_attn.indexer.index_qk_proj.weight") : nullptr;\n'
            '    const float* w_idx_norm = sparse ? mtp.tensor_f32("self_attn.indexer.k_layernorm.weight") : nullptr;\n'
            '    if (sparse && (!w_idx || !w_idx_norm || !mmq::built())) return false;')
        s = replace(s, '(q8 ? (uint64_t) mmq::q8_bytes(g.hc, Nn) + 4 * g.hc : 0);',
            '(q8 ? (uint64_t) mmq::q8_bytes(g.hc, Nn) + 4 * g.hc : 0) +\n'
            '                             (sparse ? 4 * (uint64_t) g.idx_key_dim : 0);')
        s = replace(s, '    int32_t* bnd = q8 ? (int32_t*) carve(16) : nullptr;',
            '    int32_t* bnd = q8 ? (int32_t*) carve(16) : nullptr;\n'
            '    float* idx_raw = sparse ? (float*) carve((size_t) B * g.idx_key_dim * 4) : nullptr;')
        old = '        proj(mixed, mixed_h, w_v, Vc, nb, KV, Nn, 0);'
        s = replace(s, old, old + '\n'
            '        if (sparse) {\n'
            '            // Prefill needs indexer keys only; query rows are never consumed here.\n'
            '            const auto* keys = (const uint8_t*) w_idx +\n'
            '                mmq::matrix_bytes(kQ8_0, g.idx_q_heads * g.idx_key_dim, Nn);\n'
            '            proj(mixed, mixed_h, keys, idx_raw, nb, g.idx_key_dim, Nn, 0);\n'
            '            const strata::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};\n'
            '            try {\n'
            '                strata::kernels::native_qsa_indexer_append_batch(idx_raw, nb, c0, 0, w_idx_norm, EPS, ib, s,\n'
            '                    st.max_cells, strata::kernels::rope_scaling(), m.cs);\n'
            '            } catch (const std::exception& e) { err = std::string("draft prefill indexer: ") + e.what(); return false; }\n'
            '        }')
        s = replace(s, '    mtp.ms_prefill += ms_since(t0);',
            '    // Publish the rollback state only after every batch has completed.\n'
            '    if (sparse && (!mtp.finish_batched_index(m.cs, err) || cudaStreamSynchronize(m.cs) != cudaSuccess)) {\n'
            '        if (err.empty()) err = "draft prefill index checkpoint synchronization failed";\n'
            '        return false;\n'
            '    }\n'
            '    mtp.ms_prefill += ms_since(t0);')
        (OUT / 'prefill.cpp').write_text(s)
    flags = {}
    for line in (BUILD / 'CMakeFiles/strata_prefill.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k, v = line.split(' = ', 1)
            flags[k] = shlex.split(v)
    obj = OUT / 'prefill.cpp.o'
    compile_cmd = ['/usr/bin/c++', '-I' + str(OUT / 'include'), *flags['CXX_DEFINES'],
                   *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'], '-c', str(OUT / 'prefill.cpp'), '-o', str(obj)]
    subprocess.run(compile_cmd, cwd=BUILD, check=True)
    link = [x for x in shlex.split((BUILD / 'CMakeFiles/strata.dir/link.txt').read_text())
            if not x.startswith('-Wl,--dependency-file=')]
    binary = OUT / 'strata-sparse-draft-prefill'
    link[link.index('-o') + 1] = str(binary)
    link.insert(link.index('libstrata_engine.a'), str(obj))
    subprocess.run(link, cwd=BUILD, check=True)
    manifest = dict(source_sha256=sha(source), header_sha256=sha(header),
                    generated_source_sha256=sha(OUT / 'prefill.cpp'), binary_sha256=sha(binary),
                    compile=compile_cmd, link=link, cwd=str(BUILD),
                    mechanism='Batch full-history sparse draft KV and key-only indexer projections; target unchanged',
                    archives={x: sha(BUILD / x) for x in set(link) if x.endswith('.a') and not Path(x).is_absolute()})
    (OUT / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(binary, flush=True)


if __name__ == '__main__':
    main()
