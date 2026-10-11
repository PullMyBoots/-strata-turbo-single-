"""Produce isolated sources; never edit the released numerical implementation."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'build'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Expected exactly one source anchor: ' + old[:100])
    return source.replace(old, new, 1)


def materialize():
    OUT.mkdir(exist_ok=True)
    records = {}
    pinned = json.loads((HERE / 'base-identities.json').read_text())

    def write(source_name, target_name, transform):
        source = ROOT / source_name
        if hashlib.sha256(source.read_bytes()).hexdigest() != pinned[source_name]:
            raise ValueError('Pinned baseline source changed: ' + source_name)
        target = OUT / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source.read_text()
        result = transform(original)
        target.write_text(result)
        records[source_name] = dict(
            base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            candidate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            generated=str(target.relative_to(HERE)))

    def header(s):
        s = replace_once(s, '    Verifier();\n', '''    Verifier();
    /// Exact accounting callback for all-resident routed groups. Set before init.
    void set_resident_observer(PoolMultiFn observer) { resident_observer_ = observer; }
''')
        # Append fields to preserve the offsets of every incumbent member.
        end = s.rfind('\n};')
        if end < 0:
            raise ValueError('Verifier class end missing')
        return s[:end] + '''
    bool resident_bypass_ = false;
    PoolMultiFn resident_observer_ = nullptr;
    int32_t* h_route_trace_ = nullptr;
    int32_t* m_route_trace_ = nullptr;
    int64_t route_trace_stride_ = 0;
    int64_t route_trace_steps_ = 0;
''' + s[end:]

    def verify(s):
        s = '#include "resident_dispatch.hpp"\n' + s
        s = replace_once(s, '    peer_decode_.reset();\n',
                         '    if (h_route_trace_) cudaFreeHost(h_route_trace_);\n    peer_decode_.reset();\n')
        s = replace_once(s, '    if (device_plan_) {\n        bool ok2 =', '''    const char* bypass_env = std::getenv("STRATA_RESIDENT_HOST_BYPASS");
    resident_bypass_ = bypass_env && std::atoi(bypass_env) == 1 && resident_observer_ &&
                       !device_only_ && !hits.peer && lb_ == 0 && le_ == g.n_layers && g_test_stall == 0;
    if (resident_bypass_) device_plan_ = true;
    if (device_plan_) {
        bool ok2 =''')
        s = replace_once(s, '    std::fprintf(stderr, "strata verify: window up to %d tokens,', '''    if (resident_bypass_ && device_plan_) {
        route_trace_stride_ = ((2 + (int64_t) max_t_ * ss.k + 15) / 16) * 16;
        route_trace_steps_ = 2 * (le_ - lb_);
        if (!mapped((size_t) route_trace_steps_ * route_trace_stride_ * sizeof(int32_t),
                    (void**) &h_route_trace_, (void**) &m_route_trace_)) {
            cudaGetLastError();
            resident_bypass_ = false;
        }
        std::fprintf(stderr, "strata verify: exact resident host bypass %s\\n",
                     resident_bypass_ ? "enabled" : "unavailable (mapped allocation failed)");
    } else resident_bypass_ = false;
    std::fprintf(stderr, "strata verify: window up to %d tokens,''')
        s = replace_once(s, '''        if (!device_only_)
            doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                             m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);''', '''        if (resident_bypass_) {
            const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
            resident_dispatch_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K,
                                      m_x_ + tb * N, m_ids_ + tb * K, m_w_ + tb * K, m_seq_, skip_ + grp,
                                      ring, m_route_trace_ + (size_t) (ring - 1) * route_trace_stride_, cs);
        } else if (!device_only_)
            doorbell_publish(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
                             m_ids_ + tb * K, m_w_ + tb * K, m_seq_, cs);''')
        s = replace_once(s, '    *(volatile uint32_t*) h_seq_ = 0;\n', '''    if (resident_bypass_)
        std::memset(h_route_trace_, 0, (size_t) route_trace_steps_ * route_trace_stride_ * sizeof(int32_t));
    *(volatile uint32_t*) h_seq_ = 0;
''')
        s = replace_once(s, '        while (*seq < want) {', '''        int32_t* const trace = resident_bypass_ ? h_route_trace_ + k * route_trace_stride_ : nullptr;
        volatile uint32_t* const ready = trace ? (volatile uint32_t*) trace : seq;
        while (*ready < want) {''')
        s = replace_once(s, 'if (q != cudaErrorNotReady && *seq < want)',
                         'if (q != cudaErrorNotReady && *ready < want)')
        s = replace_once(s, '''        set_plan_slot(grp);
        const int tb = gtb[grp], n = gte[grp] - gtb[grp];''', '''        const int tb = gtb[grp], n = gte[grp] - gtb[grp];
        std::atomic_thread_fence(std::memory_order_acquire);
        if (trace && trace[1] == 1) {
            resident_observer_(user, nullptr, trace + 2, n, ss.k, nullptr, l);
            progress_tick();
            ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
            ms_pool += ms_since(b);
            continue;
        }
        set_plan_slot(grp);''')
        return s

    def entry(s):
        s = replace_once(s, '/// Layer split: every verify stage shares one Drive', '''// Exact accounting of a group already planned and computed entirely on the GPU.
// No activation preparation, job construction, plan publication or source reads.
void drive_resident_observer(void* user, const float*, const int32_t* ids,
                             int64_t n_tok, int64_t k, float*, int64_t layer) {
    Drive* t = (Drive*) user;
    const auto begin = Clock::now();
    auto& d = t->d;
    d.layers = layer;
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            d.usage[(size_t) layer * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    d.cache_hits += n_tok * k;
    d.experts += n_tok * k;
    ++d.layers;
    ++t->calls;
    if (t->routing && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

/// Layer split: every verify stage shares one Drive''')
        s = replace_once(s, '''        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err) ||''', '''        if (n_stages == 1) ver.set_resident_observer(&drive_resident_observer);
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err) ||''')
        return s

    write('include/strata/core/verify.hpp', 'include/strata/core/verify.hpp', header)
    write('src/core/verify.cpp', 'verify.cpp', verify)
    write('src/program/generate.cpp', 'generate.cpp', entry)
    (OUT / 'sources.json').write_text(json.dumps(records, indent=2) + '\n')
    return records


if __name__ == '__main__':
    print(json.dumps(materialize(), indent=2))
