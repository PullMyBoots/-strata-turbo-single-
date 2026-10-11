"""Build request-scoped native media profile repair on the direct-state candidate."""
import difflib,hashlib,json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'build';OUT.mkdir(exist_ok=True)
BASE=ROOT/'experimental/direct-slot-state/build';CMAKE=ROOT/'build-distribution/upstream-media'
old=(BASE/'generate.cpp').read_text();s=old
marker='        // #477 --expert-profile-save: what the adaptive tier learned, kept across restarts'
init='''        const char* media_profile_env = std::getenv("STRATA_MEDIA_FIXED_PROFILE");
        const bool media_fixed_profile = media_profile_env && media_profile_env[0] == '1' && o.vision &&
            !multi_gpu && !peer.valid();
        const std::vector<int32_t> media_residency = media_fixed_profile ? host_res : std::vector<int32_t>{};
        bool media_profile_hold = false;
        // Compact RAM ownership cannot be restored from an immutable full arena.
        if (media_fixed_profile && srcp != &arena_src) drive.d.usage.clear();
'''
assert s.count(marker)==1;s=s.replace(marker,init+marker)
s=s.replace('if (drain || a_win - a_last < (int64_t) o.adapt_every)', 'if (drain || media_profile_hold || a_win - a_last < (int64_t) o.adapt_every)')
marker='            // per-request sampling for the verify window\'s head (greedy when temperature is absent)'
restore='''            media_profile_hold = media_fixed_profile && (geni ||
                std::any_of(bs.begin(), bs.end(), [](const BSlot& slot) { return slot.active && slot.img; }));
            auto media_target = media_residency;
            // KV growth can reclaim the tail of the expert cache. Keep the
            // original placement only for slots that still exist.
            for (auto& slot : media_target) if (slot >= xcache.slots()) slot = -1;
            if (media_profile_hold && host_res != media_target) {
                // The preceding drains finish adaptive copies. Fence all active
                // slot readers before overwriting cache bytes or publishing a table.
                if (cudaDeviceSynchronize() != cudaSuccess) return 1;
                const auto restore_start = Clock::now();
                int64_t restored = 0;
                for (size_t i = 0; i < media_target.size(); ++i) {
                    const int32_t slot = media_target[i];
                    if (slot < 0 || host_res[i] == slot) continue;
                    const int64_t layer = (int64_t)i / g.n_expert;
                    const uint8_t* blob = srcp->blob(layer, (int64_t)i % g.n_expert);
                    if (slot >= xcache.slots() || !blob || !xcache.fill_slot_queued(slot, blob, err,
                            (int64_t)strata::kernels::cpu::expert_layout().blob_bytes(layer))) {
                        std::printf("ERR restoring media expert profile: %s\\n",err.c_str());return 1;
                    }
                    ++restored;
                }
                if (!xcache.sync_queued(err)) {
                    std::printf("ERR restoring media expert profile: %s\\n",err.c_str());return 1;
                }
                std::copy(media_target.begin(),media_target.end(),host_res.begin());
                res_upload();
                std::fill(drive.d.usage.begin(),drive.d.usage.end(),0.0f);
                std::fprintf(stderr,"strata serve: restored %lld media-profile slots in %.1f ms\\n",
                    (long long)restored,std::chrono::duration<double,std::milli>(Clock::now()-restore_start).count());
            }
'''
assert s.count(marker)==1;s=s.replace(marker,restore+marker)
a='if (!drive.d.usage.empty() && !ajob && ((rounds + 1) % o.adapt_every) == 0)';assert s.count(a)==1;s=s.replace(a,a.replace('if (','if (!media_profile_hold && ',1))
a='if (!drive.d.usage.empty() && o.adapt_every > 0 && (rounds % o.adapt_every) == 0 && pending.empty() &&';assert s.count(a)==1;s=s.replace(a,a.replace('if (','if (!media_profile_hold && ',1))
src=OUT/'generate.cpp';src.write_text(s)
(HERE/'source.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/generate.cpp',tofile='b/generate.cpp')))
record=json.loads((BASE/'build.json').read_text());compile_cmd=next(c for c in record['commands'] if '-c' in c and str(BASE/'generate.cpp') in c)
cmd=[str(src) if x==str(BASE/'generate.cpp') else str(OUT/'generate.cpp.o') if x==str(BASE/'generate.cpp.o') else x for x in compile_cmd]
subprocess.run(cmd,cwd=CMAKE,check=True)
link=record['commands'][-1];binary=OUT/'strata-media-profile';link=[str(OUT/'generate.cpp.o') if x==str(BASE/'generate.cpp.o') else str(binary) if x==str(BASE/'strata-direct-slot-state') else x for x in link]
subprocess.run(link,cwd=CMAKE,check=True)
(OUT/'build.json').write_text(json.dumps(dict(commands=[cmd,link],base_sha256=record['binary_sha256'],binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2)+'\n')
print(binary)
