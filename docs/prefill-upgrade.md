# v0.2.2 guarded prefill upgrade

The large-GPU, single-session IQ2_XS consumer profile now uses automatic prefill chunks up to 16384 tokens. The previous 1024-token owned workspace processed a cold native-window input slowly, even though output was faster than upstream.

## Cache lifecycle

The prompt path temporarily borrows expert-cache slots, while target verification is explicitly suspended. Refill waits for all expert copies into their original slots, restores the residency table and validates it against the table captured at startup. Only then can the same device-only verification graphs execute. A window with a live loan, a failed copy or an altered mapping is an error. Adaptive exchanges, peer dispatch, split execution and IPC do not opt into this contract.

Target GGUFs, full experts, INT8 KV, 262144 native context, draft settings and one-token media verification are retained. This is a cache-lifecycle and prefill scheduling improvement, not a new decoding algorithm.

## Bounded same-device comparison

L40S physical GPU0, IQ2_XS, one request, cache disabled, unchanged target weights. Short score combines three variants each of English, Chinese and code, 512 output tokens. Long input is 256013 tokens with 512 output tokens and five retrieval checkpoints. These pilot values have one service start and one native-window observation per condition; the final local ABBA certificate is recorded separately.

| Metric | v0.2.0 incumbent | Guarded 16k candidate | Pinned upstream |
|---|---:|---:|---:|
| Short output, token/s | 211.5 | 211.0 | 194.6 |
| 31,566-token prefill, token/s | 2405.8 | 4308.4 | 4542.8 |
| Native-window prefill, seconds | 128.66 | 61.95 | 60.09 |
| Native-window output, token/s | 175.5 | 172.1 | 137.4 |
| Native-window request, seconds | 132.15 | 65.51 | 64.01 |

Prefill is approximately 2.1x faster than the incumbent at the native window. Short output changed by less than 1%, and native-window output by about 2%, in this pilot. Native-window complete-request latency is close to upstream, not faster: the remaining input-stage difference outweighs the generation saving for a 512-token answer. Nine short responses match the incumbent exactly. Longer responses can differ because different prefill chunk sizes change floating-point batching; checkpoint retrieval is checked separately. This is not a complete intelligence-equivalence evaluation.

The 8192, 16384 and 32768 caps were explored. The 16k cap was selected for balanced medium/long input timing, decode retention and smaller cache loans. Swift's experimental 32k cap failed a native video color test and was rejected. Swift/IQ3_S and concurrent profiles retain their previous prefill settings.

## Final installer and launcher verification

The final portable v0.2.2 engine passed fresh local ABBA qualification using the same comparison code as `./setup.sh --verify --verify-long --no-start`. IQ2_XS used L40S physical GPU0. ABBA repeats include six observations per short task family and two approximately 32k prefill observations per arm. The native-window workload has one observation per arm, so it does not establish a confidence interval.

| Metric | Pinned upstream | v0.2.2 IQ2_XS |
|---|---:|---:|
| Short output, token/s | 195.0 | 212.1 |
| 31,566-token prefill, token/s | 4534.2 | 4318.0 |
| 256,013-token prefill, seconds | 60.02 | 62.17 |
| Native-window output, token/s | 137.3 | 170.7 |
| Native-window request, seconds | 63.93 | 65.78 |

Short output is 8.8% ahead of upstream. Native-window output is 24.3% ahead of upstream. Against the v0.2.0 incumbent replay, native-window prefill is 2.07x faster and complete-request latency is 2.01x faster, while native-window output changed by -2.7%. A zero-loss decode claim at every context is not supported. Faster output alone does not guarantee lower total latency: input processing still dominates this 512-token answer.

The actual `./start.sh` reused the final matching certificate, generated `auto:16384` with the restore guard, and passed arithmetic, 32k retrieval, exact-request cache reuse and changed-suffix checkpoint reuse. An exact replay reused 31,559 tokens in 25.1 ms of input processing. A changed suffix reused a 16,384-token checkpoint. Image, video, forced tools and Anthropic API checks passed for both IQ2_XS arms. Test services were stopped.

Fresh Swift and IQ3_S qualification also selected Turbo, with 10.3% and 18.2% short-output gains against their own paired upstream runs. Swift used physical GPU0 and IQ3_S used physical GPU1; absolute rates across these placements are not a controlled model comparison. An IQ3_S diagnostic with briefly overlapping NUMA workers was excluded and repeated without that overlap.

The v0.2.1 Swift video failure (`0:00` instead of red) is preserved as a failing control. Version 0.2.2 adds request-scoped original-profile restoration, retains adaptive text caching, and passes repeated RGB, temporal-order and video-after-native-window checks. See [the Swift repair](swift-video-fix.md) for its diagnostic limits and current speed evidence. The unchanged Swift/IQ3_S prefill chunk policy was retained.

Exact binaries, configurations, raw responses, rejected candidates and the source delta are recorded in [measured evidence](../evidence/prefill-upgrade.json) and [raw runs](../evidence/prefill-upgrade-runs). These are measurement reports, not activation certificates.

## Interpreting prefill speed

Prefill throughput is not a constant. It depends on the number of newly processed tokens, prefix length, chunking, attention implementation and cache location. Batched input processing can amortize work better than single-token decode, but new attention queries still need the relevant prior context. This distinction is described in [NVIDIA's attention analysis](https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference/); dense-attention scaling should not be applied blindly to this engine's hybrid/indexed attention.

Cold 32k versus cold 256k timing is not a controlled cached-prefix experiment. In the final run, cold input throughput changed from about 4318 to 4118 token/s, but that does not guarantee the same behavior when appending a small suffix to an already cached long conversation. With a cache hit, throughput counts newly processed tokens only. Seven tokens in 25.5 ms is about 275 token/s, despite the input stage being much quicker in wall time than a cold 32k prompt. Compare newly processed token count, cached token count and latency together.

## Local qualification

The ABBA gate separately checks output rates and prefill rates. It requires at least 3% output gain, no output-family regression over 3%, and no tested-prefill regression over 10%. The explicit prefill tolerance permits small input-stage differences while rejecting the old roughly 53% loss. It does not assert that input processing is always faster than upstream. `--verify-long` adds the native-window workload; default setup measures approximately 32k prefill as part of recall.

Actual prefill, speculation, cache-restore flags and serving options now participate in certificate identity. Hardware, source/binary or relevant configuration changes require local requalification. Certificates are excluded from archives. Only L40S is physically tested; small-memory resource rules and unrelated concurrency evidence are not new performance measurements for this upgrade.
