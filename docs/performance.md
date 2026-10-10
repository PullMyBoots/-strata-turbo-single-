# Historical single-GPU performance

These IQ2_XS comparisons use unchanged target weights, native 262144 context, INT8 KV and one request. On the same L40S GPU1, a working older single-GPU configuration scored 164.2501 token/s versus 203.1452/200.8600 for the optimized configuration: +23.68%/+22.29%. All nine short requests and full replies matched between those configurations.

A separate earlier native-context configuration scored 114.5 token/s after the same 256013-token input, compared with 162.3/161.6 for the optimized setup: +41.75%/+41.14%. Both passed five retrieval checkpoints; their complete long answers differed. Prefill changed from 2048 to 1024 and took approximately 96 seconds versus 131 seconds. This is an aggregate configuration/engine improvement, not the isolated contribution of one kernel.

A still earlier misconfigured run scored 69.35 token/s because prefill8192 displaced expert cache. Comparing it with 203.15 mixes configuration repair with subsequent optimization; the working 164.3 baseline is the preferred short-input comparison.

A rebuild in the independent project on GPU0/NUMA0 scored 202.2243 short and 162.0 cold-long token/s. Nine short replies and one complete long reply matched the optimized reference. This checks the standalone build; it does not turn GPU0/GPU1 differences into an algorithmic speedup.

The combined implementation includes smaller prefill/residency planning, native target verification, exact parallel QSA top-k, higher-precision Q5 auxiliary experts, draft-only low-rank candidate routing, PLE overlap, short-window HC/expert fusion, and adaptive speculative depth. There is no full per-component ablation, so additive percentages are not claimed. Experimental training, online draft correction and unrelated dual-GPU routes are not enabled in this single-GPU preset.

Evidence: `evidence/performance.json`, `evidence/baseline-normal/`, `evidence/baseline-native-long/`, `evidence/single-v2/`, `evidence/single-v2-repeat/`, and `evidence/release-build-summary.json`. Capability checks covered reasoning, tools, images, native 8fps video, Anthropic API and queued requests. These checks are not a complete intelligence evaluation. Current three-weight evidence is in [three-weight tuning](three-weight-tuning.md); new portable binaries must be qualified separately.
