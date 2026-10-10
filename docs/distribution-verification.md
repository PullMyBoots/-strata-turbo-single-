# v0.2.0 distribution verification

This release was verified with freshly compiled Linux sm89 portable-CPU binaries. Paired output rates, actual request timing and exact binary hashes are recorded in `evidence/distribution-verification.json`.

These are historical v0.2.0 measurements. The IQ2_XS prefill restriction was subsequently removed in v0.2.1; see [prefill upgrade](prefill-upgrade.md) for the new configuration and measurements.

## Measured results

| Weights | GPU | Official short | Turbo short | Gain |
|---|---|---:|---:|---:|
| iq2_xs | L40S 0 | 194.5 | 211.1 | +8.5% |
| swift | L40S 1 | 119.1 | 142.0 | +19.2% |
| iq3_s | L40S 0 | 144.0 | 163.1 | +13.3% |

IQ2 native-window output: official **137.3** vs Turbo **175.3** token/s. One 256013-token/512-output observation per arm, after short/recall/media/API checks.

- official: prefill 60.3 s; generation 3.7 s; request wall time 64.2 s.
- turbo: prefill 129.2 s; generation 2.9 s; request wall time 132.7 s.

These are paired hardware-specific comparisons. Swift ran on physical GPU1; IQ2/IQ3 ran on GPU0. Swift and IQ3 comparisons ran concurrently on different NUMA nodes. Absolute rates across those GPUs should not be used to rank model performance. Each short arm has six observations per task family across two service starts; no confidence interval is claimed.

Six-slot IQ2 aggregate throughput: official **178.8** vs batch-MTP **167.4** token/s (median of two warmed batches). Both passed isolation, the seventh queued request, image and video checks. This tuning is not promoted as faster for that workload; the local gate can choose upstream.


## Checks

- Linux L40S inference through the consumer installer and launcher.
- Native 262144 context cap, unchanged target GGUFs, all experts and INT8 KV.
- Three weight families; short ABBA comparisons and native-window IQ2 recall.
- Text, OpenAI/Anthropic APIs, synthetic image and native video checks.
- Six shared serving slots, session isolation and a seventh queued request.
- A clean Python 3.13 environment installed from cached official wheels in a path containing spaces.
- Offline reconstruction of target pack and MTP runtimes; reconstructed Q5 expert hash and dense runtime match the existing assets.
- Archive file hashes, build-dependency inclusion and exclusion of private environments/model assets/local settings.

## Scope

Only NVIDIA L40S is physically measured. The 12/16/24/32/48/80 GB resource matrix is a policy test, not physical inference validation on each GPU size. Native Windows, AMD and six simultaneous native-window prompts are unqualified. Internet model downloads were not repeated; previously downloaded shards and pinned tensors were reused. The relocated installation used a local wheel cache because this server cannot reach the public package/model hosts reliably.

A synthetic color test is a capability smoke, not a complete multimodal quality evaluation. Main weights, expert count and KV precision are preserved; a complete intelligence-equivalence benchmark is not claimed.

## Media verification guard

The earlier optimized Swift configuration returned only `0:00` for a native video request. The same optimized engine answered the color question correctly when target verification used one token per step. The release therefore limits optimized media requests to one-token verification and retains tuned speculative windows for text. This guard reduces media speculation; the published text throughput score does not imply a media throughput advantage.

## Timing definitions

Single-session throughput is output-stage tokens/s, excluding prompt reading. Short scores combine task-family medians (English, Chinese and code). Long input has 256013 prompt tokens and 512 output tokens, with no prompt cache. Prefill and total latency are reported separately: IQ2's tuned 1024-token prefill can be slower than the official auto-prefill even when its output stage is faster.

Concurrent aggregate throughput is total generated tokens divided by batch wall time, including admission/prefill. The six-slot smoke is a single service start per arm with one warm and two measured batches. It is not an ABBA qualification or a confidence interval. If the on-device gate finds no consistent advantage, startup selects the validated upstream configuration.
