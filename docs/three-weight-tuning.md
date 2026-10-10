# Three-weight single-GPU tuning: 2026-10-10

Historical development-engine evidence. All tests used the same L40S physical GPU0, native 262144 context cap, unchanged target GGUF weights, full experts and INT8 KV. These numbers are not universal hardware guarantees, community SOTA claims or complete intelligence-equivalence evidence. Portable distribution builds have separate qualification records.

## Single-session output speed

Units: output-stage tokens/s, excluding prefill. Short score: geometric mean of task-family medians over Chinese/English/code, three prompt variants each, 512 generated tokens, temperature 0 and prompt cache disabled. Selected presets were independently replayed after tuning. Long input: 256013 tokens, 512 output tokens, cache hit 0. Swift/IQ3_S both arms started with a fresh service directly for long tests; IQ2 both arms ran short tasks before the long prompt. Long conditions have one observation each.

| Weights | Official short | Tuned short | Gain | Official long | Tuned long | Gain |
|---|---:|---:|---:|---:|---:|---:|
| GSQ-RCO IQ2_XS | 187.1 | 206.4 | +10.3% | 137.3 | 174.3 | +26.9% |
| Swift IQ3_XXS | 139.9 | 174.6 | +24.8% | 100.9 | 140.4 | +39.1% |
| GSQ-RCO IQ3_S | 119.0 | 163.1 | +37.1% | 83.4 | 123.3 | +47.8% |

Long prefill times, official/tuned: IQ2_XS 60.7/129.4 seconds, Swift 60.1/66.7 seconds, IQ3_S 61.5/68.1 seconds. Faster output does not imply lower total latency for every prompt.

## Native concurrent batch throughput

Aggregate throughput is total generated tokens divided by batch wall time, including prefill and admission. It is not the single-session output-stage metric. Each group ran two batches; the table reports the second, warmed-expert batch. Every request produced 256 tokens. Prompt caching was disabled; serving slot counts were verified at 2/4/6.

| Weights | Concurrency | Official aggregate | Tuned aggregate | Gain | Official median per-request generation | Tuned median per-request generation |
|---|---:|---:|---:|---:|---:|---:|
| GSQ-RCO IQ2_XS | 2 | 130.4 | 150.5 | +15.4% | 78.3 | 94.4 |
| GSQ-RCO IQ2_XS | 4 | 159.1 | 175.0 | +10.0% | 47.0 | 51.6 |
| GSQ-RCO IQ2_XS | 6 | 152.5 | 169.5 | +11.1% | 29.6 | 34.5 |
| Swift IQ3_XXS | 2 | 76.5 | 127.2 | +66.4% | 43.1 | 78.1 |
| Swift IQ3_XXS | 4 | 92.8 | 141.5 | +52.5% | 25.5 | 43.6 |
| Swift IQ3_XXS | 6 | 107.4 | 143.8 | +33.9% | 20.0 | 27.9 |
| GSQ-RCO IQ3_S | 2 | 59.0 | 112.3 | +90.3% | 32.7 | 68.0 |
| GSQ-RCO IQ3_S | 4 | 76.8 | 121.8 | +58.5% | 21.2 | 35.9 |
| GSQ-RCO IQ3_S | 6 | 89.4 | 119.9 | +34.1% | 16.8 | 23.9 |

## Concurrent long prompts

Each request had 32278 input tokens and 256 output tokens; every session retained the 262144 cap. Six full native-window sessions were not tested. Each condition ran one batch. Per-request generation rates include interference from admission/prefill; do not sum them to infer aggregate throughput. Five retrieval checkpoints passed.

| Weights | Concurrency | Official batch seconds | Tuned batch seconds | Official median per-request token/s | Tuned median per-request token/s | Official median TTFT seconds | Tuned median TTFT seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSQ-RCO IQ2_XS | 2 | 27.90 | 26.50 | 55.0 | 86.0 | 15.84 | 15.62 |
| GSQ-RCO IQ2_XS | 4 | 49.20 | 46.81 | 16.1 | 34.9 | 29.84 | 29.47 |
| GSQ-RCO IQ2_XS | 6 | 70.52 | 66.57 | 11.3 | 34.7 | 40.42 | 39.33 |
| Swift IQ3_XXS | 2 | 29.23 | 26.23 | 34.6 | 68.1 | 15.48 | 15.38 |
| Swift IQ3_XXS | 4 | 50.28 | 46.32 | 14.2 | 23.5 | 29.09 | 28.94 |
| Swift IQ3_XXS | 6 | 71.62 | 66.61 | 9.3 | 21.5 | 39.35 | 39.20 |
| GSQ-RCO IQ3_S | 2 | 32.10 | 27.44 | 25.7 | 58.0 | 15.88 | 15.86 |
| GSQ-RCO IQ3_S | 4 | 53.89 | 48.28 | 12.4 | 21.6 | 29.87 | 29.81 |
| GSQ-RCO IQ3_S | 6 | 76.12 | 69.04 | 8.4 | 18.6 | 40.65 | 40.26 |

## Implementation and limits

IQ2_XS retained its optimized single-session preset. Swift/IQ3_S use dynamic expert caching, 32 CPU workers, the Q5 auxiliary runtime and adaptive speculative windows. Main weights, expert count, KV precision and accepted context cap stay unchanged. Concurrent tuning uses upstream native shared batching, batch-MTP, smaller resident KV requests and 1400 MiB reserve. Effective KV residency is rounded by the engine.

The concurrent engine uses pinned upstream numerical libraries with an existing SVE2 embedding reader port that preserves mtmd time/spatial positions. No new decoding algorithm is claimed. All three models passed native video requests; IQ3_S also passed video with forced tool output, and IQ2 passed the Anthropic API. Simultaneous multi-video performance and a complete intelligence suite were not evaluated. Audio is unsupported.

NVML/nvidia-smi had a userspace/kernel driver mismatch; CUDA enumeration and inference worked. The driver was not changed. Tests ran sequentially on one GPU without confidence intervals; small differences are not a stable guarantee.

## Historical development concurrency routing

Retained IQ2 IPC-pool short aggregate rates at 2/4/6 requests were 129.9/155.3/174.1 token/s, versus native batch 150.5/175.0/169.5. The development launcher therefore keeps IPC at 5/6 requests and native batch at 2..4; 3/5 were supported but not independently benchmarked. At six long requests, IPC median per-request generation was 58.8 versus native 34.7 token/s, while total batch time was 80.8 versus 66.6 seconds.

The portable consumer distribution uses one shared native-batch engine and requalifies its selected concurrency on each host. It does not import these historical MPS results as its own certificate.

## Evidence

See `evidence/three-weight-tuning-20261010.json`, the recorded binaries in `engine/manifest.json`, and the original experiment directory referenced by that evidence. Raw matrix JSON/CSV includes request rates, TTFT, retrieval, both warm-up batches, configuration and engine logs. Historical services were stopped and GPU memory was released after testing.
