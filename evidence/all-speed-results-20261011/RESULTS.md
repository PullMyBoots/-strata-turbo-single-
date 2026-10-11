# Complete speed measurements

All measured configurations are included. This report does not select README highlights.

CSV: [Decode](decode.csv) · [Concurrency](concurrency.csv) · [Prefill](prefill.csv) · [Pelican](pelican.csv).

One L40S; EPYC 9754; same selected target weights within each pair; native 262,144-token cap; INT8 KV. Strata is pinned to fb58e0d. Turbo uses sparse/fused prefill for one request and direct slot-state transfer for multiple requests. These experimental builds are not the installer default. Resource settings match within each pair; Swift/IQ3_S use 2,048 MiB headroom in the latest campaign. Swift uses minimum 128 visual tokens in both arms.

## Decode

Output-stage tokens/s, excluding prefill. Two 512-token outputs per task/length, median shown; temperature zero, thinking off. Prefix caching is enabled; first request at each size is cold. A single service start per engine/weight; two responses do not establish independent-start confidence intervals. The CSV includes both samples.

| Weight | Background tokens | Task | Strata | Strata-Turbo-Single | Gain |
|---|---|---|---|---|---|
| IQ2_XS | 0 | english | 189.7 | 210.1 | +10.8% |
| IQ2_XS | 0 | code | 231.6 | 276.2 | +19.2% |
| IQ2_XS | 0 | chinese | 172.1 | 172.1 | +0.0% |
| IQ2_XS | 8192 | english | 179.4 | 202.6 | +13.0% |
| IQ2_XS | 8192 | code | 215.9 | 265.5 | +23.0% |
| IQ2_XS | 8192 | chinese | 160.8 | 188.4 | +17.2% |
| IQ2_XS | 65536 | english | 168.1 | 199.8 | +18.9% |
| IQ2_XS | 65536 | code | 223.0 | 252.1 | +13.0% |
| IQ2_XS | 65536 | chinese | 166.6 | 179.2 | +7.6% |
| IQ2_XS | 253952 | english | 151.3 | 178.1 | +17.7% |
| IQ2_XS | 253952 | code | 194.0 | 247.8 | +27.7% |
| IQ2_XS | 253952 | chinese | 141.8 | 167.2 | +18.0% |
| Swift IQ3_XXS | 0 | english | 145.1 | 155.9 | +7.5% |
| Swift IQ3_XXS | 0 | code | 188.6 | 228.1 | +20.9% |
| Swift IQ3_XXS | 0 | chinese | 139.9 | 144.7 | +3.4% |
| Swift IQ3_XXS | 8192 | english | 145.2 | 161.4 | +11.1% |
| Swift IQ3_XXS | 8192 | code | 191.8 | 223.1 | +16.3% |
| Swift IQ3_XXS | 8192 | chinese | 140.9 | 153.6 | +8.9% |
| Swift IQ3_XXS | 65536 | english | 148.3 | 167.1 | +12.6% |
| Swift IQ3_XXS | 65536 | code | 173.4 | 226.1 | +30.4% |
| Swift IQ3_XXS | 65536 | chinese | 137.0 | 146.8 | +7.2% |
| Swift IQ3_XXS | 253952 | english | 129.9 | 151.3 | +16.5% |
| Swift IQ3_XXS | 253952 | code | 169.4 | 220.2 | +30.1% |
| Swift IQ3_XXS | 253952 | chinese | 123.7 | 149.2 | +20.6% |
| IQ3_S | 0 | english | 135.9 | 147.9 | +8.9% |
| IQ3_S | 0 | code | 171.0 | 175.3 | +2.5% |
| IQ3_S | 0 | chinese | 120.9 | 129.3 | +7.0% |
| IQ3_S | 8192 | english | 133.4 | 151.4 | +13.5% |
| IQ3_S | 8192 | code | 168.0 | 177.1 | +5.4% |
| IQ3_S | 8192 | chinese | 120.2 | 128.9 | +7.3% |
| IQ3_S | 65536 | english | 134.7 | 142.3 | +5.7% |
| IQ3_S | 65536 | code | 165.8 | 180.4 | +8.8% |
| IQ3_S | 65536 | chinese | 121.1 | 132.4 | +9.4% |
| IQ3_S | 253952 | english | 126.2 | 154.8 | +22.7% |
| IQ3_S | 253952 | code | 159.0 | 187.8 | +18.1% |
| IQ3_S | 253952 | chinese | 113.7 | 126.2 | +11.0% |

## Concurrent throughput

Aggregate output tokens / wall time, including prefill and admission. Two independent starts per engine in ABBA order, three rotated English/code/Chinese waves per start, 256 output tokens per request, cache/thinking off. Per-request decode is a different metric and is retained separately in the CSV.

| Weight | Requests | Strata aggregate | Turbo aggregate | Gain |
|---|---|---|---|---|
| IQ2_XS | 1 | 165.1 | 183.0 | +10.8% |
| IQ2_XS | 2 | 134.7 | 149.1 | +10.6% |
| IQ2_XS | 4 | 164.0 | 186.9 | +14.0% |
| IQ2_XS | 6 | 183.7 | 211.8 | +15.3% |
| Swift IQ3_XXS | 1 | 129.6 | 142.8 | +10.2% |
| Swift IQ3_XXS | 2 | 104.1 | 113.1 | +8.7% |
| Swift IQ3_XXS | 4 | 135.2 | 150.2 | +11.2% |
| Swift IQ3_XXS | 6 | 151.1 | 172.4 | +14.1% |
| IQ3_S | 1 | 119.5 | 127.8 | +6.9% |
| IQ3_S | 2 | 90.1 | 96.0 | +6.6% |
| IQ3_S | 4 | 114.4 | 124.2 | +8.6% |
| IQ3_S | 6 | 125.3 | 136.8 | +9.2% |

## Prefill

Input tokens/s with zero cache hits. Mean of two independent single-request starts.

| Weight | Input tokens | Strata | Strata-Turbo-Single | Gain |
|---|---|---|---|---|
| IQ2_XS | 31566 | 4627 | 4962 | +7.2% |
| IQ2_XS | 256013 | 4231 | 4583 | +8.3% |
| Swift IQ3_XXS | 31566 | 4625 | 4956 | +7.2% |
| Swift IQ3_XXS | 256013 | 4256 | 4373 | +2.7% |
| IQ3_S | 31566 | 4535 | 4860 | +7.2% |
| IQ3_S | 256013 | 4155 | 4270 | +2.8% |

## IQ2_XS pelican animation

> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.

Claude Code 2.1.290 (Claude Code); maximum effort, same initial rendered prompt; fresh sessions and workspaces; ABBA order. Wall time includes thinking, tool calls and retries, excludes model loading. Outputs and tool trajectories can differ.

| Engine | Order | Wall time | Model calls | Output tokens |
|---|---|---|---|---|
| Strata | 0 | 407.4 s | 11 | 95336 |
| Strata-Turbo-Single | 1 | 180.2 s | 7 | 37142 |
| Strata-Turbo-Single | 2 | 120.2 s | 7 | 23418 |
| Strata | 3 | 351.6 s | 18 | 84373 |

Median task times: Strata 379.5 s; Strata-Turbo-Single 150.2 s. Output volumes differ substantially; this is not an isolated numerical-engine speed ratio. [All four animations](../../docs/showcase/pelican-latest-20261011/index.html).

## Evidence

[Decode raw data](../readme-latest-curves-20261011/summary.json) · [Concurrency and Prefill raw data](../readme-full-matrix-20261011/summary.json) · [Pelican raw data](../pelican-latest-20261011/pelican-summary.json).

All 48 matrix starts passed image/video/tool, isolation and 32k retrieval checks. Native retrieval passed in one- and six-slot configurations for all weights, additionally two/four slots for IQ2_XS. These checks do not establish comprehensive model-quality equivalence or simultaneous near-full-window performance.
