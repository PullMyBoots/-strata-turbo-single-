# Complete concurrency and Prefill matrix

ABBA fresh starts per count and weight; arithmetic mean of two start rates; each start rate is actual output tokens / total three-wave wall time. 256 output tokens per request; three rotated English/code/Chinese waves; thinking/cache off; native cap 262144; shared resources; GPU0 L40S. Latest single uses sparse/fused prefill; latest multi uses target-only direct state transfer and singleton commit. Capability failures retained, not qualified. No cross-hardware or full-window concurrent performance claim.

| Weight | Engine | 1 request | 2 requests | 4 requests | 6 requests | 6-slot median request decode |
|---|---|---:|---:|---:|---:|---:|
| IQ2_XS | Strata | 165.1 | 134.7 | 164.0 | 183.7 | 35.8 |
| IQ2_XS | Strata-Turbo-Single | 183.0 | 149.1 | 186.9 | 211.8 | 39.9 |
| Swift IQ3_XXS | Strata | 129.6 | 104.1 | 135.2 | 151.1 | 29.1 |
| Swift IQ3_XXS | Strata-Turbo-Single | 142.8 | 113.1 | 150.2 | 172.4 | 32.5 |
| IQ3_S | Strata | 119.5 | 90.1 | 114.4 | 125.3 | 23.9 |
| IQ3_S | Strata-Turbo-Single | 127.8 | 96.0 | 124.2 | 136.8 | 25.3 |

## Prefill

| Weight | Engine | 31,566 input tokens (t/s) | 256,013 input tokens (t/s) | 256,013-token prefill time |
|---|---|---:|---:|---:|
| IQ2_XS | Strata | 4,627 | 4,231 | 60.5 s |
| IQ2_XS | Strata-Turbo-Single | 4,962 | 4,583 | 55.9 s |
| Swift IQ3_XXS | Strata | 4,625 | 4,256 | 60.1 s |
| Swift IQ3_XXS | Strata-Turbo-Single | 4,956 | 4,373 | 58.5 s |
| IQ3_S | Strata | 4,535 | 4,155 | 61.6 s |
| IQ3_S | Strata-Turbo-Single | 4,860 | 4,270 | 60.0 s |

## Conditions and checks

Strata uses unchanged numerical engine SHA `54bf7a29420294216f78fb72e1c46f7496fb54ec809a5c5c464b9d4607a9acb9` with the common media adapter. Turbo uses sparse/fused prefill SHA `c89835335d1a26ba00c404bc2e7fed95c3f8d32b21f28e0e978260da73db173c` at one request, and direct-state SHA `dc0065aadc397a2af333e3db7fb33f7c7ab77a0b862992dcf2d5c8e496ef552d` for multiple requests. These are experimental executables, not the installer default.

Both arms receive identical resource settings within each model/count pair: 32 workers, native 262,144-token cap, INT8 KV, slot-dependent KV residency. Swift/IQ3_S reserve 2,048 MiB; IQ2_XS reserves 700 MiB at one request and 1,400 MiB at multiple requests. Swift minimum visual tokens is 128 in both arms. Batch MTP is disabled in both arms for multiple requests.

All 48 starts passed image/video/tool, session-isolation and 32k retrieval checks. Native 256,013-token retrieval passed at counts 1/6 for every weight and additionally 2/4 for IQ2_XS. These bounded checks do not establish comprehensive quality equivalence or six simultaneous near-full-window performance.

Failed/interrupted pre-fix Swift cohorts are archived with exclusion notes; no partial or selected starts were inserted into the final matrix. See [video repair](../swift-native-video-resolution-20261011/REPORT.md).

[Audited summary](summary.json) · [Raw records, commands and logs](raw.tar.gz) · [File hashes](manifest.json).
