# Decoder optimization follow-up

This directory preserves the 2026-10-10 decoder experiments. No experimental
kernel is a distribution default. The installed v0.2.2 executable and all three
default model profiles are unchanged. Target weights, complete expert routing,
262,144 context positions, INT8 KV, and the existing image/video adapter are
retained.

The requested material additional throughput improvement has **not been
established**. A controlled IQ3_S diagnostic measured **2.72%** from extended
expert producer fusion, selective CPU activation quantization, and mixed expert
combination. All three fixed-window outputs matched exactly. Its absolute rates
are diagnostic: disabling adaptive cache swaps makes that configuration slower
than the consumer profile. [Evidence](../../evidence/post-publication/expert-fixed-diagnostic.json).

| Candidate | Measured result | Decision |
|---|---|---|
| Interleaved dense projections | Under 1% whole-engine difference; exact outputs | Retain incumbent |
| Tiled recurrent verification | Under 1% whole-engine difference; exact outputs | Retain incumbent |
| Pipelined recurrence and fused quantization | Component improved; whole engine unchanged | Retain incumbent |
| Expert fusion with virtual-warp projections | +1.29% versus bracketed controls; adaptive outputs differ | No promotion |
| More GPU processing of nonresident experts | PCIe fractions .75 and 1.0 lost 4.3% and 9.8% geometric throughput | Retain .55 |
| Auxiliary head shortlists 128 / 512 / 2048 | Whole-engine -14.55% / -3.85% / -1.69%; exact outputs | Retain 8192 |
| 64-row expert tiles | Exact kernel results, slower component timings | No engine promotion |
| Paired gate/up accumulation | Component improved, whole engine -0.90%; exact outputs | No promotion |
| Exact fused HC norm/down, sequential tokens | All 32 direct/graph cases match; slower | No promotion |
| Exact fused HC norm/down, parallel tokens | All 32 direct/graph cases match; slower than incumbent | No promotion |

These percentages use each experiment's own paired controls. Workloads and
output lengths vary between experiments; absolute rates cannot be combined
across protocols. Component timings are not decoder throughput.

Compact reports are in [evidence/post-publication](../../evidence/post-publication).
Full local requests, responses, configurations, process records and build logs
are retained under `runtime/`. An incomplete 1024-token native-context campaign
is excluded: its last official output ended naturally at 914 tokens. The cores
do not implement the API's `ignore_eos` hint. Subsequent timing runs assert the
actual 512-token output budget and retain failures before raising.

The final official / incumbent / fusion / fusion / incumbent / official
comparison completed on one L40S, with two fresh starts per arm and English,
Python and Chinese tasks at short and approximately 254k input lengths.
All 72 timing responses contain exactly 512 output tokens. All 36 capability
checks passed, including a cold 256,013-token recall request in every start;
all six owned services stopped. These are smoke checks, not a comprehensive
model-quality evaluation.

| IQ3_S engine | Short input, token/s | Near-native input, token/s |
|---|---:|---:|
| Pinned upstream | 143.4 | 128.6 |
| Released incumbent | 158.4 | 149.0 |
| Experimental expert fusion | 159.8 | 152.0 |

Values average each start's three task medians. The experimental increment
over the incumbent is **0.90% short / 2.02% long**, below the material-gain
target. It is **not promoted**. Adaptive output text varies even between
incumbent restarts, so this experiment does not establish exact numerical
equivalence; the fixed-state diagnostic above is the matched-output test.
Prefix-cache hits are recorded rather than presented as cold prefill.
[Complete six-arm report](../../evidence/post-publication/expert-native-512-six-arm.json).

The API repair now fingerprints both adapters and their prompt templates in
local performance certificates. A fresh IQ2_XS official/Turbo/Turbo/official
qualification passes on the saved installation, with an 8.7% geometric output
advantage across the three task families. The launcher still selects the
released Turbo executable; saved settings and its executable hash are unchanged.
This short-input qualification validates the API repair, not the experimental
decoder or another GPU/model profile. It does not replace the six-arm native
context comparison. [Launcher requalification](../../evidence/post-publication/api-launcher-requalification.json).

## Reproduce an experiment

Prepare assets and the installed distribution using the normal setup first.
Apply the archived patch to the recorded base, then build an experimental
executable. This patch includes failed candidates so each route can be reviewed;
their switches are opt-in. Its manifest records the source paths and patch hash.

```bash
git apply --check experimental/decode-followup-20261010/decoder-exploration.patch
git apply experimental/decode-followup-20261010/decoder-exploration.patch
cmake -S . -B build-exploration -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-exploration --target strata iq_multi_parity \
  hc_norm_down_parity -j4
.venv/bin/python tools/probe_decode_candidates.py \
  --output runtime/independent-expert-fusion --gpu 0 --weights iq3_s \
  --candidate-exe build-exploration/strata --tokens 512 --repeats 2 \
  --arms incumbent expert-fusion incumbent --native-context --media
```

Each run selects one GPU, holds the benchmark lock, uses isolated ports, and
stops its own services. It does not change saved installation settings or issue
an activation certificate. The patch consolidates final experimental source;
the individually recorded binary hashes belong to their original build stages.
A rebuilt candidate requires a new measurement.
