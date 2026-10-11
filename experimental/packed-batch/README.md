# Concurrency scheduling follow-up

These are isolated experiments against the released v0.2.2 engines, not installed
presets. All target GGUF shards and experts, INT8 KV, the native 262,144-token
limit and the original image/video path are retained. Only GPU 0 is used.

## Mechanisms and decisions

| Candidate | Mechanism | Decision |
|---|---|---|
| Packed verified proposals | Give every active session one target row, then spend the eight-row budget on proposals ranked by prefix confidence; allow longer proposals with two sessions | Reject: more verification work costs more than the accepted-token gain |
| Grouped GDN recurrence | Run independent session recurrence/commit groups in one grid with the original floating-point operation order | Reject standalone: a large component gain gives under 1% whole-engine gain |
| Adaptive paired verification | Use paired target/proposal rows with at most four active sessions; use one target row per session above four, while maintaining each draft state for later load changes | Reject: draft-state maintenance consumes the saved work |
| Target-only six-slot batching | Existing upstream shared batching without batch MTP | Useful reference: faster at full load, but disabling proposals for the whole service can lose low-load benefits |

Every speculative token remains checked against the complete target sampler.
No target expert, vocabulary entry, input or generated token is discarded to
improve a rate. Calibration prompts differ from the README holdouts. No new
decoding algorithm or intelligence-equivalence claim is made.

Completed calibration results and each process/configuration are retained in
[the follow-up evidence](../../evidence/concurrency-followup/summary.json).
Six-session answer text varies even between unchanged incumbent restarts;
identical complete answers are therefore not used as a six-slot acceptance claim.
Two-session calibration answers match the corresponding unchanged incumbent.

## Numerical check

The grouped GDN test compares the original kernel with the candidate over
144 direct/graph cases, two head geometries, one through six groups, mixed
one-through-four-token prefixes and varying accepted-prefix lengths. It uses the
actual one-row commit path. All **1,342,832,640 output/state bytes** match.
This checks the tested recurrence arithmetic; it is not a comprehensive model
evaluation. Component timings are recorded separately from engine throughput.

## Rebuild and reproduce

First build the pinned upstream/media libraries with the normal distribution
builder. Candidate builds link those libraries and write only this directory.
The source patches verify their base and generated SHA256 identities; installed
binaries and the comparator are never overwritten by this script.

```bash
.venv/bin/python experimental/packed-batch/build.py
.venv/bin/python experimental/packed-batch/build.py --gdn
.venv/bin/python experimental/packed-batch/build.py --adaptive
.venv/bin/python experimental/packed-batch/build.py --gdn --parity
CUDA_VISIBLE_DEVICES=0 experimental/packed-batch/gdn-parity

.venv/bin/python tools/probe_packed_batch.py \
  --output runtime/packed-reproduction --gpu 0 --weights iq2_xs \
  --counts 2 6 --arms incumbent control packed incumbent \
  --candidate-exe experimental/packed-batch/strata-packed

.venv/bin/python tools/probe_packed_batch.py \
  --output runtime/grouped-reproduction --gpu 0 --weights iq2_xs \
  --counts 2 6 --arms incumbent control gdn incumbent \
  --candidate-exe experimental/packed-batch/strata-gdn

.venv/bin/python tools/probe_packed_batch.py \
  --output runtime/adaptive-reproduction --gpu 0 --weights iq2_xs \
  --counts 6 2 --arms incumbent control adaptive incumbent \
  --candidate-exe experimental/packed-batch/strata-adaptive
```

The local screen uses CUDA architecture 89; `--arch` and `--nvcc` are explicit
build options. Other hardware is unqualified. Runs hold the same benchmark lock,
start one owned service at a time and stop it in a `finally` block. Media/long
holdouts are reserved for a candidate that first clears the material-gain gate.
No newly implemented candidate here clears that gate, so none changes the distributed engine.
