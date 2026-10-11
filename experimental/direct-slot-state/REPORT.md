# Direct slot-state transfer with singleton commit

IQ2_XS passed the held-out six-way end-to-end performance gate: **184.25 ->
213.10 aggregate output tokens/s (+15.66%)** against pinned upstream with the
same resources. All four independent service starts passed isolation, image,
video, tool, 32k retrieval and 256013-token native retrieval checks.

IQ3_S also passed: **125.72 -> 139.76 aggregate token/s (+11.17%)**, with all
four starts passing the same capability checks. Swift is not qualified: the earlier upstream
batch engine failed its video smoke before the candidate could be compared.
This experimental source build has not replaced any installed/default engine.

## Mechanism and attribution

1. Singleton batch commit removes redundant GDN/conv state replay when every
   independent slot has exactly one mandatory token. This extends upstream's
   single-sequence self-commit technique. Repeated-slot speculative layouts
   retain their original verifier path.
2. Direct live-state transfer replaces GPU -> pageable host checkpoint -> GPU
   copies at slot admission/restoration with direct copies of the same bytes.
   The PLE history, QSA tail/dead/position and spare pooled row follow the same
   restore semantics. Historical checkpoint restore and all KV paging/ring
   save/restore code remain unchanged. All validation precedes destination writes.

The second mechanism's off/on/off calibration holds singleton commit enabled:
190.16 / 206.96 / 190.98 aggregate token/s, an **8.60% incremental gain** versus
the bracketed control mean. The combined +15.66% above uses different held-out
prompts and its own fresh upstream ABBA cohort; do not multiply percentages
from different cohorts to reconstruct a result.

No weights, experts, arithmetic kernel, sampling policy, context cap, KV format,
resource allocation or request admission order is changed. Neither mechanism
branches on a hardware product name. Only an L40S host has been measured.
This is execution and transfer elimination, not a new model decoding algorithm.

## Numerical and operational checks

Direct transfer passed 104 cases / 6,337,466,720 compared bytes with zero mismatches
against upstream save/restore, including full/partial layer carves, PLE absent
and present, token/block edges through 262144, host token history, the spare pooled
row and malformed-carve rejection without writes. Alternating component timings
were 91.36/91.71 ms via host and 0.4485/0.4486 ms directly. In the engine, full
admission still includes KV work: approximately 100–120 ms became 1–7 ms for the
short calibration prompts. The component ratio is not an engine speedup.

The singleton commit's GDN/conv and QSA state checks are documented in the
[previous report](../batch-self-commit/REPORT.md). Its prototype retains the
same restriction to synchronous, single-GPU batch verification; asynchronous
full-model pipeline entry requires its commit guard before wider promotion.

The held-out comparison uses official/candidate/candidate/official fresh starts,
three six-request mixed English/code/Chinese waves each, exactly 256 output
tokens per request, temperature zero, thinking and prefix caching disabled,
native 262144 context cap, identical resource arguments, INT8 KV and media adapter.
Both arms disable batch MTP to isolate this target-only mode.

IQ2_XS official starts: 183.5537, 184.9476; candidate: 214.1607, 212.0462.
IQ3_S official starts: 126.2059, 125.2326; candidate: 138.0626, 141.4584.
The initial IQ3_S cohort was interrupted after one official start and one
partial candidate wave. It remains archived and is excluded; the complete
repeated ABBA cohort supplies the IQ3_S result.
Across the 36 IQ2_XS requests per arm, median first-content latency was
1.0057 -> 0.7425 seconds and median request completion was 8.0909 -> 6.9732
seconds. These are descriptive medians from the same six-way workload.

Each token/s figure is aggregate end-to-end output, including prefill/admission;
it is not a single-request decode rate. Two starts per arm do not establish a
confidence interval. Full response text varies between unchanged concurrent
controls, so full-answer identity is not asserted. The capability checks do not
replace comprehensive intelligence evaluation. Native retrieval is a capability
check, not six near-full-context sessions in parallel.

## Reproduction

```bash
.venv/bin/python experimental/batch-self-commit/build.py
.venv/bin/python experimental/direct-slot-state/build.py
CUDA_VISIBLE_DEVICES=0 experimental/direct-slot-state/build/parity
.venv/bin/python experimental/direct-slot-state/build.py --engine
.venv/bin/python tools/probe_packed_batch.py \
  --output runtime/direct-state-reproduction --gpu 0 --weights iq2_xs \
  --counts 6 --arms no-mtp direct-state direct-state no-mtp \
  --candidate-exe experimental/direct-slot-state/build/strata-direct-slot-state \
  --holdout --media --native-context
```

Candidate SHA256: `dc0065aadc397a2af333e3db7fb33f7c7ab77a0b862992dcf2d5c8e496ef552d`.
Upstream SHA256: `54bf7a29420294216f78fb72e1c46f7496fb54ec809a5c5c464b9d4607a9acb9`.
The installed release, single-request decode/prefill/pelican charts and production
launchers remain unchanged. No all-metric superiority or 280–300 single-request
token/s result is claimed.

## Archived evidence

[Structured final summary](../../evidence/direct-slot-state-20261011/summary.json),
[calibration](../../evidence/direct-slot-state-20261011/calibration.json),
[parity result](../../evidence/direct-slot-state-20261011/parity.json),
[raw requests, logs and build manifests](../../evidence/direct-slot-state-20261011/raw.tar.gz),
[per-file hashes](../../evidence/direct-slot-state-20261011/manifest.json).
