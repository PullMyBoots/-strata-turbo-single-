# Batched sparse-drafter prefill (experimental)

The released sparse drafter keeps full-context keys and an auxiliary pooled
index. Its KV mode is 1, which the ordinary batched draft prefill rejects. The
fallback feeds small groups through graphs and computes indexer query rows
that prefill never consumes. A 31,566-token phase diagnostic charged 657 ms to
Turbo's chunk callbacks, versus 45 ms in the windowed upstream drafter.

This isolated candidate extends batched prefill to sparse mode 1. It reuses the
existing projection and KV append implementations, projects only the indexer
key rows, updates the pooled index in batches, and saves the index rollback
checkpoint after all appends. The original path remains the default and fallback.
No device-name dispatch or hardware parameter search is involved.

Target weights, target numerical kernels, native context and target sampling
are unchanged. Draft projections use different batching and can produce
different floating-point results and proposals. Acceptance and total latency
must therefore be measured together. No target-quality equivalence claim follows
merely from a successful build or a checkpoint-retrieval smoke.

Build with the existing root numerical archives:

```sh
.venv/bin/python experimental/sparse-draft-prefill/build.py
```

The builder writes only its ignored `build/` directory and records source,
archive and executable hashes. It does not replace an installed executable.
Enable the candidate with `STRATA_MTP_SPARSE_BATCH=1`. The initial control uses
the same executable with the flag set to zero. Unsupported modes retain the
original implementation.

Validation begins with a cold 32k five-code retrieval task, then balanced
repeats and native-window/media/cache checks only after a useful speed gain.
Results remain experimental until that qualification is complete.

Measured results, attribution and limitations: [qualification report](REPORT.md).

The sparse-drafter changes are now present in `src/prefill/prefill.cpp` and
`include/strata/core/mtp.hpp`, behind the same default-off flag. Their contents
match the qualified overlay exactly; `source-integration.json` records the
hashes. The isolated builder accepts both the integrated tree and its historical
unmodified parent. Existing release binaries and saved launch configurations
are unchanged. Normal source builds can opt into the path with
`STRATA_MTP_SPARSE_BATCH=1`; the measured combined profile also uses
`STRATA_PF_FUSED=1`, which selects the existing upstream fused expert kernels.
