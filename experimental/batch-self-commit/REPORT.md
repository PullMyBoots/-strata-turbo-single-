# Independent-slot self-commit: qualification results

The six-way IQ2_XS target-only candidate passes the declared 5% whole-engine
performance gate. IQ3_S does not. Swift qualification stopped at an upstream
video failure before any candidate run. No default release promotion is made.

| Weight | Upstream aggregate token/s | Candidate aggregate token/s | Change | Decision |
|---|---:|---:|---:|---|
| IQ2_XS | 184.20 | 197.13 | +7.02% | Qualified in the tested six-slot mode |
| IQ3_S | 127.08 | 131.29 | +3.31% | Below the 5% performance gate |
| Swift IQ3_XXS | — | — | — | Official video smoke failed; cohort incomplete |

These are aggregate client end-to-end output rates, not individual decode rates.
For each complete weight: official/candidate/candidate/official service starts,
three held-out waves per start, six simultaneous mixed English/code/Chinese
requests, 256 actual output tokens per request, temperature zero, thinking and
prefix cache off. Same native 262144 cap, resource arguments, model, INT8 KV and
media adapter. Both arms disable batch MTP for this mechanism isolation. The
existing release batch-MTP charts are a different configuration and remain intact.

IQ2_XS official starts: 185.3431, 183.0579; candidate: 197.3301, 196.9359.
IQ3_S official starts: 129.9469, 124.2193; candidate: 130.7210, 131.8622.
Every complete start passed isolation, images, video, tools, 32k retrieval and
256013-token native retrieval. Native retrieval is a capability check, not
six-way full-context throughput. Two starts per arm do not establish confidence
intervals or a universal hardware claim. This host has an L40S GPU.

Swift's first unchanged upstream baseline answered `0:00` to the red-frame video
fixture. Other checks passed. Preserve the failure and do not fill in a candidate
comparison or weaken the fixture. The single-session custom entry's documented
media profile repair is not present in this upstream batch entry.

## Correctness and fallback

GDN/convolution: 144 direct/captured cases, 1,397,587,968 bytes compared, zero
mismatches. QSA append versus save/restore/replay: 528 cases, 10,274,880 bytes,
zero mismatches. Both include state, not just emitted tokens. All experts,
weights, arithmetic kernels and KV precision stay unchanged. This is redundant
state-computation removal, extending upstream single-token self-commit.

With capacity six but two active requests and batch MTP enabled, candidate-off /
on / off measured 143.14 / 146.41 / 145.73 aggregate token/s. Isolation and all
image/video/tool/32k checks passed. This is a fallback regression check, not a
new significant throughput claim. Full answer text varies between unchanged
controls as well, so answer identity is not asserted for these concurrent runs.

The isolated prototype is limited to synchronous single-GPU `run_slot_rows`.
Before enabling other entry points, asynchronous full-model `batch_launch` must
skip the redundant commit too; its original unconditional commit would replay
an already advanced state. Partial-model pipeline stages and repeated-slot
speculative layouts stay on the original path. The prototype is default-off.

## Evidence and reproduction

[Summary](../../evidence/batch-self-commit-20261011/summary.json),
[fallback](../../evidence/batch-self-commit-20261011/fallback.json),
[all raw logs, requests, build identities and tests](../../evidence/batch-self-commit-20261011/raw.tar.gz),
[per-file hashes](../../evidence/batch-self-commit-20261011/manifest.json).

Candidate SHA256: `451563acc6f09e43d9dc641f8c00ad4f7a0df2cb74256a300126716b0a897b9b`.
Upstream SHA256: `54bf7a29420294216f78fb72e1c46f7496fb54ec809a5c5c464b9d4607a9acb9`.

```bash
.venv/bin/python experimental/batch-self-commit/build.py
.venv/bin/python tools/probe_packed_batch.py \
  --output runtime/self-commit-reproduction --gpu 0 --weights iq2_xs \
  --counts 6 --arms no-mtp self-commit self-commit no-mtp \
  --candidate-exe experimental/batch-self-commit/build/strata-batch-self-commit \
  --holdout --media --native-context
```

No new single-request gain, pelican time, full intelligence score, or 280–300
individual token/s result is claimed by this experiment.
