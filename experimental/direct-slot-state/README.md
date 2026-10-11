# Direct running-state transfer between batch slots

Bounded candidate: replace checkpoint save-to-host then restore-to-device with
same-device state copies at batch admission and live slot restoration. Keep KV
save/restore, historical checkpoint restoration, ordering, sampling, weights,
experts, native context and memory policy unchanged. This removes a transfer,
not model arithmetic, and does not select hardware by product name.

Observed admission logs show approximately 100–115 ms per short-input slot.
The fraction caused by running-state transfer has not yet been measured. Require
exact state parity, including PLE history and QSA spare pooled row at block edges,
then a component timing gate before rebuilding a full engine. Qualification must
compare against the already improved singleton commit engine as well as upstream.
No end-to-end benefit is claimed in advance. Leave production defaults unchanged.

Reject malformed/mismatched layer carves before any destination write. Caller
synchronizes before transfer; synchronize after it as upstream restore does.
Paged KV, differing KV pool capacities and draft KV retain upstream code.

## Component result

The gate passed 104 cases and 6,337,466,720 compared bytes with zero mismatches.
Whole-model and partial layer carves, PLE present/absent, zero/one-token history,
QSA block edges, 32768 and 262144 positions, and prevalidation rejection were
checked against upstream checkpoint save/restore. Running-state transfer took
91.36/91.71 ms through host memory and 0.4485/0.4486 ms directly in an alternating
component measurement. This is a transfer measurement, not a decode speedup.

Opt in with `STRATA_DIRECT_SLOT_STATE=1`; otherwise use the unchanged snapshot
roundtrip. The isolated build combines this with the previously measured
singleton commit object. Its six-slot calibration compares the same executable
with direct transfer off/on/off, holding singleton commit enabled throughout.
No numeric kernel, KV page map, expert residency or admission ordering is changed.

```bash
.venv/bin/python experimental/direct-slot-state/build.py
CUDA_VISIBLE_DEVICES=0 experimental/direct-slot-state/build/parity
# Only after the component gate passes:
.venv/bin/python experimental/direct-slot-state/build.py --engine
```

## Calibration result

Six simultaneous IQ2_XS requests, three mixed-task waves per start, same candidate
binary off/on/off: 190.1636 / 206.9649 / 190.9757 aggregate client token/s.
The incremental gain versus the bracketed control mean is 8.60%, passing the 5%
screen gate. All isolation checks passed. Identical token count (256 each),
no prefix cache, no thinking, same native cap and resources. Per-slot admission
logs fell from roughly 100–120 ms to roughly 1–7 ms. End-to-end improvement is
much smaller than the isolated transfer ratio, as expected.

Independent-task ABBA qualification is complete: IQ2_XS +15.66%, IQ3_S +11.17%
versus upstream. All media/tools and native-context checks passed in both complete
cohorts. Swift remains unqualified after its earlier official media failure.
See [the complete report](REPORT.md). Do not replace release
charts with these calibration numbers.
