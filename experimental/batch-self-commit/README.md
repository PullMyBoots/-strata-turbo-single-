# Commit independent one-token batch slots during their forward pass

Candidate: remove recurrent replay when every active slot has exactly one token.
The target-only batch path currently runs its recurrence without updating state,
then repeats it in commit. Every one-row slot must retain that row. Use the
existing single-token conv/state commit capabilities in the forward graph and
skip the redundant commit graph. QSA index tails and PLE history already advance
in the forward pass; retain them instead of snapshot/restore/replay. Host PLE
token history still advances at the normal commit boundary.

This is an extension of upstream's single-sequence one-token self-commit idea,
not a new numerical kernel or a claim of original speculative decoding.

Keep weights, experts, KV format, native context, sampling, admission order and
output tokens unchanged. Default off (`STRATA_BATCH_SELF_COMMIT=1` opts in).
Enable only when all rows belong to distinct slots, the verifier covers the
whole model on one GPU, and existing one-token commit support is enabled.
Any repeated slot (speculative proposals) retains the old path. No hardware-name
or parameter search. On a failed window the service's existing error handling
applies; this is the same eager state principle as upstream's single-token path.

Lead choice: repeated state computation is measured around 2.7 ms of a 25 ms
six-slot window. Scheduling changes have repeatedly failed, and generic residual
Tensor Core arithmetic failed its speed gate. A theoretical 10% ceiling is not
a measured speed claim. Select only this bounded candidate.

Gate: byte-exact conv/intermediate/recurrent state/output comparison against
forward-then-commit using actual upstream kernels, direct and captured execution,
two head geometries and 1..6 slots. Then candidate-off/on/off calibration against
target-only batching, including isolation. Require >=5% aggregate gain before
native/media/holdout qualification. Check the speculative duplicate-slot fallback
at two sessions, even though it is not accelerated. Do not mutate the baseline.

## Initial evidence

The direct/captured GDN and convolution comparison passed 144 cases and
1,397,587,968 compared bytes with zero mismatches. The six-slot local component
measurement was 50.39 -> 27.93 microseconds; it is not an engine throughput rate.
A candidate-off/on/off IQ2_XS screen measured 180.40 / 191.49 / 180.01 aggregate
client token/s: +6.26% against the bracketed control mean. Isolation passed.
Full answer text also differs between the two controls, so six-slot complete
answer identity is not used as a quality claim.

Held-out ABBA and fallback checks are complete; see [the report](REPORT.md).
IQ2_XS gained 7.02%; IQ3_S gained 3.31%, below the performance gate.
Swift stopped at an official video-check failure before candidate qualification.
Only the tested target-only batch mode is accelerated. The batch-MTP configuration
with repeated slot rows remains on the old commit path. No default mode selection
or production rollout has occurred.
