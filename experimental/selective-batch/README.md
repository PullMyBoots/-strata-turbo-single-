# Selective proposal computation (experimental)

The earlier packed scheduler computed a proposal for every live session, even
when six base rows left room to verify only two proposals. The adaptive paired
variant also computed unused proposals while maintaining the draft state.

This candidate reserves one target row for every live session, then computes at
most one proposal for each session that fits the remaining verification budget.
Other sessions update their complete draft K/V through the existing prefill
operation, omitting proposal attention, experts and vocabulary projection. They
can resume proposal generation when load falls without reconstructing history.
Stable slot order chooses proposal owners; every session still receives a target
row, and the benchmark rotates task families across slots.

No target kernel, expert, context limit, vocabulary, sampling rule or generated
token is removed. The implementation reuses the pinned upstream numerical
libraries and is enabled only by both `STRATA_BATCH_PACKED_MTP=1` and
`STRATA_BATCH_SELECTIVE_MTP=1`. It does not inspect GPU names or tune hardware
hyperparameters. It has not yet established a performance benefit.

```sh
.venv/bin/python experimental/selective-batch/build.py
.venv/bin/python tools/probe_packed_batch.py \
  --output runtime/selective-batch-screen --gpu 0 --weights iq2_xs \
  --counts 6 --arms no-mtp selective no-mtp \
  --candidate-exe experimental/selective-batch/build/strata-selective-batch
# Only after a useful six-slot result, guard the existing low-load MTP path:
.venv/bin/python tools/probe_packed_batch.py \
  --output runtime/selective-batch-low-load --gpu 0 --weights iq2_xs \
  --counts 2 --arms incumbent selective incumbent \
  --candidate-exe experimental/selective-batch/build/strata-selective-batch
```

The initial six-slot calibration must beat target-only batching materially,
pass request isolation and preserve the low-load MTP benefit. Only a passing
screen justifies independent holdouts, load transitions, cancellation, cache,
native-context and media checks. All failed measurements remain in the evidence.
