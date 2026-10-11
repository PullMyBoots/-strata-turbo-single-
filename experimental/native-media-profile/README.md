# Native batch media profile repair

Swift's native batch engine can return only `0:00` for a video depending on prior request history. In the controlled sequence, the unmodified engine failed fresh/after-text red-video checks; disabling adaptive expert placement passed all four checks. A request-scoped repair also passed those checks while retaining adaptation for text.

The repair drains pending swaps, fences device readers, restores original expert bytes and their residency table before media, and suppresses adaptive exchanges while a media request or media slot is active. Cache slots reclaimed by elastic VRAM resizing remain nonresident. Compact RAM mode, without an immutable full expert arena, retains fixed placement throughout instead of violating its ownership mapping. The flag is opt-in: `STRATA_MEDIA_FIXED_PROFILE=1`.

This identifies adaptive placement/history sensitivity and repairs the observed behavior. It does not establish the exact logit-level cause or broad model-quality equivalence. Target weights, context, KV precision, output limits and native video preprocessing are unchanged.

Build after the direct-slot-state candidate:

```bash
.venv/bin/python experimental/native-media-profile/build.py
.venv/bin/python tools/probe_swift_media_state.py \
  --output runtime/swift-media-reproduce \
  --exe experimental/native-media-profile/build/strata-media-profile \
  --fixed --regression
```

The baseline executable is not modified. Compile/link commands and hashes are recorded in `build/build.json`. The source patch is relative to the direct-slot-state generated entry. This is a single-GPU repair; multi-GPU activation is disabled.
