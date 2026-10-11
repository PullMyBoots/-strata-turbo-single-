# Resident dispatch result, 2026-10-11

Decision: retain this isolated prototype; do not promote it or change README
performance claims. Its measured incremental gain is below the 5% gate.

The user requested a mechanism reusable across hardware rather than a device
parameter sweep. This prototype uses actual route residency to avoid redundant
host work and activation transfer. Its decisions contain no device names or
hardware-specific performance thresholds. The floating-point kernels, target
weights, all routed experts, INT8 KV and native context limit remain unchanged.

| Three-family mean output rate, token/s | Installed engine, bracketed | Candidate disabled | Candidate enabled | Incremental mechanism gain |
|---|---:|---:|---:|---:|
| Normal adaptive execution, two samples/family | 156.73 | 159.43 | 159.98 | +0.345% |
| Fixed four-row verification, one sample/family | 128.62 | 126.77 | 126.97 | +0.158% |

Each arm starts a fresh GPU0 service, performs the same 64-token warmup and
completes English, code and Chinese requests with exactly 512 output tokens.
Prompt caching is disabled. The context limit is 262144; these are short-input
screens, not native-window measurements. The fixed execution diagnostic disables
adaptive lookahead, cache swaps and suffix proposals equally in every arm. The
fixed setting is a comparability check, not a performance preset.

All three fixed-window outputs match byte-for-byte between the candidate's on
and off modes. Normal adaptive answers differ between arms, including incumbent
restarts, so the adaptive screen alone does not isolate mathematical effects.
Automatic cache allocation also gives the installed engine 18104 slots and the
candidate 18103 slots; this placement difference can affect CPU/GPU arithmetic
and is retained rather than hidden. The fixed English answer differs between the
installed and candidate-disabled engines, so the mechanism's incremental effect
is reported against the same candidate executable with the flag disabled.

The conditional publication check passes 960 direct/graph cases, including
mutable residency and all-hit/all-miss/mixed sequences: 9216 resident and 6144
fallback groups, with 58,558,464 checked payload bytes. The existing integer
resident plans match the CPU reference in group order, entries and pointers.
Eight owned services were stopped. The installed executable is unchanged.

The mechanism reduced host preparation, but that work largely overlaps GPU
execution. It does not remove the dominant target computation and therefore
does not establish a substantial complete-engine acceleration. It offers no
additional path for IQ2_XS when every expert is already resident.

Only an L40S was measured. A hardware-independent dispatch rule is not evidence
of universally positive speedups. No native-window, image/video/tool or broad
quality campaign was run for this rejected candidate; publication as an
experimental artifact is not release qualification.

The changed conditional publication kernel also compiles for SM75, SM80, SM86,
SM89 and SM90. This checks compilation of the mechanism across CUDA targets;
the rest of the engine and performance were not validated on those devices.

Raw requests, responses, timing, source/library identities and process cleanup
records are in [evidence](../../evidence/resident-bypass-20261011/report.json).
Do not repeat this unmodified bypass or attribute the small gain to a new
intelligence-preserving decoding algorithm. Further search needs a route that
reduces the dominant verified target work rather than merely its overlapped
host preparation.
