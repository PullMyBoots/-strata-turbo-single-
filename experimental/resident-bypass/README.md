# Exact resident-group dispatch (experimental)

This prototype is isolated from the installed engine. It changes scheduling and
mapped staging, not target weights or floating-point kernels. It is not yet a
qualified release optimization.

For each actual routed group, the existing GPU resident planner checks every
expert. When all are resident, the prototype publishes the actual expert IDs in
a per-step record and skips host activation quantization, plan/job construction,
CPU dispatch and activation transfer. Usage counts still reach the adaptive
cache. Any missing expert selects the original complete host path. The existing
global device-only path is unchanged.

Per-step records matter: a fast GPU can complete several resident groups before
the host reads a shared staging buffer. A shared skip flag alone cannot safely
identify the route the host should count. The final completion ring publishes
the payload after system fences from all mapped-memory writers.

The decision contains no GPU model names, device-specific thresholds or timing
hyperparameters. It uses the CUDA mapped-memory synchronization already required
by the engine. Builds inherit the architecture flags of the selected build tree.
This implementation is CUDA-only; portability of the dispatch rule does not
establish speed gains on untested devices or an implemented HIP port.

The prototype is disabled on split/peer verifiers and watchdog-stall tests. It
falls back if mapped trace allocation fails. Enable only on the isolated binary:
`STRATA_RESIDENT_HOST_BYPASS=1`. Leave it unset or set to `0` for the control.

```bash
python3 experimental/resident-bypass/build.py
python3 experimental/resident-bypass/build.py --parity
```

The build verifies pinned source identities, compiles an entry/verifier overlay
and a conditional doorbell, and links the unchanged numerical archives. Generated
sources, objects, executables and their manifests are under ignored `build/`.
The parity check covers direct launches and graph replays, mutable residency,
repeated IDs, slot offsets, all-hit, all-miss and mixed step sequences. Whole
engine controls, native context and capability qualification remain separate.
