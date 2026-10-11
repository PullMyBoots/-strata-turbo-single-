# Resource policy versus engine preset

The original README decode, prefill and concurrency curves use **upstream with
shared resource tuning**. Both arms already receive the distribution's worker,
resident-KV and VRAM-reserve settings. That is a stricter comparison than the
historical default-resource runs, which is why their gain percentages differ.

The new three-arm comparison measures those contributions separately. It uses
two simultaneous requests, all three weights and the README's English, code and
Chinese holdout prompts. Every arm has two fresh process starts, in balanced
default/shared/Turbo/Turbo/shared/default order. Each start gets the same
64-token warm-up and three rotated batches of 256 output tokens per request.
Temperature is zero; thinking and prefix caching are disabled. The native
262,144-token cap and INT8 KV remain unchanged.

| Arm | CPU expert workers | Configured resident KV budget | VRAM reserve | Batch MTP |
|---|---|---:|---:|---|
| Upstream resource defaults | Engine automatic selection | 32,768 | 700 MiB | Off |
| Upstream + shared tuning | 32 | 16,384 | 1,400 MiB | Off |
| Turbo concurrent preset | 32 | 16,384 | 1,400 MiB | On |

These are the pinned launcher's resource defaults on this non-hybrid EPYC host,
with no saved upstream calibration. The pinned setup leaves worker selection
automatic on this CPU, adds a 32,768 resident-KV budget for a large native context,
and uses a 700 MiB vision reserve. The numerical executable, target GGUF,
expert profile, RT drafter, API/media reader, GPU and NUMA placement are common
across all three arms. It is a controlled resource ablation, not an untouched
installation test. Upstream also supports its own optional calibration.

The main rate is total generated tokens divided by batch wall time, including
prefill, admission and waiting for the last request. The summary gives the
median and observed range of two process-level rates; this is not a confidence
interval. No generated answer is cached or shortened to inflate throughput.
Raw data include actual completion counts, zero cache-hit checks, per-request
rates, configurations, binary hashes and stopped-process records.

The default-to-Turbo difference includes hardware resource policy and upstream's
existing batch-MTP feature. It is not evidence of a new CUDA kernel or decoding
algorithm. The shared-to-Turbo difference isolates the remaining batch-MTP
preset effect under common resource settings. The broader original 1/2/4/6-slot
curves remain in the README, including six-slot regressions. These measurements
do not validate six simultaneous near-full-context sessions.

[All measurements](../evidence/resource-comparison/summary.json) ·
[Figure](figures/resource-comparison.svg) ·
[Original methodology](benchmark-methodology.md) ·
[Rejected follow-up candidates](../experimental/packed-batch/README.md).

## Reproduce

Prepare all three weights, then run sequentially on the chosen idle GPU. Do not
render animations or run another inference campaign while this suite is active.
The shared benchmark lock enforces one campaign at a time.

```bash
for strata_weight in iq2_xs swift iq3_s
do
  .venv/bin/python tools/probe_packed_batch.py \
    --output "runtime/resource-comparison-${strata_weight}" \
    --gpu 0 --weights "$strata_weight" --counts 2 \
    --arms default shared incumbent incumbent shared default \
    --tokens 256 --waves 3 --seed 20261010 --holdout
done

.venv/bin/python tools/export_resource_comparison.py \
  --input runtime/resource-comparison-iq2_xs \
          runtime/resource-comparison-swift \
          runtime/resource-comparison-iq3_s
.venv/bin/python docs/figures/scripts/plot_resources.py
```

No candidate compilation is required. Inspect the PNG after regenerating the
figure. Absolute rates depend on this complete L40S/EPYC/RAM system.
