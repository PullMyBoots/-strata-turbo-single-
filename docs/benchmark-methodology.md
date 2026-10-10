# README benchmark methodology

This report records fresh measurements of the **v0.2.2** distribution on 2026-10-10. Historical results elsewhere in the repository use different builds and protocols; they are not substituted into these charts. An initial campaign overlapped throughput and agent inference on different GPUs. Because they can share CPU and memory resources, its IQ2_XS pairs, upstream Swift runs, and all pelican trials were excluded and rerun sequentially. Already completed Swift Turbo and IQ3_S runs taken after the agent suite ended were retained. [Rerun provenance](../evidence/showcase-pilot-audit.json) records this decision; the excluded pilot is not used for README claims.

## Hardware and comparator

Throughput experiments use one NVIDIA L40S, GPU 0 (`0000:01:00.0`), on NUMA node 0. The host has AMD EPYC 9754 CPUs and approximately 1 TB RAM. The allocator selects 32 CPU workers and records CPU affinity, available memory, and VRAM in each configuration. Pelican experiments use a different single L40S, GPU 2, on the same host. Do not compare absolute rates across GPU placements as if they were a paired experiment.

Both arms load the same target GGUF shards, tokenizer, expert profile, and projector within each weight pair. Both retain the native 262,144-token context limit and INT8 KV storage. MTP/router assets and speculative settings are arm-specific release presets: the single-request Turbo path uses its native Q5 draft and model-matched router, while upstream uses its RT draft. Draft proposals are verified against the selected target model. Runtime residency and execution settings also follow the release's hardware-selected presets; this is a distribution comparison, not a single-kernel ablation.

The upstream source is pinned to Strata commit `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. The common media reader changes input embedding handling while preserving upstream numerical kernels. The single-request Turbo binary is the optimized release binary. For multiple slots, both arms use native shared batching; the Turbo preset additionally enables batch MTP. All build hashes are in [the summary](../evidence/showcase/benchmark-summary.json) and [binary manifest](../distribution/build.json).

## Exact Claude Code animation task

```text
Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.
```

The [harness](../tools/benchmark_pelican.py) runs official, Turbo, Turbo, official in that order. Each trial uses a freshly loaded service, the same 64-token warm-up, the same emptied working directory, and an isolated Claude Code configuration. Claude Code 2.1.290 uses `--bare`, no session persistence, no MCP servers, and the Write/Read/Bash/Edit tools. An identical added instruction saves `pelican.html` and ends the task without rendering or installing software.

Wire requests set temperature 0, seed 20261010, and prompt caching on. CLI `--effort max` sends `output_config.effort=max` with adaptive thinking; the API maps this to native `xhigh`. Each call permits 65,536 generated tokens, shared by reasoning and final output. There is no extra thinking cap. The exporter verifies identical first requests after removing session metadata. Subsequent requests differ as generated outputs and tool paths diverge.

The engine status verifies its 262,144-token limit. Claude Code reports a 200,000-token heuristic window for this custom model in its usage metadata. Pelican requests do not approach that window; this showcase does not validate Claude Code at the full native limit. The long-input throughput experiment exercises the native window directly through the API. The output-budget statement is checked on actual wire requests, rather than the harness's model metadata.

Wall time starts immediately before launching Claude Code and ends at CLI exit. It includes startup, reasoning, generation, tool execution, and retries. Model loading and warm-up are excluded. Reported output-token totals are the harness's usage values across model calls; they are not independently parsed reasoning-token counts. The local API does not charge the estimated prices displayed by the harness.

All four trials exit successfully and save an HTML file containing SVG. The median is calculated from two trials per arm. Original outputs are preserved byte-for-byte, with SHA256 hashes in the evidence. The ratio is task completion time, not a controlled fixed-token speed ratio. The small sample, differing output lengths, and tool paths limit generalization; no formal visual-quality ranking is inferred.

Chromium renders each original at 1000×720. External requests are intercepted and recorded. The original source is not modified. The capture clock advances through the first six seconds at 20 fps, with SVG/CSS and JavaScript animation timelines controlled for capture. More than ten distinct frames confirms movement. GIFs are viewable on GitHub; HTML retains the interactive original. [Render checks](showcase/pelican/render-checks.json) record animation and dependency observations.

## Decode curves

The [throughput harness](../tools/benchmark_showcase.py) runs three weights, each with upstream and Turbo. Each arm starts a fresh service and receives the same 64-token warm-up. Input sizes are traversed in increasing order, and every size has two consecutive outputs per task family:

| Task family | Requested output |
|---|---|
| English | A detailed numbered software-project organization/review guide |
| Python code | An asyncio scheduler with limits, retries, cancellation, logging, and a demo |
| Chinese | An explanation of inference bandwidth, caches, prefill, and generation |

Exact prompts are in the script and their SHA256 hashes are in the raw results. Background prefixes contain 0, 8,192, 65,536, or 253,952 tokens from [the recorded long-context corpus](../evidence/requests/long-256k.json), encoded with the model's native tokenizer. System and task text add roughly 100 tokens; charts use measured prompt-token counts, not padding sizes. Input plus output always fits the 262,144-token native limit.

All requests use temperature 0, seed 20261010, thinking disabled, EOS ignored, and exactly 512 generated tokens. Output-stage speed is the engine's `predicted_per_second` timing, excluding prefill. Each point shows the median and observed minimum–maximum of two samples. Two samples from one process are not independent-start repetitions and do not provide a confidence interval.

KV prefix caching is enabled. A distinct system prefix for each size makes the first request cold; its `cache_n=0` is asserted. Later tasks/repeats can reuse KV prefixes; cache hits are preserved in raw evidence. No complete generated answer is served from cache. Both arms use the same ordering and request bodies. This protocol resembles repeated work over a shared context; it is not a cold-start measurement for every plotted point.

The IQ2_XS and IQ3_S pairs traverse upstream then Turbo; the retained Swift Turbo run precedes its upstream recheck. This is not ABBA for each curve point. Drift and workload sensitivity remain possible. No points are removed because Turbo is slower. The fixed-length, thinking-disabled test does not estimate highest-reasoning agent completion time.

## Cold prefill

For every model/arm/size, the first English request has zero cached input. Prefill uses the engine's `prompt_per_second` and `prompt_ms`. The six-configuration chart uses the approximately 254k-token request; the README also reports approximately 8k and 64k. Each condition has one cold observation. Short-input prefill exists in raw evidence but startup-size overhead makes it less informative.

Prefill is neither output throughput nor total request time. Only input-stage timing is used, and cached repeat requests are never used to claim cold-prefill improvement.

## Concurrent throughput

Each arm/model starts a fresh service for each slot count: 1, 2, 4, and 6. After the same warm-up, three batches run with a barrier so all requests start together. Task families rotate across requests and batches. Each request generates 256 tokens, with temperature 0, seed 20261010, thinking off, EOS ignored, and prefix caching off. The harness asserts zero cache hits.

Aggregate end-to-end throughput is:

```text
sum(generated output tokens across 3 batches) / sum(batch wall time)
```

Batch wall time includes prefill, scheduling, admission, generation, HTTP overhead, and waiting for the final request. This is not the sum of engine-reported per-request decode rates. Raw reports retain per-request decode, first-content latency, batch duration, and task families. For one/two slots, a batch contains only one/two task families; the three-batch rotation covers all families. The four/six-slot batches contain mixed tasks.

These are short-input concurrency experiments. Simultaneous near-full-context sessions are not validated, and users should not infer linear scaling from the supported slot limit. Concurrency presets share one model copy; single-request and concurrent engines have different execution paths. The setup's local qualification workloads differ from these README holdout prompts, so a locally qualified preset can still regress on a chart point.

## Reproduce

Install the selected models and prepare a complete asset tree first. Run on an otherwise idle chosen GPU and avoid other inference jobs on the CPU/memory-sharing host. The scripts hold a project-wide lock so the throughput and agent suites run sequentially even on different GPUs. They select their own local ports and stop only the services they create; they do not modify saved installation settings. Use a new output directory for each campaign.

```bash
# All three weights, both arms, curves and 1/2/4/6-slot runs.
.venv/bin/python tools/benchmark_showcase.py \
  --assets /path/to/assets --gpu 0 --suite all \
  --output runtime/readme-benchmark-reproduction

# Requires locally installed Claude Code; run sequentially if only one GPU is available.
.venv/bin/python tools/benchmark_pelican.py \
  --assets /path/to/assets --gpu 0 \
  --output runtime/pelican-reproduction

# Export only after both suites finish; rejects incomplete campaigns.
.venv/bin/python tools/export_showcase.py \
  --benchmark runtime/readme-benchmark-reproduction \
  --pelican runtime/pelican-reproduction

# Optional chart dependencies, separate from engine dependencies.
.venv/bin/python -m pip install -r tools/requirements-showcase.txt
.venv/bin/python docs/figures/scripts/plot_showcase.py

# Requires Node.js/npx, Playwright CLI, and its Chromium installation.
.venv/bin/python tools/render_pelican.py \
  --input runtime/pelican-reproduction --output docs/showcase/pelican

# Refresh README tables only when evidence and rendered HTML hashes match.
.venv/bin/python tools/update_showcase_docs.py
```

Verify the asset and binary hashes before interpreting a reproduction as the same experiment. Different GPU/NUMA placement, available memory, CPU load, GPU clocks, selected presets, driver/compiler versions, and prompts can change results. Re-render and visually inspect exported charts after changing data.
