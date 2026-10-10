# Consumer distribution

## Entry points

`setup.sh` bootstraps an isolated Python environment and runs `setup.py`. `tools/distribution.py` owns installation, resource planning, saved settings, and the selected service configuration. `start.sh` uses the consumer settings when present; the historical development launcher remains available through `./start.sh --legacy`.

The pinned upstream `setup.py` is imported as a helper for the model catalog, resumable downloads, GGUF completeness checks, and supported build-tool installation. It is not allowed to overwrite the optimized engine or silently download an official engine in its place. The upstream source snapshot and installer hash are recorded in `distribution/manifest.json`.

## Asset preparation

Each selected target has two GGUF shards, its own pack/tokenizer, projector, and model-matched full-vocabulary auxiliary router. The main GGUF files are never quantized again. MTP BF16 tensors are fetched by ranges and verified against pinned hashes, then prepared as an upstream Q2 auxiliary runtime and a Q5 auxiliary runtime. The complete Q5 directory intentionally excludes `draft_vocab.bin`; the target still verifies proposals using the full model vocabulary. The router loader checks the target head/vocabulary fingerprint at startup.

The default asset location is `models-data/` inside the extracted project. Use `--gguf-dir` for already downloaded shards, or `--adopt --assets` for a complete existing asset tree. Adopted files are checked and read, not rewritten. The project archive includes the small auxiliary routers; main GGUF and MTP expert weights are downloaded separately.

Incomplete downloads are resumed by the upstream downloader. Auxiliary preparation uses a temporary directory and a final rename. A process lock prevents two installers from racing over shared preparation/configuration in the same checkout.

## Resource planning

One GPU is selected by UUID; its identity is passed to both the text engine and GPU vision encoder. Automatic first-install selection chooses the card with the most free VRAM. Subsequent starts retain the selected UUID. CPU affinity, GPU-local CPU cores, NUMA, host available RAM, and visible cgroup memory limits participate in planning.

The context stays at 262144, KV stays INT8, and every target expert remains available. KV can stream to RAM when there is sufficient room; low-RAM mode maps experts from disk. WSL does not use pinned KV streaming. Small GPUs default to CPU vision, preserving image/video support at different preprocessing speed. The engine still automatically budgets its expert cache from free VRAM.

Concurrent mode shares weights in one native batch engine and queues requests beyond the configured maximum (1..6). Every new concurrency/resource combination needs its own speed qualification. The development checkout's older IQ2 MPS pool is not the portable consumer default.

On the large-GPU IQ2_XS single-session profile, prefill can borrow expert-cache slots in automatic chunks up to 16384 tokens. Verification is suspended while a loan is active. Refill completes the queued expert copies into their original slots, uploads the original residency table, and checks it before verification resumes. Adaptive expert exchanges stay disabled for this profile. The decode graphs, speculative settings, target weights and full context remain unchanged. Smaller-memory, Swift/IQ3_S, concurrent and historical development profiles retain their previous policy.

## Qualification and fallback

The local gate runs upstream/Turbo/Turbo/upstream, three prompt families with three variants per service start, greedy output, 512 generated tokens, prompt-cache disabled, arithmetic/recall checks, native images/video, and the Anthropic API. Single-session comparison uses output-stage token/s. Concurrent comparison uses aggregate generated tokens divided by batch wall time, including prefill and waiting.

A candidate is selected if it passes capability checks, gains at least 3% in the geometric mean of task-family medians, does not regress an output family by more than 3%, and does not regress tested prefill throughput by more than 10%. The prefill tolerance is explicit: selection does not mean every input stage is faster than upstream. The approximately 32k-token recall request measures prefill without adding another workload. `--verify-long` adds the bundled 256013-token prompt, fixed retrieval checkpoints, its prefill rate and its output-stage rate to the comparison. Reports separately record prefill, first-content latency and total request time. A passing short gate alone does not certify full-window performance or six simultaneous 262k sessions.

The saved identity includes GPU/driver, CPU affinity/NUMA, effective total RAM, resource plan, actual execution flags, binary hashes, preset/API/installer versions, and model asset size/mtime. Asset size/mtime is a change detector, not a cryptographic guarantee for imported weights. The qualification report records the baseline binary hashes and raw results. A different machine cannot inherit another machine's qualification just by extracting the archive.

If speed evidence is insufficient, the consumer runs upstream numerical kernels with the existing native media reader. Both the unmodified upstream text comparator and its media-adapted variant are packaged and can be built from source. The media adapter handles SVE2 positions; it does not alter upstream numerical kernels. If neither arm passes basic workload checks, installation reports failure and leaves an actionable log path.

## Build/platform boundary

`build_distribution.py` uses `STRATA_PORTABLE=ON` for an AVX2 CPU baseline and runtime dispatch of specialized CPU kernels. It builds the optimized engine, pinned official engine, official media-adapted engine, and GPU image/video encoder. Binaries retain recorded CUDA architectures and minimum glibc. This runtime archive requires glibc 2.38 or newer; an older glibc or an unmatched GPU architecture triggers a source build. The actual binary library dependencies are checked with `ldd`; required CUDA libraries can be installed into the private Python environment without changing the system driver.

Linux NVIDIA is the release target. Native Windows/AMD require their own builds, correctness checks, and speed measurements before being advertised. WSL is a source/experimental portability route, not a measured performance guarantee. Driver installation and hardware capacity are prerequisites.

The optimized single-session engine uses one-token target verification for media requests. An observed Swift video failure with multi-token verification motivated this guard. Text retains its tuned speculative decoding. Image/video capability is tested independently; text throughput measurements do not advertise media output speed.

Swift also restores its original expert-cache profile before native media requests and holds that placement for the request. This prevents a prior text request's adaptive cache history from changing the media execution state. Text retains adaptive caching on the full immutable arena. A compact RAM source keeps placement fixed instead. See [the request-state repair](swift-video-fix.md) for evidence and limits.

## Diagnostics and updates

Use `./setup.sh --check` to inspect the plan and `./setup.sh --verify --no-start` to repeat qualification. Inspect `runtime/qualification-runs/*/{server.log,engine.log,results.json}` for failed comparisons. Binaries/models are never switched to a different precision by a recovery path. Changing files/configuration requires requalification; published speed evidence remains tied to the exact binary that produced it.
