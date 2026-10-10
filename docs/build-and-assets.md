# Build and assets

For consumer installation and model preparation, use `./setup.sh`; see [distribution](distribution.md). Main model weights and the MTP expert tensors are downloaded separately. The release carries model-matched auxiliary routers and source, but no main GGUF files or local asset links.

Build all consumer engines and the GPU native media encoder:

```bash
.venv/bin/python tools/build_distribution.py --arch 89 --nvcc /path/to/nvcc --jobs 8
```

The portable CPU baseline is AVX2; specialized CPU kernels dispatch at runtime. Set a semicolon-separated CUDA architecture list when building a multi-architecture archive. The consumer installer rebuilds for a mismatched GPU architecture. NVIDIA CUDA Toolkit, C++20 compiler and CMake >=3.24 are needed for a source build. Linux x86_64 NVIDIA is the qualified release target.

The historical development build remains available through `build.sh` and does not overwrite the consumer binaries:

```bash
CUDACXX=/path/to/nvcc STRATA_CUDA_ARCH=89 ./build.sh
./start.sh --legacy --weights iq2_xs --gpu 0
```

Typical consumer asset tree:

```text
models-data/
  models/IQ2_XS/                 # two target GGUF shards and projector
  models/Swift-1.5-IQ3_XXS/       # optional selected alternative
  models/IQ3_S/                  # optional selected alternative
  data/packs/<model>/            # native metadata, dense.bin, tokenizer/
  data/mtp/rt/                   # upstream auxiliary runtime
  data/mtp/native-q5-full/       # four complete-vocabulary auxiliary files
  data/mtp/head-router-*/        # model-matched router; about 35 MiB each
  data/expert-profile.bin
```

Do not use the IQ2 router for Swift or IQ3_S. The native loader verifies the head and vocabulary fingerprint. Main target GGUF files retain their published quantization; only auxiliary MTP experts are prepared in Q5. The full-vocabulary directory must not include the historical cropped `draft_vocab.bin`.

`--adopt --assets PATH` uses a complete tree without modifying it. `--gguf-dir PATH` imports already downloaded shards and prepares other assets. The older `tools/attach_assets.py` remains a development helper; it creates asset links, not a self-contained release package.

Create clean source/runtime archives:

```bash
.venv/bin/python tools/package_release.py
```

The archive root is `strata-turbo-single-0.2.0/`. Each contains `PACKAGE.json` with per-file SHA256. `artifacts/SHA256SUMS-0.2.0` records archive hashes. Private environments, local settings/credentials, models, symlinks, logs and unrelated experiments are excluded. The native media encoder uses bundled llama.cpp/mtmd source and does not start llama-server.
