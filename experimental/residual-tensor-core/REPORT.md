# Residual Tensor Core screen: rejected

Three implementations were evaluated on GPU0 (L40S), against the existing
`bf16_gemv_fp32_mmvf_multi` object, without modifying the installed engine.
The FP32 inputs were decomposed into three BF16 residuals; the target BF16
weights were unchanged. This is an established arithmetic construction, not
a claim of a new mathematical algorithm.

| Variant | Mechanism | Numerical cases passing | Router speed gate |
|---|---|---:|---|
| 1 | Three strided cuBLAS batches, separate split/combine | 5 / 25 | Failed |
| 2 | Residuals folded into columns, 256-element partial products, FP64 reduction | 7 / 25 | Failed |
| 3 | Shared-memory residual construction and WMMA, 128-element partial products, FP64 reduction | 25 / 25 | Failed |

Variant 3's 512x2560 router projection took 12.72 microseconds at four rows,
versus 4.68 for the incumbent; at six rows it took 12.78 versus 6.67.
The complete split/product/combine work was timed through CUDA graphs in
three ABBA rounds, each with 200 graph launches after warmup. These are
operator timings, not output token throughput.

Exact activation reconstruction did not guarantee accurate dot products:
variant 1 accumulated noticeably larger error against the double-precision
reference. Shorter products reduced that error, but added work and narrow
matrix geometry defeated speed. Variant 2's isolated larger-matrix speedups
failed the numerical gate and do not justify engine integration.

All three revisions cover five matrix shapes and 1/2/4/6/8 tokens, with normal
inputs, outliers, a zero token where applicable, and opposing weight-row pairs.
This is not exhaustive numerical validation. The 10240x320 case compares the
MMVF primitive, not the engine's fused hyper-connection implementation.
No route top-10, full-model, quality, or cross-hardware claim follows.

Decision: reject this implementation family after its declared bounded screen.
Do not promote the faster but numerically rejected cases. A future attempt needs
a materially different cost model, not another split-size sweep. The prior
prefill candidate remains the best newly validated gain.

Raw outputs, hardware records, process exits and all three source versions are
in [the evidence directory](../../evidence/residual-tensor-core-20261011/).
All test processes have exited. Rebuild the final prototype with
`.venv/bin/python experimental/residual-tensor-core/build.py`; historical
variants can be selected with `--source evidence/residual-tensor-core-20261011/v1/probe.cu`.
Use the benchmark lock and an otherwise idle GPU for reproduction.
