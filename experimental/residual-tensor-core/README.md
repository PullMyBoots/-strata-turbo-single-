# Residual Tensor Core projection candidate

Status: bounded operator screen, not a release optimization.

The verifier computes BF16-weight/FP32-activation router and indexer projections
with scalar FP32 FMAs. Multiple tokens reuse weight loads but still perform the
arithmetic on CUDA cores. This is distinct from previously rejected scheduling,
expert transfer and local tiling changes.

Candidate slate:

| Candidate | Benefit boundary | Risk | Decision |
|---|---|---|---|
| Residual BF16 activation decomposition + Tensor Core products | Router/indexer projections; potential grouped-read reuse later | Extra products, narrow matrices, changed FP32 reduction | Screen one implementation |
| Integer Tensor Core native experts | Larger compute fraction | Must preserve per-block scales and integer truncation; broad kernel rewrite | Defer until a precise equivalence contract exists |
| More speculative scheduling | Small/no observed benefit | Repeats several failed screens | Reject unchanged family |

Split each FP32 activation into three BF16 residual components. Multiply all
three against the unchanged BF16 weights, accumulate each product in FP32 and
combine residual products in increasing magnitude. No model weights, expert
count or context length changes. This is not bitwise equivalent: reductions and
Tensor Core arithmetic must be checked against both a double reference and the
incumbent kernel. No claim of an original mathematical invention is made.

The first implementation uses cuBLAS strided-batched products with shared
weights and three component batches; this avoids three host dispatches. Include
split/combine and graph launch costs in the comparison. Test native dimensions
and 1/2/4/6/8 rows, zero/outlier/cancellation inputs. Run only on GPU0 under the
shared benchmark lock. No production process changes.

Gate: finite outputs, reconstruction check, RMS error no more than twice the
incumbent plus 1e-7 relative tolerance, and at least 20% component gain on the
512x2560 router at 4 and 6 rows before an engine integration. Check route top-10
agreement if the speed gate passes. A component win is not a whole-engine claim.
Archive failures; do not tune GPU-specific parameters to rescue this screen.

Completed: all three variants rejected for speed. [Results and decision](REPORT.md).
The final source is the fused WMMA variant; previous sources are retained with evidence.
