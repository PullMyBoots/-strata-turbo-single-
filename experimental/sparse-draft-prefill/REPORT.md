# Prefill qualification report — 2026-10-11

Status: experimental candidate; the installed release has not been replaced.

## Results

Official v0.1.41 (`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`) versus the candidate on one L40S. All percentages below compare whole engine profiles. Decode gains include existing Turbo optimizations; they are not attributable solely to this prefill patch.

| Weight | Short decode, official → candidate (token/s) | 32k prefill throughput gain | Native prefill throughput gain | Native decode, official → candidate (token/s) | Native request time reduction |
|---|---:|---:|---:|---:|---:|
| IQ2_XS | 189.97 → 208.57 | 7.68% | 8.89% | 134.00 → 167.40 | 8.22% |
| Swift IQ3_XXS | 152.24 → 170.15 | 7.24% | 3.18% | 129.20 → 134.45 | 2.51% |
| IQ3_S | 133.89 → 148.00 | 7.90% | 3.76% | 114.85 → 126.50 | 3.40% |

## Mechanism and attribution

The new sparse-drafter path batches full-history KV construction and projects only the indexer keys needed during prefill. It retains the pooled index and publishes its rollback checkpoint after completion. It does not use GPU model names or remove experts/context. The target verifier remains in place.

Against the same compiled IQ3_S control, this path improved 32k/native prefill throughput by approximately 8.1%/7.8%. Batching alone closed the prefill deficit but did not consistently beat upstream. The final candidate additionally enables the existing upstream fused expert prefill implementation (`STRATA_PF_FUSED=1`). That component is upstream reuse, not a new algorithm invented here.

## Protocol

Official/candidate/candidate/official, two independent service starts per arm. Each start runs a warmup, English/code/Chinese short generation, cold 31,566-token retrieval, cold 256,013-token retrieval, image, video and tools checks. Native context is 262,144; KV is INT8, CPU workers 32, resident KV budget 32,768, reserve 700 MiB, prompt cache disabled, thinking disabled. Short decode is the geometric mean of three family means; prefill is measured from engine prompt timing. Native wall time is measured by the client.

IQ2_XS retains the shipped automatic 16k prefill cap; other profiles use 8k. Official and Turbo draft strategies also differ. This is a documented profile comparison, not a controlled attribution experiment for every optimization. It does not measure maximum-thinking Claude Code pelican completion.

The first IQ2_XS cohort had a CPU compilation overlap and is excluded in full. A complete clean ABBA replaced it. Both cohorts and the explicit caveat are archived. No confidence interval or statistical significance claim is made from two starts.

## Validation and remaining limits

All twelve service runs passed native recall and image/video/tools checks and stopped successfully. A separate IQ3_S cache-enabled test passed exact-prefix and changed-suffix continuation. Six fused expert format pairs passed the upstream numerical reference test; fused versus MMQ relative RMS was 0.00006256–0.0002998. Floating-point results can differ. These checks do not establish comprehensive intelligence equivalence.

Only L40S performance has been measured. The mechanism is hardware-independent in its decision logic, but a universal cross-hardware speedup has not been established. Six-way selective speculation lost 5.69% against bracketed target-only batching and is rejected. Neither all-metric superiority nor 280–300 token/s has been achieved. No new pelican timing is claimed.

## Reproducibility

Build instructions: [candidate README](README.md). Raw responses, engine logs, process records, excluded pilot and numerical tests are preserved in [raw-results.tar.gz](../../evidence/sparse-draft-prefill-20261011/raw-results.tar.gz), with per-file hashes in [manifest.json](../../evidence/sparse-draft-prefill-20261011/manifest.json). [Summary](../../evidence/sparse-draft-prefill-20261011/summary.json) and build manifests identify both binaries.

Recompute clean tables with `experimental/sparse-draft-prefill/summarize.py` after extracting the archive at the repository root. The summarizer rejects a marked pilot and requires all four runs to have stopped and passed capability checks.
