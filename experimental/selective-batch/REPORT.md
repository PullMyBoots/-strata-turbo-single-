# Selective batch screen — rejected

Six simultaneous IQ2_XS requests on GPU0, bracketed target-only / candidate / target-only. Three task families rotate across slots.

| Arm | Aggregate client throughput (token/s) |
|---|---:|
| Target-only, first | 180.719 |
| Selective proposals | 169.403 |
| Target-only, repeat | 178.533 |

The candidate is 5.69% slower than the mean of the controls. All isolation checks pass. Outputs also vary between the two controls, so exact answer identity is not established by this experiment.

Skipping unused proposal work did not yield a net throughput gain. The component-level cause is not isolated. The 5% improvement gate failed; two-slot and broader qualification were not run. Do not promote this candidate or claim a concurrency improvement.

Raw results are preserved in `evidence/sparse-draft-prefill-20261011/raw-results.tar.gz`, under `runtime/selective-batch-screen-20261011`.
