# Historical v0.1.1 qualification

L40S physical GPU0, single session, IQ2_XS target unchanged, native 262144 context, INT8 KV. Alternating configurations after other GPU experiments were stopped. Chinese, English and code each ran three variants, 512 output tokens, temperature 0, prompt cache disabled; short score is the geometric mean of task medians.

| Configuration | Short round 1 | Short round 2 | Cold 256013-token input output rate |
|---|---:|---:|---:|
| v0.1.0 | 200.19 | 200.47 | 154.9 |
| v0.1.1 | 208.69 | 207.78 | 158.3 |

Units: tokens/s. Short gain approximately 3.9%; the 280-300 target was not achieved. Cold-long observations: v0.1.1 [158.3,158.2,159.0], control [154.8,154.9,155.4]. Prefill medians were 133.7 and 133.2 seconds and are excluded from output-stage rates. These are not task-independent floors.

The historical short-then-long protocol scored 165.5 versus 161.1 once per configuration. This warm-up differs from direct cold-long service starts and is reported separately.

Enabled changes: verified HC scratch reuse, expert gate/up to SwiGLU/Q8 fusion, two-row MMVQ, full 248320-token auxiliary vocabulary and 8192 candidate routing. Main weights, expert count, KV precision and target verification stay unchanged. Failed tree-verification, adapter-training, nonlinear state prediction and tensor-core experiments were not enabled.

36 short responses and six long responses across the two versions matched their references; retrieval passed. Reasoning, tools, images, native 8fps video, Anthropic API and queues passed. The actual default launcher subsequently scored 209.90 short token/s with 9/9 reference equality. Audio remains unsupported; this is not a complete intelligence benchmark.

Evidence: `evidence/qualification-r2.json`, `evidence/qualification-r2/`, `engine/manifest.json`. The newer consumer distribution uses independent portable binaries and local qualification rather than reusing this historical certificate.
