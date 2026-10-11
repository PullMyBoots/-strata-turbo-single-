# Swift tiny-video input repair

A 160×90 video yielded only 15 visual tokens per encoded frame chunk. Swift sometimes answered with a timestamp instead of the requested color. Native prompt traces retained the question, frame ordering, and assistant header; missing text or frames was not established.

Setting `vision.min_tokens = 128` raises this fixture to 144 visual tokens per chunk (rounded to its aspect ratio). The prompt grows from 121 to 766 tokens. Frame rate, video bytes, target weights and output budget remain unchanged.

| Validation | Result |
|---|---|
| Unchanged Strata: fresh, after text, repeat, after 32k | 4/4 pass |
| Unchanged Strata: mixed six-request RGB videos/text IDs | Pass |
| Unchanged Strata: 256,013-token retrieval, subsequent video and mixed requests | Pass |
| Turbo direct-state: original failing six-way workload, image/video/tools/isolation/32k/native retrieval | 6/6 pass |

The same input setting is applied to both engines in the fresh Swift benchmark matrix. The production vision helper now accepts `--min-tokens`, and generated Swift distribution configurations set it to 128. Normal larger frames are unaffected unless their computed visual budget falls below the minimum. The default is empirical for this weight; it is not a claim about an official architectural requirement or comprehensive media quality.

Earlier placement, verification and prompt-tail numerical patches failed later regressions and are not selected. The unchanged Strata engine passing this intervention supports an input-resolution fix; it does not prove a unique internal logit-level cause.

[Check results](summary.json) · [Requests, traces and logs](raw.tar.gz) · [File hashes](manifest.json).
