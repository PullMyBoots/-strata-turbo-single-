# Swift native-batch video repair

| Sequence | Strata | Fixed placement control | Request-scoped repair |
|---|---|---|---|
| Video after standard 64-token warm-up | Fail (`0:00`) | Pass | Pass |
| Video after English/code/Chinese text | Fail (`0:00`) | Pass | Pass |
| Repeated video | Pass | Pass | Pass |
| Video after 31,566-token text | Pass | Pass | Pass |

The repaired engine additionally passed three simultaneous RGB-video requests plus three independent text-identifier requests, a 256,013-token retrieval request, a video after that request, and another six-way mixed batch. All media use the real native video input path; the request does not disclose the expected color.

## Cause and repair

Adaptive expert placement makes Swift's video response sensitive to prior text execution. Disabling placement changes eliminates the failure in this sequence. Restoring the startup placement only for media also eliminates it, while keeping text adaptation enabled. CPU and GPU expert paths can round differently; this is a plausible explanation for placement sensitivity, not a proved per-logit diagnosis. No corrupted cache bytes were established by this experiment.

The patch drains adaptive work, fences active device readers, restores original expert bytes, then publishes their residency table. Adaptive exchanges stay suspended while media is active, including concurrent media slots. The implementation respects reduced elastic cache capacity and keeps placement fixed in compact RAM mode, where immutable full expert bytes are unavailable.

Target GGUFs, expert count, native context, KV format, prompt, output budget and media preprocessing are unchanged. The baseline remains unmodified. This single-GPU patch is not yet installed as the default engine. Functional validation is bounded to these fixtures and retrieval checks, not comprehensive model-quality equivalence.

[Source and build](../../experimental/native-media-profile/README.md) · [All check outcomes](summary.json) · [Raw requests, responses and logs](raw.tar.gz) · [Hashes](manifest.json).
