# Swift native-video request state

Swift could answer only `0:00` to a red video after an unrelated code or 32k-token text request. A fresh engine could answer the same video correctly. This was a capability failure; increasing the output limit or replacing the answer with a canned response was not used.

## Request-scoped repair

The Swift single-session preset enables `STRATA_MEDIA_FIXED_PROFILE=1`. The serving loop saves the original expert residency at startup. Before a native image/video request it waits for pending adaptive exchanges, restores changed expert bytes to their original slots, waits for those copies, then publishes the original residency table. The request retains that placement until it finishes. Ordinary text requests continue using adaptive caching, fused expert computation and the tuned speculative decoder.

Prefill can still borrow cache slots and must refill them before decode. The target GGUFs, full expert set, INT8 KV, 262144-token context, native video reader, 8 fps default and one-token media verification are unchanged. On a compact RAM source without the immutable full expert arena, this opt-in keeps placement fixed throughout instead of corrupting the RAM/GPU ownership planner. Performance on that resource mode requires local qualification.

## Diagnostic boundaries

Disabling PLE prefetch or shared-expert overlap did not fix the failure. Fused gate/up computation passed bitwise comparison against separate kernels at the actual model dimensions, with partial groups, reused scratch and wider scales. Instrumenting the actual failing sequence also found no fused activation mismatch. Resident cache bytes matched their expected experts. Sending all media decode experts to the GPU still failed.

Disabling all adaptive exchanges passed repeated RGB, temporal-order and media-after-native-window checks, but its paired short-output gain was only 0.67%, so that whole-service configuration was rejected. The final repair limits restoration to media requests. The detailed low-level mechanism of adaptive-history sensitivity remains unresolved; the evidence supports the tested request-state repair, not a claim that an IQ arithmetic defect has been identified.

## Validation

The request-scoped candidate passed a 31566-token text request followed by video, image, video and repeated video. It restored 853 slots in 56.1 ms in that replay. The complete regression passed code generation, 32k checkpoint recall, seven red/green/blue or temporal-order video checks, image understanding, a cold 256013-token request with all five retrieval checkpoints, and final arithmetic. A cache-enabled replay also passed; the repeated video reused 1719 tokens and processed seven new tokens.

On L40S GPU0, the final ABBA comparison combined six observations per English, Chinese and code task family, with 512 output tokens each. Swift's aggregate output-stage throughput was **176.4 token/s versus 160.0 upstream (+10.3%)**. Code medians were 200.1 versus 176.4 token/s. The score excludes prefill and does not advertise video output speed; approximately 32k prefill was 8.3% below upstream, within the explicit 10% qualification tolerance. The full-window regression validates capability and has no paired official long-speed control.

Final qualification and archive checks are recorded in the accompanying evidence. Capability checks cover these fixtures and retrieval tasks; they are not a comprehensive intelligence benchmark. No activation certificate is bundled for another machine.

See [the measured report](../evidence/swift-video-fix.json) and [prefill comparisons](prefill-upgrade.md).
