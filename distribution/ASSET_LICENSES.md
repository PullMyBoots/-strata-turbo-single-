# Model-derived asset notices

The engine and model-derived auxiliary assets have separate licenses. The root MIT license covers the Strata fork's code, not model weights or derived router matrices.

- IQ2_XS and IQ3_S routers derive from ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF. Its model card states that quantized weights inherit the Qwen base-model license: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF#license
- The original Qwen3.8-Flash-Next components, including fetched MTP tensors, follow the Qwen Community License 1.0: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- The Swift router derives from UkisAI Swift 1.5 GSQ-RCO IQ3_XXS. The Swift contribution follows Swift Open License v1.0, while original Qwen components retain the Qwen license: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF#license-and-access

The original license texts and Swift NOTICE are preserved under `distribution/licenses/` with source URLs and hashes. Bundled router files are transformed auxiliary matrices, not a relicense of the original model. No target GGUF weights are bundled in the engine archives.
