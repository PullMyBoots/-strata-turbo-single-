# Three-weight pelican rerun

October 11, 2026. One new trial per weight on Strata-Turbo-Single, one L40S GPU (GPU 0), native 262,144-token context. Claude Code, max effort, thinking enabled, 65,536 output tokens per call, temperature 0 and seed 20261010. Same prompt, fresh service and empty workspace for every weight. Timing includes Claude Code, thinking, tools and retries, but excludes model loading and warm-up. These are task completion times, not decode-speed measurements.

> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.

The sparse/fused experimental build matches the latest decode measurements. VRAM reserve is 700 MiB for IQ2_XS and 2,048 MiB for Swift/IQ3_S. System NVML reports a driver/library mismatch following an automatic driver update; CUDA remains operational and is used for device discovery. No driver or system library was changed. The earlier user-interrupted IQ2_XS retry is excluded from completed trials.

| Weight | Completion (s) | Model calls | Output tokens | Thinking-limit hits |
|---|---:|---:|---:|---:|
| GSQ-RCO IQ2_XS | 349.7 | 23 | 86,567 | 1 |
| Swift IQ3_XXS | 165.6 | 7 | 30,308 | 0 |
| GSQ-RCO IQ3_S | 453.3 | 30 | 65,860 | 0 |

All generated HTML files are preserved unedited. Browser checks confirm animation and no external resource requests. One trial per weight does not establish a stable speed or quality ranking. Earlier trials remain published separately.

[Animation gallery](../../docs/showcase/pelican-three-weights-20261011/index.html) · [Raw evidence](raw.tar.gz) · [Summary](summary.json).
