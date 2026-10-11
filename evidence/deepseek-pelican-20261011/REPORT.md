# DeepSeek-V4.1-Flash pelican task

Claude Code 2.1.290, official DeepSeek Anthropic-compatible API, model `deepseek-flash`, thinking enabled, `output_config.effort=max`, 65,536 output tokens per call. Same task prompt, tool permissions and appended instructions as the local IQ2_XS test; two independent sessions in an empty workspace. Public API network and queue time are included.

> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.

| Trial | Completion time | Model calls | Output tokens |
|---|---:|---:|---:|
| 1 | 333.3 s | 21 | 80,727 |
| 2 | 181.6 s | 4 | 45,194 |

Median completion time: **257.5 seconds**. Both trials generated an HTML/SVG file successfully. All API calls returned HTTP 200. Initial requests match except metadata: **True**.

The previous local IQ2_XS medians were 379.5 s with Strata and 150.2 s with Strata-Turbo-Single. These are different models/deployments and generated different output volumes; the task times do not isolate numerical-engine speed or establish equivalent output quality. DeepSeek ignores temperature in thinking mode; this API test does not control a sampling seed. Two trials do not establish a stable cross-task speed ranking.

[Animations and original HTML](../../docs/showcase/deepseek-pelican-20261011/index.html) · [Full summary](summary.json) · [Sanitized raw evidence](raw.tar.gz).

[Official integration instructions](https://api-docs.deepseek.com/guides/coding_agents/) · [Thinking-mode parameter semantics](https://api-docs.deepseek.com/guides/thinking_mode/).

Local launch command: `claude-deepseek-flash`. Credentials are stored outside the repository and are excluded from this report and archive.
