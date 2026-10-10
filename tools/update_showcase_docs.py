"""Update README tables and the gallery from complete, rendered showcase evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'iq2_xs': 'IQ2_XS', 'swift': 'Swift IQ3_XXS', 'iq3_s': 'IQ3_S'}
ARMS = {'official': 'Strata upstream', 'turbo': 'Strata-Turbo-Single'}


def main():
    data = json.loads((ROOT / 'evidence/showcase/benchmark-summary.json').read_text())
    pelican = json.loads((ROOT / 'evidence/showcase/pelican-summary.json').read_text())
    checks = json.loads((ROOT / 'docs/showcase/pelican/render-checks.json').read_text())
    assert data['complete'] and pelican['complete'] and len(checks) == 4
    assert pelican['first_requests_match_except_metadata']
    assert pelican['effort'] == 'max' and pelican['native_reasoning_effort'] == 'xhigh'
    counts = {'official': 0, 'turbo': 0}
    runs = {}
    for row in pelican['runs']:
        arm = row['arm']
        counts[arm] += 1
        name = f'{arm}-{counts[arm]}'
        check = next(c for c in checks if c['run'] == name)
        assert check['html_sha256'] == row['html_sha256'] and check['animated'] and check['self_contained']
        runs[name] = row
    median = pelican['engine_median_s']
    reduction = 100 * (1 - median['turbo'] / median['official'])
    ratio = median['official'] / median['turbo']
    comparison = (f'{reduction:.1f}% lower ({ratio:.2f}× faster completion)' if reduction >= 0
                  else f'{-reduction:.1f}% higher')
    rows = []
    for arm, label in ARMS.items():
        first, second = runs[arm+'-1'], runs[arm+'-2']
        rows.append(f"| {label} | {first['wall_s']:.1f} s | {second['wall_s']:.1f} s | **{median[arm]:.1f} s** | {first['model_calls']} / {second['model_calls']} | {first['total_output_tokens']:,} / {second['total_output_tokens']:,} |")
    gpu = pelican['hardware']['gpu']
    section = f'''## 1. Claude Code pelican showcase

The same **GSQ-RCO IQ2_XS** model completed this exact task through **Claude Code {pelican['claude_version'].split()[0]}**, using either the pinned upstream engine or Strata-Turbo-Single:

> {pelican['task']}

Both engines used one {gpu['name']} on GPU {gpu['index']}, native context, `--effort max` (mapped to the backend's highest `xhigh` reasoning level), and **{pelican['output_limit_per_request']:,} output tokens per model call**. Thinking and final output share that budget; no separate thinking limit was imposed. Each trial started with an empty workspace and a fresh service/session.

| Pinned Strata upstream — first trial | Strata-Turbo-Single — first trial |
|:---:|:---:|
| <img src="docs/showcase/pelican/official-1/pelican.gif" width="400" alt="Upstream model-generated pelican animation"> | <img src="docs/showcase/pelican/turbo-1/pelican.gif" width="400" alt="Turbo model-generated pelican animation"> |
| **{runs['official-1']['wall_s']:.1f} s** · [Original HTML](docs/showcase/pelican/official-1/pelican.html) | **{runs['turbo-1']['wall_s']:.1f} s** · [Original HTML](docs/showcase/pelican/turbo-1/pelican.html) |

| Engine | Trial 1 | Trial 2 | Median completion time | Model calls, trials 1 / 2 | Reported output tokens, trials 1 / 2 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

In this four-trial **ABBA** test, Turbo's median task time was **{comparison}**. Timing includes CLI startup, thinking, generation, tool calls, and retries through successful completion. Model loading and the identical 64-token warm-up are excluded. The first model requests match after removing session metadata.

These are two trials per engine. Output lengths and tool paths differ, so the completion-time ratio is not a pure decoder speedup or a model-quality score. All four original HTML files are preserved without editing. Chromium checks confirm animated SVG content and no requested external dependencies; GIFs show the first six seconds at 20 fps.

View every result in the [four-trial gallery](docs/showcase/pelican/index.html), or open the second-trial originals: [upstream](docs/showcase/pelican/official-2/pelican.html), [Turbo](docs/showcase/pelican/turbo-2/pelican.html). Download HTML files and open them locally; GitHub displays their source. [Timing and API evidence](evidence/showcase/pelican-summary.json) · [Render checks](docs/showcase/pelican/render-checks.json).

'''
    path = ROOT / 'README.md'
    readme = path.read_text()
    before, rest = readme.split('## 1. Claude Code pelican showcase', 1)
    _, after = rest.split('## 2. Throughput across tasks, context, and concurrency', 1)
    readme = before + section + '## 2. Throughput across tasks, context, and concurrency' + after
    prefill = ['| Weight | Engine | ~8k tokens/s | ~64k tokens/s | ~254k tokens/s | ~254k prefill time |',
               '|---|---|---:|---:|---:|---:|']
    concurrent = ['| Weight | Engine | 1 request | 2 requests | 4 requests | 6 requests | 6-slot median request decode |',
                  '|---|---|---:|---:|---:|---:|---:|']
    decode = ['| Weight | Short upstream / Turbo | Short change | ~254k upstream / Turbo | Long change |',
              '|---|---:|---:|---:|---:|']
    for weight, name in NAMES.items():
        means = {}
        for size in (0,253952):
            rates = [sum(p['median_decode_tps'] for p in data['curves'][weight+'-'+arm]['points']
                         if p['background_tokens'] == size)/3 for arm in ARMS]
            means[size] = (rates,100*(rates[1]/rates[0]-1))
        short, long = means[0],means[253952]
        decode.append(f'| {name} | {short[0][0]:.1f} / {short[0][1]:.1f} | {short[1]:+.1f}% | {long[0][0]:.1f} / {long[0][1]:.1f} | {long[1]:+.1f}% |')
        for arm, label in ARMS.items():
            cold = {p['background_tokens']: p for p in data['curves'][weight+'-'+arm]['cold_prefill']}
            assert all(p['cached_tokens'] == 0 for p in cold.values())
            prefill.append(f"| {name} | {label} | " + ' | '.join(f"{cold[size]['prefill_tps']:,.0f}" for size in (8192,65536,253952)) + f" | {cold[253952]['prefill_s']:.1f} s |")
            slots = {p['count']: p for p in data['concurrency'][weight+'-'+arm]}
            concurrent.append(f"| {name} | {label} | " + ' | '.join(f"{slots[count]['aggregate_e2e_tps']:.1f}" for count in (1,2,4,6)) + f" | {slots[6]['median_request_decode_tps']:.1f} |")
    for key, table in [('DECODE SUMMARY',decode),('PREFILL',prefill),('CONCURRENCY',concurrent)]:
        begin, end = f'<!-- BEGIN {key} TABLE -->', f'<!-- END {key} TABLE -->'
        left, rest = readme.split(begin,1)
        _, right = rest.split(end,1)
        readme = left + begin + '\n\n' + '\n'.join(table) + '\n\n' + end + right
    path.write_text(readme)
    gallery = ROOT / 'docs/showcase/pelican/index.html'
    html = gallery.read_text()
    import re
    for arm, label in [('official','Upstream'),('turbo','Turbo')]:
        for trial in (1,2):
            html = re.sub(rf'{label} · trial {trial} · [0-9.]+ s',
                          f"{label} · trial {trial} · {runs[f'{arm}-{trial}']['wall_s']:.1f} s",html)
    html = re.sub(r'<p><strong>Median:</strong>.*?</p>',
                  f"<p><strong>Median:</strong> upstream {median['official']:.1f} s; Turbo {median['turbo']:.1f} s ({abs(reduction):.1f}% {'less' if reduction >= 0 else 'more'} time). Small-sample local measurements, not a decoder ratio or formal visual-quality ranking.</p>",html)
    gallery.write_text(html)
    print('Updated README and gallery from complete evidence; rendered HTML hashes match.')


if __name__ == '__main__':
    main()
