"""Publish the three-weight pelican rerun without replacing previous trials."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'pelican-three-weights-20261011'
source = ROOT / 'runtime' / NAME
out = ROOT / 'evidence' / NAME
showcase = ROOT / 'docs/showcase' / NAME
out.mkdir(parents=True, exist_ok=True)
rows, manifest, files = [], {}, []
for weight in ('iq2_xs', 'swift', 'iq3_s'):
    run = source / weight
    report = json.loads((run / 'results.json').read_text())
    assert len(report['runs']) == 1
    row = report['runs'][0]
    assert row['exit_code'] == 0 and row['html_svg_exists']
    assert not row['claude_result'].get('is_error')
    calls = []
    for path in (run / '0-turbo').glob('api-*.json'):
        call = json.loads(path.read_text())
        assert call.get('status') == 200, path
        if call['path'].split('?')[0] == '/v1/messages':
            request = call['request']
            assert request['output_config']['effort'] == 'max'
            assert request['thinking']['type'] in ('enabled', 'adaptive')
            assert request['max_tokens'] == 65536
            calls.append(call)
    checks = json.loads((showcase / weight / 'render-checks.json').read_text())
    assert all(c['animated'] and c['self_contained'] and c.get('visually_reviewed') for c in checks)
    rows.append(dict(weight=weight, label=report['weights'], wall_s=row['wall_s'],
                     model_calls=len(calls), output_tokens=row['claude_result']['usage']['output_tokens'],
                     effort=report['effort'], claude_version=report['claude_version'],
                     thinking_limit_hits=(run / '0-turbo/server.log').read_text().count('the reply reached max tokens while still thinking'), binary_sha256=row['binary_sha256'], render_checks=checks))
    files += [run / name for name in ('results.json', 'command.json', 'settings.json', 'hardware.json', 'build.json')]
    files += [p for p in (run / '0-turbo').iterdir() if p.is_file()]
with tarfile.open(out / 'raw.tar.gz', 'w:gz') as archive:
    for path in files:
        data = path.read_bytes()
        # Local harness uses a dummy key; never include provider credentials.
        import re
        assert not re.search(rb'sk-[A-Za-z0-9]{20,}', data), path
        name = str(path.relative_to(source))
        manifest[name] = hashlib.sha256(data).hexdigest()
        archive.add(path, arcname=name)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
text = ['# Three-weight pelican rerun', '',
        'October 11, 2026. One new trial per weight on Strata-Turbo-Single, one L40S GPU (GPU 0), native 262,144-token context. Claude Code, max effort, thinking enabled, 65,536 output tokens per call, temperature 0 and seed 20261010. Same prompt, fresh service and empty workspace for every weight. Timing includes Claude Code, thinking, tools and retries, but excludes model loading and warm-up. These are task completion times, not decode-speed measurements.', '',
        '> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.', '',
        'The sparse/fused experimental build matches the latest decode measurements. VRAM reserve is 700 MiB for IQ2_XS and 2,048 MiB for Swift/IQ3_S. System NVML reports a driver/library mismatch following an automatic driver update; CUDA remains operational and is used for device discovery. No driver or system library was changed. The earlier user-interrupted IQ2_XS retry is excluded from completed trials.', '',
        '| Weight | Completion (s) | Model calls | Output tokens | Thinking-limit hits |', '|---|---:|---:|---:|---:|']
for row in rows:
    text.append(f"| {row['label']} | {row['wall_s']:.1f} | {row['model_calls']} | {row['output_tokens']:,} | {row['thinking_limit_hits']} |")
text += ['', 'All generated HTML files are preserved unedited. Browser checks confirm animation and no external resource requests. One trial per weight does not establish a stable speed or quality ranking. Earlier trials remain published separately.', '',
         '[Animation gallery](../../docs/showcase/' + NAME + '/index.html) · [Raw evidence](raw.tar.gz) · [Summary](summary.json).', '']
(out / 'REPORT.md').write_text('\n'.join(text))
cards=[]
for row in rows:
    folder=row['weight'] + '/turbo-1'
    cards.append(f'<article><h2>{row["label"]}</h2><p>{row["wall_s"]:.1f} seconds · max effort</p><img src="{folder}/pelican.gif" alt="Pelican bicycle animation"><p><a href="{folder}/pelican.html">Original HTML</a></p></article>')
(showcase / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three-weight pelican rerun</title><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:20px;background:#f5f6fa}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}article{padding:16px;background:white;border-radius:12px}img{width:100%}</style><h1>Strata-Turbo-Single: three-weight pelican rerun</h1><p>October 11, 2026 · Claude Code · max effort · one trial per weight · original outputs</p><main>'+''.join(cards)+'</main></html>')
readme=ROOT/'README.md'
s=readme.read_text()
s=s.replace('### Pelican animation · Claude Code', '### Initial pelican comparison · Claude Code')
section=['### New pelican rerun · all three weights', '', 'One fresh trial per weight, Claude Code with `max` effort. Original animations shown below; earlier comparison trials above are retained.', '', '| | IQ2_XS | Swift IQ3_XXS | IQ3_S |', '|---|---|---|---|']
section.append('| Animation | '+' | '.join(f'![{r["label"]}](docs/showcase/{NAME}/{r["weight"]}/turbo-1/pelican.gif)' for r in rows)+' |')
section.append('| Completion time | '+' | '.join(f'**{r["wall_s"]:.1f} s**' for r in rows)+' |')
section.append('| Original HTML | '+' | '.join(f'[Open](docs/showcase/{NAME}/{r["weight"]}/turbo-1/pelican.html)' for r in rows)+' |')
section += ['', f'[View all three outputs](docs/showcase/{NAME}/index.html) · [Test records](evidence/{NAME}/REPORT.md). Single trials; output lengths differ.', '', '']
assert '### New pelican rerun' not in s
readme.write_text(s.replace('### Decode · tokens/s', '\n'.join(section)+'### Decode · tokens/s'))
print(json.dumps(rows, indent=2))
