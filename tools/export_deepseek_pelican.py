"""Export DeepSeek task timings and sanitized evidence without credentials."""
from pathlib import Path
import json,hashlib,tarfile,statistics
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'runtime/deepseek-pelican-20261011';out=ROOT/'evidence/deepseek-pelican-20261011';out.mkdir(parents=True,exist_ok=True)
j=json.loads((run/'results.json').read_text());assert len(j['runs'])==2
key=(Path.home()/'.config/deepseek/claude-v41-flash.key').read_text().strip().encode()
first=[];response_models=set();files=[run/'results.json',run/'settings.json',run/'command.json']
for row in j['runs']:
 folder=run/f"{row['index']}-deepseek";files+=list(folder.iterdir());calls=[]
 assert row['exit_code']==0 and row['html_svg_exists'] and not row['claude_result'].get('is_error')
 for p in sorted(folder.glob('api-*.json'),key=lambda p:int(p.stem.split('-')[1])):
  x=json.loads(p.read_text());assert x['status']==200
  if x['path'].split('?')[0]=='/v1/messages':
   b=x['request'];assert b['model']=='deepseek-flash' and b['output_config']['effort']=='max' and b['thinking']['type']=='enabled' and b['max_tokens']==65536;calls.append(x)
 for p in folder.glob('api-*.response.txt'):
  for line in p.read_text().splitlines():
   if line.startswith('data: '):
    try:event=json.loads(line[6:])
    except ValueError:continue
    if event.get('type')=='message_start':response_models.add(event['message'].get('model'))
 initial=dict(calls[0]['request']);initial.pop('metadata',None);first.append(initial)
 row['model_calls']=len(calls);row['output_tokens']=row['claude_result'].get('usage',{}).get('output_tokens')
j['first_requests_match_except_metadata']=first[0]==first[1];j['response_models']=sorted(response_models);j['complete']=True
j['documentation']=['https://api-docs.deepseek.com/guides/coding_agents/','https://api-docs.deepseek.com/guides/thinking_mode/'];manifest={}
with tarfile.open(out/'raw.tar.gz','w:gz') as archive:
 for p in files:
  if p.is_file():
   data=p.read_bytes();assert key not in data, 'Credential detected; refusing export'
   name=str(p.relative_to(ROOT));manifest[name]=hashlib.sha256(data).hexdigest();archive.add(p,arcname=name)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'summary.json').write_text(json.dumps(j,indent=2)+'\n')
rows=['| Trial | Completion time | Model calls | Output tokens |','|---|---:|---:|---:|']
for r in j['runs']:rows.append(f"| {r['index']+1} | {r['wall_s']:.1f} s | {r['model_calls']} | {r['output_tokens']:,} |")
(out/'REPORT.md').write_text('''# DeepSeek-V4.1-Flash pelican task

Claude Code 2.1.290, official DeepSeek Anthropic-compatible API, model `deepseek-flash`, thinking enabled, `output_config.effort=max`, 65,536 output tokens per call. Same task prompt, tool permissions and appended instructions as the local IQ2_XS test; two independent sessions in an empty workspace. Public API network and queue time are included.

> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.

'''+ '\n'.join(rows)+f"\n\nMedian completion time: **{j['median_s']:.1f} seconds**. Both trials generated an HTML/SVG file successfully. All API calls returned HTTP 200. Initial requests match except metadata: **{j['first_requests_match_except_metadata']}**.\n\n"+'''The previous local IQ2_XS medians were 379.5 s with Strata and 150.2 s with Strata-Turbo-Single. These are different models/deployments and generated different output volumes; the task times do not isolate numerical-engine speed or establish equivalent output quality. DeepSeek ignores temperature in thinking mode; this API test does not control a sampling seed. Two trials do not establish a stable cross-task speed ranking.

[Animations and original HTML](../../docs/showcase/deepseek-pelican-20261011/index.html) · [Full summary](summary.json) · [Sanitized raw evidence](raw.tar.gz).

[Official integration instructions](https://api-docs.deepseek.com/guides/coding_agents/) · [Thinking-mode parameter semantics](https://api-docs.deepseek.com/guides/thinking_mode/).

Local launch command: `claude-deepseek-flash`. Credentials are stored outside the repository and are excluded from this report and archive.
''')
gallery=ROOT/'docs/showcase/deepseek-pelican-20261011';gallery.mkdir(exist_ok=True,parents=True)
cards=[]
for r in j['runs']:
 name=f"deepseek-{r['index']+1}";cards.append(f'<article><h2>Trial {r["index"]+1}: {r["wall_s"]:.1f} s</h2><p>{r["model_calls"]} model calls · {r["output_tokens"]:,} output tokens</p><img src="{name}/pelican.gif" alt="DeepSeek pelican animation"><p><a href="{name}/pelican.html">Original HTML</a></p></article>')
(gallery/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DeepSeek pelican comparison</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px;background:#f5f5f2}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:24px}article{background:white;padding:20px;border-radius:12px}img{width:100%}</style><h1>DeepSeek-V4.1-Flash: pelican animation</h1><p>Claude Code 2.1.290 · max effort · official API · network included</p><main>'+''.join(cards)+'</main></html>')
print('Median seconds',round(j['median_s'],1),'initial request parity',j['first_requests_match_except_metadata'],'response models',j['response_models'])
