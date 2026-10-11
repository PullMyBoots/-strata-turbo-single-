"""Export all completed measurements for user-selected presentation; do not edit README."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/all-speed-results-20261011'
NAMES={'iq2_xs':'IQ2_XS','swift':'Swift IQ3_XXS','iq3_s':'IQ3_S'}
curves=json.loads((ROOT/'evidence/readme-latest-curves-20261011/summary.json').read_text())
matrix=json.loads((ROOT/'evidence/readme-full-matrix-20261011/summary.json').read_text())
pelican=json.loads((ROOT/'evidence/pelican-latest-20261011/pelican-summary.json').read_text())
assert curves['complete'] and matrix['complete'] and pelican['complete']
assert matrix['starts']==48 and not matrix['failures']
OUT.mkdir(exist_ok=True,parents=True)
def save_csv(name,rows):
 with (OUT/name).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def markdown(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
decode=[]
for weight in NAMES:
 for size in (0,8192,65536,253952):
  for family in ('english','code','chinese'):
   points=[next(p for p in curves['curves'][f'{weight}-{arm}']['points'] if p['background_tokens']==size and p['family']==family) for arm in ('official','turbo')]
   a,b=points;decode.append(dict(weight=NAMES[weight],background_tokens=size,prompt_tokens=a['prompt_tokens'],task=family,strata_tps=a['median_decode_tps'],turbo_tps=b['median_decode_tps'],gain_pct=100*(b['median_decode_tps']/a['median_decode_tps']-1),strata_sample1=a['samples'][0],strata_sample2=a['samples'][1],turbo_sample1=b['samples'][0],turbo_sample2=b['samples'][1]))
save_csv('decode.csv',decode)
concurrent=[]
for weight in NAMES:
 for count in (1,2,4,6):
  a,b=[next(r for r in matrix['cells'] if r['weight']==weight and r['count']==count and r['arm']==arm) for arm in ('shared','latest')]
  concurrent.append(dict(weight=NAMES[weight],requests=count,strata_aggregate_tps=a['aggregate_e2e_tps'],turbo_aggregate_tps=b['aggregate_e2e_tps'],gain_pct=100*(b['aggregate_e2e_tps']/a['aggregate_e2e_tps']-1),strata_median_request_decode=a['median_request_decode'],turbo_median_request_decode=b['median_request_decode'],strata_start1=a['start_rates'][0],strata_start2=a['start_rates'][1],turbo_start1=b['start_rates'][0],turbo_start2=b['start_rates'][1]))
save_csv('concurrency.csv',concurrent)
prefill=[]
for weight in NAMES:
 for field,tokens in (('short_tps',31566),('native_tps',256013)):
  a,b=[next(r for r in matrix['prefill'] if r['weight']==weight and r['arm']==arm) for arm in ('shared','latest')]
  prefill.append(dict(weight=NAMES[weight],input_tokens=tokens,strata_tps=a[field],turbo_tps=b[field],gain_pct=100*(b[field]/a[field]-1)))
save_csv('prefill.csv',prefill)
trials=[]
for r in pelican['runs']:
 trials.append(dict(engine='Strata' if r['arm']=='official' else 'Strata-Turbo-Single',order=r['index'],wall_s=r['wall_s'],model_calls=r['model_calls'],output_tokens=r['total_output_tokens'],html_sha256=r['html_sha256']))
save_csv('pelican.csv',trials)
lines=['# Complete speed measurements','', 'All measured configurations are included. This report does not select README highlights.','', 'CSV: [Decode](decode.csv) · [Concurrency](concurrency.csv) · [Prefill](prefill.csv) · [Pelican](pelican.csv).','',
'One L40S; EPYC 9754; same selected target weights within each pair; native 262,144-token cap; INT8 KV. Strata is pinned to fb58e0d. Turbo uses sparse/fused prefill for one request and direct slot-state transfer for multiple requests. These experimental builds are not the installer default. Resource settings match within each pair; Swift/IQ3_S use 2,048 MiB headroom in the latest campaign. Swift uses minimum 128 visual tokens in both arms.','',
'## Decode','', 'Output-stage tokens/s, excluding prefill. Two 512-token outputs per task/length, median shown; temperature zero, thinking off. Prefix caching is enabled; first request at each size is cold. A single service start per engine/weight; two responses do not establish independent-start confidence intervals. The CSV includes both samples.','',markdown(['Weight','Background tokens','Task','Strata','Strata-Turbo-Single','Gain'],[[r['weight'],r['background_tokens'],r['task'],f"{r['strata_tps']:.1f}",f"{r['turbo_tps']:.1f}",f"{r['gain_pct']:+.1f}%"] for r in decode]),'',
'## Concurrent throughput','', 'Aggregate output tokens / wall time, including prefill and admission. Two independent starts per engine in ABBA order, three rotated English/code/Chinese waves per start, 256 output tokens per request, cache/thinking off. Per-request decode is a different metric and is retained separately in the CSV.','',markdown(['Weight','Requests','Strata aggregate','Turbo aggregate','Gain'],[[r['weight'],r['requests'],f"{r['strata_aggregate_tps']:.1f}",f"{r['turbo_aggregate_tps']:.1f}",f"{r['gain_pct']:+.1f}%"] for r in concurrent]),'',
'## Prefill','', 'Input tokens/s with zero cache hits. Mean of two independent single-request starts.','',markdown(['Weight','Input tokens','Strata','Strata-Turbo-Single','Gain'],[[r['weight'],r['input_tokens'],f"{r['strata_tps']:.0f}",f"{r['turbo_tps']:.0f}",f"{r['gain_pct']:+.1f}%"] for r in prefill]),'',
'## IQ2_XS pelican animation','', '> Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.','',f"Claude Code {pelican['claude_version']}; maximum effort, same initial rendered prompt; fresh sessions and workspaces; ABBA order. Wall time includes thinking, tool calls and retries, excludes model loading. Outputs and tool trajectories can differ.",'',markdown(['Engine','Order','Wall time','Model calls','Output tokens'],[[r['engine'],r['order'],f"{r['wall_s']:.1f} s",r['model_calls'],r['output_tokens']] for r in trials]),'',
'Median task times: Strata '+f"{pelican['engine_median_s']['official']:.1f}"+' s; Strata-Turbo-Single '+f"{pelican['engine_median_s']['turbo']:.1f}"+' s. Output volumes differ substantially; this is not an isolated numerical-engine speed ratio. [All four animations](../../docs/showcase/pelican-latest-20261011/index.html).','', '## Evidence','', '[Decode raw data](../readme-latest-curves-20261011/summary.json) · [Concurrency and Prefill raw data](../readme-full-matrix-20261011/summary.json) · [Pelican raw data](../pelican-latest-20261011/pelican-summary.json).','',
'All 48 matrix starts passed image/video/tool, isolation and 32k retrieval checks. Native retrieval passed in one- and six-slot configurations for all weights, additionally two/four slots for IQ2_XS. These checks do not establish comprehensive model-quality equivalence or simultaneous near-full-window performance.','']
(OUT/'RESULTS.md').write_text('\n'.join(lines))
print(OUT/'RESULTS.md')
