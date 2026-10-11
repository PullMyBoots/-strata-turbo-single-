"""Audit the latest six decode curves without overwriting historical cohorts."""
from pathlib import Path
import hashlib,json,statistics,tarfile
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runtime/readme-latest-curves-20261011'
OUT=ROOT/'evidence/readme-latest-curves-20261011'
NAMES={'iq2_xs':'IQ2_XS','swift':'Swift IQ3_XXS','iq3_s':'IQ3_S'}
summary={'curves':{},'complete':False,'protocol':'One fresh service per engine/weight; two 512-token responses per task and input length; prefix caching enabled; same paired requests; no independent-start confidence interval.'}
for weight in NAMES:
 pair=[]
 for arm in ('official','turbo'):
  folder=RUN/f'{weight}-{arm}-curves';r=json.loads((folder/'results.json').read_text());pair.append(r)
  expected='54bf7a29420294216f78fb72e1c46f7496fb54ec809a5c5c464b9d4607a9acb9' if arm=='official' else 'c89835335d1a26ba00c404bc2e7fed95c3f8d32b21f28e0e978260da73db173c'
  assert r['binary_sha256']==expected
  assert json.loads((folder/'process.json').read_text())['stopped']
  assert len(r['rows'])==24
  points=[]
  for size in (0,8192,65536,253952):
   for family in ('english','code','chinese'):
    rows=[x for x in r['rows'] if x['context_background_tokens']==size and x['family']==family]
    assert len(rows)==2
    assert all(x['result']['usage']['completion_tokens']==512 for x in rows)
    rates=[x['result']['timings']['predicted_per_second'] for x in rows]
    points.append(dict(background_tokens=size,family=family,prompt_tokens=rows[0]['result']['usage']['prompt_tokens'],median_decode_tps=statistics.median(rates),minimum_decode_tps=min(rates),maximum_decode_tps=max(rates),samples=rates,cached_tokens=[x['result']['timings']['cache_n'] for x in rows]))
  summary['curves'][f'{weight}-{arm}']={'points':points,'binary_sha256':expected,'protocol':r['protocol']}
 assert [x['request_sha256'] for x in pair[0]['rows']]==[x['request_sha256'] for x in pair[1]['rows']]
 for flag in ('--native','--ple-gguf','--expert-profile','--max-context','--pool-workers','--kv','--kv-resident','--vram-reserve-mib'):
  vals=[r['configuration']['args'][r['configuration']['args'].index(flag)+1] for r in pair];assert vals[0]==vals[1],(weight,flag,vals)
 assert pair[0]['configuration']['vision']==pair[1]['configuration']['vision']
summary['complete']=True
OUT.mkdir(exist_ok=True,parents=True)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['| Weight | Input | Strata | Strata-Turbo-Single | Gain |','|---|---|---:|---:|---:|']
for weight,name in NAMES.items():
 for size,label in ((0,'Short'),(253952,'~254k')):
  vals=[statistics.mean(p['median_decode_tps'] for p in summary['curves'][f'{weight}-{arm}']['points'] if p['background_tokens']==size) for arm in ('official','turbo')]
  lines.append(f'| {name} | {label} | {vals[0]:.1f} | {vals[1]:.1f} | {(vals[1]/vals[0]-1)*100:+.1f}% |')
(OUT/'table.md').write_text('\n'.join(lines)+'\n')
manifest={}
with tarfile.open(OUT/'raw.tar.gz','w:gz') as archive:
 for p in sorted(RUN.rglob('*')):
  if p.is_file():
   name=str(p.relative_to(ROOT));archive.add(p,arcname=name);manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('\n'.join(lines))
