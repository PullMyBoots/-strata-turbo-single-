"""Replay bundled requests against a running engine; report output speed and exact-text parity."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import requests

ROOT=Path(__file__).resolve().parents[1]

def session():
    s=requests.Session();s.trust_env=False;return s

def run(url, request):
    start=time.monotonic();events=[]
    with session().post(url+'/v1/chat/completions',json=request,stream=True,timeout=(10,600)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b'data: ') and line!=b'data: [DONE]':
                event=json.loads(line[6:]);events.append(event)
                if event.get('error'):raise RuntimeError(event['error'])
    text=''.join(c.get('delta',{}).get('content','') for e in events for c in e.get('choices',[]))
    timings=next(e['timings'] for e in reversed(events) if e.get('timings'))
    usage=next(e['usage'] for e in reversed(events) if e.get('usage'))
    return {'text':text,'timings':timings,'usage':usage,'wall_s':time.monotonic()-start},events

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8090')
    p.add_argument('--output',type=Path,default=ROOT/'runtime/benchmark')
    p.add_argument('--long',action='store_true')
    p.add_argument('--device-label',required=True,help='Actual GPU/hardware label; no inferred device identity.')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for _ in range(240):
        try:
            if session().get(args.url+'/health',timeout=2).json().get('loaded'):break
        except (requests.RequestException,ValueError):pass
        time.sleep(1)
    else:raise RuntimeError('Engine startup timed out')
    status=session().get(args.url+'/v1/status',timeout=5).json()
    summary={'device_label':args.device_label,'status_before':status,'all_short_text_exact':True,
             'short_request_count':0,'main_weights_changed':False}
    rates={case:[] for case in ('chinese','english','code')}
    sample=json.loads((ROOT/'evidence/requests/single-short-chinese-1.json').read_text())
    run(args.url,dict(sample['request'],max_tokens=64))
    for case in rates:
        for repeat in range(1,4):
            name=f'single-short-{case}-{repeat}'
            spec=json.loads((ROOT/'evidence/requests'/(name+'.json')).read_text())
            result,events=run(args.url,spec['request'])
            exact=result['text']==spec['expected_text']
            (args.output/(name+'.json')).write_text(json.dumps({'request':spec['request'],
                'result':result,'events':events,'exact_reference_text':exact},ensure_ascii=False,indent=2)+'\n')
            assert exact and result['timings']['cache_n']==0 and result['usage']['completion_tokens']==512,name
            rates[case].append(result['timings']['predicted_per_second']);summary['short_request_count']+=1
            print(name,result['timings']['predicted_per_second'],exact,flush=True)
    medians={case:statistics.median(values) for case,values in rates.items()}
    summary.update(short_case_medians=medians,short_geomean=math.exp(sum(map(math.log,medians.values()))/3))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if args.long:
        spec=json.loads((ROOT/'evidence/requests/long-256k.json').read_text())
        result,events=run(args.url,spec['request'])
        exact=result['text']==spec['expected_text']
        (args.output/'long-256k.json').write_text(json.dumps({'request':spec['request'],
            'result':result,'events':events,'exact_reference_text':exact},ensure_ascii=False,indent=2)+'\n')
        assert exact and result['timings']['cache_n']==0
        assert result['usage']['prompt_tokens']==256013 and result['usage']['completion_tokens']==512
        assert all(x in result['text'] for x in json.loads((ROOT/'evidence/checkpoints.json').read_text()))
        summary.update(long=result['timings'],long_full_text_exact=True,long_retrieval=True)
        print('long-256k',result['timings'],exact,flush=True)
    summary['status_after']=session().get(args.url+'/v1/status',timeout=5).json()
    summary['passed']=True
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if not k.startswith('status')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
