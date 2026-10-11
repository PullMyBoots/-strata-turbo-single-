"""Compare resource and verified-batching variants without replacing installed engines.

Every run retains full request/results, actual token counts, source identity,
and process cleanup. Calibration prompts are separate from README holdouts.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time

import distribution as d
import qualify_distribution as q
from benchmark_showcase import TASKS, measure
from showcase_lock import acquire

CALIBRATION = {
    'english': 'Write a very detailed numbered handbook for maintaining a community garden. Cover planning, budgets, tools, seasonal tasks, volunteers, mistakes, and examples. Continue for at least 100 numbered sections.',
    'code': 'Write a complete Python file implementing a bounded producer-consumer work queue with typed dataclasses, priority ordering, deadlines, retries, safe shutdown, metrics and an extensive demonstration. Include detailed comments and at least 250 lines.',
    'chinese': '用中文详细介绍一座城市如何规划公共交通、步行和自行车网络，分析预算、换乘、无障碍、运营、维护与居民需求，给出具体例子，连续写至少2000字。',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--weights', choices=tuple(d.MODELS), default='iq2_xs')
    p.add_argument('--counts', nargs='+', type=int, choices=range(2,7), default=[2,6])
    p.add_argument('--capacity', type=int, choices=range(2,7), help='Keep this server capacity while varying active request counts')
    p.add_argument('--arms', nargs='+', choices=('default','shared','incumbent','legacy','no-mtp','control','packed','gdn','gdn-plain','packed-gdn','adaptive','adaptive-gdn','selective','self-commit','self-commit-control','self-commit-mtp','direct-state','direct-state-control','profile'),
                   default=['incumbent','packed','incumbent'])
    p.add_argument('--candidate-exe', type=Path)
    p.add_argument('--tokens', type=int, default=256)
    p.add_argument('--waves', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--holdout', action='store_true')
    p.add_argument('--media', action='store_true')
    p.add_argument('--native-context', action='store_true')
    a = p.parse_args()
    if a.capacity is not None and a.capacity < max(a.counts):
        p.error('--capacity must accommodate every requested count')
    if a.candidate_exe is None and any(arm in ('control','packed','gdn','gdn-plain',
                                              'packed-gdn','adaptive','adaptive-gdn','selective','self-commit','self-commit-control','self-commit-mtp','direct-state','direct-state-control') for arm in a.arms):
        p.error('--candidate-exe is required for isolated candidate arms')
    lock = acquire()
    out = a.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    hw = d.detect(a.gpu)
    d.save(out/'hardware.json', hw)
    tasks = TASKS if a.holdout else CALIBRATION
    references = {}
    summary = []
    for count in a.counts:
        for index, arm in enumerate(a.arms):
            cfg = d.build_config(d.ROOT/'assets', hw, a.weights,
                                 d.resource_plan(hw,a.weights,a.capacity or count),
                                 arm='official' if arm in ('default','shared') else 'turbo')
            d.check_config(cfg)
            cfg = copy.deepcopy(cfg)
            cfg['args'] = d.set_arg(cfg['args'],'--prompt-cache','0')
            if arm == 'default':
                cfg['args'] = d.set_arg(cfg['args'],'--pool-workers')
                cfg['args'] = d.set_arg(cfg['args'],'--kv-resident','32768')
                cfg['args'] = d.set_arg(cfg['args'],'--vram-reserve-mib','700')
            if arm == 'legacy' and '--batch-mtp' not in cfg['args']:
                cfg['args'].append('--batch-mtp')
            if arm == 'no-mtp':
                cfg['args'] = d.set_arg(cfg['args'],'--batch-mtp')
            if arm in ('control','packed','gdn','gdn-plain','packed-gdn','adaptive','adaptive-gdn','selective','self-commit','self-commit-control','self-commit-mtp','direct-state','direct-state-control'):
                cfg['exe'] = str(a.candidate_exe.absolute())
                cfg['env']['STRATA_BATCH_PACKED_MTP'] = '1' if arm in ('packed','packed-gdn','selective') else '0'
                cfg['env']['STRATA_BATCH_SELECTIVE_MTP'] = '1' if arm == 'selective' else '0'
                cfg['env']['STRATA_BATCH_GDN_FUSED'] = '1' if arm in ('gdn','gdn-plain','packed-gdn','adaptive-gdn') else '0'
                cfg['env']['STRATA_BATCH_MTP_ADAPT'] = '1' if arm in ('adaptive','adaptive-gdn') else '0'
                if arm == 'gdn-plain':
                    cfg['args'] = d.set_arg(cfg['args'],'--batch-mtp')
            if arm in ('self-commit','self-commit-control','self-commit-mtp','direct-state','direct-state-control'):
                cfg['exe'] = str(a.candidate_exe.absolute())
                cfg['env']['STRATA_BATCH_SELF_COMMIT'] = '0' if arm == 'self-commit-control' else '1'
                if arm != 'self-commit-mtp':
                    cfg['args'] = d.set_arg(cfg['args'],'--batch-mtp')
            if arm in ('direct-state','direct-state-control'):
                cfg['exe'] = str(a.candidate_exe.absolute())
                cfg['env']['STRATA_BATCH_SELF_COMMIT'] = '1'
                cfg['env']['STRATA_DIRECT_SLOT_STATE'] = '1' if arm == 'direct-state' else '0'
                cfg['args'] = d.set_arg(cfg['args'],'--batch-mtp')
            if arm == 'profile':
                cfg['env']['STRATA_VERIFY_PROFILE']='1'

            def run(url,config,folder):
                rows=[]
                families=list(tasks)
                for wave in range(a.waves):
                    gate=threading.Barrier(count+1)
                    def one(i):
                        family=families[(i+wave)%3]
                        req=q.body(config['model_name'],tasks[family],a.tokens)
                        req.pop('ignore_eos',None)
                        req['seed']=a.seed
                        gate.wait()
                        result=q.request(url,req)
                        signature=hashlib.sha256(result['text'].encode()).hexdigest()
                        key=(count,wave,i)
                        if index==0: references[key]=signature
                        row=dict(family=family,wave=wave,index=i,request=req,result=result,
                                 text_sha256=signature,equals_first_arm=signature==references[key])
                        d.save(folder/f'wave{wave}-request{i}.json',row)
                        assert result['usage']['completion_tokens']==a.tokens,result['usage']
                        assert result['timings']['cache_n']==0,result['timings']
                        return row
                    with ThreadPoolExecutor(max_workers=count) as pool:
                        jobs=[pool.submit(one,i) for i in range(count)]
                        begin=time.monotonic();gate.wait()
                        requests=[j.result() for j in jobs]
                        wall=time.monotonic()-begin
                    row=dict(wave=wave,wall_s=wall,requests=requests,
                             aggregate_e2e_tps=count*a.tokens/wall)
                    rows.append(row)
                    d.save(folder/'waves.json',rows)
                    print(a.weights,count,arm,wave,round(row['aggregate_e2e_tps'],2),
                          'equal',all(r['equals_first_arm'] for r in requests),flush=True)

                checks={}
                # Simultaneous requests must return their own distinct task IDs.
                expected=[f'SESSION-{i}-NOVA-{8191+137*i}' for i in range(count)]
                gate=threading.Barrier(count+1)
                def isolated(i):
                    req=q.body(config['model_name'],f'Repeat this exact identifier and nothing else: {expected[i]}',64)
                    req.pop('ignore_eos',None);gate.wait()
                    result=q.request(url,req)
                    passed=expected[i] in result['text'] and not any(code in result['text'] for j,code in enumerate(expected) if j!=i)
                    return dict(passed=passed,request=req,result=result)
                with ThreadPoolExecutor(max_workers=count) as pool:
                    jobs=[pool.submit(isolated,i) for i in range(count)]
                    gate.wait();isolation=[job.result() for job in jobs]
                checks['isolation']=dict(passed=all(r['passed'] for r in isolation),requests=isolation)
                if a.media:
                    for name,fun in [('image',q.image_smoke),('video',q.video_smoke),('tools',q.tool_smoke)]:
                        try: checks[name]=fun(url,config['model_name'])
                        except (RuntimeError,ValueError) as e: checks[name]=dict(passed=False,error=str(e))
                    recall,codes=q.recall_body(config['model_name']);recall.pop('ignore_eos',None)
                    result=q.request(url,recall)
                    checks['recall']=dict(passed=all(code in result['text'] for code in codes),result=result)
                if a.native_context:
                    request=json.loads((d.ROOT/'evidence/requests/long-256k.json').read_text())
                    body=request['request'];body.update(model=config['model_name'],seed=20261011)
                    body.pop('ignore_eos',None);result=q.request(url,body)
                    import re
                    codes=re.findall(r'QF-[1-5]-[A-F0-9]{12}',request['expected_text'])
                    checks['native_recall']=dict(passed=len(codes)==5 and all(code in result['text'] for code in codes)
                        and result['usage']['prompt_tokens']==256013 and result['timings']['cache_n']==0,
                        result=result)
                d.save(folder/'capability-checks.json',checks)
                aggregate=sum(count*a.tokens for _ in rows)/sum(r['wall_s'] for r in rows)
                return dict(rows=rows,checks=checks,aggregate_e2e_tps=aggregate,
                            all_equal=all(r['equals_first_arm'] for row in rows for r in row['requests']),
                            screening_only=True,holdout=a.holdout,configured_capacity=a.capacity or count,
                            protocol='Fresh service per count/arm; identical 64-token warm-up; three-family rotated simultaneous batches; aggregate includes prefill/admission; temperature zero; thinking/cache off.')

            report=measure(cfg,out/f'c{count}-{index}-{arm}',run)
            row=dict(count=count,index=index,arm=arm,aggregate_e2e_tps=report['aggregate_e2e_tps'],
                     per_family_decode={family:statistics.median(r['result']['timings']['predicted_per_second']
                        for wave in report['rows'] for r in wave['requests'] if r['family']==family)
                        for family in tasks},equal=report['all_equal'],checks=report['checks'])
            summary.append(row);d.save(out/'summary.json',summary)
            assert all(check['passed'] for check in report['checks'].values()),report['checks']
    lock.close()
    print(json.dumps([{k:v for k,v in row.items() if k!='checks'} for row in summary],indent=2),flush=True)


if __name__=='__main__':
    main()
