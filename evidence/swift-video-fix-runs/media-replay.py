import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import distribution as d
import qualify_distribution as q

name,gpu,binary=sys.argv[1:]
folder=ROOT/'runtime/swift-video-debug'/name
folder.mkdir(parents=True,exist_ok=False)
hw=d.detect(gpu)
cfg=d.build_config(ROOT/'assets',hw,'swift',d.resource_plan(hw,'swift'))
cfg['exe']=str(Path(binary).absolute())
if name.endswith('-parity'):
    cfg['env']['STRATA_DEBUG_FUSED_PARITY']='1'
if name.endswith('-cache'):
    cfg['env']['STRATA_DEBUG_CACHE_BYTES']='1'
if name.endswith('-all-gpu'):
    original_request=q.request
    def media_gpu_request(url, request, *args, **kwargs):
        if any(isinstance(message.get('content'),list) for message in request.get('messages',[])):
            request=dict(request,strata_tune={'pcie_frac':1.0})
        return original_request(url,request,*args,**kwargs)
    q.request=media_gpu_request
if name.endswith('-fixed-cache'):
    cfg['args']=d.set_arg(cfg['args'],'--adapt-swaps','0')
cfg.update(port=q.free_port(),host='127.0.0.1',log=str(folder/'engine.log'))
cfg['args']+=['--prompt-cache','8' if name.endswith('-cached') else '0']
if name.endswith('-cached'):
    original_cached_request=q.request
    def cached_request(url, request, *args, **kwargs):
        request=dict(request,cache_prompt=True)
        request['messages']=[{'role':'system','content':
            'Carefully answer the user and inspect supplied media. Use only the visible evidence. '*100},
            *request['messages']]
        return original_cached_request(url,request,*args,**kwargs)
    q.request=cached_request
d.save(folder/'config.json',cfg)
command,cwd=d.server_command(cfg,folder/'config.json')
url='http://127.0.0.1:'+str(cfg['port'])
report={'binary_sha256':d.digest(Path(cfg['exe'])),'cases':[]}
env=d.child_environment(cfg)
env['STRATA_DEBUG']='1'
with (folder/'server.log').open('w') as log:
    server=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        for _ in range(900):
            if server.poll() is not None:raise RuntimeError('Server exited')
            try:
                if q.session().get(url+'/v1/status',timeout=2).json().get('loaded'):break
            except Exception:pass
            time.sleep(1)
        else:raise RuntimeError('Startup timed out')
        if name.startswith('long-'):
            request,codes=q.recall_body(cfg['model_name'])
            result=q.request(url,request)
            assert all(code in result['text'] for code in codes)
            report['long_prefix']=result
        elif name.startswith('short-'):
            request=q.body(cfg['model_name'],'Write a complete Python program implementing a persistent key-value database with transactions, recovery, examples and tests. Output code only.')
            report['short_prefix']=q.request(url,request)
        for case,fun in [('video-first',q.video_smoke),('image',q.image_smoke),('video-after-image',q.video_smoke),('video-repeat',q.video_smoke)]:
            try:
                item=dict(case=case,passed=True,result=fun(url,cfg['model_name']))
            except Exception as error:
                item=dict(case=case,passed=False,error=str(error))
            report['cases'].append(item)
            print(json.dumps(item),flush=True)
        req=q.body(cfg['model_name'],'What is 2 plus 2? Answer with the number only.',16)
        req.pop('ignore_eos')
        result=q.request(url,req)
        report['post_media_text']=result
        assert '4' in result['text']
    finally:
        q.stop(server)
        report['stopped']=True
        d.save(folder/'results.json',report)
