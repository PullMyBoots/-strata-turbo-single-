"""Reproduce native-batch media failures after text and compare state policies."""
import argparse
import base64,json,re,subprocess,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import distribution as d
import qualify_distribution as q
from benchmark_showcase import TASKS,measure
from showcase_lock import acquire

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--exe',type=Path);p.add_argument('--fixed',action='store_true');p.add_argument('--no-adapt',action='store_true');p.add_argument('--regression',action='store_true');p.add_argument('--reserve-mib',type=int);p.add_argument('--trace',action='store_true');p.add_argument('--vision-min-tokens',type=int);a=p.parse_args()
with acquire():
 hw=d.detect('0');cfg=d.build_config(d.ROOT/'assets',hw,'swift',d.resource_plan(hw,'swift',6),arm='official')
 cfg['args']=d.set_arg(cfg['args'],'--prompt-cache','0')
 if a.exe:
  cfg['exe']=str(a.exe.absolute());cfg['env'].update(STRATA_BATCH_SELF_COMMIT='1',STRATA_DIRECT_SLOT_STATE='1')
 if a.fixed:cfg['env']['STRATA_MEDIA_FIXED_PROFILE']='1'
 if a.reserve_mib:cfg['args']=d.set_arg(cfg['args'],'--vram-reserve-mib',str(a.reserve_mib))
 if a.trace:cfg['env']['STRATA_MEDIA_TRACE_DIR']=str(a.output.absolute()/'media-trace')
 if a.vision_min_tokens is not None:
  cfg['vision']['min_tokens']=a.vision_min_tokens
 if a.no_adapt:cfg['args']=d.set_arg(cfg['args'],'--adapt-every','0')
 def run(url,config,folder):
  checks=[]
  def video(label):
   try:r=q.video_smoke(url,config['model_name'])
   except (RuntimeError,ValueError) as e:r=dict(passed=False,error=str(e))
   checks.append(dict(label=label,**r));d.save(folder/'checks.json',checks);print(label,r,flush=True)
  video('fresh')
  for family,task in TASKS.items():
   req=q.body(config['model_name'],task,256);req.pop('ignore_eos',None)
   d.save(folder/(family+'.json'),q.request(url,req))
  video('after-text');video('repeated')
  req,codes=q.recall_body(config['model_name']);req.pop('ignore_eos',None)
  r=q.request(url,req);d.save(folder/'recall.json',r)
  checks.append(dict(label='recall',passed=all(c in r['text'] for c in codes)))
  video('after-32k')
  if a.regression:
   requests=[]
   for color in ('red','green','blue'):
    path=folder/(color+'.mp4')
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','lavfi','-i',f'color=c={color}:s=96x64:r=8:d=1','-c:v','libx264','-pix_fmt','yuv420p',str(path)],check=True)
    req=q.body(config['model_name'],'Describe the video.',256);req.pop('ignore_eos',None)
    req['messages'][0]['content']=[dict(type='text',text='What is the dominant color throughout this video? Answer in English.'),dict(type='video_url',video_url=dict(url='data:video/mp4;base64,'+base64.b64encode(path.read_bytes()).decode()))]
    requests.append((color,req))
   for i in range(3):
    expected=f'SESSION-MIX-{i}-NOVA-{8719+i*17}'
    req=q.body(config['model_name'],'Repeat exactly: '+expected,64);req.pop('ignore_eos',None)
    requests.append((expected,req))
   def mixed(label):
    gate=threading.Barrier(7)
    def one(item):
     expected,req=item;gate.wait();result=q.request(url,req)
     return dict(expected=expected,passed=expected.lower() in result['text'].lower(),request=req,result=result)
    with ThreadPoolExecutor(max_workers=6) as pool:
     jobs=[pool.submit(one,item) for item in requests];gate.wait();results=[j.result() for j in jobs]
    d.save(folder/(label+'.json'),results)
    checks.append(dict(label=label,passed=all(r['passed'] for r in results)))
    print(label,[(r['expected'],r['passed']) for r in results],flush=True)
   mixed('mixed-six')
   original=json.loads((d.ROOT/'evidence/requests/long-256k.json').read_text())
   req=original['request'];req.update(model=config['model_name']);req.pop('ignore_eos',None)
   result=q.request(url,req);d.save(folder/'native.json',result)
   codes=re.findall(r'QF-[1-5]-[A-F0-9]{12}',original['expected_text'])
   checks.append(dict(label='native-recall',passed=len(codes)==5 and all(c in result['text'] for c in codes) and result['usage']['prompt_tokens']==256013))
   video('after-native');mixed('mixed-after-native')
   d.save(folder/'checks.json',checks)
  return dict(checks=checks)
 measure(cfg,a.output.absolute(),run)
