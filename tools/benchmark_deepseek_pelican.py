"""Run the same isolated Claude Code pelican harness against DeepSeek's API."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,os,shutil,subprocess,threading,time,hashlib,statistics
import requests
import distribution as d
import qualify_distribution as q
from benchmark_pelican import PROMPT
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runtime/deepseek-pelican-20261011'
OUT.mkdir(exist_ok=False)
work=OUT/'workspace';work.mkdir()
key=(Path.home()/'.config/deepseek/claude-v41-flash.key').read_text().strip()
state={'folder':None,'counter':0}
class Proxy(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_POST(self):self.forward()
 def do_GET(self):self.forward()
 def forward(self):
  body=self.rfile.read(int(self.headers.get('Content-Length',0)));payload=json.loads(body) if body else None
  if self.path.split('?')[0]=='/v1/messages' and payload:
   payload.update(model='deepseek-flash',temperature=0,max_tokens=65536,thinking={'type':'enabled'},output_config={'effort':'max'})
   body=json.dumps(payload).encode()
  state['counter']+=1;index=state['counter'];folder=state['folder'];begin=time.monotonic();chunks=[]
  headers={k:v for k,v in self.headers.items() if k.lower() not in ('host','content-length','connection','accept-encoding','authorization','x-api-key')}
  headers['Authorization']='Bearer '+key
  session=requests.Session();session.trust_env=False
  try:
   with session.request(self.command,'https://api.deepseek.com/anthropic'+self.path,headers=headers,data=body,stream=True,timeout=(30,1800)) as response:
    self.send_response(response.status_code);self.send_header('Content-Type',response.headers.get('Content-Type','application/json'));self.send_header('Connection','close');self.end_headers();self.close_connection=True
    for chunk in response.iter_content(chunk_size=1):
     if chunk:
      chunks.append(chunk);self.wfile.write(chunk)
      if chunk==b'\n':self.wfile.flush()
    self.wfile.flush()
    if folder:
     (folder/f'api-{index}.response.txt').write_bytes(b''.join(chunks).replace(key.encode(),b'[REDACTED]'))
     d.save(folder/f'api-{index}.json',dict(path=self.path,request=payload,status=response.status_code,wall_s=time.monotonic()-begin))
  except Exception as e:
   if folder:d.save(folder/f'api-{index}-error.json',{'type':type(e).__name__,'message':str(e).replace(key,'[REDACTED]')})
   raise
  finally:session.close()
proxy=ThreadingHTTPServer(('127.0.0.1',0),Proxy);threading.Thread(target=proxy.serve_forever,daemon=True).start()
env=dict(os.environ)
for name in list(env):
 if name.startswith(('STRATA_','CUDA_MPS_','ANTHROPIC_')) or name in ('CUDA_VISIBLE_DEVICES','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','CLAUDE_CODE_OAUTH_TOKEN'):env.pop(name,None)
local={'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{proxy.server_port}','ANTHROPIC_API_KEY':'local-harness-test','CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1','DISABLE_AUTOUPDATER':'1','CLAUDE_CODE_MAX_OUTPUT_TOKENS':'65536','CLAUDE_CODE_EFFORT_LEVEL':'max'}
env.update(local,CLAUDE_CONFIG_DIR=str(OUT/'claude-config'))
d.save(OUT/'settings.json',{'env':local,'alwaysThinkingEnabled':True})
command=json.loads((ROOT/'runtime/readme-latest-pelican-20261011/command.json').read_text())
command[command.index('--settings')+1]=str(OUT/'settings.json');command[command.index('--model')+1]='deepseek-flash[1m]'
d.save(OUT/'command.json',command)
report=dict(task=PROMPT,model='deepseek-flash',model_family='DeepSeek-V4.1-Flash',effort='max',output_limit_per_request=65536,claude_version=subprocess.check_output(['claude','--version'],text=True).strip(),temperature_requested=0,temperature_effective='Ignored in DeepSeek thinking mode per provider documentation',seed='Not controlled by this provider API',network_included=True,runs=[])
try:
 for index in range(2):
  folder=OUT/f'{index}-deepseek';folder.mkdir()
  for child in work.iterdir():shutil.rmtree(child) if child.is_dir() else child.unlink()
  state.update(folder=folder,counter=0);begin=time.monotonic()
  with (folder/'claude.jsonl').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
   cli=subprocess.Popen(command,cwd=work,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
   try:code=cli.wait(timeout=1800)
   except subprocess.TimeoutExpired:q.stop(cli);raise RuntimeError('Claude Code task exceeded 30 minutes')
  elapsed=time.monotonic()-begin;state['folder']=None
  events=[json.loads(s) for s in (folder/'claude.jsonl').read_text().splitlines() if s.strip()]
  final=next((e for e in reversed(events) if e.get('type')=='result'),{})
  artifact=work/'pelican.html';valid=artifact.exists() and '<svg' in artifact.read_text().lower()
  if artifact.exists():shutil.copy2(artifact,folder/'pelican.html')
  row=dict(index=index,arm='deepseek',wall_s=elapsed,exit_code=code,html_svg_exists=valid,claude_result=final,api_requests=len(list(folder.glob('api-[0-9]*.json'))),html_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None)
  report['runs'].append(row);d.save(folder/'summary.json',row);d.save(OUT/'results.json',report)
  print('deepseek',index,round(elapsed,2),'HTML',valid,'exit',code,flush=True)
  assert code==0 and valid and not final.get('is_error'), 'Claude Code did not complete the task; inspect saved sanitized logs'
 report['median_s']=statistics.median(x['wall_s'] for x in report['runs']);d.save(OUT/'results.json',report)
finally:proxy.shutdown()
