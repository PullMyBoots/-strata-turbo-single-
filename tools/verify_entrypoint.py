"""Verify this workspace's own Python/start.sh, then release the test GPU."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import requests

ROOT=Path(__file__).resolve().parents[1]

def main():
    summary={'passed':False,'test_gpu':0,'port':18084,'test_server_stopped':False}
    s=requests.Session();s.trust_env=False
    process=None
    try:
        with (ROOT/'runtime/entrypoint-server.log').open('w') as log:
            env=dict(os.environ);env.pop('STRATA_PYTHON',None)
            process=subprocess.Popen([str(ROOT/'start.sh'),'--gpu','0','--port','18084'],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(240):
                if process.poll() is not None:raise RuntimeError('project entrypoint exited')
                try:
                    if s.get('http://127.0.0.1:18084/health',timeout=2).json().get('loaded'):break
                except (requests.RequestException,ValueError):pass
                time.sleep(1)
            else:raise RuntimeError('project entrypoint startup timeout')
            cfg=json.loads((ROOT/'runtime/current.json').read_text())
            assert cfg['gpu']==[0] and cfg['parallel_workers']==1
            assert Path(cfg['exe']).name=='strata-single-built'
            request={'model':'qwen3.8 flash','messages':[{'role':'user','content':'Calculate 17 times 23. Reply with the number only.'}],
                     'temperature':0,'max_tokens':128,'reasoning_effort':'none','cache_prompt':False}
            response=s.post('http://127.0.0.1:18084/v1/chat/completions',json=request,timeout=60)
            response.raise_for_status();data=response.json()
            assert '391' in (data['choices'][0]['message'].get('content') or ''),data
            summary.update(passed=True,response=data,health=s.get('http://127.0.0.1:18084/health').json(),
                           executable=os.readlink('/proc/'+str(process.pid)+'/exe'))
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGINT)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
        summary['test_server_stopped']=process is not None and process.poll() is not None
        (ROOT/'evidence/entrypoint-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'passed':summary['passed'],'test_server_stopped':summary['test_server_stopped']}))

if __name__=='__main__':main()
