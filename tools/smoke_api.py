"""Check real single-GPU text, thinking, tools, images, native video and queueing."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time
import requests
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8090')
    p.add_argument('--output',type=Path,default=ROOT/'runtime/smoke')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    summary={'passed':False,'checks':{}}
    def session():
        s=requests.Session();s.trust_env=False;return s
    def call(name,messages,**extra):
        body={'model':'qwen3.8 flash','messages':messages,'max_tokens':256,'temperature':0,
              'reasoning_effort':'none','cache_prompt':False,'seed':20261008}
        body.update(extra)
        r=session().post(args.url+'/v1/chat/completions',json=body,timeout=(10,300))
        data=r.json();(args.output/(name+'.json')).write_text(json.dumps({'request':body,
               'response':data,'http_status':r.status_code},ensure_ascii=False,indent=2)+'\n')
        r.raise_for_status();assert 'error' not in data,data
        print(name,data.get('usage'),flush=True)
        return data['choices'][0]['message']
    try:
        result=call('thinking',[{'role':'user','content':'计算17*23，检查一次，只给简短结论。'}],
                    reasoning_effort='low',max_tokens=2048)
        assert '391' in (result.get('content') or '') and result.get('reasoning_content'),result
        summary['checks']['thinking']=True
        tools=[{'type':'function','function':{'name':'lookup_weather','description':'Get weather for a city.',
               'parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}]
        result=call('tools',[{'role':'user','content':'Use lookup_weather to get the weather in Beijing.'}],
                    tools=tools,tool_choice={'type':'function','function':{'name':'lookup_weather'}})
        calls=result.get('tool_calls') or []
        assert calls and calls[0]['function']['name']=='lookup_weather',result
        assert 'Beijing' in json.loads(calls[0]['function']['arguments'])['city']
        summary['checks']['tools']=True
        image=Image.new('RGB',(640,360),'white');ImageDraw.Draw(image).ellipse((230,90,410,270),fill='red')
        data=io.BytesIO();image.save(data,format='PNG')
        encoded='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
        result=call('image',[{'role':'user','content':[{'type':'text','text':'背景和图形是什么颜色？图形是什么形状？简短回答。'},
                     {'type':'image_url','image_url':{'url':encoded}}]}])
        text=result.get('content') or '';assert '红' in text and ('圆' in text or '圈' in text),result
        summary['checks']['image']=text
        video=ROOT/'evidence/fixtures/probe-360p-5s.mp4'
        encoded='data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()
        result=call('video-8fps',[{'role':'user','content':[{'type':'text','text':'简短描述视频画面的变化。'},
                      {'type':'video_url','video_url':{'url':encoded}}]}],max_tokens=512)
        assert (result.get('content') or '').strip(),result
        summary['checks']['video_8fps']=result['content']
        request={'model':'qwen3.8 flash','max_tokens':128,'thinking':{'type':'disabled'},
                 'messages':[{'role':'user','content':'只原样回复 ANTHROPIC_READY，不要添加文字。'}]}
        r=session().post(args.url+'/v1/messages',json=request,timeout=60);data=r.json()
        (args.output/'anthropic.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        r.raise_for_status()
        assert any('ANTHROPIC_READY' in (c.get('text') or '') for c in data['content']),data
        summary['checks']['anthropic_api']=True
        def queued(i):
            secret=f'QUEUE_CHECK_{i}_8417'
            result=call('queued-'+str(i),[{'role':'user','content':f'只原样回复字符串 {secret}。'}])
            assert secret in (result.get('content') or ''),result
            return True
        with ThreadPoolExecutor(max_workers=3) as pool:assert all(pool.map(queued,range(3)))
        summary['checks']['single_capacity_queue']=True
        summary['status_after']=session().get(args.url+'/v1/status',timeout=5).json()
        assert summary['status_after']['loaded'] and summary['status_after']['activity']['in_flight']==0
        summary['passed']=True
    finally:
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
