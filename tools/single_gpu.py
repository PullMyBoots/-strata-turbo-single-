"""Standalone single-GPU launcher; it never controls the production service."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CONTEXT=262144
STEM='Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS'

def gpu_inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total',
                                 '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(row[0]),uuid=row[1].strip(),pci=row[2].strip(),name=row[3].strip(),
                 memory_mib=int(row[4])) for row in csv.reader(io.StringIO(raw))]

def select_gpu(value, inventory):
    found=[g for g in inventory if value in (str(g['index']),g['uuid'])]
    if len(found)!=1:raise ValueError('Select exactly one available GPU index or full UUID.')
    return found[0]

def numa_node(gpu):
    pci=gpu['pci'];domain,bus,slot=pci.split(':')
    path=Path('/sys/bus/pci/devices')/(f'{int(domain,16):04x}:{bus}:{slot}')/'numa_node'
    try:return max(-1,int(path.read_text()))
    except (OSError,ValueError):return -1

def expand(value, assets):
    return str(value).format(root=ROOT,assets=assets)

def build_config(assets, gpu, concurrency=1, engine='built', port=8090, host='127.0.0.1'):
    if type(concurrency) is not int or not 1<=concurrency<=6:raise ValueError('Concurrency must be an integer from 1 to 6.')
    if engine not in ('validated','built'):raise ValueError('Unknown engine selection.')
    if concurrency>1 and engine!='validated':raise ValueError('The fresh source build is validated for single-request mode; use --engine validated for concurrent mode.')
    assets=Path(assets).absolute()
    preset=json.loads((ROOT/'configs/iq2-single.json').read_text())
    native=assets/'models/IQ2_XS'/(STEM+'-00001-of-00002.gguf')
    ple=assets/'models/IQ2_XS'/(STEM+'-00002-of-00002.gguf')
    pack=assets/'data/packs/iq2_xs'
    env={key:expand(value,assets) for key,value in preset['env'].items()}
    options={key:expand(value,assets) for key,value in preset['args'].items()}
    binary='strata-single-validated' if engine=='validated' else 'strata-single-built'
    cfg={'exe':str(ROOT/'engine'/binary),'cwd':str(ROOT),'gpu':[gpu['index']],
         'gpus_asked':True,'layer_split':'auto','backend':'cuda','host':host,'port':port,
         'model_name':'qwen3.8-flash-next-ud-q3-k-xl',
         'aliases':['qwen3.8 flash','Qwen3.8-Flash-Next',STEM],
         'draft_vocab':'cjk',
         'sampling':{'temperature':0.3,'top_p':0.95,'top_k':20,'min_p':0.0,
                     'presence_penalty':0.0,'repetition_penalty':1.0},
         'tokenizer':str(pack/'tokenizer'),'log':str(ROOT/'runtime/engine.log'),
         'lib_dirs':[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'lib64')],
         'parallel_workers':concurrency}
    if concurrency>1:
        pool=json.loads((ROOT/'configs/concurrency-upstream.json').read_text())['profiles']['iq2_xs']
        binary='strata-concurrent-validated';cfg['exe']=str(ROOT/'engine'/binary)
        env={k:'1' for k in ('STRATA_SPLIT_ASYNC_COMMIT','STRATA_RESIDENT_PLAN_PARALLEL',
                             'STRATA_GR_DOWN_MAX4','STRATA_VERIFY_DEVICE_ONLY','STRATA_EPHEMERAL_PREFILL')}
        options={'--prefill':str(pool['prefill']),'--spec':'4','--spec-min-p':'0.5',
                 '--kv-resident':'32768','--mtp':str(assets/'data/mtp/rt'),'--mtp-window':'32768',
                 '--vram-reserve-mib':'700'}
        cfg.update(parallel_worker_active_thread_pct=pool['worker_active_thread_pct'],
                   parallel_device_max_connections=pool['device_max_connections'])
    env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];cfg['env']=env
    cfg['args']=['--pack',str(pack),'--native',str(native),'--ple-gguf',str(ple),
        '--expert-profile',str(assets/'data/expert-profile.bin'),'--expert-cache','auto',
        '--max-context',str(CONTEXT),'--kv','int8','--vision','--adapt-swaps','0',
        '--rope-scaling','none','--no-prefill-borrow']
    for flag,value in options.items():cfg['args'].extend([flag,str(value)])
    cfg['vision']={'exe':str(ROOT/'engine/strata-vision-native'),
        'mmproj':str(assets/'models/IQ2_XS/mmproj-Qwen3.8-Flash-Next-BF16.gguf'),
        'model':str(native),'gpu':True,'cuda_device':gpu['uuid'],'threads':32,
        'max_tokens':0,'flash_attn':'on','native_video':{'fps':8.0,'timestamp_interval_ms':5000}}
    cfg['deployment']={'project':'strata-single-GPU','weights':'iq2_xs','context':CONTEXT,
        'physical_gpus':[gpu['index']],'gpu_uuids':[gpu['uuid']],
        'max_concurrent_requests':concurrency,'main_weights_changed':False,
        'input_modalities':['text','image','video'],'audio_input':False,
        'engine_selection':engine,'numa_node':numa_node(gpu)}
    return cfg

def check_assets(cfg):
    manifest=json.loads((ROOT/'engine/manifest.json').read_text())['binaries']
    for path in (Path(cfg['exe']),Path(cfg['vision']['exe'])):
        if not path.is_file():raise ValueError('Missing engine: '+str(path))
        expected=manifest.get(path.name,{}).get('sha256')
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Binary is not the recorded release: '+str(path))
    def arg(flag):return Path(cfg['args'][cfg['args'].index(flag)+1])
    for path in (arg('--native'),arg('--ple-gguf'),Path(cfg['vision']['mmproj'])):
        if not path.is_file():raise ValueError('Missing GGUF: '+str(path))
        with path.open('rb') as f:
            if f.read(4)!=b'GGUF':raise ValueError('Invalid GGUF: '+str(path))
    for path in (arg('--pack')/'native_experts.txt',arg('--pack')/'index.txt',arg('--pack')/'dense.bin',
                 Path(cfg['tokenizer'])/'tokenizer.json',arg('--mtp')/'dense.txt',arg('--mtp')/'dense.bin',
                 arg('--mtp')/'experts.bin',arg('--expert-profile')):
        if not path.is_file():raise ValueError('Missing runtime asset: '+str(path))
    if cfg['parallel_workers']==1:
        router=Path(cfg['env']['STRATA_MTP_HEAD_ROUTER'])
        for name in ('router.txt','projection-f32.bin','coarse-q8.bin','bias-f32.bin'):
            if not (router/name).is_file():raise ValueError('Missing draft router: '+str(router/name))
        if not (arg('--mtp')/'experts-native.txt').is_file():raise ValueError('Missing native auxiliary expert metadata.')

def clean_environment():
    for key in list(os.environ):
        if key.startswith('STRATA_') and key not in {'STRATA_API_KEY'}:os.environ.pop(key)
    for key in ('CUDA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY',
                'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','CUDA_DEVICE_MAX_CONNECTIONS'):os.environ.pop(key,None)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=Path(os.environ.get('STRATA_ASSET_DIR',ROOT/'assets')))
    parser.add_argument('--gpu',default='0',help='Exactly one physical index or full UUID (default: 0).')
    parser.add_argument('--max-concurrency',type=int,choices=range(1,7),default=1)
    parser.add_argument('--engine',choices=['validated','built'],default=None,
                        help='Single request: built by default; concurrent mode: validated by default.')
    parser.add_argument('--port',type=int,default=8090)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--api-key',default=os.environ.get('STRATA_API_KEY',''))
    parser.add_argument('--dry-run',action='store_true',help='Print config without writing or loading weights.')
    parser.add_argument('--prepare',action='store_true',help='Validate assets and save config without starting.')
    args=parser.parse_args()
    try:
        if not 1<=args.port<=65535:raise ValueError('Invalid port.')
        if args.host not in ('127.0.0.1','localhost','::1') and not args.api_key:
            raise ValueError('A non-loopback listener requires an API key.')
        gpu=select_gpu(args.gpu,gpu_inventory())
        engine=args.engine or ('built' if args.max_concurrency==1 else 'validated')
        cfg=build_config(args.assets,gpu,args.max_concurrency,engine,args.port,args.host)
        if args.dry_run:
            print(json.dumps(cfg,ensure_ascii=False,indent=2));return 0
        check_assets(cfg)
        (ROOT/'runtime').mkdir(exist_ok=True)
        config=ROOT/'runtime/current.json';temp=config.with_suffix('.json.tmp')
        temp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');temp.replace(config)
        if args.prepare:
            print(json.dumps({'config':str(config),'assets_checked':True}));return 0
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex(('127.0.0.1',args.port))==0:raise ValueError('Port already in use; choose another --port.')
        clean_environment()
        if args.api_key:os.environ['STRATA_API_KEY']=args.api_key
        command=[sys.executable,'-m','serve.local_api','--engine','strata','--config',str(config),
                 '--host',args.host,'--port',str(args.port)]
        node=cfg['deployment']['numa_node']
        if node>=0 and shutil.which('numactl'):
            command=['numactl',f'--cpunodebind={node}',f'--membind={node}']+command
        os.chdir(ROOT)
        os.execvp(command[0],command)
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        parser.exit(2,str(error)+'\n')

if __name__=='__main__':raise SystemExit(main())
