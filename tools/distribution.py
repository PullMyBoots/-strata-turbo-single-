"""Portable consumer installer and launcher for Strata-Turbo-Single.

The upstream downloader/model catalog is vendored at a fixed commit. Target
weights, expert count, INT8 KV and the native context are never changed here.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import single_gpu

ROOT = Path(__file__).resolve().parents[1]
NATIVE_CONTEXT = 262144
MODELS = {'iq2_xs': ('qwen', 'IQ2_XS'), 'swift': ('swift', 'IQ3_XXS'),
          'iq3_s': ('qwen', 'IQ3_S')}
INSTALL = ROOT / 'runtime/install.json'
BUILD = ROOT / 'distribution/build.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def upstream():
    path = ROOT / 'distribution/upstream/setup.py'
    expected = json.loads((ROOT / 'distribution/manifest.json').read_text())['upstream_setup_sha256']
    if digest(path) != expected:
        raise ValueError('Bundled upstream installer checksum mismatch.')
    spec = importlib.util.spec_from_file_location('strata_pinned_setup', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(command, **kwargs):
    print('  ' + ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def cuda_details(gpu):
    cuda = ctypes.CDLL('libcuda.so.1')
    if cuda.cuInit(0):
        raise ValueError('The NVIDIA driver could not initialize CUDA.')
    device = ctypes.c_int()
    domain, bus, slot = gpu['pci'].split(':')
    pci = f'{int(domain, 16):04x}:{bus}:{slot}'.encode()
    if cuda.cuDeviceGetByPCIBusId(ctypes.byref(device), pci):
        raise ValueError('Cannot access the selected CUDA device.')
    attrs = []
    for attribute in (75, 76):
        val = ctypes.c_int()
        if cuda.cuDeviceGetAttribute(ctypes.byref(val), attribute, device):
            raise ValueError('Cannot read CUDA compute capability.')
        attrs.append(val.value)
    driver = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(driver))
    # A temporary context measures *free* VRAM; it is destroyed before loading a model.
    context = ctypes.c_void_p()
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device):
        raise ValueError('Selected GPU cannot create a CUDA context.')
    try:
        if cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)):
            raise ValueError('Cannot read available VRAM.')
    finally:
        cuda.cuCtxDestroy_v2(context)
    return {'arch': attrs[0] * 10 + attrs[1], 'cuda_driver': driver.value,
            'free_vram_mib': free.value // 2**20}


def effective_memory(total, available, cgroup=Path('/sys/fs/cgroup')):
    """Respect Docker/systemd limits as well as the host's MemAvailable."""
    pairs = [('memory.max', 'memory.current'),
             ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes')]
    # Include parents of the current unified-cgroup path when visible.
    roots = [cgroup]
    try:
        relative = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                        if line.startswith('0::'))
        child = cgroup / relative.lstrip('/')
        if '..' not in child.relative_to(cgroup).parts:
            roots += [child, *[p for p in child.parents if p != cgroup and cgroup in p.parents]]
    except (OSError, StopIteration, ValueError):
        pass
    for root in roots:
        for cap_name, used_name in pairs:
            try:
                cap = int((root / cap_name).read_text().strip())
                used = int((root / used_name).read_text().strip())
                if 0 < cap < total:
                    total, available = cap, min(available, max(0, cap - used))
            except (OSError, ValueError):
                pass
    return total, available


def detect(gpu_value='auto'):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('This release supports Linux x86_64 NVIDIA. Windows users can use WSL2; native Windows builds are not qualified.')
    # Preserve installer options, but never inherit other experiment/GPU masks.
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'):
        os.environ.pop(key, None)
    cards = single_gpu.gpu_inventory()
    if not cards:
        raise ValueError('No NVIDIA GPU is available. Install the NVIDIA driver first.')
    if gpu_value == 'auto':
        available = []
        for card in cards:
            try:
                available.append(dict(card, **cuda_details(card)))
            except ValueError:
                continue
        if not available:
            raise ValueError('No GPU can create a CUDA context. Free GPU memory and rerun setup.')
        gpu = max(available, key=lambda g: g['free_vram_mib'])
    else:
        card = single_gpu.select_gpu(gpu_value, cards)
        gpu = dict(card, **cuda_details(card))
    if gpu['arch'] < 75:
        raise ValueError('The optimized release requires NVIDIA compute capability 7.5 or newer.')
    if gpu['free_vram_mib'] < 7168:
        raise ValueError('Less than 7 GiB free on the selected GPU. Free GPU memory or select another --gpu.')
    # First-run hardware inspection must work before private pip dependencies exist.
    memory = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
    total, available = effective_memory(memory['MemTotal'], memory['MemAvailable'])
    cpu_text = Path('/proc/cpuinfo').read_text()
    flags = next((s.split(':', 1)[1].split() for s in cpu_text.splitlines() if s.startswith('flags')), [])
    if 'avx2' not in flags:
        raise ValueError('The portable optimized build requires an AVX2-capable CPU.')
    cpus = sorted(os.sched_getaffinity(0))
    node = single_gpu.numa_node(gpu)
    try:
        local = upstream().cpulist((Path('/sys/devices/system/node') / f'node{node}/cpulist').read_text())
        local = sorted(set(cpus) & local)
        if not local:
            node = -1
    except (OSError, ValueError):
        local = cpus
    physical = set()
    for cpu in local or cpus:
        base = Path('/sys/devices/system/cpu') / f'cpu{cpu}/topology'
        try:
            physical.add(((base / 'physical_package_id').read_text().strip(), (base / 'core_id').read_text().strip()))
        except OSError:
            physical.add(cpu)
    return {'gpu': gpu, 'ram_total_bytes': total, 'ram_available_bytes': available,
            'cpu_model': next((s.split(':', 1)[1].strip() for s in cpu_text.splitlines() if s.startswith('model name')), ''),
            'cpu_affinity': cpus, 'local_cpu_affinity': local or cpus,
            'local_physical_cores': len(physical), 'numa_node': node,
            'os': platform.platform(), 'wsl': Path('/dev/dxg').exists()}


def resource_plan(hw, weights, concurrency=1, vision='auto'):
    if type(concurrency) is not int or not 1 <= concurrency <= 6:
        raise ValueError('Maximum concurrency must be 1..6.')
    catalog = upstream().MODELS[MODELS[weights][1]]
    vram = hw['gpu']['free_vram_mib'] / 1024
    ram = hw['ram_available_bytes'] / 1e9
    if ram < 16:
        raise ValueError('Native-context full-expert inference needs at least 16 GB available RAM; current resources are insufficient.')
    workers = max(1, min(32, hw['local_physical_cores']))
    low_ram = ram < catalog['arena_gb'] + 12
    kv_stream = not hw['wsl'] and ram >= catalog['arena_gb'] + 12 + 3.7 * concurrency
    # Keep all modalities when VRAM is small: move the encoder, never remove it.
    on_gpu = (vram >= 20) if vision == 'auto' else vision == 'gpu'
    if vision not in ('auto', 'gpu', 'cpu'):
        raise ValueError('Vision must be auto, gpu or cpu.')
    return {'context': NATIVE_CONTEXT, 'kv': 'int8', 'workers': workers,
            'vision_gpu': on_gpu, 'low_ram': low_ram, 'kv_streaming': kv_stream,
            'kv_resident': (32768 if concurrency == 1 else 16384 if concurrency == 2 else 8192) if kv_stream else 0,
            'reserve_mib': 700 if concurrency == 1 and vram >= 20 else 1400 if on_gpu else 512,
            'large_gpu_turbo': vram >= 40 and not low_ram and workers >= 16,
            'concurrency': concurrency, 'full_experts': True,
            'input_modalities': ['text', 'image', 'video'], 'audio_input': False}


def lib_dirs():
    paths = [Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'lib64']
    for base in sys.path:
        for name in ('cuda_runtime', 'cublas'):
            path = Path(base) / 'nvidia' / name / 'lib'
            if path.is_dir():
                paths.append(path)
    return list(dict.fromkeys(str(p) for p in paths if p.is_dir()))


def prepare_dependencies():
    import importlib.metadata
    requirements = (ROOT / 'requirements.txt').read_text().splitlines()
    missing = []
    for line in requirements:
        if not line or line.startswith('#'):
            continue
        name, version = line.split('==')
        try:
            if importlib.metadata.version(name) != version:
                missing.append(line)
        except importlib.metadata.PackageNotFoundError:
            missing.append(line)
    if missing:
        run([sys.executable, '-m', 'pip', 'install', *missing])
    # Private CUDA runtime, no driver/system CUDA replacement.
    try:
        ctypes.CDLL('libcudart.so.12')
        ctypes.CDLL('libcublas.so.12')
    except OSError:
        run([sys.executable, '-m', 'pip', 'install', 'nvidia-cuda-runtime-cu12==12.5.82',
             'nvidia-cublas-cu12==12.5.3.2'])


def prepare_assets(assets, weights, gguf_dir=None, adopt=False):
    assets = Path(assets).absolute()
    if adopt:
        return assets
    official = upstream()
    family, quant = MODELS[weights]
    catalog = official.FAMILIES[family]
    folder, stem, pack_name, _, mmproj = single_gpu.WEIGHTS[weights]
    model_dir = assets / 'models' / folder
    model_dir.mkdir(parents=True, exist_ok=True)
    shards = [model_dir / (stem + f'-0000{i}-of-00002.gguf') for i in (1, 2)]
    for i, shard in enumerate(shards, 1):
        if gguf_dir:
            source = Path(gguf_dir).absolute() / shard.name
            if not source.is_file():
                raise ValueError('Missing GGUF shard: ' + str(source))
            if not shard.exists():
                shard.symlink_to(source)
        elif not official.whole_shard(shard):
            official.download(catalog['hf'].format(q=quant) + official.model_file(catalog, quant, i), shard)
        if not official.whole_shard(shard):
            raise ValueError('Incomplete or invalid GGUF shard: ' + str(shard))
    # Store the projector under its actual upstream precision/name.
    projector = model_dir / catalog['mmproj']
    if not projector.exists():
        official.download(catalog.get('mmproj_hf', catalog['hf']).format(q=quant) + catalog['mmproj'], projector)
    pack = assets / 'data/packs' / pack_name
    tool_root = ROOT / 'distribution/upstream/tools'
    tool_env = dict(os.environ, STRATA_GGUF_PY=str(ROOT / 'third_party/llama.cpp/gguf-py'))
    if not all((pack / name).is_file() for name in ('native_experts.txt', 'dense.bin', 'tokenizer/tokenizer.json')):
        run([sys.executable, tool_root / 'iq_pack.py', '--gguf', shards[0], '--out', pack], env=tool_env)
    profile = assets / 'data/expert-profile.bin'
    if not profile.exists():
        profile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'distribution/upstream/data/expert-profile.bin', profile)
    mtp = assets / 'data/mtp'
    rt = mtp / 'rt'
    if not all((rt / name).is_file() for name in ('dense.txt', 'dense.bin', 'experts.bin')):
        run([sys.executable, tool_root / 'mtp_fetch.py', 'fetch', '--out', mtp], env=tool_env)
        run([sys.executable, tool_root / 'mtp_fetch.py', 'verify', '--out', mtp], env=tool_env)
        run([sys.executable, tool_root / 'mtp_pack.py', '--src', mtp, '--experts', 'q2_0', '--out', mtp / 'mtp-q2_0.gguf'], env=tool_env)
        run([sys.executable, tool_root / 'mtp_rt.py', '--gguf', mtp / 'mtp-q2_0.gguf', '--out', rt], env=tool_env)
    full = mtp / 'native-q5-full'
    if not all((full / name).is_file() for name in ('dense.txt', 'dense.bin', 'experts.bin', 'experts-native.txt')):
        # No cropped draft_vocab.bin is admitted to the full-vocabulary preset.
        if full.exists():
            raise ValueError('Incomplete auxiliary preparation: remove only ' + str(full) + ' and rerun setup.')
        with tempfile.TemporaryDirectory(prefix='q5-prepare-', dir=mtp) as tmp:
            out = Path(tmp) / 'auxiliary'
            run([sys.executable, ROOT / 'tools/mtp_native_rt.py', '--src', mtp, '--base', rt,
                 '--out', out, '--format', 'q5_0'], env=tool_env)
            for name in ('dense.txt', 'dense.bin', 'experts.bin', 'experts-native.txt'):
                file = out / name
                if file.is_symlink():
                    data = file.resolve()
                    file.unlink()
                    shutil.copy2(data, file)
            cropped = out / 'draft_vocab.bin'
            if cropped.exists() or cropped.is_symlink():
                cropped.unlink()
            out.rename(full)
    router_name = 'head-router-full-r128-k8192' if weights == 'iq2_xs' else f'head-router-{weights}-full-r128-k8192'
    router = mtp / router_name
    manifest = json.loads((ROOT / 'distribution/manifest.json').read_text())
    for name in ('router.txt', 'projection-f32.bin', 'coarse-q8.bin', 'bias-f32.bin'):
        source = ROOT / 'distribution/routers' / router_name / name
        key = 'routers/' + router_name + '/' + name
        if digest(source) != manifest['router_files'][key]['sha256']:
            raise ValueError('Bundled router checksum mismatch: ' + key)
        router.mkdir(exist_ok=True)
        if not (router / name).exists():
            shutil.copy2(source, router / name)
    return assets


def set_arg(args, flag, value=None):
    args = list(args)
    while flag in args:
        index = args.index(flag)
        size = 1 if flag in ('--no-prefill-borrow', '--vision', '--mmap-experts', '--batch-mtp') else 2
        del args[index:index + size]
    if value is not None:
        args += [flag, str(value)]
    return args


def build_config(assets, hw, weights, plan, arm='turbo', port=8090, host='127.0.0.1'):
    cfg = single_gpu.build_config(assets, hw['gpu'], weights=weights, port=port, host=host)
    if arm not in ('turbo', 'official'):
        raise ValueError('Unknown engine arm.')
    portable = json.loads(BUILD.read_text())
    cfg['exe'] = str(ROOT / portable['binaries']['turbo' if arm == 'turbo' else 'official_media']['path'])
    cfg['vision']['exe'] = str(ROOT / portable['binaries']['vision']['path'])
    cfg['lib_dirs'] = lib_dirs()
    cfg['vision'].update(gpu=plan['vision_gpu'], threads=plan['workers'])
    if weights == 'swift':
        # Tiny native video frames need enough spatial samples for reliable recognition.
        cfg['vision']['min_tokens'] = 128
    projector = Path(cfg['vision']['mmproj'])
    if not projector.exists():
        model_dir = Path(assets) / 'models' / single_gpu.WEIGHTS[weights][0]
        projector = model_dir / upstream().FAMILIES[MODELS[weights][0]]['mmproj']
        cfg['vision']['mmproj'] = str(projector)
    args = cfg['args']
    if arm == 'official' or not plan['large_gpu_turbo'] or plan['concurrency'] > 1:
        keep = ('--pack', '--native', '--ple-gguf', '--expert-profile', '--expert-cache', '--max-context', '--kv', '--rope-scaling')
        args = [item for flag in keep for item in (flag, args[args.index(flag) + 1])]
        cfg['env'] = {'CUDA_VISIBLE_DEVICES': hw['gpu']['uuid']}
        args += ['--vision', '--prefill', 'auto', '--spec', '4', '--spec-min-p', '0.5', '--mtp', str(Path(assets) / 'data/mtp/rt')]
        if arm == 'turbo' and plan['concurrency'] == 1:
            # Conservative low-memory preset, locally qualified before selection.
            preset = json.loads((ROOT / 'configs' / ('iq2-single.json' if weights == 'iq2_xs' else weights + '-single.json')).read_text())
            cfg['env'].update({k: single_gpu.expand(v, Path(assets).absolute()) for k, v in preset['env'].items()
                               if k not in ('STRATA_VERIFY_DEVICE_ONLY', 'STRATA_VERIFY_PRECAPTURE')})
            args = set_arg(args, '--mtp', Path(assets) / 'data/mtp/native-q5-full')
    if plan['concurrency'] > 1:
        # Share one weight copy; do not spawn six full engines or require MPS.
        cfg['exe'] = str(ROOT / portable['binaries']['official_media']['path'])
        cfg.update(parallel=plan['concurrency'], parallel_workers=1)
        if arm == 'turbo':
            args += ['--batch-mtp']
    if arm == 'turbo' and plan['large_gpu_turbo'] and plan['concurrency'] == 1 and weights == 'iq2_xs':
        # Keep the tuned decode path and restore every borrowed expert before its captured graphs run.
        # The engine sizes loans from actual free cache space and uses smaller layouts for short inputs.
        args = set_arg(args, '--no-prefill-borrow')
        args = set_arg(args, '--prefill', 'auto:16384')
        cfg['env']['STRATA_VERIFY_PREFILL_RESTORE'] = '1'
    for flag, value in (('--pool-workers', plan['workers']), ('--kv-resident', plan['kv_resident']),
                        ('--vram-reserve-mib', plan['reserve_mib'])):
        args = set_arg(args, flag, value)
    if plan['low_ram']:
        args += ['--mmap-experts']
    cfg['args'] = args
    cfg['model_name'] = {'iq2_xs': 'qwen3.8-flash-iq2-xs', 'swift': 'qwen3.8-flash-swift-iq3-xxs', 'iq3_s': 'qwen3.8-flash-iq3-s'}[weights]
    cfg['aliases'] += ['qwen3.8-flash-next-ud-q3-k-xl']
    cfg['log'] = str(ROOT / 'runtime' / ('consumer-' + arm + '.log'))
    cfg['deployment'].update(project='strata-turbo-single', engine_selection=arm, resource_plan=plan,
                             max_concurrent_requests=plan['concurrency'], numa_node=hw.get('numa_node', cfg['deployment']['numa_node']))
    cfg['deployment']['cpu_affinity'] = hw.get('local_cpu_affinity', hw.get('cpu_affinity', []))
    # Producer/consumer versions must match. Both arms get the same API and media adapter.
    cfg['server_module'] = 'serve.local_api' if arm == 'turbo' and plan['concurrency'] == 1 else 'experimental-native-api'
    return cfg


def performance_options(cfg):
    """Fingerprint actual execution flags; paths/ports are not tuning settings."""
    paths = {'--native', '--ple-gguf', '--mtp', '--pack', '--expert-profile'}
    flags = []
    i = 0
    while i < len(cfg['args']):
        if cfg['args'][i] in paths:
            i += 2
        else:
            flags.append(cfg['args'][i])
            i += 1
    return {'args': flags, 'env': {k: v for k, v in cfg['env'].items()
                                  if k not in ('CUDA_VISIBLE_DEVICES', 'STRATA_MTP_HEAD_ROUTER')},
            'draft_vocab': cfg.get('draft_vocab'), 'server_module': cfg.get('server_module'),
            'parallel': cfg.get('parallel', 1), 'parallel_workers': cfg.get('parallel_workers', 1),
            'vision': {k: cfg['vision'].get(k) for k in
                       ('gpu', 'threads', 'min_tokens', 'max_tokens', 'flash_attn', 'native_video')}}


def api_source_hashes(root=None):
    """Include adapters and prompt templates in a performance certificate.

    A stable entrypoint is insufficient: frontend changes can alter the model's
    actual tokens or cache hits without changing the entrypoint or executable.
    """
    root = ROOT if root is None else Path(root)
    hashes = {}
    for relative in ('serve', 'experimental/native-batch/serve'):
        folder = root / relative
        files = [p for p in folder.glob('*.py') if not p.name.startswith('test_')]
        files += list(folder.glob('*.jinja'))
        for path in sorted(files):
            hashes[str(path.relative_to(root))] = digest(path)
    return hashes


def identity(hw, cfg):
    files = [Path(cfg['exe']), Path(cfg['vision']['exe'])]
    assets = {}
    for flag in ('--native', '--ple-gguf', '--mtp', '--pack', '--expert-profile'):
        path = Path(cfg['args'][cfg['args'].index(flag) + 1])
        if path.is_file():
            paths = sorted(path.parent.glob(path.name.replace('-00001-of-00002', '-*-of-00002')))
        else:
            paths = [path / name for name in ('dense.txt', 'dense.bin', 'experts.bin', 'experts-native.txt', 'native_experts.txt', 'tokenizer/tokenizer.json') if (path / name).is_file()]
        assets[flag] = [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in paths]
    projector = Path(cfg['vision']['mmproj'])
    assets['vision'] = {'name': projector.name, 'bytes': projector.stat().st_size,
                        'mtime_ns': projector.stat().st_mtime_ns}
    router = cfg['env'].get('STRATA_MTP_HEAD_ROUTER')
    if router:
        assets['router'] = {p.name: digest(p) for p in sorted(Path(router).iterdir()) if p.is_file()}
    return {'gpu': {k: hw['gpu'][k] for k in ('uuid', 'name', 'arch', 'memory_mib', 'cuda_driver')},
            'cpu_model': hw['cpu_model'], 'cpu_affinity': hw['cpu_affinity'], 'numa': hw['numa_node'],
            'ram_total': hw['ram_total_bytes'], 'os': hw['os'], 'wsl': hw['wsl'],
            'binaries': {p.name: digest(p) for p in files}, 'assets': assets,
            'distribution_binaries': {k: v['sha256'] for k, v in json.loads(BUILD.read_text())['binaries'].items()},
            'plan': cfg['deployment']['resource_plan'], 'weights': cfg['deployment']['weights'],
            'execution_options': performance_options(cfg),
            'api_sha256': digest(ROOT / 'serve/local_api.py'),
            'api_source_hashes': api_source_hashes(),
            'installer_sha256': digest(Path(__file__)),
            'qualification_sha256': digest(ROOT / 'tools/qualify_distribution.py'),
            'native_api_sha256': digest(ROOT / 'experimental/native-batch/serve/local_api.py'),
            'preset_sha256': digest(ROOT / 'configs' / ('iq2-single.json' if cfg['deployment']['weights'] == 'iq2_xs' else cfg['deployment']['weights'] + '-single.json'))}


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_config(cfg):
    manifest = json.loads(BUILD.read_text())
    entries = {str(ROOT / b['path']): b for b in manifest['binaries'].values()}
    for path in (cfg['exe'], cfg['vision']['exe']):
        if path not in entries or digest(path) != entries[path]['sha256']:
            raise ValueError('Release binary checksum mismatch: ' + path)
    def arg(flag):
        return Path(cfg['args'][cfg['args'].index(flag) + 1])
    required = [arg('--native'), arg('--ple-gguf'), Path(cfg['vision']['mmproj']),
                arg('--expert-profile'), Path(cfg['tokenizer']) / 'tokenizer.json',
                arg('--pack') / 'native_experts.txt', arg('--pack') / 'dense.bin',
                arg('--mtp') / 'dense.txt', arg('--mtp') / 'dense.bin', arg('--mtp') / 'experts.bin']
    for path in required:
        if not path.is_file():
            raise ValueError('Missing model/runtime asset: ' + str(path))
    for path in (arg('--native'), arg('--ple-gguf'), Path(cfg['vision']['mmproj'])):
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError('Invalid GGUF: ' + str(path))
    if '--mmap-experts' in cfg['args'] and not (arg('--pack') / 'experts.bin').is_file():
        raise ValueError('Low-RAM deployment requires a prepared pack/experts.bin; rerun setup without --adopt.')
    router = cfg['env'].get('STRATA_MTP_HEAD_ROUTER')
    if router:
        for name in ('router.txt', 'projection-f32.bin', 'coarse-q8.bin', 'bias-f32.bin'):
            if not (Path(router) / name).is_file():
                raise ValueError('Missing full-vocabulary auxiliary router: ' + router)
        if (arg('--mtp') / 'draft_vocab.bin').exists():
            raise ValueError('Cropped draft vocabulary present in the full-vocabulary auxiliary directory.')
    env = child_environment(cfg)
    for path in (cfg['exe'], cfg['vision']['exe']):
        result = subprocess.run(['ldd', path], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode or 'not found' in result.stdout:
            raise ValueError('Missing binary dependencies; rerun setup.\n' + result.stdout)


def child_environment(cfg):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(('STRATA_', 'CUDA_MPS_')) or name in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS'):
            env.pop(name)
    env['LD_LIBRARY_PATH'] = ':'.join(cfg['lib_dirs'] + [env.get('LD_LIBRARY_PATH', '')])
    env['STRATA_ALLOW_PARALLEL'] = '1'
    return env


def server_command(cfg, path):
    if cfg['server_module'] == 'experimental-native-api':
        command = [sys.executable, '-m', 'serve.local_api', '--engine', 'strata', '--config', str(path),
                   '--host', cfg['host'], '--port', str(cfg['port'])]
        cwd = ROOT / 'experimental/native-batch'
    else:
        command = [sys.executable, '-m', 'serve.local_api', '--engine', 'strata', '--config', str(path),
                   '--host', cfg['host'], '--port', str(cfg['port'])]
        cwd = ROOT
    # Bind CPUs to the GPU node; allow memory fallback instead of forcing a depleted node.
    if cfg['deployment']['numa_node'] >= 0 and shutil.which('numactl'):
        affinity = cfg['deployment'].get('cpu_affinity', [])
        binding = '--physcpubind=' + ','.join(map(str, affinity)) if affinity else '--cpunodebind=' + str(cfg['deployment']['numa_node'])
        command = ['numactl', binding,
                   '--preferred=' + str(cfg['deployment']['numa_node'])] + command
    return command, cwd


def ensure_build(hw, force=False):
    if BUILD.exists() and not force:
        meta = json.loads(BUILD.read_text())
        current = tuple(map(int, platform.libc_ver()[1].split('.')))
        minimum = tuple(map(int, meta.get('minimum_glibc', '2.38').split('.')))
        complete = all((ROOT / item['path']).is_file() for item in meta['binaries'].values())
        if complete and hw['gpu']['arch'] in meta['cuda_architectures'] and current >= minimum:
            return
        print('Packaged binaries are missing or the GPU architecture/Linux C library does not match. Building for this machine.', flush=True)
    nvcc = shutil.which('nvcc')
    if not nvcc:
        official = upstream()
        nvcc, _ = official.find_nvcc()
        if not nvcc:
            gpu = dict(hw['gpu'], arch=str(hw['gpu']['arch']), vram_gb=hw['gpu']['memory_mib'] / 1024)
            nvcc, _ = official.install_build_tools(gpu, True)
    if not shutil.which('cmake'):
        run([sys.executable, '-m', 'pip', 'install', 'cmake>=3.24,<5', 'ninja'])
    run([sys.executable, ROOT / 'tools/build_distribution.py', '--arch', str(hw['gpu']['arch']), '--nvcc', str(nvcc)])


def ensure_runtime():
    """Match the private runtime to the actual build, including source builds."""
    meta = json.loads(BUILD.read_text())
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(lib_dirs() + [os.environ.get('LD_LIBRARY_PATH', '')]))
    outputs = [subprocess.check_output(['ldd', str(ROOT / b['path'])], env=env, text=True) for b in meta['binaries'].values()]
    missing = '\n'.join(s for s in outputs if 'not found' in s)
    if 'libcudart.so.13' in missing or 'libcublas.so.13' in missing:
        run([sys.executable, '-m', 'pip', 'install', *upstream().CUDA_WHEELS])
    elif 'libcudart.so.12' in missing or 'libcublas.so.12' in missing:
        run([sys.executable, '-m', 'pip', 'install', 'nvidia-cuda-runtime-cu12==12.5.82', 'nvidia-cublas-cu12==12.5.3.2'])


def selected_config(settings, hw, port=None, host=None, api_key=None):
    plan = resource_plan(hw, settings['weights'], settings['max_concurrency'], settings['vision'])
    common = dict(port=port or settings['port'], host=host or settings['host'])
    turbo = build_config(settings['assets'], hw, settings['weights'], plan, **common)
    token = key(identity(hw, turbo))
    record = ROOT / 'runtime/qualification' / (token + '.json')
    report = json.loads(record.read_text()) if record.exists() else {}
    if report.get('state') == 'failed' and not report.get('fallback_validated'):
        raise ValueError('Neither engine has passed the local workload checks. See ' + str(record) + '; run ./setup.sh --verify after fixing the reported error.')
    if report.get('identity_key') != token or report.get('selection') != 'turbo':
        cfg = build_config(settings['assets'], hw, settings['weights'], plan, arm='official', **common)
        print('Turbo has not passed qualification on this hardware. Using upstream kernels; run ./setup.sh --verify to compare again.', flush=True)
    else:
        cfg = turbo
        print(f"Turbo qualified on this hardware: local qualification throughput is {report['gain_percent']:.1f}% higher ({report['metric']}).", flush=True)
    cfg['qualification'] = {'identity_key': token, 'report': str(record), 'state': report.get('state', 'unverified')}
    if api_key:
        cfg['api_key'] = api_key
    if cfg['host'] not in ('127.0.0.1', 'localhost', '::1') and not api_key:
        raise ValueError('Use STRATA_API_KEY when serving beyond localhost.')
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description='Strata-Turbo-Single: install, inspect, launch, and verify performance on your hardware.')
    p.add_argument('--weights', choices=MODELS)
    p.add_argument('--gpu', default=None, help='One index/UUID; auto chooses the most free VRAM on first install.')
    p.add_argument('--assets', type=Path)
    p.add_argument('--gguf-dir', type=Path)
    p.add_argument('--adopt', action='store_true', help='Reuse a complete existing asset tree without modifying it.')
    p.add_argument('--max-concurrency', type=int, choices=range(1, 7), default=None)
    p.add_argument('--vision', choices=('auto', 'gpu', 'cpu'), default=None)
    p.add_argument('--port', type=int, default=None)
    p.add_argument('--host', default=None)
    p.add_argument('--yes', action='store_true')
    p.add_argument('--check', action='store_true', help='Read-only hardware/resource check.')
    p.add_argument('--verify', action='store_true', help='Repeat same-device speed comparison.')
    p.add_argument('--verify-long', action='store_true', help='Also compare a native-window 256k-token prompt.')
    p.add_argument('--skip-verify', action='store_true', help='Install without a speed claim; use upstream kernels until verified.')
    p.add_argument('--no-start', action='store_true')
    p.add_argument('--start', action='store_true')
    p.add_argument('--build', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    old = json.loads(INSTALL.read_text()) if INSTALL.exists() else {}
    weights = a.weights or old.get('weights')
    hw = detect(a.gpu or old.get('gpu', 'auto'))
    if weights is None:
        weights = 'iq2_xs'
        if not a.yes and sys.stdin.isatty() and not a.check:
            choice = input('Select weights: 1 IQ2_XS (fastest, default) / 2 Swift IQ3_XXS / 3 IQ3_S: ').strip()
            weights = {'': 'iq2_xs', '1': 'iq2_xs', '2': 'swift', '3': 'iq3_s'}.get(choice)
            if weights is None:
                p.error('Select 1, 2, or 3.')
    settings = {'weights': weights, 'assets': str((a.assets or Path(old.get('assets', ROOT / 'models-data'))).absolute()),
                'gpu': hw['gpu']['uuid'], 'max_concurrency': a.max_concurrency or old.get('max_concurrency', 1),
                'vision': a.vision or old.get('vision', 'auto'), 'port': a.port or old.get('port', 8090),
                'host': a.host or old.get('host', '127.0.0.1')}
    if not 1 <= settings['port'] <= 65535:
        p.error('Invalid port.')
    plan = resource_plan(hw, weights, settings['max_concurrency'], settings['vision'])
    if a.check or a.dry_run:
        print(json.dumps({'hardware': hw, 'settings': settings, 'resource_plan': plan,
                          'download_gb': upstream().MODELS[MODELS[weights][1]]['download_gb'],
                          'speed_support': 'L40S measured; other hardware needs local verification'}, ensure_ascii=False, indent=2))
        return 0
    lock = ROOT / 'runtime/install.lock'
    lock.parent.mkdir(exist_ok=True)
    import fcntl
    lock_stream = lock.open('a')
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('Another installer/qualification is active. Wait for it to finish.')
    install = not a.start and not (a.verify and INSTALL.exists())
    if install:
        print(f"Installing {weights} on one {hw['gpu']['name']}; native context {NATIVE_CONTEXT}; image and video input retained.", flush=True)
        prepare_dependencies()
        ensure_build(hw, a.build)
        assets = Path(settings['assets'])
        if not a.adopt:
            assets.mkdir(parents=True, exist_ok=True)
            catalog = upstream().MODELS[MODELS[weights][1]]
            existing = sum(p.stat().st_size for p in (assets / 'models' / single_gpu.WEIGHTS[weights][0]).glob('*.gguf'))
            expected = max(0, catalog['download_gb'] * 1e9 - existing) + 14e9
            if plan['low_ram']:
                expected += catalog['arena_gb'] * 1e9
            if not a.gguf_dir and shutil.disk_usage(assets).free < expected:
                raise ValueError('Insufficient free disk for model and auxiliary preparation: need about ' + str(round(expected / 1e9)) + ' GB on first install.')
        prepare_assets(assets, weights, a.gguf_dir, a.adopt)
        if plan['low_ram'] and not a.adopt:
            pack = assets / 'data/packs' / single_gpu.WEIGHTS[weights][2]
            native = assets / 'models' / single_gpu.WEIGHTS[weights][0] / (single_gpu.WEIGHTS[weights][1] + '-00001-of-00002.gguf')
            if not (pack / 'experts.bin').exists():
                run([sys.executable, ROOT / 'distribution/upstream/tools/iq_pack.py', '--gguf', native, '--out', pack, '--experts-bin'],
                    env=dict(os.environ, STRATA_GGUF_PY=str(ROOT / 'third_party/llama.cpp/gguf-py')))
    else:
        ensure_build(hw, a.build)
    ensure_runtime()
    turbo = build_config(settings['assets'], hw, weights, plan, port=settings['port'], host=settings['host'])
    official = build_config(settings['assets'], hw, weights, plan, arm='official', port=settings['port'], host=settings['host'])
    check_config(turbo)
    check_config(official)
    if install:
        save(INSTALL, settings)
    token = key(identity(hw, turbo))
    record = ROOT / 'runtime/qualification' / (token + '.json')
    if a.verify or a.verify_long or (install and not a.skip_verify and not record.exists()):
        from qualify_distribution import qualify
        report = qualify(hw, turbo, official, record, long=a.verify_long)
        if report['state'] == 'failed' and not report.get('fallback_validated'):
            raise ValueError('Local qualification failed. See ' + str(record))
    if a.no_start or a.verify or a.verify_long:
        print('Installation/qualification complete. Run ./start.sh to start; Ctrl+C to stop.', flush=True)
        return 0
    cfg = selected_config(settings, hw, api_key=os.environ.get('STRATA_API_KEY'))
    check_config(cfg)
    path = ROOT / 'runtime/consumer-current.json'
    save(path, cfg)
    cmd, cwd = server_command(cfg, path)
    print(f"API: http://{cfg['host']}:{cfg['port']}/v1; model: {cfg['model_name']}; maximum concurrency: {settings['max_concurrency']}", flush=True)
    os.chdir(cwd)
    lock_stream.close()
    os.execvpe(cmd[0], cmd, child_environment(cfg))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('Installation/startup failed: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
