"""Measure current release presets on one GPU; export raw timing, not certificates."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import distribution as d
import qualify_distribution as q
from strata_tokenizer import Tokenizer
from showcase_lock import acquire

TASKS = {
    'english': 'Write a detailed numbered guide to organizing and reviewing a software project. Include concrete examples. Continue for at least 100 items.',
    'code': 'Write a complete Python asyncio task scheduler with concurrency limits, retries with exponential backoff, cancellation, structured logging, and a runnable demonstration. Explain the design after the code. Produce at least 200 lines.',
    'chinese': '用中文详细解释大语言模型推理的内存带宽、缓存、预填充和输出生成机制，给出计算例子和优化步骤。连续撰写至少1500字。',
}


def ready(server, url):
    for _ in range(900):
        if server.poll() is not None:
            raise RuntimeError('Engine exited before readiness')
        try:
            status = q.session().get(url + '/v1/status', timeout=2).json()
            if status.get('loaded'):
                assert status['context']['max_positions'] == 262144
                return status
        except (q.requests.RequestException, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError('Engine startup timed out')


def measure(cfg, folder, fun):
    folder.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(cfg)
    cfg.update(port=q.free_port(), host='127.0.0.1', log=str(folder / 'engine.log'))
    d.save(folder / 'config.json', cfg)
    cmd, cwd = d.server_command(cfg, folder / 'config.json')
    url = f"http://127.0.0.1:{cfg['port']}"
    with (folder / 'server.log').open('w') as log:
        server = subprocess.Popen(cmd, cwd=cwd, env=d.child_environment(cfg),
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = ready(server, url)
            d.save(folder / 'status.json', status)
            q.request(url, q.body(cfg['model_name'], 'Explain speculative decoding.', 64))
            report = fun(url, cfg, folder)
            report.update(binary_sha256=d.digest(Path(cfg['exe'])), configuration=cfg,
                          status=status, measurement_only=True)
            d.save(folder / 'results.json', report)
        finally:
            q.stop(server)
            d.save(folder / 'process.json', {'pid': server.pid, 'stopped': True,
                                           'exit_code': server.returncode})
    return report


def curves(url, cfg, folder):
    pack = Path(cfg['args'][cfg['args'].index('--pack') + 1])
    directory = pack / 'tokenizer'
    vocab = json.loads((directory / 'vocab.json').read_text())
    vocabulary = [None] * len(vocab)
    for token, index in vocab.items():
        vocabulary[index] = token
    tok = Tokenizer(vocabulary, (directory / 'merges.txt').read_text().split('\n'),
                    json.loads((directory / 'token_type.json').read_text()))
    corpus = json.loads((d.ROOT / 'evidence/requests/long-256k.json').read_text())['request']['messages'][-1]['content']
    tokens = tok.encode(corpus)
    rows = []
    for size in (0, 8192, 65536, 253952):
        background = tok.decode(tokens[:size]) if size else ''
        prefix = 'The following records are background context. Complete the task at the end.\n' + background + '\nTASK:\n'
        for family, task in TASKS.items():
            req = q.body(cfg['model_name'], prefix + task)
            # A different system prefix makes each context's first request cold.
            # Later tasks may reuse KV prefixes; no generated answer is cached.
            req.update(seed=20261010, cache_prompt=True)
            req['messages'].insert(0, {'role': 'system', 'content': f'Benchmark context size {size}. Follow the task accurately.'})
            for sample in range(2):
                result = q.request(url, req)
                assert result['usage']['completion_tokens'] == 512
                assert result['timings']['prompt_n'] + 512 <= 262144
                cold = family == 'english' and sample == 0
                if cold:
                    assert result['timings']['cache_n'] == 0, result['timings']
                row = {'context_background_tokens': size, 'family': family,
                       'sample': sample, 'cold_prefill_sample': cold,
                       'request_sha256': hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest(),
                       'result': result}
                rows.append(row)
                d.save(folder / f'{size}-{family}-{sample}.json', dict(row, request=req))
                print(cfg['deployment']['weights'], cfg['deployment']['engine_selection'], size,
                      family, sample, result['timings']['predicted_per_second'],
                      'cache', result['timings']['cache_n'], flush=True)
    return {'protocol': 'One fresh service per arm/model; identical 64-token warm-up; increasing input sizes; two 512-token outputs per family/size; temperature 0; thinking disabled; KV-prefix caching enabled; first request at each size is cold.',
            'rows': rows}


def concurrent(url, cfg, folder):
    count = cfg['deployment']['max_concurrent_requests']
    rows = []
    families = list(TASKS)
    for wave in range(3):
        gate = threading.Barrier(count + 1)
        def one(i):
            family = families[(i + wave) % 3]
            req = q.body(cfg['model_name'], TASKS[family], 256)
            req['seed'] = 20261010
            gate.wait()
            result = q.request(url, req)
            assert result['usage']['completion_tokens'] == 256
            assert result['timings']['cache_n'] == 0
            d.save(folder / f'wave{wave}-request{i}.json', dict(request=req, result=result, family=family))
            return dict(result, family=family)
        with ThreadPoolExecutor(max_workers=count) as pool:
            jobs = [pool.submit(one, i) for i in range(count)]
            begin = time.monotonic()
            gate.wait()
            results = [job.result() for job in jobs]
            wall = time.monotonic() - begin
        row = {'wave': wave, 'count': count, 'wall_s': wall,
               'aggregate_e2e_tps': sum(r['usage']['completion_tokens'] for r in results) / wall,
               'requests': results}
        rows.append(row)
        print(cfg['deployment']['weights'], cfg['deployment']['engine_selection'],
              'concurrency', count, 'wave', wave, row['aggregate_e2e_tps'], flush=True)
    return {'protocol': 'Fresh service per slot count/arm/model; 64-token warm-up; three mixed English/code/Chinese batches; 256 tokens per request; thinking and KV-prefix cache disabled; aggregate includes prefill/admission.',
            'rows': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--latest', action='store_true', help='Use the sparse/fused single-request candidate and matrix headroom')
    p.add_argument('--suite', choices=('curves', 'concurrency', 'all'), default='all')
    p.add_argument('--weights', nargs='+', choices=tuple(d.MODELS), default=list(d.MODELS))
    p.add_argument('--arms', nargs='+', choices=('official', 'turbo'), default=['official', 'turbo'])
    a = p.parse_args()
    benchmark_lock = acquire()
    a.assets = a.assets.absolute()
    a.output = a.output.absolute()
    a.output.mkdir(parents=True, exist_ok=True)
    hw = d.detect(a.gpu)
    d.save(a.output / 'hardware.json', hw)
    d.save(a.output / 'build.json', json.loads(d.BUILD.read_text()))
    for weight in a.weights:
        for arm in a.arms:
            if a.suite in ('all', 'curves'):
                cfg = d.build_config(a.assets, hw, weight, d.resource_plan(hw, weight), arm=arm)
                d.check_config(cfg)
                if a.latest:
                    if weight != 'iq2_xs':
                        cfg['args'] = d.set_arg(cfg['args'], '--vram-reserve-mib', '2048')
                    if arm == 'turbo':
                        cfg['exe'] = str(d.ROOT/'experimental/sparse-draft-prefill/build/strata-sparse-draft-prefill')
                        cfg['env'].update(STRATA_MTP_SPARSE_BATCH='1', STRATA_PF_FUSED='1')
                measure(cfg, a.output / f'{weight}-{arm}-curves', curves)
            if a.suite in ('all', 'concurrency'):
                for count in (1, 2, 4, 6):
                    cfg = d.build_config(a.assets, hw, weight, d.resource_plan(hw, weight, count), arm=arm)
                    cfg['args'] += ['--prompt-cache', '0']
                    d.check_config(cfg)
                    measure(cfg, a.output / f'{weight}-{arm}-c{count}', concurrent)
    print('All requested current-release measurements completed; owned services stopped.', flush=True)
    benchmark_lock.close()


if __name__ == '__main__':
    main()
