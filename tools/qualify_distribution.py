"""Qualify performance locally with a same-device upstream/Turbo ABBA comparison.

The gate uses all three task families, a 32k recall check, correctness/API/media
smokes, and optional native-window evaluation. It never imports another GPU's
performance certificate. A failed gate selects the upstream media adapter.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import base64
import copy
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import tempfile
import time

import requests

from distribution import ROOT, child_environment, identity, key, save, server_command


def decide(official, turbo, correct, min_gain=0.03, *, official_prefill=None, turbo_prefill=None,
           max_prefill_regression=0.10):
    """Pure decision function: no successful speed claim without valid evidence."""
    if not correct or set(official) != set(turbo) or not official:
        return {'selection': 'official', 'state': 'failed', 'gain_percent': 0.0,
                'reason': 'Correctness, media, workload completeness or configuration checks failed.'}
    ratios = {name: statistics.median(turbo[name]) / statistics.median(official[name]) for name in official}
    ratio = math.exp(statistics.mean(math.log(v) for v in ratios.values()))
    prefill_ratios = {}
    if official_prefill is not None or turbo_prefill is not None:
        if (not official_prefill or not turbo_prefill or set(official_prefill) != set(turbo_prefill)
                or any(not values or any(not math.isfinite(v) or v <= 0 for v in values)
                       for arm in (official_prefill, turbo_prefill) for values in arm.values())):
            return {'selection': 'official', 'state': 'failed', 'gain_percent': 0.0,
                    'reason': 'Prefill comparisons are incomplete or invalid.'}
        prefill_ratios = {name: statistics.median(turbo_prefill[name]) / statistics.median(official_prefill[name])
                          for name in official_prefill}
    leading = (ratio >= 1 + min_gain and min(ratios.values()) >= 0.97
               and all(v >= 1 - max_prefill_regression for v in prefill_ratios.values()))
    return {'selection': 'turbo' if leading else 'official',
            'state': 'qualified' if leading else 'no-confirmed-advantage',
            'gain_percent': (ratio - 1) * 100, 'ratios': ratios, 'prefill_ratios': prefill_ratios,
            'max_prefill_regression': max_prefill_regression,
            'reason': 'At least 3% output gain, no output-family regression over 3%, tested-prefill regression within the recorded tolerance.' if leading
                      else 'The local workload did not confirm a consistent output advantage with preserved prefill speed.'}


def session():
    s = requests.Session()
    s.trust_env = False
    return s


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(url, body):
    events = []
    begin = time.monotonic()
    first_content_s = None
    with session().post(url + '/v1/chat/completions', json=dict(body, stream=True,
                          stream_options={'include_usage': True}), stream=True, timeout=(10, 1800)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b'data: ') and line != b'data: [DONE]':
                event = json.loads(line[6:])
                if event.get('error'):
                    raise ValueError(str(event['error']))
                if first_content_s is None and any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                    first_content_s = time.monotonic() - begin
                events.append(event)
    text = ''.join(c.get('delta', {}).get('content') or '' for e in events for c in e.get('choices', []))
    timings = next((e['timings'] for e in reversed(events) if e.get('timings')), {})
    usage = next((e['usage'] for e in reversed(events) if e.get('usage')), {})
    finish = next((c.get('finish_reason') for e in reversed(events) for c in e.get('choices', []) if c.get('finish_reason')), None)
    return {'text': text, 'timings': timings, 'usage': usage,
            'finish_reason': finish, 'wall_s': time.monotonic() - begin, 'first_content_s': first_content_s}


def body(model, text, maximum=512):
    return {'model': model, 'messages': [{'role': 'user', 'content': text}],
            'max_tokens': maximum, 'temperature': 0, 'reasoning_effort': 'none',
            'cache_prompt': False, 'ignore_eos': True}


def recall_body(model):
    names = ['CEDAR-712', 'ORBIT-935', 'MANGO-246', 'RIVER-508', 'TULIP-163']
    pieces = ['Read the records. Return the five secret codes in order, then explain the indexing rules in detail.\n']
    for block, name in enumerate(names):
        pieces.extend(f'Record {block}:{i} has ordinary data red green blue yellow.\n' for i in range(400))
        pieces.append(f'Checkpoint {block + 1} secret code = {name}.\n')
    pieces.append('Write all five checkpoint secret codes, then a detailed explanation.\n')
    return body(model, ''.join(pieces), 256), names


def image_smoke(url, model):
    from PIL import Image
    out = io.BytesIO()
    Image.new('RGB', (64, 64), (255, 0, 0)).save(out, format='PNG')
    data = base64.b64encode(out.getvalue()).decode()
    req = body(model, 'What is the dominant color?', 256)
    req.pop('ignore_eos')
    req['messages'][0]['content'] = [{'type': 'text', 'text': 'What is the dominant color? Answer in English.'},
                                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + data}}]
    result = request(url, req)
    if 'red' not in result['text'].lower():
        raise ValueError('Image smoke failed: expected red; received ' + repr(result['text'][:400]))
    return {'passed': True, 'answer': result['text']}


def video_smoke(url, model):
    data = base64.b64encode((ROOT / 'distribution/fixtures/red-video.mp4').read_bytes()).decode()
    req = body(model, 'Describe the video.', 256)
    req.pop('ignore_eos')
    req['messages'][0]['content'] = [{'type': 'text', 'text': 'What is the dominant color throughout this video? Answer in English.'},
                                    {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + data}}]
    result = request(url, req)
    if 'red' not in result['text'].lower():
        raise ValueError('Video smoke failed: expected red frames; received ' + repr(result['text'][:400]))
    return {'passed': True, 'answer': result['text'], 'fps': 8}


def tool_smoke(url, model):
    req = body(model, 'Use lookup_weather to get the weather in Sydney.', 256)
    req.pop('ignore_eos')
    req['tools'] = [{'type': 'function', 'function': {'name': 'lookup_weather',
        'description': 'Look up weather for a city.', 'parameters': {'type': 'object',
        'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
    req['tool_choice'] = {'type': 'function', 'function': {'name': 'lookup_weather'}}
    response = session().post(url + '/v1/chat/completions', json=req, timeout=120)
    response.raise_for_status()
    message = response.json()['choices'][0]['message']
    calls = message.get('tool_calls') or []
    if not calls or calls[0]['function']['name'] != 'lookup_weather' or 'Sydney' not in calls[0]['function']['arguments']:
        raise ValueError('Forced function-call smoke failed: ' + repr(message))
    return {'passed': True, 'calls': calls}


def stop(server):
    if server.poll() is not None:
        return
    server.send_signal(signal.SIGINT)
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=10)


def evaluate(cfg, folder, long=False, media=False):
    folder = Path(folder).absolute()
    cfg = copy.deepcopy(cfg)
    cfg.update(port=free_port(), host='127.0.0.1')
    cfg['args'] += ['--prompt-cache', '0']
    cfg['log'] = str(folder / 'engine.log')
    path = folder / 'config.json'
    save(path, cfg)
    command, cwd = server_command(cfg, path)
    url = 'http://127.0.0.1:' + str(cfg['port'])
    model = cfg['model_name']
    rates = {}
    report = {'configuration': cfg, 'short': {}, 'correct': False}
    with (folder / 'server.log').open('w') as log:
        server = subprocess.Popen(command, cwd=cwd, env=child_environment(cfg), stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        save(folder / 'process.json', {'pid': server.pid, 'started': time.time()})
        try:
            for _ in range(900):
                if server.poll() is not None:
                    raise ValueError('Qualification service exited; see ' + str(folder / 'server.log'))
                try:
                    status = session().get(url + '/v1/status', timeout=2).json()
                    if status.get('loaded'):
                        break
                except (requests.RequestException, ValueError):
                    pass
                time.sleep(1)
            else:
                raise ValueError('Qualification startup timed out.')
            report['status'] = status
            count = cfg['deployment']['resource_plan']['concurrency']
            if (status.get('context', {}).get('max_positions') != 262144
                    or status.get('concurrency', {}).get('serving') != count
                    or not status.get('vision', {}).get('available')):
                raise ValueError('The native context, requested serving slots or vision encoder are unavailable.')
            request(url, body(model, 'Write a short numbered list about geometry.', 64))
            tasks = {'english': 'Write a long, detailed numbered guide to organizing and reviewing a software project. Continue for at least 100 items.',
                     'code': 'Write a complete Python program implementing a persistent key-value database with transactions, recovery, examples and tests. Output code only.',
                     'chinese': '\u8bf7\u8be6\u7ec6\u4ecb\u7ecd\u5982\u4f55\u8bbe\u8ba1\u4e00\u4e2a\u53ef\u9760\u7684\u8f6f\u4ef6\u9879\u76ee\uff0c\u6301\u7eed\u5217\u4e3e\u5b9e\u73b0\u6b65\u9aa4\u3001\u6d4b\u8bd5\u65b9\u6cd5\u548c\u7ef4\u62a4\u5efa\u8bae\u3002\u81f3\u5c11\u5199\u4e00\u767e\u9879\u3002'}
            for case, prompt in tasks.items():
                results = []
                for repeat in range(3):
                    req = body(model, prompt)
                    req['messages'].insert(0, {'role': 'system', 'content': f'Qualification variant {repeat}: complete the task carefully.'})
                    if count == 1:
                        results.append(request(url, req))
                    else:
                        started = time.monotonic()
                        with ThreadPoolExecutor(max_workers=count) as pool:
                            batch = list(pool.map(lambda _: request(url, req), range(count)))
                        elapsed = time.monotonic() - started
                        results.append({'batch': batch, 'wall_s': elapsed,
                                        'aggregate_tps': sum(r['usage'].get('completion_tokens', 0) for r in batch) / elapsed})
                flat = results if count == 1 else [r for result in results for r in result['batch']]
                if any(r['usage'].get('completion_tokens') != 512 or r['timings'].get('cache_n', 0) != 0 or not r['text'] for r in flat):
                    raise ValueError('Invalid token count, cached prompt or empty output in ' + case)
                rates[case] = [r['timings']['predicted_per_second'] for r in results] if count == 1 else [r['aggregate_tps'] for r in results]
                report['short'][case] = results
                save(folder / 'results.json', report)
                print(f"  {folder.name} {case}: {statistics.median(rates[case]):.1f} token/s", flush=True)
            answer = request(url, body(model, 'What is 17 plus 25? Answer with the number only.', 32))
            if '42' not in answer['text']:
                raise ValueError('Arithmetic smoke failed.')
            recall, names = recall_body(model)
            recalled = request(url, recall)
            if not all(name in recalled['text'] for name in names) or recalled['timings'].get('cache_n') != 0:
                raise ValueError('Long-prompt recall smoke failed.')
            report['recall'] = recalled
            if media:
                report['image'] = image_smoke(url, model)
                report['video'] = video_smoke(url, model)
                report['tools'] = tool_smoke(url, model)
                r = session().post(url + '/v1/messages', json={'model': model, 'max_tokens': 48,
                    'thinking': {'type': 'disabled'}, 'messages': [{'role': 'user', 'content': 'What is 2 plus 2? Answer with the number only.'}]}, timeout=120)
                r.raise_for_status()
                if '4' not in str(r.json().get('content', '')):
                    raise ValueError('Anthropic API smoke failed.')
                report['anthropic'] = {'passed': True}
            if long:
                sample = json.loads((ROOT / 'evidence/requests/long-256k.json').read_text())
                req = dict(sample['request'], model=model)
                result = request(url, req)
                names = json.loads((ROOT / 'evidence/checkpoints.json').read_text())
                if (result['usage'].get('prompt_tokens') != 256013 or result['usage'].get('completion_tokens') != 512
                        or result['timings'].get('cache_n') != 0 or not all(n in result['text'] for n in names)):
                    raise ValueError('Native-window recall qualification failed.')
                rates['native_long'] = [result['timings']['predicted_per_second']]
                report['native_long'] = result
            report.update(correct=True, rates=rates)
            save(folder / 'results.json', report)
            return report
        finally:
            stop(server)
            save(folder / 'process.json', {'pid': server.pid, 'exit_code': server.returncode, 'stopped': True})


def qualify(hw, turbo, official, record, long=False):
    stamp = time.strftime('%Y%m%d-%H%M%S')
    token = key(identity(hw, turbo))
    output = ROOT / 'runtime/qualification-runs' / (stamp + '-' + token[:12])
    output.mkdir(parents=True)
    arms = {'official': official, 'turbo': turbo}
    collected = {arm: {} for arm in arms}
    collected_prefill = {arm: {} for arm in arms}
    details = []
    result = {'identity_key': token, 'identity': identity(hw, turbo),
              'official_binary_sha256': identity(hw, official)['binaries'],
              'output': str(output), 'context': 262144, 'main_weights_changed': False,
              'protocol': 'ABBA; 3 task families x 3 requests x 2 service starts per arm; no prompt cache',
              'metric': 'output-stage token/s' if turbo['deployment']['resource_plan']['concurrency'] == 1 else 'concurrent aggregate end-to-end token/s',
              'native_long_checked': long, 'quality_scope': 'arithmetic, checkpoint recall, image, video and Anthropic smokes; not a comprehensive intelligence evaluation'}
    print('Comparing upstream and Turbo on this GPU. Model loading is repeated; no other service is stopped.', flush=True)
    try:
        for i, arm in enumerate(('official', 'turbo', 'turbo', 'official')):
            folder = output / f'{i}-{arm}'
            folder.mkdir()
            report = evaluate(arms[arm], folder, long=long and i in (0, 1), media=i in (0, 1))
            details.append({'arm': arm, 'path': str(folder), 'correct': report['correct']})
            for task, values in report['rates'].items():
                collected[arm].setdefault(task, []).extend(values)
            for task in ('recall', 'native_long'):
                if task in report:
                    collected_prefill[arm].setdefault(task, []).append(report[task]['timings']['prompt_per_second'])
        result.update(decide(collected['official'], collected['turbo'], all(r['correct'] for r in details),
                             official_prefill=collected_prefill['official'], turbo_prefill=collected_prefill['turbo']))
    except Exception as error:
        result.update(selection='official', state='failed', gain_percent=0.0, reason=str(error))
        print('Qualification failed: ' + str(error), flush=True)
    result.update(rates=collected, prefill_rates=collected_prefill, runs=details,
                  fallback_validated=any(r['arm'] == 'official' and r['correct'] for r in details))
    save(record, result)
    save(output / 'summary.json', result)
    print(f"Qualification: {result['state']}; selected {result['selection']}; gain {result['gain_percent']:.1f}%.\nReport: {record}", flush=True)
    return result
