"""Native media after code, 32k recall, and a cold native-window request.

Owns only its temporary server. All requests use the consumer preset, without
special routing, retries, semantic-answer substitution, or a smaller context.
"""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import distribution as d
import qualify_distribution as q

folder = ROOT / 'runtime/swift-video-debug' / (sys.argv[2] if len(sys.argv)>2 else 'history-fixed')
folder.mkdir(parents=True, exist_ok=False)
hw = d.detect(sys.argv[1])
cfg = d.build_config(ROOT / 'assets', hw, 'swift', d.resource_plan(hw, 'swift'))
if len(sys.argv)>3:
    cfg['exe']=str(Path(sys.argv[3]).absolute())
cfg.update(port=q.free_port(), host='127.0.0.1', log=str(folder / 'engine.log'))
cfg['args'] += ['--prompt-cache', '0']
d.save(folder / 'config.json', cfg)
d.save(folder / 'hardware.json', hw)
report = {'binary_sha256': d.digest(Path(cfg['exe'])), 'context': 262144,
          'main_weights_changed': False, 'cases': [], 'passed': False}

def video_fixture(color):
    path = folder / (color + '.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=' + color + ':s=160x90:r=8:d=1',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path

fixtures = {color: video_fixture(color) for color in ('red', 'green', 'blue')}
sequence = folder / 'red-green-blue.mp4'
subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error',
                '-f', 'lavfi', '-i', 'color=c=red:s=160x90:r=8:d=2',
                '-f', 'lavfi', '-i', 'color=c=green:s=160x90:r=8:d=2',
                '-f', 'lavfi', '-i', 'color=c=blue:s=160x90:r=8:d=2',
                '-filter_complex', '[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]',
                '-map', '[v]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(sequence)], check=True)

command, cwd = d.server_command(cfg, folder / 'config.json')
url = 'http://127.0.0.1:' + str(cfg['port'])

def check_video(path, expected, label, ordered=False):
    data = base64.b64encode(path.read_bytes()).decode()
    prompt = ('List the main colors in the order they appear in this video. Answer briefly in English.'
              if ordered else 'What is the dominant color throughout this video? Answer briefly in English.')
    req = q.body(cfg['model_name'], prompt, 256)
    req.pop('ignore_eos')
    req['messages'][0]['content'] = [
        {'type': 'text', 'text': prompt},
        {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + data}}]
    result = q.request(url, req)
    answer = result['text'].lower()
    good = all(color in answer for color in expected)
    if ordered and good:
        positions = [answer.index(color) for color in expected]
        good = positions == sorted(positions)
    item = {'case': label, 'passed': good, 'expected': expected, 'result': result}
    report['cases'].append(item)
    d.save(folder / 'results.json', report)
    print(json.dumps({'case': label, 'passed': good, 'answer': result['text']}), flush=True)
    if not good:
        raise ValueError('Native video regression failed: ' + label)

with (folder / 'server.log').open('w') as log:
    server = subprocess.Popen(command, cwd=cwd, env=d.child_environment(cfg),
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(900):
            if server.poll() is not None:
                raise RuntimeError('Server exited during startup')
            try:
                if q.session().get(url + '/v1/status', timeout=2).json().get('loaded'):
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Server startup timed out')
        req = q.body(cfg['model_name'], 'Write a complete Python persistent key-value database with transactions, recovery, examples and tests. Output code only.')
        report['code_prefix'] = q.request(url, req)
        req, names = q.recall_body(cfg['model_name'])
        report['recall_prefix'] = q.request(url, req)
        assert all(name in report['recall_prefix']['text'] for name in names)
        for color, path in fixtures.items():
            check_video(path, [color], 'after-code-and-32k-' + color)
        report['image'] = q.image_smoke(url, cfg['model_name'])
        check_video(sequence, ['red', 'green', 'blue'], 'temporal-order', ordered=True)
        sample = json.loads((ROOT / 'evidence/requests/long-256k.json').read_text())
        req = dict(sample['request'], model=cfg['model_name'])
        report['native_long'] = q.request(url, req)
        names = json.loads((ROOT / 'evidence/checkpoints.json').read_text())
        assert report['native_long']['usage']['prompt_tokens'] == 256013
        assert report['native_long']['timings']['cache_n'] == 0
        assert all(name in report['native_long']['text'] for name in names)
        for color, path in reversed(list(fixtures.items())):
            check_video(path, [color], 'after-native-window-' + color)
        report['post_media_text'] = q.request(url, q.body(cfg['model_name'], 'What is 17 times 19? Answer with the number only.', 32))
        assert '323' in report['post_media_text']['text']
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        q.stop(server)
        report['process'] = {'pid': server.pid, 'stopped': True, 'exit_code': server.returncode}
        d.save(folder / 'results.json', report)
