"""Run an exact HTML animation task through isolated local Claude Code sessions."""
from __future__ import annotations

import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import distribution as d
import qualify_distribution as q
from showcase_lock import acquire

PROMPT = 'Create a self-contained HTML file with a 2D SVG animation of a pelican riding a bicycle.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--gpu', default='2')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--claude', default='claude')
    a = p.parse_args()
    benchmark_lock = acquire()
    out = a.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    work = out / 'workspace'
    work.mkdir()
    hw = d.detect(a.gpu)
    d.save(out / 'hardware.json', hw)
    d.save(out / 'build.json', json.loads(d.BUILD.read_text()))
    state = {'backend': '', 'case': None, 'counter': 0}

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *_):
            pass
        def do_GET(self):
            self.forward()
        def do_POST(self):
            self.forward()
        def forward(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            payload = json.loads(body) if body else None
            if self.path.split('?')[0] == '/v1/messages' and body:
                payload.update(temperature=0, seed=20261010, cache_prompt=True)
                body = json.dumps(payload).encode()
            index = state['counter'] = state['counter'] + 1
            case = state['case']
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ('host', 'content-length', 'connection', 'accept-encoding')}
            begin = time.monotonic()
            chunks = []
            try:
                with q.session().request(self.command, state['backend'] + self.path,
                                         data=body, headers=headers, stream=True, timeout=(10, 1800)) as response:
                    self.send_response(response.status_code)
                    self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    for chunk in response.iter_content(chunk_size=1):
                        if chunk:
                            chunks.append(chunk)
                            self.wfile.write(chunk)
                            if chunk == b'\n':
                                self.wfile.flush()
                    self.wfile.flush()
                    if case:
                        (case / f'api-{index}.response.txt').write_bytes(b''.join(chunks))
                        d.save(case / f'api-{index}.json', {'path': self.path, 'request': payload,
                                                           'status': response.status_code,
                                                           'wall_s': time.monotonic() - begin})
            except Exception as error:
                if case:
                    d.save(case / f'api-{index}-error.json', {'error': str(error)})
                raise

    proxy = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    base_url = f'http://127.0.0.1:{proxy.server_port}'
    model = 'qwen3.8-flash-iq2-xs'
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(('STRATA_', 'CUDA_MPS_', 'ANTHROPIC_')) or name in (
            'CUDA_VISIBLE_DEVICES', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
            'http_proxy', 'https_proxy', 'all_proxy', 'CLAUDE_CODE_OAUTH_TOKEN'):
            env.pop(name, None)
    local = {'ANTHROPIC_BASE_URL': base_url, 'ANTHROPIC_API_KEY': 'local-harness-test',
             'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_AUTOUPDATER': '1',
             'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '65536'}
    env.update(local, CLAUDE_CONFIG_DIR=str(out / 'claude-config'))
    d.save(out / 'settings.json', {'env': local, 'alwaysThinkingEnabled': True})
    command = [a.claude, '-p', PROMPT, '--bare', '--setting-sources', '',
               '--settings', str(out / 'settings.json'), '--model', model, '--effort', 'max',
               '--tools', 'Write,Read,Bash,Edit', '--allowedTools', 'Write', 'Read', 'Bash', 'Edit',
               '--permission-mode', 'acceptEdits', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
               '--disable-slash-commands', '--no-session-persistence', '--output-format', 'stream-json', '--verbose',
               '--append-system-prompt', 'Save the requested complete self-contained HTML file as pelican.html in the current directory. Finish after saving it. Do not install software, render previews, or delegate the task.']
    d.save(out / 'command.json', command)
    report = {'task': PROMPT, 'weights': 'GSQ-RCO IQ2_XS', 'context_limit': 262144,
              'claude_version': subprocess.check_output([a.claude, '--version'], text=True).strip(),
              'effort': 'max', 'output_limit_per_request': 65536, 'temperature': 0, 'seed': 20261010,
              'model_loading_included': False, 'cli_startup_thinking_tools_retries_included': True,
              'protocol': 'ABBA, two fresh services/sessions per engine; identical warm-up; empty same workspace; prompt cache on; original HTML preserved.',
              'runs': []}
    try:
        for index, arm in enumerate(('official', 'turbo', 'turbo', 'official')):
            folder = out / f'{index}-{arm}'
            folder.mkdir()
            cfg = d.build_config(a.assets.absolute(), hw, 'iq2_xs', d.resource_plan(hw, 'iq2_xs'), arm=arm)
            cfg.update(port=q.free_port(), host='127.0.0.1', log=str(folder / 'engine.log'))
            d.check_config(cfg)
            d.save(folder / 'config.json', cfg)
            state['backend'] = f"http://127.0.0.1:{cfg['port']}"
            cmd, cwd = d.server_command(cfg, folder / 'config.json')
            with (folder / 'server.log').open('w') as log:
                server = subprocess.Popen(cmd, cwd=cwd, env=d.child_environment(cfg), stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    from benchmark_showcase import ready
                    status = ready(server, state['backend'])
                    d.save(folder / 'status.json', status)
                    q.request(state['backend'], q.body(model, 'Explain speculative decoding.', 64))
                    for child in work.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                    state.update(case=folder, counter=0)
                    begin = time.monotonic()
                    with (folder / 'claude.jsonl').open('w') as stdout, (folder / 'stderr.log').open('w') as stderr:
                        cli = subprocess.Popen(command, cwd=work, env=env, stdout=stdout, stderr=stderr,
                                               start_new_session=True)
                        try:
                            code = cli.wait(timeout=1800)
                        except subprocess.TimeoutExpired:
                            q.stop(cli)
                            raise RuntimeError('Claude Code task exceeded 30 minutes')
                    elapsed = time.monotonic() - begin
                    state['case'] = None
                    artifact = work / 'pelican.html'
                    events = [json.loads(s) for s in (folder / 'claude.jsonl').read_text().splitlines() if s.strip()]
                    final = next((e for e in reversed(events) if e.get('type') == 'result'), {})
                    if artifact.exists():
                        shutil.copy2(artifact, folder / 'pelican.html')
                    valid = artifact.exists() and '<svg' in artifact.read_text().lower()
                    row = {'arm': arm, 'index': index, 'wall_s': elapsed, 'exit_code': code,
                           'html_svg_exists': valid, 'claude_result': final,
                           'api_requests': len(list(folder.glob('api-[0-9]*.json'))),
                           'binary_sha256': d.digest(Path(cfg['exe']))}
                    report['runs'].append(row)
                    d.save(folder / 'summary.json', row)
                    d.save(out / 'results.json', report)
                    print(arm, index, elapsed, 'HTML', valid, 'exit', code, flush=True)
                    assert code == 0 and valid and not final.get('is_error'), row
                finally:
                    state['case'] = None
                    q.stop(server)
                    d.save(folder / 'process.json', {'pid': server.pid, 'stopped': True, 'exit_code': server.returncode})
    finally:
        proxy.shutdown()
    print('Claude Code ABBA tasks complete; all owned engines stopped.', flush=True)


if __name__ == '__main__':
    main()
