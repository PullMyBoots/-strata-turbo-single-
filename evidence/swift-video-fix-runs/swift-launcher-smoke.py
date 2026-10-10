"""Exercise the repaired Swift preset through the actual consumer entrypoint."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import distribution as d
import qualify_distribution as q

folder = ROOT / 'runtime/swift-video-debug/actual-launcher'
folder.mkdir(exist_ok=True)
port = q.free_port()
url = f'http://127.0.0.1:{port}'
before = (ROOT / 'runtime/install.json').read_bytes()
report = {'entrypoint': './start.sh --weights swift --gpu 0', 'passed': False}
with (folder / 'server.log').open('w') as log:
    server = subprocess.Popen([str(ROOT / 'start.sh'), '--weights', 'swift', '--gpu', '0',
                               '--port', str(port)], cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(900):
            if server.poll() is not None:
                raise RuntimeError('Launcher exited before readiness.')
            try:
                status = q.session().get(url + '/v1/status', timeout=2).json()
                if status.get('loaded'):
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Launcher readiness timed out.')
        cfg = json.loads((ROOT / 'runtime/consumer-current.json').read_text())
        assert cfg['qualification']['state'] == 'qualified'
        assert cfg['exe'].endswith('/strata-turbo')
        assert cfg['env']['STRATA_MEDIA_FIXED_PROFILE'] == '1'
        assert status['context']['max_positions'] == 262144
        report.update(configuration=cfg, status=status,
                      binary_sha256=d.digest(Path(cfg['exe'])))
        request, codes = q.recall_body(cfg['model_name'])
        text = q.request(url, request)
        assert all(code in text['text'] for code in codes)
        report.update(recall=text, video=q.video_smoke(url, cfg['model_name']),
                      image=q.image_smoke(url, cfg['model_name']))
        assert (ROOT / 'runtime/install.json').read_bytes() == before
        report['passed'] = True
    finally:
        q.stop(server)
        report['process'] = {'pid': server.pid, 'stopped': True, 'exit_code': server.returncode}
        d.save(folder / 'results.json', report)
print(json.dumps({'passed': report['passed'], 'video': report.get('video'),
                  'weights_saved_unchanged': True}, indent=2))
