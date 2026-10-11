"""Consolidate current-release measurements into portable README evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ('iq2_xs', 'swift', 'iq3_s')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path)
    p.add_argument('--pelican', type=Path, required=True)
    p.add_argument('--pelican-only', action='store_true')
    p.add_argument('--pelican-output', type=Path, default=ROOT / 'evidence/showcase')
    p.add_argument('--allow-incomplete', action='store_true')
    a = p.parse_args()
    if a.pelican_only:
        export_pelican(a.pelican, a.pelican_output, a.allow_incomplete)
        return
    if a.benchmark is None:
        p.error('--benchmark is required unless --pelican-only is selected')
    out = ROOT / 'evidence/showcase'
    build = json.loads((ROOT / 'distribution/build.json').read_text())
    summary = {'version': (ROOT / 'VERSION').read_text().strip(),
               'created_utc': datetime.now(timezone.utc).isoformat(),
               'measurement_only': True, 'build': build,
               'hardware': json.loads((a.benchmark / 'hardware.json').read_text()),
               'curves': {}, 'concurrency': {}, 'missing': []}
    for weight in WEIGHTS:
        for arm in ('official', 'turbo'):
            name = f'{weight}-{arm}'
            source = a.benchmark / (name + '-curves')
            if not (source / 'results.json').exists():
                summary['missing'].append(name + '-curves')
            else:
                report = json.loads((source / 'results.json').read_text())
                assert report['binary_sha256'] == build['binaries']['turbo' if arm == 'turbo' else 'official_media']['sha256']
                points = []
                for size in (0, 8192, 65536, 253952):
                    for family in ('english', 'code', 'chinese'):
                        rows = [r for r in report['rows'] if r['context_background_tokens'] == size and r['family'] == family]
                        rates = [r['result']['timings']['predicted_per_second'] for r in rows]
                        assert len(rates) == 2
                        points.append({'background_tokens': size, 'family': family,
                                       'prompt_tokens': rows[0]['result']['usage']['prompt_tokens'],
                                       'median_decode_tps': statistics.median(rates),
                                       'minimum_decode_tps': min(rates), 'maximum_decode_tps': max(rates),
                                       'samples': rates, 'cached_tokens': [r['result']['timings']['cache_n'] for r in rows]})
                cold = [{'background_tokens': r['context_background_tokens'],
                         'prompt_tokens': r['result']['usage']['prompt_tokens'],
                         'prefill_tps': r['result']['timings']['prompt_per_second'],
                         'prefill_s': r['result']['timings']['prompt_ms']/1000,
                         'cached_tokens': r['result']['timings']['cache_n']}
                        for r in report['rows'] if r['cold_prefill_sample']]
                summary['curves'][name] = {'points': points, 'cold_prefill': cold,
                                           'protocol': report['protocol']}
                for filename in ('config.json', 'results.json', 'status.json', 'process.json'):
                    if (source / filename).exists():
                        save(out / 'raw' / source.name / filename, json.loads((source / filename).read_text()))
            for count in (1, 2, 4, 6):
                source = a.benchmark / f'{name}-c{count}'
                if not (source / 'results.json').exists():
                    summary['missing'].append(source.name)
                    continue
                report = json.loads((source / 'results.json').read_text())
                rows = report['rows']
                assert len(rows) == 3
                tokens = sum(r['usage']['completion_tokens'] for row in rows for r in row['requests'])
                requests = [r for row in rows for r in row['requests']]
                summary['concurrency'].setdefault(name, []).append({
                    'count': count, 'aggregate_e2e_tps': tokens/sum(r['wall_s'] for r in rows),
                    'wave_e2e_tps': [r['aggregate_e2e_tps'] for r in rows],
                    'median_request_decode_tps': statistics.median(r['timings']['predicted_per_second'] for r in requests),
                    'median_ttft_s': statistics.median(r['first_content_s'] for r in requests),
                    'output_tokens': tokens, 'protocol': report['protocol']})
                for filename in ('config.json', 'results.json', 'status.json', 'process.json'):
                    if (source / filename).exists():
                        save(out / 'raw' / source.name / filename, json.loads((source / filename).read_text()))
    summary['complete'] = not summary['missing']
    pairs = {}
    for weight in WEIGHTS:
        paths = [a.benchmark / f'{weight}-{arm}-curves/results.json' for arm in ('official', 'turbo')]
        if all(path.exists() for path in paths):
            reports = [json.loads(path.read_text()) for path in paths]
            hashes = [[r['request_sha256'] for r in report['rows']] for report in reports]
            assert hashes[0] == hashes[1], f'Mismatched paired prompts: {weight}'
            pairs[weight] = {'paired_requests_match': True, 'requests_per_arm': len(hashes[0])}
            configurations = [report['configuration'] for report in reports]
            target_assets = {}
            for flag in ('--native', '--ple-gguf', '--expert-profile'):
                values = [cfg['args'][cfg['args'].index(flag) + 1] for cfg in configurations]
                assert values[0] == values[1], f'Mismatched target assets: {weight} {flag}'
                target_assets[flag] = {'path': values[0], 'bytes': Path(values[0]).stat().st_size}
            pairs[weight]['same_target_asset_paths'] = target_assets
    summary['paired_prompt_validation'] = pairs
    if not a.allow_incomplete:
        assert summary['complete'], summary['missing']
    save(out / 'benchmark-summary.json', summary)

    export_pelican(a.pelican, a.pelican_output, a.allow_incomplete)
    print('Benchmark complete:', summary['complete'])


def export_pelican(source_root, out, allow_incomplete=False):
    """Export a separate campaign without overwriting throughput evidence."""
    report = json.loads((source_root / 'results.json').read_text()) if (source_root / 'results.json').exists() else {'runs': []}

    if len(report['runs']) != 4:
        if not allow_incomplete:
            raise RuntimeError('Pelican ABBA is incomplete')
        return
    first_requests = []
    rendered_prompts = []
    for row in report['runs']:
        source = source_root / f"{row['index']}-{row['arm']}"
        payloads = [json.loads(f.read_text()) for f in sorted(source.glob('api-[0-9]*.json'), key=lambda f: int(f.stem.split('-')[1]))]
        calls = [p for p in payloads if p['path'].split('?')[0] == '/v1/messages']
        assert all(p['request']['output_config']['effort'] == 'max' and
                   p['request']['max_tokens'] == 65536 and
                   not p['request'].get('reasoning_budget_tokens') for p in calls)
        first = dict(calls[0]['request'])
        first.pop('metadata', None)
        first_requests.append(first)
        cfg = json.loads((source / 'config.json').read_text())
        adapter = ROOT / ('experimental/native-batch/serve/frontend.py'
                          if cfg['server_module'] == 'experimental-native-api' else 'serve/frontend.py')
        module_name = f"showcase_frontend_{row['index']}"
        spec = importlib.util.spec_from_file_location(module_name, adapter)
        frontend = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = frontend
        spec.loader.exec_module(frontend)
        template = Path(cfg['tokenizer']) / 'chat_template.jinja'
        if not template.exists():
            template = adapter.parent / 'chat_template.jinja'
        messages, tools, options = frontend.anthropic_to_messages(first)
        prompt = frontend.ChatTemplate(template).render(messages, tools=tools, **options).encode()
        rendered_prompts.append({'arm': f"{row['index']}-{row['arm']}",
                                 'bytes': len(prompt), 'sha256': hashlib.sha256(prompt).hexdigest(),
                                 'wire_request_sha256': hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(),
                                 'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
                                 'template_sha256': hashlib.sha256(template.read_bytes()).hexdigest(),
                                 'render_options': options})
        assert first['output_config']['effort'] == 'max' and first['max_tokens'] == 65536
        row['model_calls'] = len(calls)
        row['total_output_tokens'] = row['claude_result'].get('usage', {}).get('output_tokens')
        row['html_sha256'] = hashlib.sha256((source / 'pelican.html').read_bytes()).hexdigest()
        dest = out / 'pelican' / source.name
        for filename in ('config.json', 'summary.json', 'status.json', 'process.json'):
            save(dest / filename, json.loads((source / filename).read_text()))
        for filename in ('claude.jsonl', 'engine.log', 'server.log', 'stderr.log'):
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / filename, dest / (filename + '.txt'))
        for file in source.glob('api-*'):
            shutil.copy2(file, dest / file.name)
    report['first_requests_match_except_metadata'] = all(r == first_requests[0] for r in first_requests)
    assert report['first_requests_match_except_metadata']
    report['initial_prompt_parity'] = {'rows': rendered_prompts,
                                      'same': len({r['sha256'] for r in rendered_prompts}) == 1,
                                      'scope': 'Offline rendering with the recorded tokenizer template and current API adapters; adapter/template hashes bind the check.'}
    report['native_reasoning_effort'] = 'xhigh'
    report['all_model_calls_max_effort_and_output_budget'] = True
    report['complete'] = True
    report['hardware'] = json.loads((source_root / 'hardware.json').read_text())
    report['engine_median_s'] = {arm: statistics.median(r['wall_s'] for r in report['runs'] if r['arm'] == arm)
                               for arm in ('official', 'turbo')}
    save(out / 'pelican-summary.json', report)
    save(out / 'pelican' / 'initial-request.json', first_requests[0])
    save(out / 'pelican' / 'command.json', json.loads((source_root / 'command.json').read_text()))
    print(json.dumps({'pelican_complete': True,
                      'pelican_medians': report['engine_median_s'],
                      'rendered_initial_prompts_match': report['initial_prompt_parity']['same']}, indent=2))


if __name__ == '__main__':
    main()
