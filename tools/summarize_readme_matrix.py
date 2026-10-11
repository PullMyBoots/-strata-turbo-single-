"""Audit and render the complete three-weight ABBA concurrency matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'iq2_xs': 'IQ2_XS', 'swift': 'Swift IQ3_XXS', 'iq3_s': 'IQ3_S'}


def summarize(run):
    cells = []
    prefill = []
    failures = []
    resource_references = {}
    for weight in NAMES:
        for count in (1, 2, 4, 6):
            for arm, indices in [('shared', (0, 3)), ('latest', (1, 2))]:
                rates, decodes, paths, checks = [], [], [], []
                prefill_starts = []
                for index in indices:
                    folder = run / weight / f'c{count}-{index}-{arm}'
                    result = json.loads((folder / 'results.json').read_text())
                    cfg = result['configuration']
                    args = cfg['args']
                    assert args[args.index('--max-context') + 1] == '262144'
                    assert args[args.index('--prompt-cache') + 1] == '0'
                    resource = {flag: args[args.index(flag)+1] for flag in ('--max-context','--kv','--pool-workers','--kv-resident','--vram-reserve-mib','--native')}
                    resource['vision'] = cfg['vision']
                    key = (weight,count)
                    if key not in resource_references: resource_references[key] = resource
                    assert resource == resource_references[key], (key,resource)
                    expected_binary = ('54bf7a29420294216f78fb72e1c46f7496fb54ec809a5c5c464b9d4607a9acb9' if arm == 'shared' else
                                       'c89835335d1a26ba00c404bc2e7fed95c3f8d32b21f28e0e978260da73db173c' if count == 1 else
                                       'dc0065aadc397a2af333e3db7fb33f7c7ab77a0b862992dcf2d5c8e496ef552d')
                    assert result['binary_sha256'] == expected_binary, folder
                    if weight == 'swift': assert cfg['vision']['min_tokens'] == 128
                    assert len(result['rows']) == 3
                    assert json.loads((folder / 'process.json').read_text())['stopped']
                    for wave in result['rows']:
                        assert len(wave['requests']) == count
                        for req in wave['requests']:
                            response = req['result']
                            assert response['usage']['completion_tokens'] == 256
                            assert response['timings']['cache_n'] == 0
                            assert req['request']['reasoning_effort'] == 'none'
                            assert req['request']['temperature'] == 0
                            decodes.append(response['timings']['predicted_per_second'])
                    rates.append(result['aggregate_e2e_tps'])
                    paths.append(str(folder.relative_to(ROOT)))
                    checks.append({k: v['passed'] for k, v in result['checks'].items()})
                    if arm == 'latest':
                        assert all(checks[-1].values()), ('Candidate capability failure requires repair', folder, checks[-1])
                    required = {'isolation', 'image', 'video', 'tools', 'recall'}
                    if weight == 'iq2_xs' or count in (1,6): required.add('native_recall')
                    assert set(checks[-1]) == required
                    if count == 1:
                        timing = {name: result['checks'][name]['result']['timings'] for name in ('recall','native_recall')}
                        assert timing['recall']['prompt_n']==31566 and timing['native_recall']['prompt_n']==256013
                        assert all(t['cache_n']==0 for t in timing.values())
                        prefill_starts.append(timing)
                    for name, passed in checks[-1].items():
                        if not passed:
                            failures.append(dict(weight=weight, count=count, arm=arm, start=index,
                                                 check=name, evidence=paths[-1], detail=result['checks'][name]))
                if count == 1:
                    prefill.append(dict(weight=weight,arm=arm,starts=prefill_starts,
                        short_tps=statistics.mean(t['recall']['prompt_per_second'] for t in prefill_starts),
                        native_tps=statistics.mean(t['native_recall']['prompt_per_second'] for t in prefill_starts),
                        native_s=statistics.mean(t['native_recall']['prompt_ms']/1000 for t in prefill_starts)))
                cells.append(dict(weight=weight, count=count, arm=arm,
                                  aggregate_e2e_tps=statistics.mean(rates), start_rates=rates,
                                  median_request_decode=statistics.median(decodes),
                                  capability_passed=all(all(x.values()) for x in checks), checks=checks,
                                  source_paths=paths))
    return dict(complete=True, starts=48, cells=cells, prefill=prefill, failures=failures,
                protocol='ABBA fresh starts per count and weight; arithmetic mean of two start rates; each start rate is actual output tokens / total three-wave wall time. 256 output tokens per request; three rotated English/code/Chinese waves; thinking/cache off; native cap 262144; shared resources; GPU0 L40S. Latest single uses sparse/fused prefill; latest multi uses target-only direct state transfer and singleton commit. Capability failures retained, not qualified. No cross-hardware or full-window concurrent performance claim.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.run.absolute())
    out = a.output.absolute(); out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['| Weight | Engine | 1 request | 2 requests | 4 requests | 6 requests | 6-slot median request decode |',
             '|---|---|---:|---:|---:|---:|---:|']
    for weight, name in NAMES.items():
        for arm, label in [('shared', 'Strata'), ('latest', 'Strata-Turbo-Single')]:
            rows = [next(r for r in result['cells'] if r['weight']==weight and r['arm']==arm and r['count']==c) for c in (1,2,4,6)]
            values = [f"{r['aggregate_e2e_tps']:.1f}" + (' †' if not r['capability_passed'] else '') for r in rows]
            lines.append('| ' + ' | '.join([name, label, *values, f"{rows[-1]['median_request_decode']:.1f}"]) + ' |')
    (out / 'table.md').write_text('\n'.join(lines) + '\n')
    prefill_lines=['| Weight | Engine | 31,566 input tokens (t/s) | 256,013 input tokens (t/s) | 256,013-token prefill time |','|---|---|---:|---:|---:|']
    for r in result['prefill']:
        prefill_lines.append('| '+ ' | '.join([NAMES[r['weight']], 'Strata' if r['arm']=='shared' else 'Strata-Turbo-Single', f"{r['short_tps']:,.0f}",f"{r['native_tps']:,.0f}", f"{r['native_s']:.1f} s"])+' |')
    (out/'prefill-table.md').write_text('\n'.join(prefill_lines)+'\n')
    manifest = {}
    with tarfile.open(out / 'raw.tar.gz', 'w:gz') as archive:
        for path in sorted(a.run.absolute().rglob('*')):
            if path.is_file():
                relative = str(path.relative_to(ROOT))
                manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
                archive.add(path, arcname=relative, recursive=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('\n'.join(lines)); print('Failed checks:', len(result['failures']))


if __name__ == '__main__':
    main()
