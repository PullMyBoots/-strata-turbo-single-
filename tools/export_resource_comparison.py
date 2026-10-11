"""Export a complete default/shared-resource/Turbo concurrency comparison.

Keep every model and every selected slot count. Two independent process starts
per arm provide an observed range, not a confidence interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('default', 'shared', 'incumbent')
WEIGHTS = ('iq2_xs', 'swift', 'iq3_s')


def arg(cfg, flag):
    return cfg['args'][cfg['args'].index(flag) + 1]


def export(inputs, destination, counts=(2,)):
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    hardware = []
    for source in inputs:
        hardware.append(json.loads((source / 'hardware.json').read_text()))
        rows = json.loads((source / 'summary.json').read_text())
        for row in rows:
            assert row['arm'] in ARMS
            folder = source / f"c{row['count']}-{row['index']}-{row['arm']}"
            report = json.loads((folder / 'results.json').read_text())
            cfg = report['configuration']
            weight = cfg['deployment']['weights']
            assert weight in WEIGHTS
            assert report['holdout'] and len(report['rows']) == 3
            assert arg(cfg, '--max-context') == '262144' and arg(cfg, '--kv') == 'int8'
            assert arg(cfg, '--prompt-cache') == '0'
            assert all(c['passed'] for c in report['checks'].values())
            process = json.loads((folder / 'process.json').read_text())
            assert process['stopped'] and process['exit_code'] == 0
            requests = [r for batch in report['rows'] for r in batch['requests']]
            assert len(requests) == row['count'] * 3
            assert all(r['result']['usage']['completion_tokens'] == 256 and
                       r['result']['timings']['cache_n'] == 0 for r in requests)
            assert all(r['request']['seed']==20261010 and r['request']['temperature']==0 and
                       r['request']['reasoning_effort']=='none' and
                       r['request']['cache_prompt'] is False for r in requests)
            if row['arm'] == 'default':
                assert '--pool-workers' not in cfg['args']
                assert arg(cfg, '--kv-resident') == '32768'
                assert arg(cfg, '--vram-reserve-mib') == '700'
            else:
                assert arg(cfg, '--pool-workers') == '32'
                assert arg(cfg, '--kv-resident') == ('16384' if row['count'] == 2 else '8192')
                assert arg(cfg, '--vram-reserve-mib') == '1400'
            assert ('--batch-mtp' in cfg['args']) == (row['arm'] == 'incumbent')
            raw = destination / 'raw' / weight / folder.name
            raw.mkdir(parents=True, exist_ok=True)
            for name in ('config.json', 'results.json', 'status.json', 'process.json',
                         'capability-checks.json', 'waves.json'):
                shutil.copy2(folder / name, raw / name)
            log = (folder / 'engine.log').read_text()
            (raw / 'engine-start.txt').write_text('\n'.join(log.splitlines()[:80])+'\n')
            match = re.search(r'CPU pool tasks/phase:.*participating threads: (\d+)', log)
            reports.setdefault((weight, row['count'], row['arm']), []).append(dict(
                aggregate_e2e_tps=report['aggregate_e2e_tps'],
                wave_e2e_tps=[r['aggregate_e2e_tps'] for r in report['rows']],
                raw=str(raw.relative_to(ROOT)),
                engine_sha256=report['binary_sha256'],
                target=[arg(cfg, f) for f in ('--native', '--ple-gguf', '--expert-profile')],
                prompt_sha256=[hashlib.sha256(json.dumps(r['request'], sort_keys=True).encode()).hexdigest()
                               for r in requests],
                participating_cpu_threads=int(match.group(1)) if match else None))
    assert len({h['gpu']['uuid'] for h in hardware}) == 1
    table = []
    assert set(reports) == {(w,n,a) for w in WEIGHTS for n in counts for a in ARMS}
    for weight in WEIGHTS:
        for count in counts:
            arms = {}
            expected_prompts = expected_targets = expected_engine = None
            for arm in ARMS:
                runs = reports[weight, count, arm]
                assert len(runs) == 2, (weight, count, arm, len(runs))
                for run in runs:
                    if expected_prompts is None:
                        expected_prompts, expected_targets, expected_engine = (
                            run['prompt_sha256'], run['target'], run['engine_sha256'])
                    assert run['prompt_sha256'] == expected_prompts
                    assert run['target'] == expected_targets
                    # All three arms use the same pinned numerical executable.
                    assert run['engine_sha256'] == expected_engine
                rates = [r['aggregate_e2e_tps'] for r in runs]
                arms[arm] = dict(median_e2e_tps=statistics.median(rates),
                                 minimum_e2e_tps=min(rates), maximum_e2e_tps=max(rates), runs=runs)
            turbo = arms['incumbent']['median_e2e_tps']
            table.append(dict(weight=weight, count=count, arms=arms,
                              gain_vs_default_percent=100*(turbo/arms['default']['median_e2e_tps']-1),
                              gain_vs_shared_percent=100*(turbo/arms['shared']['median_e2e_tps']-1)))
    summary = dict(complete=True, release='0.2.2', hardware=hardware[0], rows=table,
                   arm_order_per_count=['default','shared','incumbent','incumbent','shared','default'],
                   protocol='Each arm has two independent service starts, identical 64-token warm-up and three rotated English/code/Chinese simultaneous batches. Actual 256 output tokens per request; thinking and prefix cache off; seed 20261010; native 262144 context; INT8 KV. Median and observed range of process-level aggregate end-to-end rates. Includes prefill/admission and all requests.',
                   limitations=['Default resource policy is reproduced with a common pinned executable, native media/API adapter and controlled GPU/NUMA placement; this is not an untouched installation test.',
                                'Default workers/KV/reserve are an explicit resource ablation. Their benefit is not a new numerical algorithm.',
                                'Two independent starts per arm provide no confidence interval.',
                                'These are short-input concurrent workloads; six simultaneous near-native contexts are unvalidated.',
                                'Target verification is retained; these smokes do not establish comprehensive intelligence equivalence.'])
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='arms'} for row in table], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, nargs=3, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'evidence/resource-comparison')
    p.add_argument('--counts', type=int, nargs='+', choices=(2,6), default=[2])
    a = p.parse_args()
    export([p.absolute() for p in a.input], a.output.absolute(), a.counts)


if __name__ == '__main__':
    main()
