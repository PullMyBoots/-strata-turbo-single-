"""Summarize complete six-slot held-out ABBA runs, retaining every weight."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def cohort(path, candidate='self-commit'):
    names = ('c6-0-no-mtp', f'c6-1-{candidate}', f'c6-2-{candidate}', 'c6-3-no-mtp')
    groups = {'official': [], 'candidate': []}
    configs = []
    for index, name in enumerate(names):
        folder = path / name
        process = read(folder / 'process.json')
        if not process['stopped'] or process['exit_code'] != 0:
            raise ValueError('Incomplete or failed service: ' + str(folder))
        row = read(folder / 'results.json')
        if not row['holdout'] or len(row['rows']) != 3:
            raise ValueError('Expected three held-out waves: ' + str(folder))
        if row['status']['context']['max_positions'] != 262144:
            raise ValueError('Context cap changed')
        for name in ('isolation', 'image', 'video', 'tools', 'recall', 'native_recall'):
            if not row['checks'][name]['passed']:
                raise ValueError('Failed capability: ' + name)
        if row['checks']['native_recall']['result']['usage']['prompt_tokens'] != 256013:
            raise ValueError('Wrong native input length')
        for wave in row['rows']:
            if len(wave['requests']) != 6:
                raise ValueError('Expected six concurrent requests')
            for request in wave['requests']:
                result = request['result']
                if result['usage']['completion_tokens'] != 256 or result['timings']['cache_n'] != 0:
                    raise ValueError('Token count or caching differs')
                if request['request']['temperature'] != 0 or request['request']['reasoning_effort'] != 'none':
                    raise ValueError('Sampling protocol differs')
        cfg = row['configuration']
        configs.append(cfg)
        groups['official' if index in (0, 3) else 'candidate'].append(row)
    # Same numerical request settings and resources; the candidate only adds its
    # feature flag and disables unrelated experimental flags.
    reference = configs[0]
    for cfg in configs[1:]:
        for key in ('args', 'parallel', 'parallel_workers', 'model_name', 'server_module', 'vision'):
            if cfg.get(key) != reference.get(key):
                raise ValueError('Configuration changed: ' + key)
    metrics = {}
    for arm, rows in groups.items():
        metrics[arm] = dict(
            aggregate_e2e_tps=statistics.mean(r['aggregate_e2e_tps'] for r in rows),
            starts_tps=[r['aggregate_e2e_tps'] for r in rows],
            binary_sha256=sorted({r['binary_sha256'] for r in rows}),
            request_count=sum(len(wave['requests']) for row in rows for wave in row['rows']),
            median_first_content_s=statistics.median(request['result']['first_content_s']
                for row in rows for wave in row['rows'] for request in wave['requests']),
            median_request_wall_s=statistics.median(request['result']['wall_s']
                for row in rows for wave in row['rows'] for request in wave['requests']),
            per_family_decode_tps={family: statistics.mean(
                statistics.median(request['result']['timings']['predicted_per_second']
                                  for wave in row['rows'] for request in wave['requests']
                                  if request['family'] == family) for row in rows)
                for family in ('english', 'code', 'chinese')})
    return dict(raw_directory=str(path), model=reference['model_name'], metrics=metrics,
                gain_percent=100 * (metrics['candidate']['aggregate_e2e_tps'] / metrics['official']['aggregate_e2e_tps'] - 1),
                limitations=['Two independent service starts per arm; no confidence interval claim.',
                             'Only target-only six-slot throughput is qualified here; this cohort does not qualify batch-MTP throughput.',
                             'Native context is a capability check, not six-way native-input throughput.',
                             'L40S only; no measured universal hardware claim.',
                             'Capability checks are not comprehensive intelligence evaluation.',
                             'Complete response text varies between unchanged six-slot controls.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cohorts', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--candidate-arm', choices=('self-commit','direct-state'), default='self-commit')
    args = p.parse_args()
    rows = [cohort(path, args.candidate_arm) for path in args.cohorts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(row['model'], row['gain_percent'], row['metrics'], flush=True)


if __name__ == '__main__':
    main()
