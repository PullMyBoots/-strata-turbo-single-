"""Summarize complete, explicitly selected ABBA cohorts without dropping losses."""
import argparse
import json
import math
from pathlib import Path
import statistics as st


def cohort(path):
    names = ('00-official', '01-fused-prefill', '02-fused-prefill', '03-official')
    if (path / 'measurement-caveat.json').exists():
        raise ValueError('This cohort has a measurement caveat; retain it as pilot evidence: ' + str(path))
    groups = {'official': [], 'candidate': []}
    for name in names:
        row = json.loads((path / name / 'results.json').read_text())
        process = json.loads((path / name / 'process.json').read_text())
        assert process['stopped'] and process['exit_code'] == 0
        assert row['result']['usage']['prompt_tokens'] == 31566
        assert row['native_long']['usage']['prompt_tokens'] == 256013
        assert row['native_long']['usage']['completion_tokens'] == 512
        assert all(row[k]['passed'] for k in ('image', 'video', 'tools'))
        groups['official' if name.endswith('official') else 'candidate'].append(row)
    metrics = {}
    for arm, rows in groups.items():
        family = {name: st.mean(next(x['result']['timings']['predicted_per_second'] for x in r['short']
                                    if x['family'] == name) for r in rows)
                  for name in ('english', 'code', 'chinese')}
        metrics[arm] = dict(
            short_family_decode_tps=family,
            short_geomean_decode_tps=math.exp(st.mean(math.log(x) for x in family.values())),
            prefill_32k_ms=st.mean(r['result']['timings']['prompt_ms'] for r in rows),
            prefill_native_ms=st.mean(r['native_long']['timings']['prompt_ms'] for r in rows),
            native_decode_tps=st.mean(r['native_long']['timings']['predicted_per_second'] for r in rows),
            native_client_wall_s=st.mean(r['native_long']['wall_s'] for r in rows),
            binary_sha256=sorted({r['binary_sha256'] for r in rows}),
            independent_service_starts=len(rows))
    a, b = metrics['official'], metrics['candidate']
    return dict(raw_directory=str(path), metrics=metrics, gains_percent=dict(
        short_decode=100 * (b['short_geomean_decode_tps'] / a['short_geomean_decode_tps'] - 1),
        prefill_32k=100 * (a['prefill_32k_ms'] / b['prefill_32k_ms'] - 1),
        prefill_native=100 * (a['prefill_native_ms'] / b['prefill_native_ms'] - 1),
        native_decode=100 * (b['native_decode_tps'] / a['native_decode_tps'] - 1),
        native_client_time_reduction=100 * (1 - b['native_client_wall_s'] / a['native_client_wall_s'])),
        limitations=['Two independent starts per arm; no confidence interval claim.',
                     'Capability smokes and operator numerical checks are not a comprehensive intelligence evaluation.',
                     'L40S hardware only; no measured cross-hardware speed claim.',
                     'Candidate includes existing decode presets, new sparse draft batching and upstream optional expert fusion.',
                     'IQ2_XS retains its existing automatic 16k prefill cap; other profiles use auto 8k.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cohorts', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = [cohort(path) for path in args.cohorts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
