"""Isolated single-session stage diagnostic; target/presets are unchanged."""
import argparse
import json
from pathlib import Path

import distribution as d
import qualify_distribution as q
from benchmark_showcase import TASKS, measure
from showcase_lock import acquire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', choices=tuple(d.MODELS), default='iq3_s')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    lock = acquire()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    hardware = d.detect(args.gpu)
    d.save(output / 'hardware.json', hardware)
    cfg = d.build_config(d.ROOT / 'assets', hardware, args.weights,
                         d.resource_plan(hardware, args.weights), arm='turbo')
    d.check_config(cfg)
    cfg['args'] = d.set_arg(cfg['args'], '--prompt-cache', '0')
    cfg['env']['STRATA_VERIFY_PROFILE'] = '1'
    cfg['env']['STRATA_DECODE_TIMING'] = '1'

    def run(url, config, folder):
        rows = []
        for family, task in TASKS.items():
            req = q.body(config['model_name'], task, 512)
            req.pop('ignore_eos', None)
            req.update(seed=20261010, cache_prompt=False)
            result = q.request(url, req)
            row = dict(family=family, request=req, result=result)
            d.save(folder / (family + '.json'), row)
            assert result['usage']['completion_tokens'] == 512
            assert result['timings']['cache_n'] == 0
            rows.append(row)
            print(family, result['timings']['predicted_per_second'], flush=True)
        return dict(rows=rows, diagnostic_only=True,
                    warning='Profiling changes execution overhead; not a new speed claim.')

    result = measure(cfg, output / 'diagnostic', run)
    d.save(output / 'summary.json', result)
    print('Stage diagnostic complete; owned service stopped.', flush=True)


if __name__ == '__main__':
    main()
