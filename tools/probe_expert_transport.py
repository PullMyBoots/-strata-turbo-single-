"""Calibrate CPU versus PCIe execution of nonresident experts, using all experts.

Calibration tasks differ from the README holdout. No target weight, context or
KV precision changes. A winning setting still requires a fresh-engine holdout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import distribution as d
import qualify_distribution as q
from benchmark_showcase import measure
from showcase_lock import acquire

TASKS = {
    'english': 'Write a long, detailed numbered guide to organizing and reviewing a software project. Continue for at least 100 items.',
    'code': 'Write a complete Python program implementing a persistent key-value database with transactions, recovery, examples and tests. Output code only.',
    'chinese': '请详细介绍如何设计一个可靠的软件项目，持续列举实现步骤、测试方法和维护建议。至少写一百项。',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', choices=tuple(d.MODELS), default='iq3_s')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--candidate-exe', type=Path, required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.55, 0.75, 1.0])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if .55 not in args.fractions or len(set(args.fractions)) != len(args.fractions) or any(not 0 <= p <= 1 for p in args.fractions):
        raise ValueError('Distinct PCIe fractions in 0..1 including control 0.55 are required.')
    lock = acquire()
    folder = args.output.absolute()
    folder.mkdir(parents=True, exist_ok=False)
    hardware = d.detect(args.gpu)
    d.save(folder / 'hardware.json', hardware)
    cfg = d.build_config(d.ROOT / 'assets', hardware, args.weights,
                         d.resource_plan(hardware, args.weights), arm='turbo')
    d.check_config(cfg)
    cfg['exe'] = str(args.candidate_exe.absolute())
    cfg['env'].update(STRATA_CUDA_EXPERT_FUSED_ALL='1', STRATA_CPU_MISS_ACTQ='1', STRATA_MOE_DIRECT_COMBINE='1')
    cfg['args'] = d.set_arg(cfg['args'], '--prompt-cache', '0')

    def run(url, config, output):
        rows = []
        for repeat in range(args.repeats):
            for family_index, (family, text) in enumerate(TASKS.items()):
                shift = (repeat + family_index) % len(args.fractions)
                order = args.fractions[shift:] + args.fractions[:shift]
                if repeat % 2:
                    order = list(reversed(order))
                for fraction in order:
                    req = q.body(config['model_name'], text, 512)
                    req.pop('ignore_eos', None)
                    req.update(seed=20261010, strata_tune={'pcie_frac': fraction})
                    result = q.request(url, req)
                    row = dict(pcie_fraction=fraction, family=family, repeat=repeat, request=req, result=result)
                    rows.append(row)
                    d.save(output / f'{repeat}-{family}-{fraction}.json', row)
                    assert result['usage']['completion_tokens'] == 512 and result['timings']['cache_n'] == 0
                    print(fraction, family, repeat, result['timings']['predicted_per_second'], flush=True)
        rates = {str(fraction): {family: statistics.median(row['result']['timings']['predicted_per_second']
                 for row in rows if row['pcie_fraction'] == fraction and row['family'] == family)
                 for family in TASKS} for fraction in args.fractions}
        ratios = {fraction: {family: rates[fraction][family] / rates['0.55'][family] for family in TASKS} for fraction in rates}
        gains = {fraction: math.exp(statistics.mean(math.log(ratio) for ratio in values.values())) - 1
                 for fraction, values in ratios.items()}
        eligible = [p for p in rates if gains[p] >= .08 and min(ratios[p].values()) >= .97]
        selected = max(eligible, key=gains.get) if eligible else '0.55'
        report = dict(rows=rows, rates=rates, ratios=ratios, geometric_gain=gains,
                      selected_for_holdout=float(selected), calibration_only=True,
                      protocol='One fresh single-GPU service; Latin-rotation order within task/repeat; three task families; 512 cold tokens; target/context/KV unchanged.')
        print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
        return report

    measure(cfg, folder / 'service', run)
    lock.close()


if __name__ == '__main__':
    main()
