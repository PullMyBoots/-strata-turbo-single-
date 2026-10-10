"""Replay a bounded, same-device prefill candidate with exact configuration evidence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import distribution as dist
import qualify_distribution as qualify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--weights', choices=dist.MODELS, default='iq2_xs')
    parser.add_argument('--variant', choices=('old', 'official', 'dynamic', 'phase', 'phase16', 'phase32'), required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--media', action='store_true')
    args = parser.parse_args()
    args.output = args.output.absolute()
    args.output.mkdir(parents=True, exist_ok=False)
    hw = dist.detect(args.gpu)
    plan = dist.resource_plan(hw, args.weights)
    if args.base_config:
        cfg = json.loads(args.base_config.read_text())
        cfg['gpu'] = [hw['gpu']['index']]
        cfg['env']['CUDA_VISIBLE_DEVICES'] = hw['gpu']['uuid']
        cfg['vision']['cuda_device'] = hw['gpu']['uuid']
        cfg['deployment']['physical_gpus'] = [hw['gpu']['index']]
        cfg['deployment']['gpu_uuids'] = [hw['gpu']['uuid']]
        cfg['deployment']['numa_node'] = hw['numa_node']
        cfg['deployment']['cpu_affinity'] = hw['local_cpu_affinity']
    else:
        settings = json.loads((dist.ROOT / 'runtime/install.json').read_text())
        cfg = dist.build_config(settings['assets'], hw, args.weights, plan,
                                arm='official' if args.variant == 'official' else 'turbo')
    cfg = copy.deepcopy(cfg)
    if cfg['deployment']['weights'] != args.weights:
        raise ValueError('The saved base configuration does not match --weights.')
    if args.binary:
        cfg['exe'] = str(args.binary.absolute())
    if args.variant in ('dynamic', 'phase', 'phase16', 'phase32'):
        cfg['args'] = dist.set_arg(cfg['args'], '--no-prefill-borrow')
        chunk = {'phase16': 'auto:16384', 'phase32': 'auto:32768'}.get(args.variant, 'auto')
        cfg['args'] = dist.set_arg(cfg['args'], '--prefill', chunk)
        if args.variant == 'dynamic':
            cfg['env'].pop('STRATA_VERIFY_DEVICE_ONLY', None)
            cfg['env'].pop('STRATA_VERIFY_PRECAPTURE', None)
            cfg['env']['STRATA_VERIFY_DEVICE_PLAN'] = '1'
        elif args.weights == 'iq2_xs':
            cfg['env']['STRATA_VERIFY_PREFILL_RESTORE'] = '1'
    dist.save(args.output / 'hardware.json', hw)
    dist.save(args.output / 'input-config.json', cfg)
    dist.save(args.output / 'manifest.json', {
        'variant': args.variant, 'weights': args.weights,
        'binary_sha256': dist.digest(Path(cfg['exe'])),
        'context': dist.single_gpu.CONTEXT, 'prompt_cache': False,
        'long_checked': args.long, 'media_checked': args.media,
        'main_weights_changed': False,
    })
    report = qualify.evaluate(cfg, args.output, long=args.long, media=args.media)
    rates = report['rates']
    summary = {'variant': args.variant, 'weights': args.weights,
               'short_tps': qualify.math.exp(qualify.statistics.mean(
                   qualify.math.log(qualify.statistics.median(rates[case]))
                   for case in ('english', 'chinese', 'code'))),
               'recall': report['recall']['timings'], 'correct': report['correct']}
    if args.long:
        summary['native_long'] = {k: v for k, v in report['native_long'].items() if k != 'text'}
    dist.save(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
