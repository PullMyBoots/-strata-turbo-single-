"""Collect cold prefill phase evidence without changing numerical presets."""
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
    parser.add_argument('--arms', nargs='+', choices=('official', 'turbo', 'draft-f16', 'batch-control', 'sparse-batch', 'fused-prefill'),
                        default=('official', 'turbo'))
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--long', action='store_true')
    parser.add_argument('--media', action='store_true')
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--cache', action='store_true', help='Enable and check exact/changed-suffix prefix reuse')
    args = parser.parse_args()
    with acquire():
        output = args.output.absolute()
        output.mkdir(parents=True, exist_ok=False)
        hardware = d.detect(args.gpu)
        d.save(output / 'hardware.json', hardware)
        rows = []
        for index, arm in enumerate(args.arms):
            cfg = d.build_config(d.ROOT / 'assets', hardware, args.weights,
                                 d.resource_plan(hardware, args.weights),
                                 arm='official' if arm == 'official' else 'turbo')
            cfg['args'] = d.set_arg(cfg['args'], '--prompt-cache', '2' if args.cache else '0')
            d.check_config(cfg)
            if not args.no_profile:
                cfg['env']['STRATA_PREFILL_TIMING'] = '1'
                cfg['env']['STRATA_DRAFT_TIMING'] = '1'
            if arm == 'draft-f16':
                cfg['env']['STRATA_MTP_BATCH_F16'] = '1'
            if arm in ('batch-control', 'sparse-batch', 'fused-prefill'):
                cfg['exe'] = str(d.ROOT / 'experimental/sparse-draft-prefill/build/strata-sparse-draft-prefill')
                cfg['env']['STRATA_MTP_SPARSE_BATCH'] = '0' if arm == 'batch-control' else '1'
            if arm == 'fused-prefill':
                cfg['env']['STRATA_PF_FUSED'] = '1'

            def run(url, config, folder):
                short = []
                if args.short:
                    for family, task in TASKS.items():
                        request = q.body(config['model_name'], task, 512)
                        request.pop('ignore_eos', None)
                        result = q.request(url, request)
                        d.save(folder / (family + '.json'), dict(request=request, result=result))
                        assert result['usage']['completion_tokens'] == 512
                        assert result['timings']['cache_n'] == 0
                        short.append(dict(family=family, result=result))
                request, codes = q.recall_body(config['model_name'])
                request.pop('ignore_eos', None)
                if args.cache:
                    request['cache_prompt'] = True
                result = q.request(url, request)
                d.save(folder / 'response.json', dict(request=request, result=result))
                assert result['timings']['cache_n'] == 0
                assert all(code in result['text'] for code in codes)
                print(arm, result['timings'], flush=True)
                report = dict(result=result, short=short, diagnostic_only=not args.no_profile,
                              limitation='A bounded mechanism screen, not release qualification or intelligence equivalence.')
                if args.cache:
                    for label, suffix in (('exact', ''), ('changed-suffix', '\nAlso write CHECKPOINT-OK after the codes.')):
                        cached_request = dict(request, messages=[dict(request['messages'][0])])
                        cached_request['messages'][0]['content'] += suffix
                        cached_result = q.request(url, cached_request)
                        d.save(folder / ('cache-' + label + '.json'), dict(request=cached_request, result=cached_result))
                        assert cached_result['timings']['cache_n'] > 0
                        assert all(code in cached_result['text'] for code in codes)
                        if suffix:
                            assert 'CHECKPOINT-OK' in cached_result['text']
                        report['cache-' + label] = cached_result
                if args.long:
                    request = json.loads((d.ROOT / 'evidence/requests/long-256k.json').read_text())['request']
                    request = dict(request, model=config['model_name'])
                    long_result = q.request(url, request)
                    d.save(folder / 'native-long.json', dict(request=request, result=long_result))
                    codes = json.loads((d.ROOT / 'evidence/checkpoints.json').read_text())
                    assert long_result['usage']['prompt_tokens'] == 256013
                    assert long_result['usage']['completion_tokens'] == 512
                    assert long_result['timings']['cache_n'] == 0
                    assert all(code in long_result['text'] for code in codes)
                    report['native_long'] = long_result
                    print(arm, 'native-long', long_result['timings'], flush=True)
                if args.media:
                    for name, test in (('image', q.image_smoke), ('video', q.video_smoke), ('tools', q.tool_smoke)):
                        report[name] = test(url, config['model_name'])
                        d.save(folder / (name + '.json'), report[name])
                return report

            rows.append(measure(cfg, output / f'{index:02d}-{arm}', run))
        d.save(output / 'summary.json', rows)


if __name__ == '__main__':
    main()
