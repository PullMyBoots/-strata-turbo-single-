"""Isolated decoder candidate screening; every arm preserves target/context/KV.

This is a screening protocol, not release qualification. All raw requests and
responses are retained, including candidate regressions and output differences.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics

import distribution as d
import qualify_distribution as q
from benchmark_showcase import TASKS, measure
from strata_tokenizer import Tokenizer
from showcase_lock import acquire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--weights', choices=tuple(d.MODELS), default='iq2_xs')
    parser.add_argument('--arms', nargs='+', default=['incumbent', 'interleaved', 'incumbent'])
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--candidate-exe', type=Path)
    parser.add_argument('--transport-fraction', type=float, default=1.0,
                        help='PCIe fraction for the expert-transport arm only')
    parser.add_argument('--native-context', action='store_true',
                        help='Also measure all families near the unchanged native context limit')
    parser.add_argument('--media', action='store_true',
                        help='Run image, video, tools, arithmetic and recall checks after timing')
    parser.add_argument('--fixed-window', type=int, choices=range(2,9),
                        help='Numerical diagnostic: no adaptive lookahead, cache swaps or suffix proposals')
    args = parser.parse_args()
    lock = acquire()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    hardware = d.detect(args.gpu)
    d.save(output / 'hardware.json', hardware)
    base = d.build_config(d.ROOT / 'assets', hardware, args.weights,
                          d.resource_plan(hardware, args.weights), arm='turbo')
    d.check_config(base)
    reference = {}
    summary = []
    for index, arm in enumerate(args.arms):
        cfg = copy.deepcopy(base)
        if arm == 'interleaved':
            cfg['env']['STRATA_MMVQ_IL'] = '1'
        elif arm == 'gdn-tiled':
            cfg['env']['STRATA_GDN_VERIFY_TILED'] = '1'
        elif arm == 'pipeline':
            if not args.candidate_exe:
                raise ValueError('Pipeline screening requires --candidate-exe')
            cfg['exe'] = str(args.candidate_exe.absolute())
            cfg['env']['STRATA_GDN_PIPELINE'] = '1'
        elif arm in ('fused-all', 'miss-actq', 'mixed-combine', 'expert-fusion', 'fusion-virtual', 'expert-transport', 'virtual2', 'fused-and-lazy', 'candidate-control', 'expert-wide', 'expert-pair', 'norm-down', 'norm-down-pair', 'resident-bypass'):
            if not args.candidate_exe:
                raise ValueError('Candidate screening requires --candidate-exe')
            cfg['exe'] = str(args.candidate_exe.absolute())
            if arm in ('resident-bypass', 'candidate-control'):
                cfg['env']['STRATA_RESIDENT_HOST_BYPASS'] = '1' if arm == 'resident-bypass' else '0'
            if arm in ('fused-all', 'fused-and-lazy', 'expert-fusion', 'fusion-virtual', 'expert-transport'):
                cfg['env']['STRATA_CUDA_EXPERT_FUSED_ALL'] = '1'
            if arm in ('miss-actq', 'fused-and-lazy', 'expert-fusion', 'fusion-virtual', 'expert-transport'):
                cfg['env']['STRATA_CPU_MISS_ACTQ'] = '1'
            if arm in ('mixed-combine', 'expert-fusion', 'fusion-virtual', 'expert-transport'):
                cfg['env']['STRATA_MOE_DIRECT_COMBINE'] = '1'
            if arm in ('virtual2', 'fusion-virtual'):
                cfg['env']['STRATA_MMVQ_VIRTUAL2'] = '1'
            if arm == 'expert-wide':
                cfg['env']['STRATA_CUDA_EXPERT_TILE64'] = '1'
            if arm in ('expert-pair','norm-down-pair'):
                cfg['env']['STRATA_CUDA_EXPERT_PAIR'] = '1'
            if arm in ('norm-down','norm-down-pair'):
                cfg['env']['STRATA_GR_NORM_DOWN'] = '1'
            if arm == 'expert-transport':
                if not 0 <= args.transport_fraction <= 1:
                    raise ValueError('Transport fraction must be in 0..1')
                cfg['args'] = d.set_arg(cfg['args'],'--pcie-frac',args.transport_fraction)
        elif arm == 'commit-only':
            cfg['env']['STRATA_GDN_COMMIT_ONLY'] = '1'
        elif arm == 'common-api':
            cfg['server_module'] = 'experimental-native-api'
        elif arm in ('draft128', 'draft512', 'draft2048'):
            # Only the auxiliary head shortlist changes. Keep the calibrated
            # projection, vocabulary and complete target verifier untouched.
            source = Path(cfg['env']['STRATA_MTP_HEAD_ROUTER'])
            fields = (source / 'router.txt').read_text().split()
            if len(fields) != 9 or fields[0] != 'SRH1':
                raise ValueError('Unexpected auxiliary router profile')
            fields[2] = arm.removeprefix('draft')
            target = output / f'{index}-{arm}-router'
            target.mkdir()
            (target / 'router.txt').write_text(' '.join(fields) + '\n')
            identity = {}
            for name in ('projection-f32.bin', 'coarse-q8.bin', 'bias-f32.bin'):
                original = source / name
                (target / name).symlink_to(original.absolute())
                identity[name] = d.digest(original)
            d.save(target / 'source.json', dict(source=str(source), auxiliary_only=True,
                                               unchanged_asset_sha256=identity))
            cfg['env']['STRATA_MTP_HEAD_ROUTER'] = str(target)
        elif arm == 'official':
            cfg = d.build_config(d.ROOT / 'assets', hardware, args.weights,
                                 d.resource_plan(hardware, args.weights), arm='official')
        elif arm != 'incumbent':
            raise ValueError('Unknown screening arm: ' + arm)
        cfg['args'] = d.set_arg(cfg['args'], '--prompt-cache', '4' if args.native_context else '0')
        if args.fixed_window:
            for name,value in (('--spec',str(args.fixed_window)),('--spec-min-p','0'),
                               ('--adapt-swaps','0'),('--suffix-draft','0')):
                cfg['args'] = d.set_arg(cfg['args'],name,value)
            cfg['env']['STRATA_MTP_ADAPTIVE'] = '0'

        def run(url, config, folder):
            rows = []
            contexts = [(0, '')]
            if args.native_context:
                pack = Path(config['args'][config['args'].index('--pack') + 1])
                directory = pack / 'tokenizer'
                vocab = json.loads((directory / 'vocab.json').read_text())
                vocabulary = [None] * len(vocab)
                for token, number in vocab.items():
                    vocabulary[number] = token
                tokenizer = Tokenizer(vocabulary, (directory / 'merges.txt').read_text().split('\n'),
                                      json.loads((directory / 'token_type.json').read_text()))
                corpus = json.loads((d.ROOT / 'evidence/requests/long-256k.json').read_text())['request']['messages'][-1]['content']
                background = tokenizer.decode(tokenizer.encode(corpus)[:253952])
                contexts.append((253952, 'The following records are background context. Complete the task at the end.\n' + background + '\nTASK:\n'))
            for context_size, prefix in contexts:
                for family, task in TASKS.items():
                    for sample in range(args.repeats):
                        req = q.body(config['model_name'], prefix + task, args.tokens)
                        req.pop('ignore_eos', None)  # Not implemented by either compared core.
                        req.update(seed=20261010, cache_prompt=bool(context_size))
                        result = q.request(url, req)
                        # Prefix caching is process-wide in these engine versions.
                        # The context campaign records every reuse; only each
                        # context's first English request must be cold.
                        cold = (family == 'english' and sample == 0) if args.native_context else True
                        key = (context_size, family, sample)
                        signature = hashlib.sha256(result['text'].encode()).hexdigest()
                        if index == 0:
                            reference[key] = signature
                        row = dict(context_background_tokens=context_size, family=family, sample=sample, cold_prefill_sample=cold, request=req, result=result,
                                   text_sha256=signature, equals_first_arm=signature == reference[key])
                        d.save(folder / f'{context_size}-{family}-{sample}.json', row)
                        # Retain failures too; never silently drop shortened outputs.
                        assert result['usage']['completion_tokens'] == args.tokens
                        if cold:
                            assert result['timings']['cache_n'] == 0
                        assert result['timings']['prompt_n'] + args.tokens <= 262144
                        rows.append(row)
                        print(arm, context_size, family, sample, result['timings']['predicted_per_second'],
                              'same', row['equals_first_arm'], flush=True)
            checks = {}
            if args.media:
                answer = q.request(url, q.body(config['model_name'], 'What is 17 plus 25? Answer with the number only.', 32))
                checks['arithmetic'] = {'passed': '42' in answer['text'], 'result': answer}
                recall, names = q.recall_body(config['model_name'])
                recalled = q.request(url, recall)
                checks['recall'] = {'passed': all(name in recalled['text'] for name in names), 'result': recalled}
                if args.native_context:
                    sample = json.loads((d.ROOT / 'evidence/requests/long-256k.json').read_text())
                    native_request = sample['request']
                    native_request.update(model=config['model_name'], seed=20261010)
                    native_result = q.request(url, native_request)
                    names = re.findall(r'QF-[1-5]-[A-F0-9]{12}', sample['expected_text'])
                    checks['native_recall'] = {'passed': len(names) == 5 and all(name in native_result['text'] for name in names)
                        and native_result['usage']['prompt_tokens'] == 256013
                        and native_result['usage']['completion_tokens'] == 512
                        and native_result['timings']['cache_n'] == 0, 'result': native_result}
                for name, fun in (('image',q.image_smoke),('video',q.video_smoke),('tools',q.tool_smoke)):
                    try:
                        checks[name] = fun(url,config['model_name'])
                    except (ValueError, RuntimeError) as exc:
                        checks[name] = {'passed': False, 'error': str(exc)}
                d.save(folder / 'capability-checks.json', checks)
                print(arm, 'capabilities', {name: value.get('passed', True) for name, value in checks.items()}, flush=True)
            return dict(rows=rows, checks=checks, screening_only=True)

        report = measure(cfg, output / f'{index}-{arm}', run)
        medians = {family: statistics.median(row['result']['timings']['predicted_per_second']
                   for row in report['rows'] if row['family'] == family and row['context_background_tokens'] == 0) for family in TASKS}
        long_medians = {family: statistics.median(row['result']['timings']['predicted_per_second']
                        for row in report['rows'] if row['family'] == family and row['context_background_tokens']) for family in TASKS} if args.native_context else {}
        summary.append(dict(index=index, arm=arm, medians=medians,
                            mean=statistics.mean(medians.values()),
                            native_context_medians=long_medians,
                            native_context_mean=statistics.mean(long_medians.values()) if long_medians else None,
                            checks=report['checks'],
                            equal=all(row['equals_first_arm'] for row in report['rows'])))
        d.save(output / 'summary.json', summary)
    lock.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
