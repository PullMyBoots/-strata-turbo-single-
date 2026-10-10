"""Attach model assets, including the optional full-vocabulary draft router."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATHS=('models/IQ2_XS','data/packs/iq2_xs','data/mtp/native-q5',
       'data/mtp/head-router-iq2-r128-k8192','data/mtp/rt','data/expert-profile.bin')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='asset root containing models/ and data/')
    parser.add_argument('--destination',type=Path,default=ROOT/'assets')
    parser.add_argument('--weights',choices=('iq2_xs','swift','iq3_s'),default='iq2_xs')
    parser.add_argument('--full-router',type=Path,help='Full-vocabulary SRH1 router directory for the default source-built engine.')
    args=parser.parse_args()
    source=args.source.resolve(strict=True); destination=args.destination.absolute()
    paths=PATHS
    if args.weights!='iq2_xs':
        model_paths=('models/Swift-1.5-IQ3_XXS','data/packs/swift-1.5-iq3_xxs') if args.weights=='swift' else \
                    ('models/IQ3_S','models/Vision','data/packs/iq3_s')
        paths=model_paths+('data/mtp/native-q5','data/mtp/rt','data/expert-profile.bin')
    planned=[(destination/name,(source/name).resolve(strict=True)) for name in paths]
    router_name='head-router-full-r128-k8192' if args.weights=='iq2_xs' else f'head-router-{args.weights}-full-r128-k8192'
    full_router=args.full_router or source/'data/mtp'/router_name
    has_full=full_router.is_dir()
    if args.full_router and not has_full:raise SystemExit('Missing full router: '+str(full_router))
    if args.weights!='iq2_xs' and not has_full:raise SystemExit('This weight requires its model-matched --full-router directory.')
    if has_full:
        full_router=full_router.resolve(strict=True)
        for name in ('router.txt','projection-f32.bin','coarse-q8.bin','bias-f32.bin'):
            if not (full_router/name).is_file():raise SystemExit('Missing full router file: '+name)
        header=(full_router/'router.txt').read_text().split()
        if len(header)!=9 or header[:5]!=['SRH1','128','8192','248320','2560']:
            raise SystemExit('Expected the validated 248320-token rank-128, top-8192 router.')
        planned.append((destination/'data/mtp'/router_name,full_router))
        for name in ('dense.txt','dense.bin','experts-native.txt','experts.bin'):
            planned.append((destination/'data/mtp/native-q5-full'/name,(source/'data/mtp/native-q5'/name).resolve(strict=True)))
    for target,asset in planned:
        if target.exists() or target.is_symlink():
            if target.resolve()!=asset:raise SystemExit('Existing destination differs: '+str(target))
    for target,asset in planned:
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() and not target.is_symlink():target.symlink_to(asset,target_is_directory=asset.is_dir())
    print(json.dumps({'assets':str(destination),'weights':args.weights,'links':[str(t.relative_to(destination)) for t,_ in planned],
        'source_changed':False,'built_engine_assets_ready':has_full,
        'next_step':'./start.sh' if has_full else 'Use --full-router to attach the new router, or start with --engine validated.'}))

if __name__=='__main__':main()
