"""Attach existing model assets using individual links, without changing their source."""
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
    args=parser.parse_args()
    source=args.source.resolve(strict=True); destination=args.destination.absolute()
    for name in PATHS:
        asset=source/name
        if not asset.exists():raise SystemExit('Missing asset: '+str(asset))
        target=destination/name
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve()!=asset.resolve():
                raise SystemExit('Existing destination differs: '+str(target))
    for name in PATHS:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.is_symlink():target.symlink_to((source/name).resolve(),target_is_directory=(source/name).is_dir())
    print(json.dumps({'assets':str(destination),'links':list(PATHS),'source_changed':False}))

if __name__=='__main__':main()
