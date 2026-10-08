"""Record exact binaries without treating a new source build as a historical benchmark."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built',action='store_true')
    args=parser.parse_args()
    path=ROOT/'engine/manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else {'schema':1,'binaries':{}}
    for name,role in [('strata-single-validated','historically benchmarked single-request engine'),
                      ('strata-concurrent-validated','historically validated concurrency engine'),
                      ('strata-vision-native','native image/video encoder'),
                      ('strata-single-built','fresh build of bundled source; benchmark separately')]:
        file=ROOT/'engine'/name
        if file.exists():
            manifest['binaries'][name]={'sha256':digest(file),'bytes':file.stat().st_size,'role':role}
    if args.built:assert 'strata-single-built' in manifest['binaries']
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest':str(path),'binaries':list(manifest['binaries'])}))

if __name__=='__main__':main()
