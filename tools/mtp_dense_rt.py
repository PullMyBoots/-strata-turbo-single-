"""Restore selected auxiliary projections from checksum-verified BF16 tensors.

Target GGUF weights are untouched. Other auxiliary tensors retain their base
runtime bytes (including already adjusted RMSNorm weights).
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(8<<20):h.update(data)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--group',choices=['fc','attention','all'],default='all')
    a=ap.parse_args()
    fc={'fc_embedding.weight','fc_hidden.weight'}
    attn={f'self_attn.{x}_proj.weight' for x in 'qkvo'}
    selected=fc if a.group=='fc' else attn if a.group=='attention' else fc|attn
    manifest=json.loads((a.src/'mtp-manifest.json').read_text())
    source={}
    for row in manifest:
        name=row['name'].removeprefix('mtp.').removeprefix('layers.0.')
        if name in selected:source[name]=row
    if set(source)!=selected:raise ValueError('missing source projection')
    base=(a.base/'dense.bin').read_bytes()
    lines=[];out=bytearray();report=[];seen=set()
    for line in (a.base/'dense.txt').read_text().splitlines():
        name,kind,rows,cols,offset,size=line.split()
        rows,cols,offset,size=map(int,(rows,cols,offset,size))
        if name in seen or offset<0 or size<0 or offset+size>len(base):raise ValueError('invalid base index')
        seen.add(name)
        data=base[offset:offset+size]
        if name in selected:
            item=source[name];path=a.src/item['file']
            if item['dtype']!='BF16' or item['shape']!=[rows,cols] or path.stat().st_size!=rows*cols*2:
                raise ValueError('source shape or precision mismatch')
            sha=digest(path)
            if sha!=item['sha256']:raise ValueError('source checksum mismatch')
            data=path.read_bytes();kind='bf16'
            report.append({'name':name,'source':str(path.resolve()),'sha256':sha,'shape':[rows,cols]})
        out.extend(bytes((-len(out))%64))
        lines.append(f'{name} {kind} {rows} {cols} {len(out)} {len(data)}')
        out.extend(data)
    if not selected<=seen:raise ValueError('projection absent from base')
    a.out.mkdir(exist_ok=False)
    for name in ['experts.bin','experts-native.txt','draft_vocab.bin']:
        p=a.base/name
        if p.exists():(a.out/name).symlink_to(p.resolve(strict=True))
    (a.out/'dense.bin').write_bytes(out)
    report={'target_weights_changed':False,'group':a.group,'projections':report,
            'dense_bytes':len(out),'dense_sha256':hashlib.sha256(out).hexdigest(),
            'base_dense_sha256':hashlib.sha256(base).hexdigest()}
    (a.out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    # The index is written last, after all verified tensor bytes and provenance.
    (a.out/'dense.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
