"""Create local source/runtime release archives, excluding model data and local environments."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','assets','runtime','artifacts','__pycache__'}

def files(with_binaries):
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if any(x in SKIP for x in rel.parts):continue
        if any(x.startswith('build') for x in rel.parts[:-1]):continue
        if path.is_symlink() or not path.is_file() or path.suffix in ('.pyc','.gguf','.log'):continue
        if rel.parts[0]=='engine' and path.name!='manifest.json' and not with_binaries:continue
        yield path,rel

def main():
    version=(ROOT/'VERSION').read_text().strip();destination=ROOT/'artifacts';destination.mkdir(exist_ok=True)
    names=[]
    for kind,binaries in [('source',False),('linux-x86_64-sm89',True)]:
        archive=destination/f'strata-single-GPU-{version}-{kind}.tar.gz'
        with archive.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
                with tarfile.open(fileobj=gz,mode='w') as tar:
                    for file,rel in files(binaries):
                        info=tar.gettarinfo(str(file),arcname=f'strata-single-GPU-{version}/{rel.as_posix()}')
                        info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
                        with file.open('rb') as content:tar.addfile(info,content)
        names.append(archive)
    sums=[]
    for archive in names:
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(8<<20),b''):h.update(block)
        sums.append(h.hexdigest()+'  '+archive.name)
    (destination/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    print(json.dumps({'archives':[{'path':str(p),'bytes':p.stat().st_size} for p in names]}))

if __name__=='__main__':main()
