"""Create clean source/runtime distributions with per-file and archive SHA256."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git', '.venv', '__pycache__', 'runtime', 'assets', 'models-data', 'artifacts', '.pytest_cache', 'node_modules'}
ALLOWED = {'src', 'include', 'cmake', 'tools', 'serve', 'third_party', 'configs', 'docs', 'evidence', 'distribution', 'experimental', 'engine'}
ROOT_FILES = {'README.md', 'LICENSE', 'VERSION', 'CMakeLists.txt', 'requirements.txt', 'setup.py', 'setup.sh', 'start.sh', 'START-HERE.bat', 'install-python.sh', 'build.sh', 'RELEASE.json', 'SOURCE_PROVENANCE.json'}


def files(with_binaries):
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in ALLOWED and str(rel) not in ROOT_FILES:
            continue
        if any(part in EXCLUDE for part in rel.parts) or any(part.startswith('build-') for part in rel.parts[:-1]):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix in ('.pyc', '.o', '.a', '.so', '.log', '.gguf', '.pid'):
            continue
        if rel.parts[0] == 'experimental':
            if len(rel.parts) < 3 or rel.parts[1] != 'native-batch':
                continue
            if rel.parts[2] not in ('serve', 'tools', 'source'):
                continue
            if rel.name in ('test_native_media', 'generate.cpp.o'):
                continue
        if rel.parts[0] == 'engine':
            if not with_binaries or len(rel.parts) < 3 or rel.parts[1] != 'distribution':
                continue
        yield path, rel


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    version = (ROOT / 'VERSION').read_text().strip()
    destination = ROOT / 'artifacts'
    destination.mkdir(exist_ok=True)
    build = json.loads((ROOT / 'distribution/build.json').read_text())
    arch = '-'.join('sm' + str(a) for a in build['cuda_architectures'])
    archives = []
    for kind, binaries in [('source', False), ('linux-x86_64-' + arch, True)]:
        archive = destination / f'strata-turbo-single-{version}-{kind}.tar.gz'
        entries = list(files(binaries))
        manifest = {'project': 'strata-turbo-single', 'version': version, 'kind': kind,
                    'model_weights_bundled': False, 'local_qualification_bundled': False,
                    'files': {str(rel): {'bytes': file.stat().st_size, 'sha256': digest(file)} for file, rel in entries}}
        prefix = f'strata-turbo-single-{version}'
        with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=6) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
            for file, rel in entries:
                info = tar.gettarinfo(str(file), arcname=prefix + '/' + rel.as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                with file.open('rb') as content:
                    tar.addfile(info, content)
            raw_manifest = (json.dumps(manifest, indent=2) + '\n').encode()
            info = tarfile.TarInfo(prefix + '/PACKAGE.json')
            info.size = len(raw_manifest)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(raw_manifest))
        archives.append({'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive)})
    (destination / ('SHA256SUMS-' + version)).write_text('\n'.join(a['sha256'] + '  ' + Path(a['path']).name for a in archives) + '\n')
    (destination / ('distribution-' + version + '.json')).write_text(json.dumps(archives, indent=2) + '\n')
    print(json.dumps(archives, indent=2))


if __name__ == '__main__':
    main()
