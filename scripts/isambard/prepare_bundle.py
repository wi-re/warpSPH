#!/usr/bin/env python3
"""Package an ARM64 source context plus the staged harness for manual SSH upload."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

HERE = Path(__file__).resolve().parent

DEFINITION = '''Bootstrap: docker
From: BASE_IMAGE

%files
    containers /opt/warpsph/containers
    sources /opt/warpsph/sources
    scripts /opt/warpsph/scripts
    stack-manifest.json /opt/warpsph/stack-manifest.json
    constraints.txt /opt/warpsph/constraints.txt

%post
    set -eu
    export PIP_NO_CACHE_DIR=1
    cd /opt/warpsph
    chmod -R a+rX /opt/warpsph
    python - <<'PY'
import importlib.metadata as m,json,subprocess,sys
from pathlib import Path
manifest=json.loads(Path('stack-manifest.json').read_text())
Path('base-constraints.txt').write_text('torch=='+m.version('torch')+'\\n')
subprocess.run([sys.executable,'-m','pip','install','-c','base-constraints.txt','-c','constraints.txt',
               'warp-lang=='+manifest['warpVersion'],'setuptools','wheel'],check=True)
subprocess.run([sys.executable,'containers/install_stack.py','simulation'],check=True)
if manifest['target']=='with-remote':
    subprocess.run([sys.executable,'containers/install_stack.py','remote'],check=True)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
Path('installed-arm64.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
PY
    apt-get update
    apt-get install -y --no-install-recommends libgl1 libegl1 libglib2.0-0 ffmpeg xvfb
    if python -c "import importlib.util; raise SystemExit(importlib.util.find_spec('warpsph_remote') is None)"; then
        apt-get install -y --no-install-recommends nginx-light
    fi
    rm -rf /var/lib/apt/lists/* /work/cache
    mkdir -p /work
    chmod 1777 /work

%environment
    export HOME=/work USER=warpsph LOGNAME=warpsph PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    export XDG_STATE_HOME=/work/state XDG_CACHE_HOME=/work/cache WARPSPH_REMOTE_DATA=/work/remote-library
    export WARP_CACHE_PATH=/work/cache/warp NUMBA_CACHE_DIR=/work/cache/numba
    export TORCHINDUCTOR_CACHE_DIR=/work/cache/torchinductor MPLCONFIGDIR=/work/cache/matplotlib

%runscript
    exec python /opt/warpsph/containers/entrypoint.py "$@"
'''


def prepare(context, output, archive=None):
    context, output = Path(context).resolve(), Path(output).resolve()
    manifest = json.loads((context/'stack-manifest.json').read_text())
    if manifest.get('platform') != 'linux/arm64':
        raise ValueError('Prepare --platform linux/arm64; do not ship the local x86-64 image to Isambard')
    base = manifest['baseImage']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/@:+-]*', base):
        raise ValueError('Base must be a plain OCI image reference')
    if output.exists() or (archive and Path(archive).exists()):
        raise ValueError('Bundle/archive output must be fresh')
    for path in context.rglob('*'):
        if path.is_symlink():
            raise ValueError('Build context symlinks are not supported')
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(context, output/'context')
    (output/'context/sources').mkdir(exist_ok=True)
    shutil.copytree(HERE, output/'harness', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (output/'context/isambard.def').write_text(DEFINITION.replace('BASE_IMAGE', base))
    hashes = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.rglob('*')) if path.is_file()}
    (output/'upload-manifest.json').write_text(json.dumps(hashes, indent=2, sort_keys=True)+'\n')
    if archive:
        archive = Path(archive).resolve()
        if output in archive.parents:
            raise ValueError('Put the archive outside the bundle directory')
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, 'w:gz') as file:
            file.add(output, arcname='warpsph-isambard')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print('Prepared', prepare(args.context, args.output, args.archive), flush=True)


if __name__ == '__main__':
    main()
