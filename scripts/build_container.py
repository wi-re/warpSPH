#!/usr/bin/env python3
"""Prepare a private, source-pinned OCI build context; optionally build it with Docker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('warpSPHCore', 'warpSPHIntegrators', 'warpSPHPlotting', 'warpSPH', 'warpSPHRemote')
METADATA = {'pyproject.toml', 'setup.py', 'setup.cfg', 'README.md', 'README.rst',
            'LICENSE', 'LICENSE.md', 'license.md', 'MANIFEST.in'}


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments])


def snapshot(root, destination):
    """Copy package inputs, including dirty/new files but no ignored caches/secrets."""
    root = root.resolve()
    if Path(git(root, 'rev-parse', '--show-toplevel').decode().strip()).resolve() != root:
        raise ValueError(f'Expected repository root: {root}')
    paths = git(root, 'ls-files', '-z', '--cached', '--others', '--exclude-standard').split(b'\0')
    files = {}
    destination.mkdir(parents=True)
    for raw in sorted(set(paths)):
        if not raw:
            continue
        relative = Path(raw.decode())
        if relative.parts[0] not in ('src', 'python') and str(relative) not in METADATA:
            continue
        source = root / relative
        # Deleted tracked files are intentionally absent in a dirty snapshot.
        if not source.exists() and not source.is_symlink():
            continue
        if not source.is_file() or source.is_symlink() or any(
                parent.is_symlink() for parent in source.parents if parent != root and root in parent.parents):
            raise ValueError(f'Package snapshots require ordinary files: {relative}')
        content = source.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(source.stat().st_mode & 0o777)
        files[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    if 'pyproject.toml' not in files:
        raise ValueError(f'Missing package metadata in {root}')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(kind='source', commit=git(root, 'rev-parse', 'HEAD').decode().strip(),
                dirty=bool(git(root, 'status', '--porcelain', '--untracked-files=normal').strip()),
                digest=digest, files=files)


def prepare(context, sources, requirements, *, with_remote, base_image, platform,
            warp_version, constraints=None):
    context = context.resolve()
    if context.exists():
        raise ValueError('Use a fresh build-context directory; existing contents are never overwritten.')
    context.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.warpsph-context-', dir=context.parent))
    try:
        packages = {}
        for name in PACKAGES if with_remote else PACKAGES[:-1]:
            if name in requirements:
                packages[name] = dict(kind='distribution', requirement=requirements[name])
            else:
                path = sources.get(name, ROOT if name == 'warpSPH' else ROOT.parent / name)
                if not path.is_dir():
                    raise ValueError(f'Supply --source {name}=/path/to/repository or --package {name}==VERSION')
                packages[name] = snapshot(path, staging / 'sources' / name)
                metadata = tomllib.loads((staging / 'sources' / name / 'pyproject.toml').read_text())
                if metadata.get('project', {}).get('name') != name:
                    raise ValueError(f'--source {name} must contain that package in pyproject.toml')
                packages[name]['snapshot'] = 'sources/' + name
        shutil.copytree(ROOT / 'containers', staging / 'containers',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        scripts = staging / 'scripts'
        scripts.mkdir()
        shutil.copy2(ROOT / 'scripts' / 'run_distributed.py', scripts)
        (staging / 'constraints.txt').write_bytes(constraints.read_bytes() if constraints else b'')
        manifest = dict(formatVersion=1, createdUtc=datetime.now(timezone.utc).isoformat(),
                        baseImage=base_image, platform=platform, warpVersion=warp_version,
                        target='with-remote' if with_remote else 'simulation', packages=packages,
                        qualification='Build recipe only; validate each CPU/GPU/site target before production.')
        manifest['buildFiles'] = {
            str(path.relative_to(staging)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staging.rglob('*')) if path.is_file() and 'sources' not in path.relative_to(staging).parts
        }
        (staging / 'stack-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        (staging / '.dockerignore').write_text('**/__pycache__\n**/*.pyc\n**/.git\n')
        staging.rename(context)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True, help='Fresh private output directory')
    parser.add_argument('--base-image', required=True, help='CUDA/PyTorch OCI base; pin a digest for releases')
    parser.add_argument('--platform', choices=('linux/amd64', 'linux/arm64'), required=True)
    parser.add_argument('--source', action='append', default=[], metavar='NAME=PATH',
                        help='Exact local repository, including uncommitted package files')
    parser.add_argument('--package', action='append', default=[], metavar='NAME==VERSION',
                        help='Use an exact public distribution instead of local source')
    parser.add_argument('--with-remote', action='store_true')
    parser.add_argument('--warp-version', default='1.17.0')
    parser.add_argument('--constraints', type=Path, help='Additional dependency constraints for the target platform')
    parser.add_argument('--prepare-only', action='store_true', help='No container engine or network access needed')
    parser.add_argument('--image', default='warpsph:local')
    args = parser.parse_args()
    sources, requirements = {}, {}
    for value in args.source:
        name, separator, path = value.partition('=')
        if name not in PACKAGES or not separator or not path or name in sources:
            parser.error('--source expects a unique known package NAME=PATH')
        sources[name] = Path(path).expanduser()
    for value in args.package:
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9]*)==([0-9][A-Za-z0-9.!+_-]*)', value)
        if not match or match[1] not in PACKAGES or match[1] in requirements:
            parser.error('--package expects a unique known package NAME==VERSION')
        requirements[match[1]] = value
    if set(sources) & set(requirements):
        parser.error('Choose source or distribution once per package')
    if not args.with_remote and 'warpSPHRemote' in sources.keys() | requirements.keys():
        parser.error('warpSPHRemote requires --with-remote')
    if not args.prepare_only and not shutil.which('docker'):
        parser.error('Docker is unavailable; use --prepare-only to export the context')
    try:
        manifest = prepare(args.context, sources, requirements, with_remote=args.with_remote,
                           base_image=args.base_image, platform=args.platform,
                           warp_version=args.warp_version, constraints=args.constraints)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f'Prepared {args.context.resolve()} ({manifest["target"]}, {args.platform})', flush=True)
    if not args.prepare_only:
        subprocess.run(['docker', 'build', '--platform', args.platform,
                        '--build-arg', 'BASE_IMAGE=' + args.base_image,
                        '--build-arg', 'WARP_VERSION=' + args.warp_version,
                        '--target', manifest['target'], '-f', str(args.context / 'containers' / 'Dockerfile'),
                        '-t', args.image, str(args.context)], check=True)


if __name__ == '__main__':
    main()
