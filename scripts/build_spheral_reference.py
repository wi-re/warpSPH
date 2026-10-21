"""Isolated, resumable pinned Spheral reference build (CPU, Python, 1D/2D/3D).

Uses Spheral's own pinned Spack/TPL manager; never modifies the warp Python
or the original source checkout. Logs and build state remain under results/.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

PIN = '116c71f2ea559d569456ba694194e5a6de3c75f7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/materialEOS/spheral-build'))
    parser.add_argument('--source', type=Path, default=Path('/home/lu26029/dev/spheral'))
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = root/'source'
    manifest_path = root/'build.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(pin=PIN, stages={})
    if manifest['pin'] != PIN:
        raise ValueError('Build root belongs to another reference revision')
    env = {**os.environ, 'SPACK_BUILD_JOBS': str(args.jobs), 'CMAKE_BUILD_PARALLEL_LEVEL': str(args.jobs),
           'SPACK_USER_CACHE_PATH': str(root/'tpls/misc'), 'SPACK_DISABLE_LOCAL_CONFIG': 'true',
           'PIP_CACHE_DIR': str(root/'pip-cache'), 'PYTHONUNBUFFERED': '1'}

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')

    def stage(name, command, cwd=root):
        prior = manifest['stages'].get(name, {})
        if prior.get('status') == 'completed':
            return
        log = root/(name+'.log')
        record = dict(status='running', startedAt=time.time(), command=list(map(str, command)), log=str(log))
        manifest['stages'][name] = record
        manifest['status'] = 'running'
        manifest['phase'] = name
        save()
        print(f'START {name}: {log}', flush=True)
        with log.open('a') as stream:
            child = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            last_report = time.monotonic()
            while child.poll() is None:
                time.sleep(1)
                if time.monotonic() - last_report >= 10:
                    tail = log.read_text(errors='replace').splitlines()[-2:]
                    print(f'{name}: '+ ' | '.join(line[-240:] for line in tail), flush=True)
                    last_report = time.monotonic()
        record.update(status='completed' if child.returncode == 0 else 'failed', returncode=child.returncode, finishedAt=time.time())
        manifest['status'] = record['status']
        save()
        if child.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
        print(f'END {name}', flush=True)

    if not source.exists():
        stage('clone', ['git', 'clone', '--shared', str(args.source.resolve()), str(source)])
    stage('pin', ['git', 'checkout', '--detach', PIN], source)
    stage('submodules', ['git', 'submodule', 'update', '--init', '--recursive', '--jobs', '4'], source)
    # The pinned external-build helper passes a dict where Spack requires an
    # upstream path string when --no-upstream is selected. Patch only this
    # isolated build helper/config, never the reference hydro implementation.
    helper = source/'scripts/devtools/tpl-manager.py'
    content = helper.read_text().replace('spack.config.set("upstreams", {"spheral_shared": {"install_tree": {}}})',
                                         'spack.config.set("upstreams", {})')
    helper.write_text(content)
    (source/'scripts/spack/configs/upstreams.yaml').write_text('upstreams: {}\n')
    config = source/'scripts/spack/configs/config.yaml'
    content = config.read_text()
    if '  build_jobs:' not in content:
        config.write_text(content.replace('config:\n', f'config:\n  build_jobs: {args.jobs}\n', 1))
    manifest['buildOnlyAdjustments'] = ['Fix invalid --no-upstream Spack schema value', 'Disable LC-only upstream',
                                      f'Limit dependency/C++ build parallelism to {args.jobs}',
                                      'Use direct CMake configuration for absolute build path',
                                      'Seed isolated venv pip without system ensurepip; use isolated pip cache']
    save()
    stage('dependencies', ['/usr/bin/python3', '-u', 'scripts/devtools/tpl-manager.py', '--spec',
        'spheral@develop~mpi+openmp+python~caliper~opensubdiv%gcc', '--tpl-dir', str(root/'tpls'), '--no-upstream'], source)
    configs = list(source.glob('*.cmake'))
    if len(configs) != 1:
        raise RuntimeError(f'Expected one generated host config, found {configs}')
    build = root/'compiled/build'
    python_cmake = source/'cmake/spheral/SpheralPRT.cmake'
    content = python_cmake.read_text().replace('-m venv ${', '-m venv --without-pip ${')
    content = content.replace('set(SPHERAL_PIP_CACHE_DIR ~/.cache/spheral_pip)',
                              f'set(SPHERAL_PIP_CACHE_DIR {root / "pip-cache"})')
    python_cmake.write_text(content)
    # Ubuntu's system Python has no ensurepip here. Seed only these isolated
    # venvs from the existing universal pip wheel, without changing system or
    # simulation packages. CMake subsequently installs its pinned requirements.
    wheels = list(Path('/home/lu26029/miniconda3/envs/warp/lib/python3.13/ensurepip/_bundled').glob('pip-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('Expected one local bootstrap pip wheel')
    for name, prefix in (('build', build), ('runtime', root/'install')):
        stage('venv-'+name, ['/usr/bin/python3', '-m', 'venv', '--without-pip', str(prefix/'.venv')])
        code = 'import sys; sys.path.insert(0, sys.argv[1]); from pip._internal.cli.main import main; sys.exit(main(["install", "--no-index", sys.argv[1]]))'
        stage('pip-'+name, [prefix/'.venv/bin/python3', '-c', code, wheels[0]])
    # Call CMake directly: the pinned host-config helper prepends its source
    # directory even to absolute --build-dir arguments.
    if manifest.get('dimensions') != [1, 2, 3]:
        for target in ('Spheral_CXX', 'install', 'python-import'):
            manifest['stages'].pop(target, None)
        manifest['dimensions'] = [1, 2, 3]
        manifest['dimensionReason'] = 'The upstream 3D-disabled library leaves unresolved 3D references during install; use full dimensions.'
        save()
    stage('configure-all-dimensions', ['cmake', '-S', str(source), '-B', str(build), '-C', str(configs[0]),
        '-DCMAKE_INSTALL_PREFIX='+str(root/'install'), '-DSPHERAL_ENABLE_3D=ON',
        '-DSPHERAL_ENABLE_TESTS=OFF', '-DSPHERAL_ENABLE_DOCS=OFF'], source)
    for target in ('python_build_env', 'python_runtime_env', 'Spheral_CXX', 'install'):
        jobs = args.jobs if target == 'Spheral_CXX' else 2
        stage(target, ['cmake', '--build', str(build), '--target', target, '-j', str(jobs)])
    stage('python-import', [root/'install/bin/spheral', '-c',
        'import Spheral1d as s; import Spheral2d; e=s.GammaLawGasMKS(1.4,1.0); assert hasattr(s,"CRKSPH"); print("Spheral 1D/2D Python interfaces and CRKSPH available")'])
    manifest['status'] = 'completed'
    save()
    print(f'Spheral installed under {root / "install"}', flush=True)


if __name__ == '__main__':
    main()
