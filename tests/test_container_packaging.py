"""Source provenance and real HTTP checks for the optional image companion.

The network check needs nginx and warpSPHRemote (no CUDA/container engine).
Set WARPSPH_TEST_NGINX and WARPSPH_TEST_REMOTE_ROOT for source installations.
"""
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def repository(path, name='warpSPHCore'):
    path.mkdir()
    subprocess.run(['git', 'init', '-q', str(path)], check=True)
    (path / 'src').mkdir()
    (path / 'src' / 'module.py').write_text('value = 1\n')
    (path / 'pyproject.toml').write_text(f'[project]\nname="{name}"\nversion="0.1.0"\n')
    (path / '.gitignore').write_text('.env\n.tmp/\nsrc/ignored.py\n')
    subprocess.run(['git', '-C', str(path), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(path), '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    return path


def test_dirty_source_snapshot_excludes_ignored_and_non_package_files(tmp_path):
    builder = load('scripts/build_container.py')
    source = repository(tmp_path / 'source')
    (source / 'src' / 'module.py').write_text('value = 2\n')
    (source / 'src' / 'new.py').write_text('new = True\n')
    (source / 'src' / 'ignored.py').write_text('private\n')
    (source / '.env').write_text('PRIVATE=secret\n')
    (source / 'unrelated.txt').write_text('not shipped\n')
    target = tmp_path / 'snapshot'
    pin = builder.snapshot(source, target)
    assert pin['dirty']
    assert (target / 'src/module.py').read_text() == 'value = 2\n'
    assert (target / 'src/new.py').is_file()
    assert not (target / '.env').exists()
    assert not (target / 'src/ignored.py').exists()
    assert not (target / 'unrelated.txt').exists()
    assert pin['files']['src/module.py'] == hashlib.sha256(b'value = 2\n').hexdigest()
    (source / 'src/module.py').unlink()
    second = builder.snapshot(source, tmp_path / 'second')
    assert 'src/module.py' not in second['files']
    assert second['digest'] != pin['digest']


def test_source_symlinks_cannot_copy_files_outside_repository(tmp_path):
    builder = load('scripts/build_container.py')
    source = repository(tmp_path / 'source')
    outside = tmp_path / 'credential'
    outside.write_text('private')
    (source / 'src/escape.py').symlink_to(outside)
    with pytest.raises(ValueError, match='ordinary files'):
        builder.snapshot(source, tmp_path / 'snapshot')


def test_mixed_public_local_context_is_atomic_and_keeps_exact_sources(tmp_path):
    builder = load('scripts/build_container.py')
    source = repository(tmp_path / 'core')
    builder.ROOT = ROOT
    public = {name: name + '==0.1.0' for name in builder.PACKAGES if name != 'warpSPHCore'}
    context = tmp_path / 'context'
    manifest = builder.prepare(context, {'warpSPHCore': source}, public,
                               with_remote=True, base_image='base@sha256:fixture',
                               platform='linux/arm64', warp_version='1.17.0')
    assert manifest['packages']['warpSPHCore']['kind'] == 'source'
    assert manifest['packages']['warpSPHRemote']['requirement'] == 'warpSPHRemote==0.1.0'
    assert manifest['platform'] == 'linux/arm64'
    assert (context / 'scripts/run_distributed.py').is_file()
    with pytest.raises(ValueError, match='fresh'):
        builder.prepare(context, {}, {}, with_remote=False, base_image='base',
                        platform='linux/amd64', warp_version='1.17.0')
    wrong = tmp_path / 'wrong'
    with pytest.raises(ValueError, match='must contain that package'):
        builder.prepare(wrong, {'warpSPH': source},
                        {name: name + '==0.1.0' for name in builder.PACKAGES if name != 'warpSPH'},
                        with_remote=False, base_image='base', platform='linux/amd64', warp_version='1.17.0')
    assert not wrong.exists()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_install_replaces_same_version_from_custom_base(tmp_path):
    environment = dict(os.environ, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                       PIP_CACHE_DIR=str(tmp_path / 'pip-cache'))
    venv = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)],
                   check=True, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    python = str(venv / 'bin/python')
    context = tmp_path / 'context'
    (context / 'containers').mkdir(parents=True)
    shutil.copy2(ROOT / 'containers/install_stack.py', context / 'containers')
    for file in ('constraints.txt', 'base-constraints.txt'):
        (context / file).write_text('')
    source = context / 'sources/fixture'
    source.mkdir(parents=True)
    (source / 'pyproject.toml').write_text(
        '[build-system]\nrequires=["setuptools", "wheel"]\nbuild-backend="setuptools.build_meta"\n'
        '[project]\nname="warpSPHContainerFixture"\nversion="0.1.0"\n'
        '[tool.setuptools]\npy-modules=["warpsph_container_fixture"]\n')
    module = source / 'warpsph_container_fixture.py'
    module.write_text('value = 1\n')
    def run(*args):
        result = subprocess.run([python, *args], env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        return result.stdout.strip()
    # Model a custom base containing a public release with the same version.
    run('-m', 'pip', 'install', '--no-deps', '--no-build-isolation', str(source))
    assert run('-c', 'import warpsph_container_fixture as f; print(f.value)') == '1'
    module.write_text('value = 200\n')
    shutil.rmtree(source / 'build', ignore_errors=True)
    (context / 'stack-manifest.json').write_text(json.dumps({'packages': {
        'fixture': {'kind': 'source', 'snapshot': 'sources/fixture'}}}))
    run(str(context / 'containers/install_stack.py'), 'simulation')
    assert run('-c', 'import warpsph_container_fixture as f; print(f.value)') == '200'
    assert len(json.loads((context / 'wheel-hashes.json').read_text())) == 1


def test_built_image_supports_arbitrary_uid_and_fresh_volume():
    image = os.environ.get('WARPSPH_TEST_IMAGE')
    if not image:
        pytest.skip('Set WARPSPH_TEST_IMAGE to test a locally built image')
    result = subprocess.run(['docker', 'run', '--rm', '--user', '65001:65001', image, 'help'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == 0, result.stdout
    assert 'Commands:' in result.stdout
    result = subprocess.run(['docker', 'run', '--rm', '--user', '65001:65001', image,
                             'exec', 'python', '-c',
                             'import warpSPH; from torch._inductor.runtime.cache_dir_utils import cache_dir; print(cache_dir())'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout
    assert '/work/cache/torchinductor' in result.stdout
    result = subprocess.run(['docker', 'run', '--rm', '--user', '65001:65001',
                             '--volume', '/work', image, 'exec', 'python', '-c',
                             "from pathlib import Path; import os; p=Path('/work/cache/volume-check'); "
                             "p.mkdir(parents=True); assert p.stat().st_uid==os.getuid()"],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == 0, result.stdout


def request(port, path, token=None, *, method='GET', origin='https://fluids.dev', host=None, body=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
    headers = {'Origin': origin}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    if host:
        headers['Host'] = host
    if body is not None:
        body = json.dumps(body)
        headers['Content-Type'] = 'application/json'
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_real_companion_proxy_auth_registration_cors_and_shutdown(tmp_path):
    nginx = os.environ.get('WARPSPH_TEST_NGINX') or shutil.which('nginx')
    remote = os.environ.get('WARPSPH_TEST_REMOTE_ROOT')
    if not nginx or (not remote and importlib.util.find_spec('warpsph_remote') is None):
        pytest.skip('Needs nginx and warpSPHRemote; see module docstring')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    (binaries / 'nginx').symlink_to(nginx)
    companion = binaries / 'warpsph-companion'
    companion.write_text(f'#!{sys.executable}\nfrom warpsph_remote.daemon import main\nmain()\n')
    companion.chmod(0o700)
    runtime = tmp_path / 'runtime'
    runtime.mkdir(mode=0o700)
    environment = dict(os.environ, PATH=str(binaries) + ':' + os.environ['PATH'],
                       XDG_RUNTIME_DIR=str(runtime), XDG_STATE_HOME=str(tmp_path / 'state'),
                       WARPSPH_REMOTE_DATA=str(tmp_path / 'library'))
    if remote:
        environment['PYTHONPATH'] = str(Path(remote) / 'python')
    viewer_port, daemon_port = free_port(), free_port()
    while viewer_port == daemon_port:
        daemon_port = free_port()
    process = subprocess.Popen([sys.executable, str(ROOT / 'containers/entrypoint.py'),
                                'companion', '--viewer-port', str(viewer_port),
                                '--daemon-port', str(daemon_port)], env=environment,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    descriptor = runtime / 'warpsph-remote/daemon.json'
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                pytest.fail('Service exited: ' + process.stderr.read().decode())
            try:
                info = json.loads(descriptor.read_text())
                if request(viewer_port, '/v1/runs', info['token'])[0] == 200:
                    break
            except (OSError, ValueError, http.client.HTTPException):
                pass
            if time.monotonic() > deadline:
                pytest.fail('Companion/proxy startup timeout')
            time.sleep(.05)
        assert request(viewer_port, '/v1/runs')[0] == 401
        assert request(viewer_port, '/v1/runs', 'wrong')[0] == 401
        assert request(viewer_port, '/v1/runs', info['token'], origin='https://untrusted.invalid')[0] == 403
        assert request(viewer_port, '/v1/runs', info['token'], host='untrusted.invalid')[0] == 403
        assert request(viewer_port, '/v1/runs', info['token'], host='localhost:19000')[0] == 200
        status, headers, _ = request(viewer_port, '/v1/runs', method='OPTIONS')
        assert status == 204 and headers['Access-Control-Allow-Origin'] == 'https://fluids.dev'
        assert request(viewer_port, '/producer/runs', info['producerToken'], method='POST', body={'name': 'fixture'})[0] == 403
        status, _, body = request(daemon_port, '/producer/runs', info['producerToken'],
                                  method='POST', body={'name': 'fixture'})
        assert status == 201
        identifier = json.loads(body)['id']
        _, _, body = request(viewer_port, '/v1/runs', info['token'])
        assert identifier.encode() in body
        process.terminate()
        assert process.wait(timeout=12) == 0
        for port in (viewer_port, daemon_port):
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', port)) != 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=12)
        process.stderr.close()
