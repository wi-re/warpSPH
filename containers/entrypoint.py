"""Portable image commands; supervise the optional daemon and viewer-only proxy."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def proxy_configuration(port, daemon_port=8766):
    # Allow the browser's loopback Host, but normalize its port for the daemon's
    # strict Host gate. Origin and Authorization are forwarded unchanged.
    return r'''
daemon off;
worker_processes 1;
pid nginx.pid;
error_log stderr warn;
events { worker_connections 128; }
http {
    access_log off;
    client_body_temp_path body;
    proxy_temp_path proxy;
    fastcgi_temp_path fastcgi;
    uwsgi_temp_path uwsgi;
    scgi_temp_path scgi;
    map $http_host $local_viewer {
        default 0;
        ~^127\.0\.0\.1(:[0-9]+)?$ 1;
        ~*^localhost(:[0-9]+)?$ 1;
        ~^\[::1\](:[0-9]+)?$ 1;
    }
    server {
        listen 0.0.0.0:PORT;
        if ($local_viewer = 0) { return 403; }
        client_max_body_size 32m;
        location /v1/ {
            proxy_pass http://127.0.0.1:PRIVATE_PORT;
            proxy_set_header Host 127.0.0.1:PRIVATE_PORT;
            proxy_set_header Connection "";
            proxy_set_header Forwarded "";
            proxy_set_header X-Forwarded-Host "";
            proxy_set_header X-Forwarded-For "";
            proxy_http_version 1.1;
            proxy_buffering off;
            proxy_request_buffering off;
            proxy_read_timeout 300s;
        }
        location / { return 403; }
    }
}
'''.replace('PRIVATE_PORT', str(daemon_port)).replace('PORT', str(port))


def companion(arguments):
    import argparse
    parser = argparse.ArgumentParser(description='Container companion with local Docker viewer access')
    parser.add_argument('--viewer-port', type=int, default=8767)
    parser.add_argument('--daemon-port', type=int, default=8766)
    parser.add_argument('--origin', action='append', default=[])
    parser.add_argument('--max-snapshot-mib', type=int, default=128)
    args = parser.parse_args(arguments)
    if not all(1024 <= port <= 65535 for port in (args.viewer_port, args.daemon_port)) or args.viewer_port == args.daemon_port:
        parser.error('Viewer/private ports must be distinct and within 1024–65535.')
    nginx = shutil.which('nginx')
    if not nginx or not shutil.which('warpsph-companion'):
        parser.error('Build the with-remote image target to use this command.')
    prefix = Path(tempfile.mkdtemp(prefix='warpsph-viewer-'))
    config = prefix / 'nginx.conf'
    config.write_text(proxy_configuration(args.viewer_port, args.daemon_port))
    children = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        subprocess.run([nginx, '-t', '-p', str(prefix) + '/', '-c', str(config)], check=True)
        command = ['warpsph-companion', '--python', sys.executable,
                   '--port', str(args.daemon_port),
                   '--max-snapshot-mib', str(args.max_snapshot_mib)]
        for origin in args.origin:
            command.extend(['--origin', origin])
        children.append(subprocess.Popen(command, start_new_session=True))
        children.append(subprocess.Popen([nginx, '-p', str(prefix) + '/', '-c', str(config)],
                                         start_new_session=True))
        print(f'Viewer: publish 127.0.0.1:8766:{args.viewer_port}; connect to '
              'http://127.0.0.1:8766 using the pairing token below. '
              'The daemon endpoint is private to this container.', flush=True)
        while not stopping and all(child.poll() is None for child in children):
            time.sleep(.1)
        if stopping:
            return 0
        return next((child.returncode or 1 for child in children if child.poll() is not None), 1)
    finally:
        # Terminate service groups. Docker's --init reaps orphans and stopping
        # the container terminates separately sessioned simulation workers too.
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 8
        for child in children:
            try:
                child.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        shutil.rmtree(prefix)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    mode = arguments.pop(0) if arguments else 'help'
    if mode == 'companion':
        return companion(arguments)
    if mode == 'manifest':
        print((ROOT / 'stack-manifest.json').read_text())
        return 0
    if mode == 'run':
        os.execvp('warpsph-run', ['warpsph-run', *arguments])
    if mode == 'distributed':
        os.execv(sys.executable, [sys.executable, '-m', 'torch.distributed.run', *arguments])
    if mode == 'exec':
        if not arguments:
            raise SystemExit('exec requires a command')
        os.execvp(arguments[0], arguments)
    if mode != 'help':
        raise SystemExit('Unknown image command: ' + mode)
    print('Commands: run [case/options], distributed [torchrun options/script], '
          'companion [--viewer-port PORT], manifest, exec COMMAND [args].\n'
          'Distributed script: /opt/warpsph/scripts/run_distributed.py\n'
          'Mount writable /work for outputs, persistent pairing/run history and caches.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
