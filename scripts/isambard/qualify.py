#!/usr/bin/env python3
"""Run one explicitly selected acceptance stage. Host Python needs only stdlib.

    single -> same-node -> restart -> multi-node

Stage jobs fail closed. --launcher local --backend gloo is CPU harness testing,
not evidence of Isambard, physical GPU, or fabric qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = {'same-node': 'single', 'restart': 'same-node', 'multi-node': 'restart'}


def atomic_json(path, data):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_layout(stage, nodes, gpus, launcher, backend, network):
    if nodes < 1 or not 1 <= gpus <= 4:
        raise ValueError('Use positive nodes and 1–4 GPUs per node')
    if stage != 'multi-node' and nodes != 1:
        raise ValueError('This stage requires exactly one node')
    if stage == 'single' and gpus != 1:
        raise ValueError('The first stage requires exactly one GPU')
    if stage in ('same-node', 'restart') and gpus < 2:
        raise ValueError('This stage requires at least two allocated GPUs')
    if stage == 'multi-node' and (nodes < 2 or network == 'none'):
        raise ValueError('Multi-node requires multiple physical nodes and an explicit network profile')
    if launcher == 'local' and (nodes != 1 or backend != 'gloo'):
        raise ValueError('Local mode only qualifies the CPU/Gloo harness on one host')
    if launcher == 'slurm' and backend != 'nccl':
        raise ValueError('Isambard GPU qualification requires NCCL')


def command(args, nodes, ranks_per_node, label, mode, script, script_args):
    if args.launcher == 'local':
        if mode == 'torchrun':
            return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--max-restarts=0', f'--nproc-per-node={ranks_per_node}', str(script), *script_args]
        return [sys.executable, str(script), *script_args]
    network = ['--network='+os.environ['QA_SLURM_NETWORK']] if os.environ.get('QA_SLURM_NETWORK') else []
    return ['srun', *network, '--kill-on-bad-exit=1', '--unbuffered', f'--nodes={nodes}',
            f'--ntasks={nodes}', '--ntasks-per-node=1', f'--gpus-per-task={args.gpus_per_node}',
            '--time=00:20:00', '--nodelist=' + ','.join(args.hosts[:nodes]),
            'bash', str(HERE / 'node.sh'), str(args.image), str(args.output), mode,
            str(nodes), str(ranks_per_node), label, script.name, *script_args]


def execute(argv, log, environment):
    print('Starting', log.stem, flush=True)
    started = time.monotonic()
    with log.open('w') as file:
        with subprocess.Popen(argv, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                file.write(line)
                file.flush()
                print(line, end='', flush=True)
            code = process.wait()
    if code:
        raise RuntimeError(f'{log.stem} exited {code}; see {log}')
    return time.monotonic()-started


def check_transport(logs, records):
    evidence = []
    for row in records:
        candidates = list(logs.glob(f'nccl-probe-*-{row["pid"]}.log'))
        # PID reuse on different nodes is allowed; host name identifies the log.
        candidates = [path for path in candidates if path.name.startswith('nccl-probe-' + row['host'] + '-')]
        if len(candidates) != 1:
            raise ValueError('Missing or ambiguous per-rank NCCL transport log')
        content = candidates[0].read_text(errors='replace')
        if not re.search(r'Using network\s+(AWS Libfabric|OFI)', content, re.I):
            raise ValueError('Expected Slingshot NCCL network was not selected: ' + candidates[0].name)
        if not re.search(r'(provider|fabric)[^\n]*\bcxi\b|\bcxi\b[^\n]*(provider|fabric)', content, re.I):
            raise ValueError('CXI provider is not evidenced in: ' + candidates[0].name)
        evidence.append(candidates[0].name)
    return evidence


def run(args):
    validate_layout(args.stage, args.nodes, args.gpus_per_node, args.launcher, args.backend, args.network)
    previous = None
    if args.stage in PREVIOUS:
        if args.previous is None:
            raise ValueError('Inspect the preceding stage and provide --previous DIRECTORY')
        previous = json.loads((args.previous / 'report.json').read_text())
        if previous.get('status') != 'passed' or previous.get('stage') != PREVIOUS[args.stage]:
            raise ValueError('Preceding stage is missing, failed or out of order')
    if args.launcher == 'slurm':
        if not os.environ.get('SLURM_JOB_ID') or int(os.environ.get('SLURM_NNODES', '0')) != args.nodes:
            raise ValueError('Launch inside the matching Slurm allocation')
        args.hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], text=True).split()
        if len(args.hosts) != args.nodes:
            raise ValueError('Slurm node list differs from the requested topology')
    else:
        args.hosts = [socket.gethostname()]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('configs', 'runs', 'probes', 'checks', 'logs'):
        (args.output / name).mkdir()
    report = dict(formatVersion=1, status='running', stage=args.stage, backend=args.backend,
                  target='Isambard' if args.launcher == 'slurm' else 'local-cpu-debug',
                  nodes=args.nodes, gpusPerNode=args.gpus_per_node, precision=args.precision,
                  jobId=os.environ.get('SLURM_JOB_ID'), imageSha256=digest(args.image) if args.image else None,
                  siteEnvironment=os.environ.get('ISAMBARD_HOST'),
                  harness={path.name: digest(path) for path in sorted(HERE.iterdir()) if path.is_file()},
                  runs={}, comparisons=[], timingPolicy='correctness only; tiny cold/JIT runs are not benchmarks')
    environment = dict(os.environ, QA_WORLD_NODES=str(args.nodes), QA_NETWORK=args.network,
                       QA_MASTER_ADDR=args.hosts[0], QA_MASTER_PORT=str(20000+int(os.environ.get('SLURM_JOB_ID', '0'))%35000),
                       OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    if args.previous:
        environment['QA_PREVIOUS'] = str(args.previous.resolve())
    if args.launcher == 'local':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    atomic_json(args.output/'report.json', report)
    def mapped(path):
        return str(path) if args.launcher == 'local' else '/work/' + str(path.relative_to(args.output))
    def launch(label, mode, script, arguments, nodes=1, ranks=1):
        return execute(command(args, nodes, ranks, label, mode, script, arguments), args.output/'logs'/f'{label}.log', environment)
    def analysis(label, *arguments):
        launch(label, 'exec', HERE/'check.py', list(arguments))
    def flow(label, steps, nodes=1, ranks=1, resume=None, rows=4, external=False):
        doc = dict(schemaVersion=1, case='drivenTurbulence', simulation=dict(
            nx=args.nx, dim=2, precision=args.precision, nSteps=steps, maxDt=.0002,
            plot=False, video=False, store=False, execution='eager', cudaGraph=False,
            progress=True, stallProgress=1e-3, velocityAlarmPlotInterval=1, integrationScheme='rungeKutta4'),
            params=dict(accelerationRMS=.1, forcingInterval=.00025, seed=1234), solver=dict(scheme='CRKSPH'))
        if resume is not None:
            if external:
                checkpoint = args.previous/'runs/distributed/checkpoint-00000002.h5'
                if not checkpoint.is_file():
                    raise ValueError('Previous job checkpoint missing: ' + str(checkpoint))
                doc['simulation']['resumeFrom'] = str(checkpoint) if args.launcher == 'local' else '/previous/runs/distributed/checkpoint-00000002.h5'
            else:
                doc['simulation']['resumeFrom'] = mapped(args.output/'runs'/resume/'finalState.h5')
        configuration = args.output/'configs'/f'{label}.json'
        atomic_json(configuration, doc)
        output = args.output/'runs'/label
        script = Path(args.distributed_script) if args.launcher == 'local' else Path('run_distributed.py')
        seconds = launch(label, 'torchrun', script, ['--configuration', mapped(configuration), '--backend', args.backend,
            '--output', mapped(output), '--edge-budget', '50000', '--max-rows', str(rows),
            '--checkpoint-every', '2', '--offset-dtype', 'int64' if rows == 3 else 'int32',
            '--replica-checks', 'exact', '--timeout-seconds', '600'], nodes, ranks)
        manifest = json.loads((output/'manifest.json').read_text())
        placement = manifest['placement']
        expected = steps + (2 if external else json.loads((args.output/'runs'/resume/'manifest.json').read_text())['completed'] if resume else 0)
        if manifest['status'] != 'complete' or manifest['completed'] != expected or len(placement) != nodes*ranks:
            raise ValueError('Run status/step/rank contract failed: ' + label)
        if len({row['host'] for row in placement}) != nodes:
            raise ValueError('Wrong physical node count in ' + label)
        if args.backend == 'nccl' and len({row['gpuUuid'] for row in placement}) != nodes*ranks:
            raise ValueError('Physical GPUs were reused in ' + label)
        if {name: item['digest'] for name, item in manifest['sources'].items()} != report['fingerprint']['sources']:
            raise ValueError('Application source changed between probe and solver')
        for rank in range(nodes*ranks):
            data = json.loads((output/f'rank-{rank:05d}.json').read_text())
            if data['status'] != 'complete' or data['completed'] != expected:
                raise ValueError('Rank did not complete: ' + label)
        report['runs'][label] = dict(ranks=nodes*ranks, nodes=nodes, completed=expected, wallSeconds=seconds)
        atomic_json(args.output/'report.json', report)
    def compare(first, second):
        output = args.output/'checks'/f'{first}-vs-{second}.json'
        analysis('compare-'+second, 'compare', mapped(args.output/'runs'/first/'finalState.h5'),
                 mapped(args.output/'runs'/second/'finalState.h5'), '--output', mapped(output))
        report['comparisons'].append(json.loads(output.read_text()))
    try:
        if environment.get('QA_DEPLOYMENT_PROFILE'):
            import importlib.util
            spec = importlib.util.spec_from_file_location('site_deployment', HERE/'deployment.py')
            deployment = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(deployment)
            captured = deployment.capture(environment['QA_DEPLOYMENT_PROFILE'], image=args.image,
                                          image_sha256=report['imageSha256'], environment=environment)
            atomic_json(args.output/'probes/deployment.json', captured)
            report['deployment'] = dict(profileSha256=captured['profileSha256'],
                                        receipt='probes/deployment.json',
                                        receiptSha256=digest(args.output/'probes/deployment.json'))
            atomic_json(args.output/'report.json', report)
        if previous and any(previous.get(key) != report.get(key) for key in ('target', 'imageSha256', 'precision', 'backend')):
            raise ValueError('Previous stage used a different target/image/precision/backend; restart at single')
        probe_output = args.output/'probes'
        launch('probe', 'torchrun', HERE/'probe.py', ['--output', mapped(probe_output), '--backend', args.backend,
               '--expected-nodes', str(args.nodes), '--expected-architecture', args.expected_architecture], args.nodes, args.gpus_per_node)
        probe = json.loads((probe_output/'probe.json').read_text())
        first = probe['ranks'][0]
        report['fingerprint'] = {name: first[name] for name in ('sources', 'packages', 'architecture')}
        if environment.get('QA_DEPLOYMENT_PROFILE') and captured['profile']['applicationFingerprint'] != report['fingerprint']:
            raise ValueError('Application/runtime fingerprint differs from pinned deployment profile')
        if previous and previous['fingerprint'] != report['fingerprint']:
            raise ValueError('Application/runtime fingerprint differs from preceding stage')
        if args.stage == 'multi-node':
            report['transportLogs'] = check_transport(args.output/'logs', probe['ranks'])
        flow('reference', 4)
        if args.stage == 'single':
            flow('prefix', 2)
            flow('resumed', 2, resume='prefix', rows=3)
            compare('reference', 'resumed')
        elif args.stage in ('same-node', 'multi-node'):
            flow('distributed', 4, args.nodes, args.gpus_per_node)
            compare('reference', 'distributed')
            if args.stage == 'multi-node':
                flow('multi-prefix', 2, args.nodes, args.gpus_per_node)
                flow('single-resume', 2, resume='multi-prefix', rows=3)
                compare('reference', 'single-resume')
        else:
            # This checkpoint was produced by the preceding Slurm job, not
            # merely another process in the current allocation.
            flow('cross-job-resume', 2, resume='previous', rows=3, external=True)
            compare('reference', 'cross-job-resume')
            flow('prefix', 2)
            flow('distributed-resume', 2, ranks=args.gpus_per_node, resume='prefix', rows=3)
            compare('reference', 'distributed-resume')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__+': '+str(error))
        raise
    finally:
        atomic_json(args.output/'report.json', report)
        try:
            analysis('seal', 'seal', mapped(args.output))
        except Exception as error:
            report.update(status='failed', artifactError=str(error))
            atomic_json(args.output/'report.json', report)
            raise RuntimeError('Artifact sealing failed') from error
    print('Stage PASSED. Retrieve and verify before submitting the next stage:', args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('single', 'same-node', 'restart', 'multi-node'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus-per-node', type=int, default=1)
    parser.add_argument('--network', choices=('none', 'host', 'custom'), default='none')
    parser.add_argument('--launcher', choices=('slurm', 'local'), default='slurm')
    parser.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl')
    parser.add_argument('--expected-architecture', default='aarch64')
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--nx', type=int, default=8)
    parser.add_argument('--distributed-script', default=str(HERE.parent/'run_distributed.py'))
    args = parser.parse_args()
    if args.launcher == 'slurm' and (args.image is None or not args.image.is_file()):
        parser.error('Provide the native ARM64 SIF image via --image')
    if not 8 <= args.nx <= 32:
        parser.error('Acceptance fixture nx must be 8–32; large runs are a separate gate')
    run(args)


if __name__ == '__main__':
    main()
