"""Failure detection for deployable site QA; no SSH, Slurm allocation or GPU required."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT/'scripts/isambard'


def load(name):
    spec = importlib.util.spec_from_file_location('isambard_'+name, HARNESS/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(path, *, reverse=False, change=None):
    check = load('check')
    with h5py.File(path, 'w') as file:
        file.attrs.update(rowChunkCheckpointVersion=1, complete=1, completedSteps=4,
                          physicalSignature='same physical fixture', time=.0008, stateClass='fixture')
        order = [1, 0] if reverse else [0, 1]
        file['state/UIDs'] = np.array([10, 20], dtype=np.int32)[order]
        file['state/positions'] = np.array([[1., 2.], [3., 4.]], dtype=np.float32)[order]
        file['state/sourceEnergyBudget'] = np.array([1e-8, 2e-8], dtype=np.float64)
        file['forcing/generator'] = np.arange(5, dtype=np.uint8)
        file['forcing/knots'] = np.arange(4, dtype=np.float64)
        file['clock/time'] = np.float32(.0008)
        file['context/initialMass'] = np.float64(2.)
        if change:
            change(file)
        file.attrs['contentChecksum'] = check.checkpoint_digest(file)


def test_checkpoint_comparison_aligns_particle_uids_and_preserves_global_fields(tmp_path):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    checkpoint(a)
    checkpoint(b, reverse=True)
    result = load('check').compare(a, b)
    assert result['passed']
    assert result['fields']['state/UIDs']['exact']
    assert result['fields']['state/sourceEnergyBudget']['maxAbsolute'] == 0


@pytest.mark.parametrize('field', ['state/positions', 'forcing/knots', 'state/sourceEnergyBudget'])
def test_numerical_or_rng_drift_fails_even_in_valid_checkpoint(tmp_path, field):
    a, b = tmp_path/'a.h5', tmp_path/'b.h5'
    checkpoint(a)
    def change(file):
        value = file[field][()]
        value.flat[0] += 1e-4 if field != 'forcing/knots' else 1e-15
        file[field][...] = value
    checkpoint(b, change=change)
    result = load('check').compare(a, b)
    assert not result['passed'] and not result['fields'][field]['passed']


def test_checksum_nonfinite_and_incomplete_states_are_rejected(tmp_path):
    check = load('check')
    path = tmp_path/'state.h5'
    checkpoint(path)
    with h5py.File(path, 'r+') as file:
        file['state/positions'][0, 0] = 99
    with h5py.File(path) as file, pytest.raises(ValueError, match='checksum'):
        check.validate_checkpoint(file)
    checkpoint(path, change=lambda file: file['state/positions'].__setitem__((0, 0), np.nan))
    with h5py.File(path) as file, pytest.raises(ValueError, match='Nonfinite'):
        check.validate_checkpoint(file)
    checkpoint(path, change=lambda file: file.attrs.__setitem__('complete', 0))
    with h5py.File(path) as file, pytest.raises(ValueError, match='incomplete'):
        check.validate_checkpoint(file)


def test_retrieval_detects_damaged_missing_and_unsafe_artifacts(tmp_path):
    check = load('check')
    (tmp_path/'runs').mkdir()
    checkpoint(tmp_path/'runs/finalState.h5')
    (tmp_path/'report.json').write_text(json.dumps({'status':'passed', 'stage':'single'}))
    check.seal(tmp_path)
    assert check.verify(tmp_path)['filesVerified'] == 2
    (tmp_path/'runs/finalState.h5').write_bytes(b'damaged transfer')
    with pytest.raises(ValueError, match='damaged'):
        check.verify(tmp_path)
    (tmp_path/'artifacts.json').write_text(json.dumps({'formatVersion':1, 'sha256': {'../outside':'unused'}}))
    with pytest.raises(ValueError, match='Unsafe'):
        check.verify(tmp_path)


def test_multinode_cannot_pass_on_socket_transport_or_missing_rank_logs(tmp_path):
    qualify = load('qualify')
    record = [{'pid':123, 'host':'nodeA'}]
    with pytest.raises(ValueError, match='Missing'):
        qualify.check_transport(tmp_path, record)
    log = tmp_path/'nccl-probe-nodeA-123.log'
    log.write_text('Using network Socket\n')
    with pytest.raises(ValueError, match='not selected'):
        qualify.check_transport(tmp_path, record)
    log.write_text('Using network AWS Libfabric\nNET/OFI Selected Provider is cxi\n')
    assert qualify.check_transport(tmp_path, record) == [log.name]
    (tmp_path/'nccl-probe-nodeAB-123.log').write_text(log.read_text())
    assert qualify.check_transport(tmp_path, record) == [log.name]


def test_apptainer_preserves_comma_separated_gpu_list(tmp_path):
    binary = tmp_path/'bin'
    binary.mkdir()
    mock = binary/'apptainer'
    mock.write_text(f'#!{sys.executable}\nimport os,json,sys\nprint(json.dumps(dict(args=sys.argv[1:],devices=os.environ.get("APPTAINERENV_CUDA_VISIBLE_DEVICES"))))\n')
    mock.chmod(0o700)
    environment = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'],
                       CUDA_VISIBLE_DEVICES='0,2,3', QA_NETWORK='none',
                       QA_MASTER_ADDR='nodeA', QA_MASTER_PORT='29000')
    result = subprocess.run(['bash',str(HARNESS/'node.sh'),'/image.sif',str(tmp_path),
                             'torchrun','1','3','probe','probe.py'],
                            env=environment,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    invocation = json.loads(result.stdout)
    assert invocation['devices'] == '0,2,3'
    assert not any('CUDA_VISIBLE_DEVICES=' in arg for arg in invocation['args'])
    env_values = [invocation['args'][index+1] for index, arg in enumerate(invocation['args']) if arg == '--env']
    assert all(not entry.startswith('HOME=') for value in env_values for entry in value.split(','))


def test_one_torchrun_parent_per_node_and_no_cpu_site_pass():
    qualify = load('qualify')
    args = SimpleNamespace(launcher='slurm', gpus_per_node=4, hosts=['nodeA','nodeB'],
                           image=Path('/image.sif'), output=Path('/shared/results'))
    argv = qualify.command(args, 2, 4, 'probe', 'torchrun', HARNESS/'probe.py', [])
    assert '--ntasks=2' in argv and '--ntasks-per-node=1' in argv and '--gpus-per-task=4' in argv
    with pytest.raises(ValueError, match='NCCL'):
        qualify.validate_layout('single', 1, 1, 'slurm', 'gloo', 'none')
    with pytest.raises(ValueError, match='network'):
        qualify.validate_layout('multi-node', 2, 1, 'slurm', 'nccl', 'none')
    with pytest.raises(ValueError, match='requires at least two'):
        qualify.validate_layout('restart', 1, 1, 'slurm', 'nccl', 'none')


def test_submit_exports_harness_path_and_rejects_failed_previous_stage(tmp_path):
    binary = tmp_path/'bin'
    binary.mkdir()
    mock = binary/'sbatch'
    mock.write_text(f'#!{sys.executable}\nimport os,json,sys\nprint(json.dumps(dict(args=sys.argv[1:],harness=os.environ.get("QA_HARNESS_ROOT"))))\n')
    mock.chmod(0o700)
    image = tmp_path/'image.sif'
    image.touch()
    env = dict(os.environ, PATH=str(binary)+':'+os.environ['PATH'], QA_IMAGE=str(image),
               QA_RESULTS_ROOT=str(tmp_path/'results'), QA_PYTHON=sys.executable)
    result = subprocess.run(['bash',str(HARNESS/'submit.sh'),'single'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    submitted = json.loads(result.stdout)
    assert submitted['harness'] == str(HARNESS)
    assert '--gpus-per-node=1' in submitted['args']
    previous = tmp_path/'previous'
    previous.mkdir()
    (previous/'report.json').write_text(json.dumps({'status':'failed','stage':'single'}))
    result = subprocess.run(['bash',str(HARNESS/'submit.sh'),'same-node'],
                            env=dict(env,QA_PREVIOUS=str(previous)),capture_output=True,text=True)
    assert result.returncode != 0 and 'no job submitted' in result.stderr


def test_arm_bundle_rejects_x86_and_generates_native_definition(tmp_path):
    prepare = load('prepare_bundle')
    context = tmp_path/'context'
    context.mkdir()
    manifest = context/'stack-manifest.json'
    manifest.write_text(json.dumps({'platform':'linux/amd64','baseImage':'nvcr.io/nvidia/pytorch:26.01-py3'}))
    with pytest.raises(ValueError, match='arm64'):
        prepare.prepare(context, tmp_path/'wrong')
    manifest.write_text(json.dumps({'platform':'linux/arm64','baseImage':'nvcr.io/nvidia/pytorch:26.01-py3'}))
    bundle = prepare.prepare(context, tmp_path/'bundle', tmp_path/'bundle.tar.gz')
    definition = (bundle/'context/isambard.def').read_text()
    assert 'From: nvcr.io/nvidia/pytorch:26.01-py3' in definition
    compile(definition.split("python - <<'PY'\n",1)[1].split('\nPY\n',1)[0], '<definition>', 'exec')
    assert (bundle/'harness/submit.sh').is_file() and (tmp_path/'bundle.tar.gz').is_file()


@pytest.mark.parametrize('scenario,expected', [('valid', 'passed'), ('old-nccl', 'failed'),
                                               ('socket', 'failed'), ('probe-failure', 'failed')])
def test_network_preflight_requires_unchanged_runtime_and_cxi_evidence(tmp_path, scenario, expected):
    import hashlib
    output, previous = tmp_path/'output', tmp_path/'previous'
    for path in (output/'logs', output/'probes', previous/'probes', previous/'runs/reference'):
        path.mkdir(parents=True)
    image = tmp_path/'image.sif'
    image.write_bytes(b'fixture image identity')
    fingerprint = dict(sources={'fixture':'source hash'}, packages={'fixture':'version'}, architecture='aarch64')
    (previous/'report.json').write_text(json.dumps(dict(stage='restart', status='passed', jobId='7989645',
        imageSha256=hashlib.sha256(image.read_bytes()).hexdigest(), fingerprint=fingerprint)))
    (previous/'probes/probe.json').write_text(json.dumps({'ranks':[{'nccl':[2,29,2]}]}))
    (previous/'runs/reference/manifest.json').write_text(json.dumps({'placement':[{'ncclRuntime':22902}]}))
    (output/'probes/runtime-libraries.json').write_text(json.dumps(dict(ncclRuntime=22606 if scenario=='old-nccl' else 22902,
        ncclLibraries=['/loaded/libnccl.so.2'])))
    rank = dict(fingerprint, backend='nccl', gpuUuid='GPU-distinct', nccl=[2,29,2],
                network={'FI_PROVIDER':'cxi'}, host='nodeA', pid=123)
    (output/'probes/probe.json').write_text(json.dumps({'status':'passed','ranks':[rank]}))
    network = 'Socket' if scenario=='socket' else 'AWS Libfabric'
    (output/'logs/nccl-probe-nodeA-123.log').write_text('Using network '+network+'\nNET/OFI Selected Provider is cxi\n')
    script = (HARNESS/'network_preflight.slurm').read_text()
    compile(script.split("<<'RUNTIME_PY'\n",1)[1].split('\nRUNTIME_PY\n',1)[0], '<runtime-capture>', 'exec')
    embedded = script.split("<<'PY'\n",1)[1].split('\nPY\n',1)[0]
    result = subprocess.run([sys.executable,'-',str(output),str(previous),str(image),str(HARNESS),
                             '2' if scenario=='probe-failure' else '0','host'],
                            input=embedded,text=True,capture_output=True)
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == expected, result.stdout+result.stderr
    assert result.returncode == (0 if expected=='passed' else 1)
    if scenario == 'old-nccl':
        assert 'Loaded NCCL changed' in report['error']
    if scenario == 'valid':
        assert report['transportLogs'] == ['nccl-probe-nodeA-123.log']


def test_host_adaptation_does_not_exec_probe_arguments(tmp_path):
    import shlex
    adapter = tmp_path/'adapt.sh'
    adapter.write_text('export ADAPT_SENTINEL=active\nexec "$@"\n')
    branch = (HARNESS/'container.sh').read_text().split('  host)',1)[1].split(';;',1)[0]
    branch = branch.replace('/host/adapt.sh', shlex.quote(str(adapter)))
    script = ('set -e\nset -- --output /work/probes\n'+branch+'\n'+shlex.quote(sys.executable)+
              ' -c '+shlex.quote('import json,os,sys; print(json.dumps(dict(args=sys.argv[1:], adapted=os.environ.get("ADAPT_SENTINEL"))))')+' "$@"')
    result = subprocess.run(['bash','-c',script],text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(args=['--output','/work/probes'],adapted='active')
