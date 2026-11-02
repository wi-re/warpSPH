#!/usr/bin/env bash
#SBATCH --job-name=wsph-network-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
# Read-only image/runtime check; this does not qualify inter-node communication.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${QA_PREVIOUS:?Provide the passed restart-stage directory}"
export QA_ROOT=${QA_ROOT:-$HOME}
export QA_BUNDLE=${QA_BUNDLE:-$QA_ROOT/warpsph-isambard}
export QA_IMAGE=${QA_IMAGE:-$QA_ROOT/images/warpsph-arm64.sif}
export QA_RESULTS_ROOT=${QA_RESULTS_ROOT:-$QA_ROOT/qualification}
export QA_NETWORK=${QA_PREFLIGHT_NETWORK:-host}
[[ $QA_NETWORK == host || $QA_NETWORK == custom ]] || { echo 'Use host or custom preflight network profile.' >&2; exit 2; }
[[ $QA_IMAGE = /* && -f $QA_IMAGE && $QA_RESULTS_ROOT = /* && $QA_PREVIOUS = /* ]] || { echo 'Use absolute paths and an existing image.' >&2; exit 2; }
command -v apptainer >/dev/null || module load apptainer
if [[ $QA_NETWORK == host ]]; then
  module load brics/apptainer-multi-node
else
  : "${QA_COMM_ROOT:?Compatible communication build required}"
  : "${QA_LIBFABRIC_HOST:?Actual host Libfabric prefix required}"
  : "${QA_NETWORK_SETUP:?Reviewed custom network script required}"
fi
output="$QA_RESULTS_ROOT/network-preflight-$SLURM_JOB_ID"
mkdir -p "$QA_RESULTS_ROOT"
mkdir "$output"
mkdir "$output/logs" "$output/probes"
# Keep the uploaded/image sources unchanged. Extend only a private probe copy.
cp -R -- "$QA_BUNDLE/harness" "$output/harness"
# Older uploaded wrappers source adapt.sh with probe arguments still in scope.
"${QA_PYTHON:-python3}" - "$output/harness/container.sh" <<'BOOTSTRAP_PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
old = 'host) source /host/adapt.sh ;;'
new = 'host) adapt_host_environment() { source /host/adapt.sh; }; adapt_host_environment ;;'
path.write_text(path.read_text().replace(old, new, 1))
BOOTSTRAP_PY
cat >> "$output/harness/probe.py" <<'RUNTIME_PY'

if __name__ == '__main__':
    import ctypes
    import json
    import os
    from pathlib import Path
    paths = {line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
             if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[5].startswith('/')}
    nccl_paths = sorted(path for path in paths if 'libnccl.so' in Path(path).name)
    if len(nccl_paths) != 1:
        raise RuntimeError('Cannot identify one loaded NCCL library: ' + repr(nccl_paths))
    library = ctypes.CDLL(nccl_paths[0], mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.ncclGetVersion.restype = ctypes.c_int
    runtime = ctypes.c_int()
    if library.ncclGetVersion(ctypes.byref(runtime)) != 0:
        raise RuntimeError('ncclGetVersion failed for the already-loaded library')
    data = dict(ncclRuntime=runtime.value, ncclLibraries=nccl_paths,
                mappedCommunicationLibraries=sorted(path for path in paths if
                    Path(path).name.startswith(('libnccl', 'libcuda', 'libfabric', 'libcxi', 'libnvidia-ptxjitcompiler'))))
    Path('/work/probes/runtime-libraries.json').write_text(json.dumps(data, indent=2)+'\n')
    print('Loaded NCCL runtime:', runtime.value, nccl_paths, flush=True)
RUNTIME_PY
export QA_MASTER_ADDR
QA_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ $QA_MASTER_ADDR != *$'\n'* ]] || { echo 'Preflight requires exactly one node.' >&2; exit 2; }
export QA_MASTER_PORT=$((20000 + SLURM_JOB_ID % 35000))
ras_port=${QA_RAS_PORT:-$((10000 + SLURM_JOB_ID % 10000))}
[[ $ras_port =~ ^[0-9]+$ ]] && (( ras_port >= 1024 && ras_port <= 65535 && ras_port != QA_MASTER_PORT )) || { echo 'Invalid RAS port or collision with rendezvous port.' >&2; exit 2; }
export APPTAINERENV_NCCL_NET='AWS Libfabric'
export APPTAINERENV_FI_PROVIDER=cxi APPTAINERENV_NCCL_SOCKET_IFNAME=hsn
export APPTAINERENV_NCCL_RAS_ADDR="127.0.0.1:$ras_port"
echo "Network preflight output: $output"
echo "Profile: $QA_NETWORK; RAS diagnostic listener: $APPTAINERENV_NCCL_RAS_ADDR"
set +e
srun --kill-on-bad-exit=1 --unbuffered --nodes=1 --ntasks=1 --gpus-per-task=1 \
  bash "$output/harness/node.sh" "$QA_IMAGE" "$output" torchrun 1 1 probe probe.py \
  --output /work/probes --backend nccl --expected-nodes 1 --expected-architecture aarch64 \
  2>&1 | tee "$output/logs/probe.log"
probe_status=${PIPESTATUS[0]}
set -e
"${QA_PYTHON:-python3}" - "$output" "$QA_PREVIOUS" "$QA_IMAGE" "$output/harness" "$probe_status" "$QA_NETWORK" <<'PY'
from pathlib import Path
import hashlib, importlib.util, json, os, re, sys

output, previous, image, harness = [Path(value) for value in sys.argv[1:5]]
code = int(sys.argv[5])
report = dict(stage='network-preflight', status='failed', jobId=os.environ.get('SLURM_JOB_ID'),
              profile=sys.argv[6], probeExitCode=code, scope='one-GPU library/provider loading; no inter-node qualification')
def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()
try:
    baseline = json.loads((previous/'report.json').read_text())
    if baseline.get('stage') != 'restart' or baseline.get('status') != 'passed':
        raise ValueError('Provide a passed restart-stage report')
    report['previousJobId'] = baseline['jobId']
    if digest(image) != baseline['imageSha256']:
        raise ValueError('SIF differs from the accepted restart stage')
    if code:
        raise RuntimeError('Container/probe failed; inspect logs/probe.log for library or plugin errors')
    probe = json.loads((output/'probes/probe.json').read_text())
    if probe.get('status') != 'passed' or len(probe['ranks']) != 1:
        raise ValueError('Expected a successful single-rank GPU probe')
    rank = probe['ranks'][0]
    report['runtime'] = rank
    if rank.get('backend') != 'nccl' or rank.get('architecture') != 'aarch64' or not rank.get('gpuUuid'):
        raise ValueError('Expected native ARM64 NCCL execution on one physical GPU')
    if {key: rank[key] for key in ('sources', 'packages', 'architecture')} != baseline['fingerprint']:
        raise ValueError('Application/runtime fingerprint changed under network profile')
    loaded = json.loads((output/'probes/runtime-libraries.json').read_text())
    report['loadedLibraries'] = loaded
    expected_nccl = json.loads((previous/'runs/reference/manifest.json').read_text())['placement'][0]['ncclRuntime']
    if loaded['ncclRuntime'] != expected_nccl:
        raise ValueError(f'Loaded NCCL changed: expected {expected_nccl}, found {loaded["ncclRuntime"]}; build a compatible OFI plugin instead of replacing the validated runtime')
    if len(loaded['ncclLibraries']) != 1:
        raise ValueError('Could not identify one loaded NCCL library')
    if report['profile'] == 'custom':
        mapped = loaded.get('mappedCommunicationLibraries', [])
        if not any(path.startswith('/opt/slingshot/aws-ofi-nccl/lib/libnccl-net') for path in mapped):
            raise ValueError('Custom OFI plugin was not loaded from the communication build')
        if not any(path.startswith('/host/libfabric/lib64/libfabric.so') for path in mapped):
            raise ValueError('Custom profile did not load the bound host Libfabric')
        if not any(Path(path).name.startswith('libcxi.so') for path in mapped):
            raise ValueError('CXI library was not loaded')
    if rank['network'].get('FI_PROVIDER') != 'cxi':
        raise ValueError('CXI provider was not explicitly selected')
    spec = importlib.util.spec_from_file_location('network_qualification', harness/'qualify.py')
    qualification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualification)
    report['transportLogs'] = qualification.check_transport(output/'logs', probe['ranks'])
    report['ncclWarnings'] = [line for name in report['transportLogs']
                              for line in (output/'logs'/name).read_text(errors='replace').splitlines()
                              if re.search(r'NCCL WARN|NCCL ERROR', line)]
    report['status'] = 'passed'
except Exception as error:
    report['error'] = type(error).__name__ + ': ' + str(error)
finally:
    temporary = output/'.report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    temporary.replace(output/'report.json')
print(json.dumps(report, indent=2), flush=True)
if report['status'] != 'passed':
    raise SystemExit(1)
print('Network preflight PASSED. This is not a multi-node qualification.', flush=True)
PY
