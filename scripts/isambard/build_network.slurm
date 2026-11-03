#!/usr/bin/env bash
#SBATCH --job-name=wsph-build-ofi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
# Build only OFI/hwloc against the accepted image; never replace its NCCL/CUDA.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
: "${QA_PREVIOUS:?Provide the accepted restart-stage directory}"
export QA_ROOT=${QA_ROOT:-$HOME}
export QA_BUNDLE=${QA_BUNDLE:-$QA_ROOT/warpsph-isambard}
export QA_IMAGE=${QA_IMAGE:-$QA_ROOT/images/warpsph-arm64.sif}
export QA_RESULTS_ROOT=${QA_RESULTS_ROOT:-$QA_ROOT/qualification}
export QA_COMM_SOURCES=${QA_COMM_SOURCES:-$QA_ROOT/communication-sources}
export QA_LIBFABRIC_HOST=${QA_LIBFABRIC_HOST:-/opt/cray/libfabric/2.3.1}
export QA_COMM_ROOT=${QA_COMM_ROOT:-$QA_ROOT/communication/ofi-$SLURM_JOB_ID}
# Slurm executes a spooled copy; sibling files must use the uploaded harness path.
here=${QA_BUILD_HARNESS:-$QA_BUNDLE/harness}
for path in "$QA_IMAGE" "$QA_PREVIOUS" "$QA_COMM_SOURCES" "$QA_LIBFABRIC_HOST" "$QA_COMM_ROOT"; do
  [[ $path = /* && $path != *:* && $path != *,* ]] || { echo 'Use absolute bind paths without colons or commas.' >&2; exit 2; }
done
[[ -f $QA_IMAGE && -f $here/build_network.sh && -f $here/communication-sources.json && -f $here/network-custom.example.sh ]] || { echo "Missing image/build inputs; QA_BUILD_HARNESS=$here" >&2; exit 2; }
[[ -d $QA_COMM_SOURCES && -d $QA_LIBFABRIC_HOST ]] || exit 2
# Exclude the incompatible site's automatic NCCL/CUDA injection, even if loaded by caller.
if [[ ${LOADEDMODULES:-} == *brics/apptainer-multi-node* ]]; then
  echo 'Unload brics/apptainer-multi-node before building the custom profile.' >&2; exit 2
fi
unset APPTAINER_BINDPATH SINGULARITY_BINDPATH
command -v apptainer >/dev/null || module load apptainer
output="$QA_RESULTS_ROOT/network-build-$SLURM_JOB_ID"
mkdir -p "$QA_RESULTS_ROOT" "$(dirname -- "$QA_COMM_ROOT")"
mkdir "$output" "$QA_COMM_ROOT"
mkdir "$output/logs" "$output/harness"
cp -- "$here/build_network.sh" "$here/communication-sources.json" "$here/network-custom.example.sh" "$output/harness/"
cp -- "$here/communication-sources.json" "$QA_COMM_ROOT/source-lock.json"
cp -- "$here/network-custom.example.sh" "$QA_COMM_ROOT/network-setup.sh"
export QA_NETWORK_BUILD_OUTPUT=$output
build_status=1
finish() {
  "${QA_PYTHON:-python3}" - "$output" "$QA_COMM_ROOT" "$build_status" <<'REPORT_PY'
import json, os, sys
from pathlib import Path
output, root = map(Path, sys.argv[1:3])
code = int(sys.argv[3])
report = dict(stage='network-build', status='passed' if code == 0 else 'failed',
              jobId=os.environ['SLURM_JOB_ID'], buildExitCode=code, communicationRoot=str(root),
              scope='OFI/hwloc build and load check only; no inter-node qualification')
if (root/'native-build.json').is_file():
    report['nativeBuild'] = json.loads((root/'native-build.json').read_text())
if code == 0 and not report.get('nativeBuild'):
    report.update(status='failed', error='Missing native build provenance')
(output/'report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps(report, indent=2), flush=True)
if report['status'] != 'passed':
    raise SystemExit(1)
REPORT_PY
}
trap finish EXIT
"${QA_PYTHON:-python3}" - "$QA_IMAGE" "$QA_PREVIOUS" "$output" <<'IDENTITY_PY'
import hashlib, json, sys
from pathlib import Path
image, previous, output = map(Path, sys.argv[1:])
report = json.loads((previous/'report.json').read_text())
assert report['stage'] == 'restart' and report['status'] == 'passed', 'Require a passed restart baseline'
h = hashlib.sha256()
with image.open('rb') as f:
    for block in iter(lambda: f.read(4*1024*1024), b''):
        h.update(block)
assert h.hexdigest() == report['imageSha256'], 'SIF differs from accepted restart'
expected = json.loads((previous/'runs/reference/manifest.json').read_text())['placement'][0]['ncclRuntime']
(output/'accepted-image.json').write_text(json.dumps(dict(imageSha256=h.hexdigest(), ncclRuntime=expected, previousJobId=report['jobId']), indent=2)+'\n')
IDENTITY_PY
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPU}"
echo "Network build output: $output; communication root: $QA_COMM_ROOT"
set +e
srun --kill-on-bad-exit=1 --unbuffered --nodes=1 --ntasks=1 --gpus-per-task=1 \
  apptainer exec --nv --cleanenv --no-home --pwd /opt/slingshot \
  --bind "$QA_COMM_ROOT:/opt/slingshot" --bind "$QA_COMM_SOURCES:/sources:ro" \
  --bind "$output:/build" --bind "$QA_LIBFABRIC_HOST:/host/libfabric:ro" \
  --bind /usr/lib64:/host/usr/lib64:ro \
  --env "BUILD_JOBS=$SLURM_CPUS_PER_TASK,QA_BUILD_JOB_ID=$SLURM_JOB_ID" \
  "$QA_IMAGE" bash /build/harness/build_network.sh 2>&1 | tee "$output/logs/build.log"
build_status=${PIPESTATUS[0]}
set -e
exit "$build_status"
