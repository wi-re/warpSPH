#!/usr/bin/env bash
#SBATCH --job-name=wsph-arm-build
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
set -euo pipefail
: "${QA_BUNDLE:?Absolute uploaded bundle directory required}"
: "${QA_SIF:?Fresh absolute output SIF path required}"
: "${SLURM_JOB_ID:?Submit through sbatch; do not build on a login node}"
[[ $(uname -m) == aarch64 ]] || { echo 'Build this image on native Isambard ARM64.' >&2; exit 2; }
[[ "$QA_BUNDLE" = /* && "$QA_SIF" = /* && ! -e "$QA_SIF" ]] || { echo 'Use absolute paths and a fresh SIF filename.' >&2; exit 2; }
command -v apptainer >/dev/null || module load apptainer
"${QA_PYTHON:-python3}" - "$QA_BUNDLE" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve()
for name,expected in json.loads((root/'upload-manifest.json').read_text()).items():
    path=root/name
    if path.is_symlink() or root not in path.resolve().parents:
        raise SystemExit('Unsafe bundle path: '+name)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Damaged uploaded source: '+name)
print('Uploaded source bundle hashes pass',flush=True)
PY
temporary_root=${QA_BUILD_TMP_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}
export APPTAINER_TMPDIR
APPTAINER_TMPDIR=$(mktemp -d "$temporary_root/warpsph-build-$SLURM_JOB_ID.XXXXXX")
trap 'rm -rf -- "$APPTAINER_TMPDIR"' EXIT
export APPTAINER_CACHEDIR="${QA_CONTAINER_CACHE:-$QA_BUNDLE/apptainer-cache}"
mkdir -p "$APPTAINER_CACHEDIR" "$(dirname -- "$QA_SIF")"
cd "$QA_BUNDLE/context"
apptainer build --fakeroot "$QA_SIF" isambard.def
sha256sum "$QA_SIF" > "$QA_SIF.sha256"
apptainer inspect --json "$QA_SIF" > "$QA_SIF.inspect.json"
echo "Built ARM64 simulation image: $QA_SIF"
echo 'Next: submit the single stage; image build success alone is not execution qualification.'
