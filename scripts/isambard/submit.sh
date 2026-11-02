#!/usr/bin/env bash
# Submit exactly one stage; no SSH credentials or automatic follow-up jobs.
set -euo pipefail
stage=${1:?Usage: bash submit.sh single|same-node|restart|multi-node}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${QA_IMAGE:?Absolute native ARM64 SIF path required}"
: "${QA_RESULTS_ROOT:?Absolute shared results parent required}"
case "$stage" in
  single) nodes=1; gpus=1 ;;
  same-node|restart) nodes=1; gpus=${QA_GPUS_PER_NODE:-2} ;;
  multi-node) nodes=${QA_NODES:-2}; gpus=${QA_GPUS_PER_NODE:-1} ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
[[ "$QA_IMAGE" = /* && -f "$QA_IMAGE" && "$QA_RESULTS_ROOT" = /* ]] || { echo 'Use absolute shared paths; image must exist.' >&2; exit 2; }
if [[ "$stage" != single ]]; then
  : "${QA_PREVIOUS:?Inspect and provide the preceding passed stage directory}"
  [[ "$QA_PREVIOUS" = /* && -f "$QA_PREVIOUS/report.json" ]] || { echo 'Previous stage report missing.' >&2; exit 2; }
  "${QA_PYTHON:-python3}" - "$QA_PREVIOUS/report.json" "$stage" <<'PY'
import json,sys
previous=json.load(open(sys.argv[1]))
expected={'same-node':'single','restart':'same-node','multi-node':'restart'}[sys.argv[2]]
if previous.get('status')!='passed' or previous.get('stage')!=expected:
    raise SystemExit('Previous stage has not passed in the required order; no job submitted.')
PY
fi
[[ "$gpus" =~ ^[1-4]$ && "$nodes" =~ ^[0-9]+$ ]] && (( nodes > 0 )) || { echo 'Invalid node/GPU request.' >&2; exit 2; }
export QA_IMAGE QA_RESULTS_ROOT QA_PREVIOUS=${QA_PREVIOUS:-} QA_NETWORK=${QA_NETWORK:-none}
export QA_HARNESS_ROOT=$here
export QA_NODES=$nodes QA_GPUS_PER_NODE=$gpus
if [[ "$stage" == multi-node && "$QA_NETWORK" == none ]]; then
  echo 'Choose QA_NETWORK=host or custom after configuring the documented Slingshot runtime.' >&2
  exit 2
fi
mkdir -p "$QA_RESULTS_ROOT/logs"
resources=(--parsable --job-name="wsph-$stage" --nodes="$nodes" --ntasks-per-node=1
  --gpus-per-node="$gpus" --cpus-per-task="${QA_CPUS_PER_NODE:-16}" --time="${QA_TIME:-00:45:00}"
  --output="$QA_RESULTS_ROOT/logs/$stage-%j.out")
[[ -z "${QA_ACCOUNT:-}" ]] || resources+=(--account="$QA_ACCOUNT")
[[ -z "${QA_PARTITION:-}" ]] || resources+=(--partition="$QA_PARTITION")
sbatch "${resources[@]}" "$here/job.slurm" "$stage"
