#!/usr/bin/env bash
# Resource flags come from submit.sh; one torchrun parent per node.
set -euo pipefail
stage=${1:?Stage required}
here=${QA_HARNESS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
# Slurm copies this file to spool. Resolve the harness using an exported path.
: "${QA_HARNESS_ROOT:?submit.sh must export the shared harness path}"
command -v apptainer >/dev/null || module load apptainer
if [[ ${QA_NETWORK:-none} == host ]]; then
  module load brics/apptainer-multi-node
elif [[ ${QA_NETWORK:-none} == custom ]]; then
  : "${QA_NETWORK_SETUP:?Absolute in-container network setup script required}"
  : "${QA_COMM_ROOT:?Host directory containing compatible nccl/aws-ofi-nccl builds required}"
  : "${QA_LIBFABRIC_HOST:?Site libfabric prefix required}"
fi
output="$QA_RESULTS_ROOT/$stage-$SLURM_JOB_ID"
args=(--stage "$stage" --output "$output" --image "$QA_IMAGE" --nodes "$QA_NODES"
  --gpus-per-node "$QA_GPUS_PER_NODE" --network "${QA_NETWORK:-none}"
  --precision "${QA_PRECISION:-float32}" --expected-architecture aarch64)
[[ -z ${QA_PREVIOUS:-} ]] || args+=(--previous "$QA_PREVIOUS")
echo "Stage output: $output"
"${QA_PYTHON:-python3}" "$here/qualify.py" "${args[@]}"
