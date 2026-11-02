#!/usr/bin/env bash
# Executed once on each selected node by srun. Do not invoke on a login node.
set -euo pipefail
image=$1; output=$2; mode=$3; nodes=$4; workers=$5; label=$6; script=$7
shift 7
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
binds=(--bind "$output:/work" --bind "$here:/qualification:ro")
[[ -z ${QA_PREVIOUS:-} ]] || binds+=(--bind "$QA_PREVIOUS:/previous:ro")
if [[ ${QA_NETWORK:-none} == custom ]]; then
  [[ -f "$QA_NETWORK_SETUP" && -d "$QA_COMM_ROOT" && -d "$QA_LIBFABRIC_HOST" ]] || { echo 'Custom communication paths missing.' >&2; exit 2; }
  binds+=(--bind "$QA_NETWORK_SETUP:/network-setup.sh:ro"
    --bind "$QA_COMM_ROOT:/opt/slingshot:ro" --bind "$QA_LIBFABRIC_HOST:/host/libfabric:ro"
    --bind /usr/lib64:/host/usr/lib64:ro)
fi
# Preserve comma-separated GPU lists literally; --env itself parses commas.
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPUs}"
# HOME is set by container.sh; Apptainer rejects HOME in --env.
args=(exec --nv --cleanenv --no-home --pwd /work "${binds[@]}"
  --env "QA_NETWORK=${QA_NETWORK:-none},QA_WORLD_NODES=$nodes,QA_NODE_RANK=${SLURM_NODEID:-0}"
  --env "QA_MASTER_ADDR=$QA_MASTER_ADDR,QA_MASTER_PORT=$QA_MASTER_PORT,QA_JOB_ID=${SLURM_JOB_ID:-local}"
  "$image" bash /qualification/container.sh "$mode" "$nodes" "$workers" "$label" "$script")
exec apptainer "${args[@]}" "$@"
