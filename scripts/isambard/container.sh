#!/usr/bin/env bash
# Trusted container-side bootstrap. Site setup is explicit and happens before Torch.
set -euo pipefail
mode=$1; nodes=$2; workers=$3; label=$4; script=$5
shift 5
case ${QA_NETWORK:-none} in
  none) ;;
  host) source /host/adapt.sh ;;
  custom) source /network-setup.sh ;;
  *) echo 'Unknown network profile' >&2; exit 2 ;;
esac
export HOME=/work USER=warpsph LOGNAME=warpsph OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export WARP_CACHE_PATH="/work/cache/$(hostname)/warp" NUMBA_CACHE_DIR="/work/cache/$(hostname)/numba"
export TORCHINDUCTOR_CACHE_DIR="/work/cache/$(hostname)/torchinductor" MPLCONFIGDIR="/work/cache/$(hostname)/matplotlib"
export XDG_STATE_HOME=/work/state XDG_CACHE_HOME=/work/cache
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE="/work/logs/nccl-$label-%h-%p.log"
if (( nodes > 1 )); then
  [[ ${QA_NETWORK:-none} != none ]] || { echo 'Multi-node requires explicit Slingshot configuration' >&2; exit 2; }
  export NCCL_NET='AWS Libfabric' FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
fi
if [[ "$script" == run_distributed.py ]]; then
  target=/opt/warpsph/scripts/run_distributed.py
else
  target="/qualification/$script"
fi
[[ -f "$target" ]] || { echo "Image/harness entrypoint missing: $target" >&2; exit 2; }
if [[ "$mode" == torchrun ]]; then
  exec python -m torch.distributed.run --nnodes "$nodes" --nproc-per-node "$workers" \
    --node-rank "$QA_NODE_RANK" --rdzv-backend c10d --rdzv-endpoint "$QA_MASTER_ADDR:$QA_MASTER_PORT" \
    --rdzv-id "$QA_JOB_ID-$label" --max-restarts 0 "$target" "$@"
fi
exec python "$target" "$@"
