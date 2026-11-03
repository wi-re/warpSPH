#!/usr/bin/env bash
# Example only: request node/GPU/time/account/partition resources with sbatch.
# One torchrun launcher per allocated node; one process per visible GPU.
# Site modules, Arm64 environment, networking and resource flags need Isambard validation.
set -euo pipefail
: "${FRONTEND_ROOT:?Absolute frontend checkout required}"
: "${CORE_ROOT:?Absolute paired core checkout required}"
: "${PYTHON:?Absolute target-node Python required}"
: "${CONFIGURATION:?Absolute JSON configuration required}"
: "${OUTPUT:?Fresh shared output directory required}"
: "${GPUS_PER_NODE:?Number of allocated visible GPUs per node required}"
: "${EDGE_BUDGET:?Per-chunk edge budget required}"
: "${SLURM_JOB_ID:?Submit using sbatch}"
: "${SLURM_NNODES:?Slurm node count required}"
: "${SLURM_JOB_NODELIST:?Slurm node list required}"
export FRONTEND_ROOT CORE_ROOT PYTHON CONFIGURATION OUTPUT GPUS_PER_NODE EDGE_BUDGET
export PYTHONPATH="$FRONTEND_ROOT/src:$CORE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mapfile -t task_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${task_nodes[0]}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
export OFFSET_DTYPE="${OFFSET_DTYPE:-int32}"
export REPLICA_CHECKS="${REPLICA_CHECKS:-exact}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c '
  exec "$PYTHON" -m torch.distributed.run --nnodes "$SLURM_NNODES" \
    --nproc-per-node "$GPUS_PER_NODE" --node-rank "$SLURM_NODEID" \
    --rdzv-backend c10d --rdzv-id "$SLURM_JOB_ID" \
    --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
    "$FRONTEND_ROOT/scripts/run_distributed.py" --configuration "$CONFIGURATION" \
    --backend nccl --output "$OUTPUT" --edge-budget "$EDGE_BUDGET" \
    --offset-dtype "$OFFSET_DTYPE" --checkpoint-every "$CHECKPOINT_EVERY" --replica-checks "$REPLICA_CHECKS"
'
