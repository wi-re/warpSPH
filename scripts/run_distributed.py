#!/usr/bin/env python3
"""Run the explicit eager row-chunk path under torchrun (JSON configuration)."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',type=Path,required=True)
    parser.add_argument('--backend',choices=('gloo','nccl'),required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh shared output directory')
    parser.add_argument('--edge-budget',type=int,required=True)
    parser.add_argument('--max-rows',type=int)
    parser.add_argument('--checkpoint-every',type=int,default=0)
    parser.add_argument('--offset-dtype',choices=('int32','int64'),default='int32')
    parser.add_argument('--timeout-seconds',type=float,default=120.)
    parser.add_argument('--shared-gpu',action='store_true',help='Experimental two-process/single-GPU NCCL debug mode')
    args=parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds<=0:
        parser.error('--timeout-seconds must be positive and finite')
    document=json.loads(args.configuration.read_text())
    precision=document.get('simulation',{}).get('precision','float32')
    if precision not in ('float32','float64'):
        parser.error('This launcher currently qualifies float32/float64 only')
    os.environ['warpSPHCore_PRECISION']=precision
    os.environ['warpSPHCore_OFFSET_DTYPE']=args.offset_dtype
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    import torch
    import torch.distributed as dist
    from warpSPH.runner import RunConfiguration
    from warpSPH.distributedRun import runDistributed
    from warpSPHCore.distributed import RowChunkProcessGroup
    if args.shared_gpu and args.backend!='nccl':
        parser.error('--shared-gpu requires NCCL')
    if args.backend=='nccl':
        if torch.version.hip is not None:
            parser.error('ROCm builds are deferred')
        localRanks=int(os.environ.get('LOCAL_WORLD_SIZE','1'))
        if args.shared_gpu:
            import ctypes
            version=ctypes.c_int()
            ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(version))
            if version.value<23000 or os.environ.get('NCCL_MULTI_RANK_GPU_ENABLE')!='1' or localRanks>2:
                parser.error('Shared-GPU debug mode requires NCCL >=2.30, NCCL_MULTI_RANK_GPU_ENABLE=1 and at most two local ranks')
            device=torch.device('cuda',0)
        else:
            if localRanks>torch.cuda.device_count():
                parser.error('One visible GPU per local rank is required; use --shared-gpu only for the qualified debug exception')
            device=torch.device('cuda',int(os.environ.get('LOCAL_RANK','0')))
        torch.cuda.set_device(device)
    else:
        device=torch.device('cpu')
    dist.init_process_group(args.backend,timeout=timedelta(seconds=args.timeout_seconds))
    try:
        communication=RowChunkProcessGroup(dist.group.WORLD,device)
        runDistributed(RunConfiguration.fromDict(document),communication,outputDirectory=args.output,
            edgeBudget=args.edge_budget,maxRows=args.max_rows,checkpointEvery=args.checkpoint_every)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
