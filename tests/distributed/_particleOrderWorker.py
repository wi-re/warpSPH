"""Two-rank owned-row Morton/rebuild/restart controls, with an empty owner."""
import argparse
from datetime import timedelta
from pathlib import Path
import json,os,sys,time

import torch
import torch.distributed as dist
import warp as wp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_chunkedParticleOrder import fixture,compare
from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
from warpSPHCore.distributed import RowChunkProcessGroup
from warpSPH.distributedRun import _sourcePins,_runtime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    os.environ['WARPSPH_TEST_DEVICE']=args.device
    if args.device.startswith('cuda'):torch.cuda.set_device(args.device)
    wp.init()
    dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo',timeout=timedelta(seconds=120))
    comm=RowChunkProcessGroup(dist.group.WORLD,args.device,replicaChecks='exact')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    records=[];started=time.monotonic()
    try:
        for scheme in ('CompSPH','CRKSPH','deltaSPH','sun2017DeltaSPH','iisph','dfsph','dfsphPredictor','dfsphShift'):
            print(f'rank{comm.rank} {scheme} reference',flush=True)
            cfg,case=fixture(scheme)
            full=cfg.run(case)
            def empty(ctx,system,completed):
                if comm.rank==1:assert not system.adjacency.chunks
            cfg,case=fixture(scheme,communication=comm,rows=1<<20,steps=2,accepted=empty)
            first=cfg.run(case)
            path=output/(scheme+'.h5')
            writeChunkedCheckpoint(path,first.ctx,first.state)
            def active(ctx,system,completed):assert system.adjacency.chunks
            cfg,case=fixture(scheme,communication=comm,rows=1,steps=2,resume=path,accepted=active)
            resumed=cfg.run(case)
            comm.localPhase('integration:compare',lambda:compare(full,resumed))
            records.append(dict(scheme=scheme,referenceWall=full.wallTime,prefixWall=first.wallTime,
                resumedWall=resumed.wallTime,sorts=resumed.ctx.scratch['nativeLayout'].sorts))
            print(f'rank{comm.rank} {scheme} passed',flush=True)
        data=dict(status='passed',wallSeconds=time.monotonic()-started,runtime=_runtime(comm),sources=_sourcePins(),records=records)
        (output/f'rank-{comm.rank}.json').write_text(json.dumps(data,indent=2))
        print('particle-order integration passed',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
