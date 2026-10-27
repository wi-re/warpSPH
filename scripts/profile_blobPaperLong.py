"""Recorded short checkpoint replay with a six-step CPU/CUDA profiler window."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys


def main(parent,root,lanes=32,profile_enabled=True):
    import numpy as np
    root.mkdir(parents=True,exist_ok=False)
    checkpoint=json.loads((parent/'checkpoints/latest.json').read_text())
    shutil.copyfile(checkpoint['path'],root/'input.npz');shutil.copyfile(parent/'tables.npz',root/'tables.npz')
    s=json.loads((parent/'settings.json').read_text());s.update(end=.01,requestedKH=.01/s['tKH'],steps=20,
        inputPath=str(root/'input.npz'),idealTable=str(root/'tables.npz'))
    (root/'settings.json').write_text(json.dumps(s,indent=2)+'\n');(root/'sourceCheckpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
    sys.path.insert(0,'/home/lu26029/dev/warpSPHRemote/python')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPHCore.autograd.lanes import setNeighborLanes
    setNeighborLanes(lanes)
    import warpSPH.runner as runner
    from warpsph_remote.publisher import SimulationPublisher
    import warpsph_remote.publisher as publisher_module
    import run_blobPaperLong as long
    original=runner.run;profiler={}
    class ProbePublisher(SimulationPublisher):
        def __init__(self,name,**kwargs):super().__init__('Performance probe | n512 | lanes '+str(lanes),**kwargs)
    publisher_module.SimulationPublisher=ProbePublisher
    def run(case,**kw):
        post=case.postStep
        def mark(c,state,step):
            post(c,state,step)
            if profile_enabled and step==4:
                torch.cuda.synchronize()
                profiler['p']=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                profiler['p'].start()
            if profile_enabled and step==10:
                torch.cuda.synchronize();p=profiler['p'];p.stop();p.export_chrome_trace(str(root/'trace.json'))
                (root/'profile.txt').write_text(p.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
                rows=[]
                for event in p.key_averages():
                    rows.append(dict(name=event.key,calls=event.count,cpuTotalUs=event.cpu_time_total,cpuSelfUs=event.self_cpu_time_total,
                        cudaTotalUs=event.device_time_total,cudaSelfUs=event.self_device_time_total))
                (root/'profile.json').write_text(json.dumps(dict(profileSteps=6,sections=rows),indent=2)+'\n')
        return original(replace(case,postStep=mark),**dict(kw,nSteps=20))
    runner.run=run
    long.run_case(root)
    save=dict(lanes=lanes,profileEnabled=profile_enabled,checkpoint=checkpoint)
    (root/'benchmark.json').write_text(json.dumps(save,indent=2)+'\n')
    print('Benchmark saved:',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--lanes',type=int,choices=(1,2,4,8,16,32),default=32);p.add_argument('--profile',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();main(a.parent.resolve(),a.output.resolve(),a.lanes,a.profile)
