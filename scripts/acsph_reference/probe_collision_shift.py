#!/usr/bin/env python3
"""Bounded reference-code diagnostic: add only native wall-collision shifting.

Keep the local Michel shift, BDF correction and fields unchanged. Native's
surface-cohesion shift, coefficient and p/v remap are intentionally separate.
This is not a production model change or a complete native shifting port.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys


def collision_displacement(x, mass, kinds, pair_i, pair_j, dx):
    import torch
    keep=(kinds[pair_i]==0)&(kinds[pair_j]==1)
    i,j=pair_i[keep],pair_j[keep]
    offset=x[i]-x[j];distance=offset.norm(dim=-1)
    keep=distance < .9*dx;i,j,offset,distance=i[keep],j[keep],offset[keep],distance[keep]
    if bool((distance==0).any()):
        raise ValueError('Coincident fluid/wall points in native collision shift')
    weighted=mass[j,None]*(distance-.9*dx)[:,None]*offset/distance[:,None]
    numerator=torch.zeros_like(x).index_add_(0,i,weighted)
    denominator=mass.clone().index_add_(0,i,mass[j])
    return -numerator/denominator[:,None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    a=p.parse_args();runtime=a.runtime.resolve();out=a.out.resolve()
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch
    import warpSPH.systems.artificialCompressible as system
    original=system.solveShifting
    count=0
    def shifting(*args,**kwargs):
        nonlocal count
        state=kwargs['systemState'];adjacency=kwargs['adjacency'];config=kwargs['config']
        delta=original(*args,**kwargs)
        i,j=(v.long() for v in adjacency.livePairs())
        collision=collision_displacement(state.positions,state.masses,state.kinds,i,j,float(config.dx))
        count+=1
        f=state.kinds==0
        magnitude=collision[f].norm(dim=-1)
        row=dict(step=count,affected=int((magnitude>0).sum()),maxDisplacementDx=float(magnitude.max()/config.dx))
        with (out/'collision.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
        if count%25==0:print('COLLISION',json.dumps(row),flush=True)
        return delta+collision
    system.solveShifting=shifting
    script=runtime/'scripts/acsph_reference/compare_lobovsky.py'
    sys.argv=[str(script),'--reference',str(a.reference.resolve()),'--out',str(out),'--steps',str(a.steps),
        '--long-run','--plot-interval','10','--precision','float32','--position-form','increment','--fused-residual',
        '--coefficients','mapped','--history','source','--wall-diffusion','source','--acoustic-damper','source',
        '--rk-treatment','source','--wall-velocity','source','--wall-bounce-ramp','0.5']
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
