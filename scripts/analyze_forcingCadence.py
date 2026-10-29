#!/usr/bin/env python3
"""Compare native work with accepted-state power and physical-clock decimations.

Requires the qualification probe's --instantaneous-power history. Does not
allocate CUDA or infer native compatible work from snapshot quadrature.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
import torch

repo=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sample_clock',repo/'src/warpSPH/utils/turbulenceBudget.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def integral(t,value,lo,hi):
    times=np.r_[lo,t[(t>lo)&(t<hi)],hi]
    return float(np.trapezoid(np.interp(times,t,value),times))

def analyze(run,baseline,window):
    h=[json.loads(s) for s in (run/'history.jsonl').read_text().splitlines()]
    b=[json.loads(s) for s in (baseline/'history.jsonl').read_text().splitlines()]
    lo,hi=window;t=np.array([r['t'] for r in h]);T=hi-lo
    if not t[0]<=lo<hi<=t[-1] or np.any(np.diff(t)<=0):raise ValueError('strict accepted times must cover assessment window')
    checked=('t','kineticEnergy','thermalEnergy','totalEnergy','forceWork','thermalExchange','minEnergy','minDensity','minPressure')
    parity={k:len(h)==len(b) and all(x[k]==y[k] for x,y in zip(h,b)) for k in checked}
    ca=json.loads((run/'input.json').read_text());cb=json.loads((baseline/'input.json').read_text())
    for c in (ca,cb):c['simulation'].pop('exportRoot')
    arrayParity={}
    for name in ('initial.npz','final.npz'):
        with np.load(run/name) as a,np.load(baseline/name) as z:
            arrayParity[name]={k:np.array_equal(a[k],z[k]) for k in a.files}
    fa=torch.load(run/'forcing.pt',map_location='cpu',weights_only=True);fb=torch.load(baseline/'forcing.pt',map_location='cpu',weights_only=True)
    forcingEqual=fa['parameters']==fb['parameters'] and torch.equal(fa['knots'],fb['knots']) and torch.equal(fa['generator'],fb['generator'])
    native={k:float(np.interp(hi,t,[r[k] for r in h])-np.interp(lo,t,[r[k] for r in h])) for k in ('forceWork','thermalExchange','kineticEnergy','totalEnergy')}
    rows=[]
    for cadence in (None,.00625,.0125,.025,.05,.1,.2,.4):
        if cadence is None:indices=list(range(len(h)));missed=0
        else:
            clock=module.AcceptedSampleClock(cadence);indices=[i for i,row in enumerate(h) if clock.due(row['t'])];missed=clock.missedTargets
            if indices[-1]!=len(h)-1:indices.append(len(h)-1)
        ti=t[indices];row=dict(requestedInterval=cadence,samples=len(indices),missedTargets=missed)
        for power,work in (('instantaneousSourcePower','forceWork'),('instantaneousSourcePowerDoubleDot','forceWork'),('instantaneousBathPower','thermalExchange')):
            q=integral(ti,np.array([r[power] for r in h])[indices],lo,hi)
            row[power]=dict(integral=q,mean=q/T,nativeWork=native[work],difference=q-native[work],relativeDifference=(q-native[work])/abs(native[work]) if native[work] else None)
        rows.append(row)
    return dict(window=window,configurationEqualExceptOutput=ca==cb,acceptedStateParity=parity,arrayParity=arrayParity,forcingCheckpointEqual=forcingEqual,
        measurementOnlyReplay=ca==cb and all(parity.values()) and all(all(x.values()) for x in arrayParity.values()) and forcingEqual,
        native=native,cadences=rows,acceptedRows=len(h),evidence={str(p):digest(p) for root in (run,baseline) for p in (root/'input.json',root/'history.jsonl',root/'forcing.pt',root/'initial.npz',root/'final.npz')},
        interpretation='Remaining accepted-power/native difference includes finite timestep source quadrature and compatible half-kick semantics; no identification as physical viscosity.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('baseline',type=Path);p.add_argument('--window',nargs=2,type=float,default=[10.,40.]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.run,a.baseline,a.window);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','arrayParity')},indent=2))
