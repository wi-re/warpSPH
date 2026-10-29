#!/usr/bin/env python3
"""Offline comparison of the short serial CRK storage timings and state parity."""
from pathlib import Path
import argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root.resolve()
rows=[];comparisons=[]
for nx in (32,96):
 for mode in ('legacy','expanded','recompute'):
  folder=r/f'timing{nx}-{mode}';reps=[json.loads(p.read_text()) for p in sorted(folder.glob('repeat*/result.json'))]
  if len(reps)!=2:raise RuntimeError(f'Missing two completed repeats: {folder}')
  rows.append(dict(nx=nx,mode=mode,steps=sum(v['nSteps'] for v in reps),
    totalRunWallSeconds=sum(v['totalWallSeconds'] for v in reps),loopWallSeconds=sum(v['loopWallSeconds'] for v in reps),
    steadyWallMs=[v['steadyWallMsPerStep'] for v in reps],nativeMs=[v['nativeMeanMs'] for v in reps],
    meanWallMs=float(np.mean([v['steadyWallMsPerStep'] for v in reps])),meanNativeMs=float(np.mean([v['nativeMeanMs'] for v in reps])),
    peakAllocatedGiB=max(v['peakTorchAllocatedBytes'] for v in reps)/2**30,
    peakReservedGiB=max(v['peakTorchReservedBytes'] for v in reps)/2**30,
    maxBudget=max(v['maxAbsoluteEnergyBudgetResidual'] for v in reps),
    valid=all(v['status']=='completed' and v['stageChecks']['bad']==0 and v['extraRHS']==0 and v['acceptedPositiveFinite'] and v['sourcesStable'] for v in reps)))
  for rep in range(2):
   if mode=='legacy' and rep==0:continue
   reference=r/f'timing{nx}-legacy/repeat0'
   target=folder/f'repeat{rep}'
   for endpoint in ('initial','final','work-sample'):
    f=reference/f'{endpoint}.npz';g=target/f'{endpoint}.npz'
    if not f.exists() or not g.exists():raise RuntimeError('Missing comparison artifact')
    with np.load(f) as x,np.load(g) as y:
     if endpoint=='work-sample' and (not np.array_equal(x['indices'],y['indices']) or not np.array_equal(x['live'],y['live'])):
      comparisons.append(dict(nx=nx,mode=mode,repeat=rep,endpoint=endpoint,passed=None,comparable=False,reason='Different live-edge list/linear sample identities after roundoff-level state differences; do not compare unrelated pairs.'))
      continue
     diffs={};passed=True
     for k in x.files:
      if k not in y.files or x[k].shape!=y[k].shape:raise RuntimeError(f'field mismatch: {k}')
      v=x[k];w=y[k];absolute=np.abs(v.astype(float)-w.astype(float));mx=float(absolute.max(initial=0));scale=float(np.abs(v.astype(float)).max(initial=0))
      pointwise=np.allclose(v,w,rtol=3e-6,atol=3e-7,equal_nan=False)
      close=pointwise if k!='divergence' else mx<=3e-7+3e-6*scale
      passed=passed and close
      diffs[k]=dict(maxAbsolute=mx,normalizedMax=mx/max(scale,1e-30),bitwise=bool(np.array_equal(v,w)),close=bool(close),pointwiseClose=bool(pointwise))
     comparisons.append(dict(nx=nx,mode=mode,repeat=rep,endpoint=endpoint,passed=bool(passed),fields=diffs))
ratios=[]
for nx in (32,96):
 base=next(x for x in rows if x['nx']==nx and x['mode']=='expanded')
 new=next(x for x in rows if x['nx']==nx and x['mode']=='recompute')
 old=next(x for x in rows if x['nx']==nx and x['mode']=='legacy')
 ratios.append(dict(nx=nx,recomputeVsExpandedWall=new['meanWallMs']/base['meanWallMs'],recomputeVsExpandedNative=new['meanNativeMs']/base['meanNativeMs'],
    expandedVsLegacyWall=base['meanWallMs']/old['meanWallMs'],memoryAllocatedRatio=new['peakAllocatedGiB']/base['peakAllocatedGiB']))
mem=[]
for mode in ('expanded','recompute'):
 folder=r/f'memory96-{mode}-recovery';s=json.loads((folder/'result.json').read_text());regions=json.loads((folder/'regions.json').read_text())
 peak=max(regions,key=lambda v:v['peakAllocated'])
 mem.append(dict(mode=mode,status=s['status'],wallSeconds=s['wallSeconds'],msPerStep=s['msPerStep'],instrumented=True,
    peakRegion=peak['name'],peakStep=peak['step'],peakStage=peak['stage'],peakAllocatedGiB=peak['peakAllocated']/2**30,
    peakRegionBeforeGiB=peak['before']['allocated']/2**30,peakRegionAfterGiB=peak['after']['allocated']/2**30,
    stats=s['stats'],maxBudget=s['maxBudget'],sourcesStable=s['sourcesStable']))
telemetry=[]
for f in list((r/'logs').glob('*.gpu.jsonl'))+list((r/'memory-recovery/logs').glob('*.gpu.jsonl')):
 records=[json.loads(x) for x in f.read_text().splitlines()]
 telemetry.append(dict(job=str(f.relative_to(r)),processPeakGiB=max((v['processMiB'] or 0) for v in records)/1024,
    unrelated=[v for v in records if v['unrelatedCompute']]))
report=dict(rows=rows,ratios=ratios,comparisons=comparisons,memoryProbes=mem,telemetry=telemetry,
    allTimingGuardsPass=all(v['valid'] for v in rows),
    allStateComparisonsPass=all(v['passed'] for v in comparisons if v['passed'] is not None),
    allPrimaryStateComparisonsPass=all(f['close'] for v in comparisons if v['passed'] is not None for name,f in v['fields'].items() if name!='divergence'),
    incompleteWorkComparisons=sum(v['passed'] is None for v in comparisons))
(r/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},indent=2))
