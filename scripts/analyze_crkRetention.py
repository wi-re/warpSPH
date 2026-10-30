#!/usr/bin/env python3
"""Offline retention A/B, allocator block states and native Torch memory-viz pages."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,pickle
import numpy as np
from torch.cuda import _memory_viz
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root.resolve();out=r/'analysis';out.mkdir(exist_ok=True)

def block_summary(s):
 states=collections.Counter();free=0;split=0;inactive=[]
 for seg in s['segments']:
  if seg['device']!=0:continue
  if seg['active_size']==0:free+=seg['total_size']
  for b in seg['blocks']:
   states[b['state']]+=b['size']
   if b['state']=='inactive':
    inactive.append(dict(address=b['address'],size=b['size'],wholeSegmentInactive=seg['active_size']==0))
    if seg['active_size']>0:split+=b['size']
 latest={};requested={};delays=[]
 for k,e in enumerate(s['device_traces'][0]):
  if e['action']=='alloc':latest[e['addr']]=e
  elif e['action']=='free_requested':requested[e['addr']]=(k,e['size'])
  elif e['action']=='free_completed':
   item=requested.pop(e['addr'],None)
   if item:delays.append(k-item[0])
 for b in inactive:
  alloc=latest.get(b['address']);b['lastAllocationStack']=alloc.get('frames',[]) if alloc else None
 return dict(states=dict(states),reserved=sum(states.values()),fullyInactiveSegments=free,inactiveInsideActiveSegments=split,
     largestInactive=sorted(inactive,key=lambda x:x['size'],reverse=True)[:8],freeCompletions=len(delays),
     freeCompletionsAdjacent=sum(x==1 for x in delays),freeCompletionsWithInterveningEvents=sum(x>1 for x in delays),pendingFreeRequestsAtSnapshot=len(requested))
rows=[];parity=[]
for nx in (32,96):
 for phase in ('before','after'):
  values=[json.loads((r/f'timing{nx}-{phase}/repeat{k}/result.json').read_text()) for k in range(2)]
  assert all(x['status']=='completed' and x['extraRHS']==0 and x['stageChecks']['bad']==0 and x['sourcesStable'] for x in values)
  rows.append(dict(nx=nx,phase=phase,wallMs=[x['steadyWallMsPerStep'] for x in values],nativeMs=[x['nativeMeanMs'] for x in values],
     meanWallMs=float(np.mean([x['steadyWallMsPerStep'] for x in values])),peakAllocatedGiB=max(x['peakTorchAllocatedBytes'] for x in values)/2**30,
     peakReservedGiB=max(x['peakTorchReservedBytes'] for x in values)/2**30,wallSeconds=sum(x['totalWallSeconds'] for x in values),maxBudget=max(x['maxAbsoluteEnergyBudgetResidual'] for x in values)))
  for rep in range(2):
   target=r/f'timing{nx}-{phase}/repeat{rep}';reference=r/f'timing{nx}-before/repeat0'
   for endpoint in ('initial','final','work-sample'):
    with np.load(reference/f'{endpoint}.npz') as x,np.load(target/f'{endpoint}.npz') as y:
     differences={k:float(np.abs(x[k].astype(float)-y[k].astype(float)).max(initial=0)) for k in x.files if k not in y.files or not np.array_equal(x[k],y[k])}
     parity.append(dict(nx=nx,phase=phase,repeat=rep,endpoint=endpoint,bitwise=not differences,differences=differences))
allocators=[];ownership=[]
for phase in ('before','after'):
 folder=r/f'trace96-{phase}/repeat0'
 with gzip.open(folder/'allocator-history.pickle.gz','rb') as f:s=pickle.load(f)
 alloc=block_summary(s);alloc.update(phase=phase,points=json.loads((folder/'allocation-points.json').read_text()))
 allocators.append(alloc)
 (out/f'{phase}-memory-timeline.html').write_text(_memory_viz.trace_plot(s))
 (out/f'{phase}-allocator-state.html').write_text(_memory_viz.segment_plot(s))
 regions=json.loads((r/f'memory96-{phase}/regions.json').read_text());peak=max(regions,key=lambda x:x['peakAllocated'])
 work=[v for v in peak['ownership']['storages'] if v['bytes']>5e9]
 ownership.append(dict(phase=phase,peakAllocatedGiB=peak['peakAllocated']/2**30,largeWorkStorages=work,
    reachableGiB=peak['ownership']['unionBytes']/2**30,groups=peak['ownership']['groups']))
telemetry=[]
for f in (r/'logs').glob('*.gpu.jsonl'):
 values=[json.loads(x) for x in f.read_text().splitlines()]
 telemetry.append(dict(job=f.name,processPeakGiB=max((x['processMiB'] or 0) for x in values)/1024,unrelatedCompute=[x for x in values if x['unrelatedCompute']]))
result=dict(timings=rows,allBitwise=all(x['bitwise'] for x in parity),parity=parity,allocators=allocators,ownership=ownership,telemetry=telemetry)
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(timings=rows,allBitwise=result['allBitwise'],allocatorStatesGiB=[dict(phase=x['phase'],states={k:v/2**30 for k,v in x['states'].items()},freeAdjacent=x['freeCompletionsAdjacent'],freeDelayed=x['freeCompletionsWithInterveningEvents']) for x in allocators]),indent=2))
