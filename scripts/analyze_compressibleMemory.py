#!/usr/bin/env python3
"""CPU-only report for a finite cross-scheme startup memory campaign."""
from pathlib import Path
import argparse,collections,hashlib,json,pickle,time
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--wait',action='store_true');a=p.parse_args();r=a.root.resolve();out=r/'analysis';out.mkdir(exist_ok=True)
def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
if a.wait:
 write(r/'analysis-status.json',dict(phase='waiting for finite GPU queue'))
 while True:
  try:
   status=json.loads((r/'status.json').read_text())
   if status['phase'].startswith('complete') or status['phase'] in ('stopped','failed'):break
  except FileNotFoundError:pass
  time.sleep(20)
plan=json.loads((r/'manifest.json').read_text());status=json.loads((r/'status.json').read_text());sourceBad=[name for name,sha in plan['sourceHashes'].items() if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha]
rows=[];captures=[];telemetry=[];failures=[]
for job in plan['jobs']:
 f=Path(job['result']);state=next((x for x in status['jobs'] if x['name']==job['name']),{})
 if not f.exists():failures.append(dict(job=job['name'],status=state.get('status','not run'),reason=state.get('reason',state.get('stopReason'))));continue
 d=json.loads(f.read_text());folder=f.parent
 if d['status']!='completed' or state.get('status')!='passed':failures.append(dict(job=job['name'],status=state.get('status'),result=d['status']))
 own=json.loads((folder/'final-ownership.json').read_text());initial=json.loads((folder/'initial-ownership.json').read_text());adj=json.loads((folder/'neighbor-counts.json').read_text())
 row=dict(job=job['name'],scheme=d['scheme'],nx=d['nx'],particles=d['particles'],instrumented=d['instrumented'],status=d['status'],seconds=d['totalWallSeconds'],loopMs=d['loopMsPerStep'],steadyMs=d['steadyWallMsPerStep'],nativeMs=d['nativeMeanMs'],actualTime=d['actualTime'],steps=d['nSteps'],memoryGiB={k:v/2**30 for k,v in d['memory'].items()},peakAllocatedKiBPerParticle=d['memory']['peakAllocated']/1024/d['particles'],initialGroups=initial['groups'],finalGroups=own['groups'],finalReachableGiB=own['unionBytes']/2**30,neighbors={k:{j:v[j] for j in ('edges','capacity','indexDtype')} for k,v in adj.items()},stageChecks=d['stageChecks'],extraRHS=d['extraRHS'],budget=d['maxAbsoluteEnergyBudgetResidual'])
 if not d['instrumented']:rows.append(row)
 else:
  from torch.cuda import _memory_viz
  with (folder/'allocator.pickle').open('rb') as stream:s=pickle.load(stream)
  (out/(d['scheme']+'-memory-timeline.html')).write_text(_memory_viz.trace_plot(s));(out/(d['scheme']+'-allocator-state.html')).write_text(_memory_viz.segment_plot(s))
  points=json.loads((folder/'allocation-points.json').read_text());peak=json.loads((folder/'peak-ownership.json').read_text());events=collections.Counter(x['action'] for x in s['device_traces'][0]);pending={};dist=[]
  for index,e in enumerate(s['device_traces'][0]):
   if e['action']=='free_requested':pending[e['addr']]=index
   if e['action']=='free_completed' and e['addr'] in pending:dist.append(index-pending.pop(e['addr']))
  before=r/'runs'/f"{d['scheme']}-96"/'step6-hashes.json';after=folder/'step6-hashes.json';prefix=(json.loads(before.read_text())==json.loads(after.read_text()))
  row.update(points=points,events=dict(events),pendingFreeRequests=len(pending),nonAdjacentFreeCompletions=sum(x>1 for x in dist),prefix6Bitwise=prefix,peakRegion=peak.get('region'),peakReachable=peak.get('inventory',{}),recordedSegmentCount=len(s['segments']))
  captures.append(row)
 for log in [r/'logs'/(job['name']+'.gpu.jsonl')]:
  if log.exists():
   samples=[json.loads(x) for x in log.read_text().splitlines()];telemetry.append(dict(job=job['name'],processPeakGiB=max((x['processMiB'] or 0) for x in samples)/1024,unrelatedCompute=[x for x in samples if x['unrelatedCompute']]))
summary=dict(queuePhase=status['phase'],sourcePins=len(plan['sourceHashes']),sourceMismatches=sourceBad,measurements=rows,captures=captures,telemetry=telemetry,failures=failures)
write(out/'summary.json',summary)
# Static scientific figure, detached from simulation processes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
schemes=list(dict.fromkeys(x['scheme'] for x in rows));fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for scheme in schemes:
 points=sorted([x for x in rows if x['scheme']==scheme],key=lambda x:x['nx'])
 axes[0].plot([x['particles'] for x in points],[x['memoryGiB']['peakAllocated'] for x in points],'-o',label=scheme)
 axes[1].plot([x['particles'] for x in points],[x['peakAllocatedKiBPerParticle'] for x in points],'-o',label=scheme)
axes[0].set_ylabel('Peak live Torch allocation (GiB)');axes[1].set_ylabel('Peak live allocation (KiB / particle)')
for ax in axes:ax.set_xscale('log');ax.set_xlabel('Particles (3D startup)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'memory-scaling.png',dpi=160);plt.close(fig)
lines=['# Native 3D compressible startup memory sweep','',f"Queue: {status['phase']}. Source mismatches: {len(sourceBad)}. Failures/skips: {len(failures)}.",'','Short startup windows; native kernels, supports and timesteps differ. Instrumented captures are separate from throughput. These are not evolved-turbulence bounds.','','|Scheme|nx|Particles|Live peak GiB|Reserved peak GiB|KiB/particle|Steady ms/step|Run seconds|Final t|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in rows:lines.append(f"|{x['scheme']}|{x['nx']}|{x['particles']:,}|{x['memoryGiB']['peakAllocated']:.3f}|{x['memoryGiB']['peakReserved']:.3f}|{x['peakAllocatedKiBPerParticle']:.2f}|{x['steadyMs']:.2f}|{x['seconds']:.2f}|{x['actualTime']:.5f}|")
lines+=['','## Separate 96³ captures','','|Scheme|Live peak GiB|Native/capture step6 fields bitwise|Post-run reachable GiB|Pending frees|Instrumented wall seconds|','|---|---:|---|---:|---:|---:|']
for x in captures:lines.append(f"|{x['scheme']}|{x['memoryGiB']['peakAllocated']:.3f}|{x['prefix6Bitwise']}|{x['finalReachableGiB']:.3f}|{x['pendingFreeRequests']}|{x['seconds']:.2f}|")
lines+=['','![Memory scaling](memory-scaling.png)','','See summary.json for ownership, edges/capacity, stage guards, native region peaks, exact allocator block states and GPU contention samples. Raw snapshots and native memory-viz HTML are retained locally.']
if failures:lines+=['','## Failures retained','',json.dumps(failures,indent=2)]
(out/'report.md').write_text('\n'.join(lines)+'\n');write(r/'analysis-status.json',dict(phase='complete',sourceMismatches=sourceBad,failures=failures,report=str(out/'report.md')))
print(json.dumps(dict(phase='complete',runs=len(rows),captures=len(captures),failures=failures,sourceMismatches=sourceBad)),flush=True)
