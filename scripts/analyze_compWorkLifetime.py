#!/usr/bin/env python3
"""Offline independent CompSPH work-storage and runner-lifetime comparisons."""
from pathlib import Path
import argparse,hashlib,json,pickle,collections
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();r=a.root.resolve();out=r/'analysis';out.mkdir(exist_ok=True)
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
plan=json.loads((r/'manifest.json').read_text());queue=json.loads((r/'status.json').read_text());rows=[];parity=[];captures=[]
sourceBad=[f for f,sha in plan['sourceHashes'].items() if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=sha]
for job in plan['jobs']:
 if job.get('kind')!='memory-window':continue
 f=Path(job['result'])
 if not f.exists():continue
 d=json.loads(f.read_text());row=dict(job=job['name'],result=d)
 tel=r/'logs'/(job['name']+'.gpu.jsonl')
 if tel.exists():
  samples=[json.loads(x) for x in tel.read_text().splitlines()]
  row.update(processPeakGiB=max((x['processMiB'] or 0) for x in samples)/1024,contended=any(x['unrelatedCompute'] for x in samples))
 rows.append(row)
 if d['instrumented']:
  from torch.cuda import _memory_viz
  with (f.parent/'allocator.pickle').open('rb') as stream:s=pickle.load(stream)
  (out/(job['name']+'-memory-timeline.html')).write_text(_memory_viz.trace_plot(s))
  (out/(job['name']+'-allocator-state.html')).write_text(_memory_viz.segment_plot(s))
  events=collections.Counter(e['action'] for e in s['device_traces'][0]);pending={};delays=[]
  for index,e in enumerate(s['device_traces'][0]):
   if e['action']=='free_requested':pending[e['addr']]=index
   if e['action']=='free_completed' and e['addr'] in pending:delays.append(index-pending.pop(e['addr']))
  captures.append(dict(job=job['name'],events=dict(events),pendingFrees=len(pending),nonAdjacentFrees=sum(x>1 for x in delays),allocationPoints=json.loads((f.parent/'allocation-points.json').read_text()),finalOwnership=json.loads((f.parent/'final-ownership.json').read_text()),peakObserver=json.loads((f.parent/'peak-ownership.json').read_text())))

def compare(left,right,expectBitwise):
 lp=r/'runs'/left/'final-fields.npz';rp=r/'runs'/right/'final-fields.npz'
 if not lp.exists() or not rp.exists():return
 fields={}
 with np.load(lp) as x,np.load(rp) as y:
  for key in sorted(set(x.files)|set(y.files)):
   if key not in x or key not in y:fields[key]=dict(missing=True,bitwise=False,close=False);continue
   av,bv=x[key],y[key]
   same=np.array_equal(av,bv)
   close=av.shape==bv.shape and np.allclose(av,bv,rtol=3e-6,atol=3e-7)
   delta=np.abs(av.astype(float)-bv.astype(float));scale=max(float(np.abs(av.astype(float)).max(initial=0)),1e-30)
   fields[key]=dict(bitwise=bool(same),close=bool(close),maxAbsolute=float(delta.max(initial=0)),maxScaled=float(delta.max(initial=0)/scale))
 work=None;wa=lp.parent/'work-sample.npz';wb=rp.parent/'work-sample.npz'
 if wa.exists() and wb.exists():
  with np.load(wa) as x,np.load(wb) as y:work=dict(bitwise=all(np.array_equal(x[k],y[k]) for k in x.files),maxAbsolute=float(np.abs(x['work']-y['work']).max(initial=0)),note='Sample order/candidate counts retained; complete pair IDs are not archived. Physical operator controls compare full live work separately.')
 parity.append(dict(left=left,right=right,expectBitwise=expectBitwise,allBitwise=all(x['bitwise'] for x in fields.values()),allClose=all(x['close'] for x in fields.values()),fields=fields,work=work))
for nx in [32,96]:
 prefix=f'CompSPH-{nx}-'
 for variant,exact in [('operator-expanded',True),('operator',False),('lifetime',True),('combined',False)]:compare(prefix+'before',prefix+variant,exact)
 compare(prefix+'operator',prefix+'combined',True)
for scheme in ['CRKSPH','Monaghan','GSPH','InutsukaGSPH','MFM','MFV']:compare(scheme+'-96-before',scheme+'-96-lifetime',True)
for scheme,variant in [('CompSPH','before'),('CompSPH','combined'),('Monaghan','lifetime'),('MFM','lifetime'),('MFV','lifetime')]:compare(f'{scheme}-96-{variant}',f'{scheme}-96-{variant}-capture',True)
accuracy={}
for variant in ['before','combined']:
 f=r/('accuracy-'+variant)/'result.json'
 if f.exists():accuracy[variant]=json.loads(f.read_text())
accuracyComparison=[]
if all(k in accuracy and accuracy[k]['status']=='passed' for k in ['before','combined']):
 for case in ['sod','gresho']:
  for name,base in accuracy['before']['metrics'][case].items():
   value=accuracy['combined']['metrics'][case][name]
   # Storage-only nonregression gate; same cases/metrics as the retained references.
   if name=='e_drift':passed=value<1e-5
   elif name=='shock_width':passed=value<=base+1
   else:passed=value<=base+.01*abs(base)+1e-7 if name!='ke_final_rel' else abs(value)<=abs(base)*1.01+1e-7
   accuracyComparison.append(dict(case=case,metric=name,before=base,combined=value,passed=bool(passed)))
failures=[dict(job=x['name'],status=x['status'],reason=x.get('reason',x.get('stopReason'))) for x in queue['jobs'] if x['status'] in ('failed','skipped','stopped')]
result=dict(queuePhase=queue['phase'],sourcePins=len(plan['sourceHashes']),sourceMismatches=sourceBad,failures=failures,windows=rows,parity=parity,captures=captures,accuracy=accuracy,accuracyComparison=accuracyComparison)
write(out/'comparison.json',result)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
names=['before','operator','lifetime','combined'];values=[]
for v in names:
 x=next((x['result'] for x in rows if x['job']=='CompSPH-96-'+v),None);values.append(x['memory']['peakAllocated']/2**30 if x else np.nan)
axes[0].bar(names,values);axes[0].set_ylabel('96³ peak live Torch allocation (GiB)');axes[0].set_title('CompSPH: independent changes')
for j,v in enumerate(values):
 if np.isfinite(v):axes[0].text(j,v+.3,f'{v:.2f}',ha='center')
schemes=['CRKSPH','Monaghan','GSPH','InutsukaGSPH','MFM','MFV'];xx=np.arange(len(schemes))
for offset,v in [(-.2,'before'),(.2,'lifetime')]:
 vals=[]
 for scheme in schemes:
  x=next((x['result'] for x in rows if x['job']==scheme+'-96-'+v),None);vals.append(x['memory']['peakAllocated']/2**30 if x else np.nan)
 axes[1].bar(xx+offset,vals,.4,label=v)
axes[1].set_xticks(xx,schemes,rotation=20);axes[1].set_ylabel('96³ peak live Torch allocation (GiB)');axes[1].set_title('Previous-step stage lifetime only');axes[1].legend();fig.tight_layout();fig.savefig(out/'memory-comparison.png',dpi=160);plt.close(fig)
lines=['# CompSPH native work and stage-lifetime comparison','',f"Queue: {queue['phase']}; source mismatches: {len(sourceBad)}; failed jobs: {len(failures)}.",'','Contended timings are observations, not throughput claims. Primary/auxiliary numerical screens and allocation captures are retained separately.','','|Window|Live peak GiB|Reserved peak GiB|Run wall s|Steady ms/step|Contended|','|---|---:|---:|---:|---:|---|']
for x in rows:
 d=x['result'];lines.append(f"|{x['job']}|{d['memory']['peakAllocated']/2**30:.3f}|{d['memory']['peakReserved']/2**30:.3f}|{d['totalWallSeconds']:.3f}|{d['steadyWallMsPerStep']:.2f}|{x.get('contended')}|")
lines+=['','![Independent changes](memory-comparison.png)','','## Numerical screens','','|Pair|Bitwise required|All stored fields bitwise|All within declared3e-6/3e-7|','|---|---|---|---|']
for x in parity:lines.append(f"|{x['left']} → {x['right']}|{x['expectBitwise']}|{x['allBitwise']}|{x['allClose']}|")
lines+=['','Actual native Sod/Gresho metrics and1% nonregression screens:','','```json',json.dumps(accuracyComparison,indent=2),'```','','Region inventories describe observer boundaries, not necessarily the allocating point of a previously reached global peak. Use native allocation stacks for transient attribution. No cache flushing or neighbour-format changes were introduced.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(phase=queue['phase'],windows=len(rows),captures=len(captures),sourceMismatches=sourceBad,failedExact=[(x['left'],x['right']) for x in parity if x['expectBitwise'] and not x['allBitwise']],failedClose=[(x['left'],x['right'],[k for k,v in x['fields'].items() if not v['close']]) for x in parity if not x['allClose']],accuracyFailed=[x for x in accuracyComparison if not x['passed']],failedJobs=failures)),flush=True)
