#!/usr/bin/env python3
"""CPU analysis of unique-storage regions and requested CUDA allocation history."""
from pathlib import Path
import argparse,gzip,hashlib,json,pickle,subprocess
from collections import defaultdict
import numpy as np


def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def reconstruct(path):
    with gzip.open(path,'rb') as f:data=pickle.load(f)
    trace=data['device_traces'][0];active={};live=0;peak=0;at=0;held={}
    for i,e in enumerate(trace):
        if e['action']=='alloc':
            assert e['addr'] not in active
            active[e['addr']]=e;live+=e['size']
        elif e['action']=='free_requested':live-=active.pop(e['addr'])['size']
        if live>peak:peak=live;held=active.copy();at=i
    return dict(peakRequested=peak,eventIndex=at,traceEntries=len(trace),allocations=list(held.values()))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root.resolve();output=root/'analysis';output.mkdir(exist_ok=True)
    results=[]
    for name in ('probe32','probe96','probe96-ownership','probe96-fullroots'):
        directory=root/name;result=json.loads((directory/'result.json').read_text());assert result['status']=='passed' and result['sourcesStable'] and result['stats']['bad']==0
        regions=json.loads((directory/'regions.json').read_text());neighbor=json.loads((directory/'neighbor-comparison.json').read_text());accepted=json.loads((directory/'accepted.json').read_text());peak=max(regions,key=lambda x:x['peakAllocated'])
        n=neighbor['particles'];cap=neighbor['bufferedCapacity'];raw=neighbor['bufferedEdges'];physical=neighbor['physicalEdges'];item=4
        metrics=dict(name=name,nx=result['nx'],particles=n,stateOnceBytes=result['initialStateInventory']['unionBytes'],stateBytesPerParticle=result['initialStateInventory']['unionBytes']/n,
            physicalEdges=physical,bufferedEdges=raw,paddedEdges=cap,physicalIndexRowBytes=8*physical+8*n,bufferedUnpaddedIndexRowBytes=8*raw+8*n,bufferedPaddedIndexRowBytes=8*cap+8*n,
            nativeListUniqueBytes=neighbor['ownership']['groups']['buffered']['uniqueBytes'],pairVectorBytes=cap*12,stagePressureViscosityPartitionBytes=cap*28,
            peakAllocated=peak['peakAllocated'],peakRegion=peak['name'],peakStep=peak['step'],peakStage=peak['stage'],peakBeforeBytes=peak['before']['allocated'],peakAfterBytes=peak['after']['allocated'],
            transientAboveBoundaryBytes=peak['transientAboveLargerBoundary'],endRegionReachableBytes=peak['ownership']['unionBytes'],accepted=accepted[-1],result=result)
        trace=reconstruct(directory/'allocator-history.pickle.gz');metrics['peakTrace']=trace
        videos=[p for p in directory.glob('exports/**/*.mp4') if p.parent.name!='images'];assert len(videos)==1
        v=videos[0];info=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(v)],capture_output=True,text=True,check=True).stdout)['streams'][0]
        from PIL import Image,ImageDraw
        indices=sorted(set([0,int(info['nb_frames'])-1]));sheet=Image.new('RGB',(960,len(indices)*505),'white')
        for i,index in enumerate(indices):
            path=output/(name+f'-video-{index}.png');subprocess.run(['ffmpeg','-v','error','-i',str(v),'-vf',f'select=eq(n\\,{index})','-frames:v','1','-y',str(path)],check=True)
            image=Image.open(path).convert('RGB');image.thumbnail((960,480));sheet.paste(image,(0,i*505+25));ImageDraw.Draw(sheet).text((8,i*505+5),f'{name}: frame{index}',fill='black')
        sheet.save(output/(name+'-contact-sheet.png'));metrics['video']=dict(path=str(v),info=info,sha256=hashlib.sha256(v.read_bytes()).hexdigest(),preparedFrames=indices)
        results.append(metrics)
    main=results[-1];regions=json.loads((root/main['name']/'regions.json').read_text());peak=max(regions,key=lambda x:x['peakAllocated'])
    old=[s for s in peak['ownership']['storages'] if any('runnerResult.state.pairWork.work'==e['path'] for e in s['examples'])]
    assert len(old)==1 and old[0]['owners']==['runnerResult'] and old[0]['bytes']==main['pairVectorBytes']
    oldAddress=old[0]['address'];categories=defaultdict(int)
    for e in main['peakTrace']['allocations']:
        names=[f['name'] for f in e.get('frames',[])];files=[f['filename'] for f in e.get('frames',[])]
        if e['addr']==oldAddress:category='Retained first-step X6 vector'
        elif '_pairVector' in names:category='X6 expression temporaries'
        elif any(x.endswith('/crk/accel.py') for x in files) and e['size']==main['pairVectorBytes']:category='Stage pressure + viscosity vectors'
        elif 'computeCompSPHBalanceTerm' in names and e['size']==main['paddedEdges']*4:category='Stage partition fractions'
        elif e['size']==main['paddedEdges']*4 and any('radiusSearch' in x for x in files):category='Neighbour indices'
        else:category='Other live allocations'
        categories[category]+=e['size']
    assert categories['Retained first-step X6 vector']==main['pairVectorBytes']
    assert categories['X6 expression temporaries']==2*main['pairVectorBytes']
    main['peakAllocationCategories']=dict(categories);main['unexpectedRetainedWork']=old[0]
    write(output/'summary.json',dict(runs=results,primary=main['name'],scope='Storage accounting and short instrumented native trajectories only; no optimized solver or long-run peak reproduction.'))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    names=['Physical support','Verlet, unpadded','Verlet + capacity'];values=[main['physicalIndexRowBytes'],main['bufferedUnpaddedIndexRowBytes'],main['bufferedPaddedIndexRowBytes']]
    axes[0].bar(names,np.array(values)/2**30,color=['#377eb8','#ff7f00','#e41a1c']);axes[0].set(ylabel='GiB',title='96³: current two-index + row-metadata format')
    for x,y in enumerate(values):axes[0].text(x,y/2**30+.04,f'{y/2**30:.2f}',ha='center')
    bottom=0
    for name,value in categories.items():axes[1].bar('Peak live storage',value/2**30,bottom=bottom,label=f'{name}: {value/2**30:.2f}GiB');bottom+=value/2**30
    axes[1].set(ylabel='GiB',title='Native X6 accumulation peak: requested allocations');axes[1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    fig.suptitle('CRKSPH memory ownership;96³ startup, native fresh midpoint, n_h4')
    fig.savefig(output/'memory-breakdown.png',dpi=150);fig.savefig(output/'memory-breakdown.pdf');plt.close(fig)
    lines=['# CRKSPH memory ownership measurement (2026-10-29)','',
        'Particle-state storage is small. Pair-sized outputs, expression temporaries and retained compatible-work vectors dominate. The native X6 cross-stage optimization is active. A separate first-step vector is retained through the runner result object; no solver optimization has been applied.','',
        '|Measurement|32³ near t4|96³ startup|','|---|---:|---:|']
    hot=results[0]
    for title,key,unit in [('Particle fields stored once','stateOnceBytes',2**20),('Physical list: indices/rows','physicalIndexRowBytes',2**30),('Verlet list: indices/rows, no padding','bufferedUnpaddedIndexRowBytes',2**30),('Verlet list: indices/rows, padded','bufferedPaddedIndexRowBytes',2**30),('One padded pair vector','pairVectorBytes',2**30),('Stage pair outputs ap/av/f','stagePressureViscosityPartitionBytes',2**30),('Maximum stage live Torch allocation','peakAllocated',2**30),('X6 transient above larger section boundary','transientAboveBoundaryBytes',2**30)]:
        suffix='MiB' if unit==2**20 else 'GiB';lines.append(f'|{title}|{hot[key]/unit:.3f}{suffix}|{main[key]/unit:.3f}{suffix}|')
    lines+=['','Full state here means the actually stored particle fields (80bytes/particle plus16bytes source ledger), excluding adjacency, derived Q output property, stages and grids. Unique storage addresses deduplicate tensors/views/shared references; this is not the64byte/particle export payload.','',
        f'96³ physical-support directed pairs: {main["physicalEdges"]:,}; buffered pairs: {main["bufferedEdges"]:,}; padded capacity: {main["paddedEdges"]:,}. Native indices are int32. The table includes both COO index arrays and CSR count/offset arrays. A single-index CSR representation would have a different cost and is not measured as an implementation change. The physical list uses exact current-position/support filtering, capacity=None; the trajectory keeps its unchanged2**(1/3) Verlet factor and capacity policy. Native list including stored geometry/hash metadata is {main["nativeListUniqueBytes"]/2**30:.3f}GiB; shared geometry is not a second particle state.','',
        '## Peak and retained ownership','',
        f'The primary96³ probe peaks at {main["peakAllocated"]/2**30:.3f}GiB during stage1 X6 accumulation. The ordered allocator trace reconstructs {main["peakTrace"]["peakRequested"]/2**30:.3f}GiB requested storage at its peak; allocator block rounding makes this slightly lower than the Torch live-allocation high-water mark.','',
        '|Live at reconstructed peak|GiB|','|---|---:|']
    for name,value in categories.items():lines.append(f'|{name}|{value/2**30:.3f}|')
    lines+=['',f'At the end of the peak section, live Torch memory is {peak["after"]["allocated"]/2**30:.3f}GiB, and reachable unique storage accounts for {peak["ownership"]["unionBytes"]/2**30:.3f}GiB. Ownership group sums overlap and must not be added. The current PairWork work tensor and runnerResult.state.pairWork.work are distinct {main["pairVectorBytes"]/2**30:.3f}GiB storages. The latter is owned exclusively by runnerResult in the audited roots; it traces to first-step accumulation.','',
        'The runner initializes RunResult.state with the original system before the loop and replaces it at completion. That reference retains the first system and its first-step PairWork.work during later steps. It is additional retention beyond the current running work vector and does not indicate a per-stage history was reintroduced. A candidate fix is to update/release the obsolete result-state reference or retire unused pair-work scratch after finalization, with separate lifecycle/numerical/graph checks.','',
        'The X6 expression f*(ap+av), followed by stage weighting, materializes full pair-vector temporaries. Two such vectors coexist at the observed peak. Other measured costs include native stage pressure/viscosity vectors, f_ij, adjacency, small particle-state copies and output/plot allocations. These are concrete optimization targets without changing neighbour support or fresh stage semantics.','',
        '## Scope and checks','',
        'Four runs were serial:32³ four native steps resuming the saved t4 checkpoint;96³ four startup steps, then three/two-step observer repeats. The first observer missed the runner frame name; the second omitted the RunResult root. All original scalar peaks/traces are retained. Observer repairs and final complete roots are recorded; no numerical changes or failed physical trajectories. Each run has VisPy z=0 video, progress, fresh native RHS, finite/positive checks and source hashes.','',
        'The physical-list filtering is after each trajectory and never replaces its evolving neighbour policy. The32³ resumed step endpoint is not a bitwise comparison against a continuous restart control.96³ measures startup geometry; it does not reproduce the historical t0–4 peak42.10GiB or explain every later rebuild transient. Synchronization instrumentation can change asynchronous allocation lifetimes; no instrumented time is a production-throughput estimate.','',
        'Tensor-storage ownership excludes allocator rounding/cached blocks, foreign-library allocations, and Python/host memory. Remaining small live allocations are visible in allocator histories; this is not a byte-exact whole-device ledger. Original physical runtime pins remain unchanged.','',
        '|Probe|Steps|Wall seconds|Instrumented ms/step|RHS valid|','|---|---:|---:|---:|---:|']
    for r in results:
        x=r['result'];lines.append(f'|{r["name"]}|{x["steps"]}|{x["wallSeconds"]:.3f}|{x["msPerStep"]:.1f}|{x["stats"]["rhs"]}|')
    lines+=['','Three CPU storage controls pass (view aliases, independent clones, cycles/computed properties/device filtering). Videos/contact sheets and memory chart are prepared for visual review.','', '![Memory breakdown]('+str(output/'memory-breakdown.png')+')','', 'Evidence: '+str(root)]
    report=root.parents[1]/'docs/crksph_memory_2026-10-29.md';report.write_text('\n'.join(lines)+'\n');print(json.dumps(dict(report=str(report),peakGiB=main['peakAllocated']/2**30,categoriesGiB={k:v/2**30 for k,v in categories.items()})),flush=True)

if __name__=='__main__':main()
