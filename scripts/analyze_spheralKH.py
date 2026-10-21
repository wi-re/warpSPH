"""Offline smooth KH growth, periodic UID/Eulerian contrasts and recordings."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request
import numpy as np
from compare_spheralMaterials import save


def periodic_delta(a,b):
    return (a-b+.5)%1-.5


def grid_fields(a,n=128,keys=('densities','pressures','velocities','internalEnergies')):
    from scipy.interpolate import LinearNDInterpolator
    pos=a['positions']%1
    # Periodic images near the box suffice for interpolation at cell centres.
    points=[];values=[]
    fields=np.column_stack([a[k] for k in keys])
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            shifted=pos+[dx,dy];mask=((shifted>=-.15)&(shifted<=1.15)).all(1)
            points.append(shifted[mask]);values.append(fields[mask])
    axis=(np.arange(n)+.5)/n;x,y=np.meshgrid(axis,axis)
    out=LinearNDInterpolator(np.concatenate(points),np.concatenate(values))(x,y)
    if not np.isfinite(out).all():raise ValueError('Incomplete periodic grid coverage')
    return out


def contrast(a,b):
    a={k:v[np.argsort(a['UIDs'])] for k,v in a.items()};b={k:v[np.argsort(b['UIDs'])] for k,v in b.items()}
    np.testing.assert_array_equal(a['UIDs'],b['UIDs'])
    out={}
    for key in ('positions','velocities','densities','pressures','internalEnergies','supports'):
        if key not in a or key not in b:continue
        d=periodic_delta(a[key],b[key]) if key=='positions' else a[key]-b[key]
        norm=np.linalg.norm(d,axis=1) if d.ndim==2 else abs(d)
        out[key]=dict(RMS=float(np.sqrt(np.mean(norm**2))),maximum=float(norm.max()))
    keys=[k for k in ('densities','pressures','velocities','internalEnergies') if k in a and k in b]
    names={'densities':['density'],'pressures':['pressure'],'velocities':['vx','vy'],'internalEnergies':['energy']}
    out['EulerianGridL1']=dict(zip([v for k in keys for v in names[k]],
        np.mean(abs(grid_fields(a,keys=keys)-grid_fields(b,keys=keys)),axis=(0,1)).tolist()))
    return out


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((root/'provenance.json').read_text());rows={};contrasts={};frames={};histories={};initials={};grids={}
    for f in provenance['rows']:
        folder=Path(f);initial=dict(np.load(folder/'input.npz'));pair={}
        for solver in ('spheral','warp'):
            data=folder/solver;name=folder.name+'-'+solver;run=json.loads((data/'comparisonRun.json').read_text())
            rows[name]=run;histories[name]=json.loads((data/'diagnosticHistory.json').read_text())
            if not run['passed']:raise ValueError(('Run failed qualification',name))
            if solver=='spheral':
                counts=json.loads((data/'overlayCalls.json').read_text());steps=run['steps']
                expected=dict(density=steps,finalEnergy=steps,rawPredictor=steps,RHS=2*steps)
                if any(counts[k]!=v for k,v in expected.items()):raise ValueError(('Unexpected stage counts',name,counts))
                rows[name]['overlayCalls']=counts
            start=dict(np.load(data/'initialState.npz'));initials[name]={k:float(abs(start[k]-initial[k]).max()) for k in initial}
            if max(initials[name].values())>1.e-12:raise ValueError(('Startup mismatch',name,initials[name]))
            # End-state NPZs have exactly matched physical times and complete
            # P/u/support fields. Regular Warp video columns start after step
            # one and omit these evolving fields, so do not use them as exact
            # interim checkpoints or substitute their initial support metadata.
            pair[solver]=dict(np.load(data/'finalState.npz'))
            frames[name]=pair[solver];grids[name]=grid_fields(frames[name])
        contrasts[folder.name]={'2.0':contrast(pair['warp'],pair['spheral'])}
        sh,wh=(histories[folder.name+'-'+s] for s in ('spheral','warp'))
        ts=np.array([r['time'] for r in sh]);tw=np.array([r['time'] for r in wh]);modeS=np.array([r['modeAmplitude'] for r in sh]);modeW=np.interp(ts,tw,[r['modeAmplitude'] for r in wh])
        contrasts[folder.name]['growth']=dict(maxAbsoluteModeDifference=float(abs(modeW-modeS).max()),
            finalModeRelativeDifference=float(abs(modeW[-1]/modeS[-1]-1)),
            earlyMaxRelativeModeDifference=float(np.max(abs(modeW[ts<=1]/modeS[ts<=1]-1))))
    refinement={};timestep={}
    for solver in ('spheral','warp'):
        refinement[solver]=dict(EulerianGridL1=dict(zip(('density','pressure','vx','vy','energy'),np.mean(abs(grids['n64-'+solver]-grids['n128-'+solver]),axis=(0,1)).tolist())),
            finalModeRatio=float(rows['n128-'+solver]['final']['modeAmplitude']/rows['n64-'+solver]['final']['modeAmplitude']))
        if 'n64-halfdt-'+solver in rows:
            timestep[solver]=dict(finalModeRelativeChange=float(rows['n64-halfdt-'+solver]['final']['modeAmplitude']/rows['n64-'+solver]['final']['modeAmplitude']-1),
                finalState=contrast(frames['n64-halfdt-'+solver],frames['n64-'+solver]))
    save(report/'comparison.json',dict(provenance=provenance,rows=rows,solverContrasts=contrasts,initialDifferences=initials,resolutionContrasts=refinement,timestepContrasts=timestep,
        metric='Periodic UID position errors use minimum-image vector norm. Eulerian fields use periodic piecewise-linear interpolation to 128^2 cell centres; comparison diagnostic, not an exact KH oracle.'))
    fig,axes=plt.subplots(2,2,figsize=(10,9),constrained_layout=True)
    for row,nx in enumerate((64,128)):
        for col,solver in enumerate(('spheral','warp')):
            ax=axes[row,col];a=frames[f'n{nx}-'+solver];p=a['positions']%1
            im=ax.scatter(p[:,0],p[:,1],c=a['densities'],s=2.5 if nx==64 else .7,vmin=1,vmax=2,cmap='viridis',linewidths=0,rasterized=True)
            ax.set(xlim=(0,1),ylim=(0,1),aspect='equal',title=f'{solver} {nx}² — density at t=2',xlabel='x',ylabel='y')
    fig.colorbar(im,ax=axes,shrink=.7,label='density');fig.savefig(report/'density.png',dpi=200);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,h in histories.items():
        nx=int(name.split('-')[0][1:]);color='C0' if name.endswith('spheral') else 'C1';style=':' if 'halfdt' in name else ('--' if nx==64 else '-')
        for ax,key in zip(axes.flat,('modeAmplitude','transverseKineticEnergy','pressureRMS','entropyChangePerMass')):
            ax.plot([r['time'] for r in h],[r[key] for r in h],label=name,color=color,ls=style,lw=1.4)
            ax.set(xlabel='t',ylabel=key);ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Smooth single-material KH — matched fresh/paper controls')
    fig.savefig(report/'growth.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(rows={k:dict(final=v['final'],wallSeconds=v['wallSeconds'],msPerStep=v['meanStepMilliseconds']) for k,v in rows.items()},contrasts=contrasts,refinement=refinement,timestep=timestep),indent=2))


def verify(root,report):
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        req=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(req,timeout=15) as response:return json.load(response)
    remote={r['id']:r for r in get('/v1/runs')['runs']};rows={}
    for data in sorted(root.glob('n*/*')):
        if data.name not in ('spheral','warp'):continue
        run=json.loads((data/'comparisonRun.json').read_text());path=data if data.name=='spheral' else Path(run['exportPath'])
        rid=run['remoteRunID'];source=HDF5Source(path/'trajectory.h5');history=get('/v1/runs/'+rid+'/history')
        np.testing.assert_allclose(history['times'],source.times,rtol=0,atol=1.e-12)
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(path/'output.mp4')]))
        count=int(video['streams'][0]['nb_frames'])
        if count!=len(source.times) or remote[rid]['status']!='completed':raise ValueError((data,count,len(source.times),remote[rid]['status']))
        if abs(source.times[-1]-2)>1.e-9:raise ValueError(('Incomplete history',data))
        rows[data.parent.name+'-'+data.name]=dict(id=rid,status=remote[rid]['status'],historyFrames=len(source.times),videoFrames=count,historyMatches=True)
    expected=2*len(json.loads((root/'provenance.json').read_text())['rows'])
    if len(rows)!=expected:raise ValueError(('Incomplete study',expected,len(rows)))
    report.mkdir(exist_ok=True,parents=True);save(report/'recordingVerification.json',rows);print('Verified',expected,'daemon histories and videos')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/smoothKH'));a=p.parse_args()
    if a.mode=='analyze':analyze(a.input.resolve(),a.report.resolve())
    else:verify(a.input.resolve(),a.report.resolve())
