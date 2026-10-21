"""Offline exact/profile, initial-array and daemon/video checks for Sod 1D."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request
import numpy as np
from compare_spheralSod1d import exact
from compare_spheralMaterials import save


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    report.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((root/'provenance.json').read_text());rows={};contrasts={};finals={};histories={};initial={}
    grid=np.linspace(-.5,.5,4097);_,waves=exact(grid,.2)
    for f in provenance['rows']:
        folder=Path(f);a=dict(np.load(folder/'input.npz'));central=abs(a['positions'][:,0])<.6
        for solver in ('spheral','warp'):
            data=folder/solver;meta=data/'comparisonRun.json'
            if not meta.exists():continue
            run=json.loads(meta.read_text());name=folder.name+'-'+solver
            b=dict(np.load(data/'finalState.npz'));h=json.loads((data/'diagnosticHistory.json').read_text())
            rows[name]=run;finals[name]=b;histories[name]=h
            start=dict(np.load(data/'initialState.npz'))
            initial[name]={k:float(np.abs(start[k]-a[k]).max()) for k in a}
            if max(initial[name].values())>1.e-12:raise ValueError(('Startup mismatch',name,initial[name]))
            order=np.argsort(b['positions'][:,0]);x=b['positions'][order,0]
            profiles={key:np.interp(grid,x,b[field][order] if field!='velocities' else b[field][order,0])
                for key,field in [('density','densities'),('pressure','pressures'),('velocity','velocities'),('energy','internalEnergies')]}
            rows[name]['uniformGridExactL1']={key:float(np.trapezoid(np.abs(profiles[key]-v),grid)) for key,v in exact(grid,.2)[0].items()}
            plateau=(grid>.03)&(grid<.12)
            rows[name]['starPlateau']=dict(pressureMean=float(profiles['pressure'][plateau].mean()),velocityMean=float(profiles['velocity'][plateau].mean()))
            mask=grid>waves['velocity']*.2+.04;xs=grid[mask];ps=profiles['pressure'][mask];threshold=.5*(waves['pressure']+.1)
            crossings=np.flatnonzero((ps[:-1]>=threshold)&(ps[1:]<threshold))
            rows[name]['shockCrossings']=[float(xs[i]+(threshold-ps[i])/(ps[i+1]-ps[i])*(xs[i+1]-xs[i])) for i in crossings]
            ids=np.argsort(b['UIDs']);bi={k:v[ids] for k,v in b.items()};contact=np.flatnonzero(np.diff(a['positions'][:,0]<0))
            rows[name]['initialContactTracerMidpoint']=float(.5*(bi['positions'][contact[0],0]+bi['positions'][contact[0]+1,0]))
        if folder.name+'-warp' not in finals:continue
        s,w=(finals[folder.name+'-'+solver] for solver in ('spheral','warp'))
        s={k:v[np.argsort(s['UIDs'])] for k,v in s.items()};w={k:v[np.argsort(w['UIDs'])] for k,v in w.items()}
        np.testing.assert_array_equal(s['UIDs'],w['UIDs'])
        contrast={}
        for key in ('positions','velocities','densities','pressures','internalEnergies','supports'):
            d=w[key][central]-s[key][central];contrast[key]=dict(RMS=float(np.sqrt(np.mean(d*d))),maximum=float(abs(d).max()))
        contrast['commonGridL1']={}
        for key in ('densities','pressures','internalEnergies','velocities'):
            def interp(b):
                o=np.argsort(b['positions'][:,0]);v=b[key][o];return np.interp(grid,b['positions'][o,0],v[:,0] if v.ndim==2 else v)
            contrast['commonGridL1'][key]=float(np.trapezoid(abs(interp(w)-interp(s)),grid))
        contrasts[folder.name]=contrast
    save(report/'comparison.json',dict(provenance=provenance,waves=waves,rows=rows,solverContrasts=contrasts,initialDifferences=initial,
        observation='Exact/profile metrics on [-.5,.5]; UID RMS on initial |x|<.6. Periodic seam at +/-1.5 remains outside the observation region.'))
    for support in ('fixed','adaptive'):
        fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        refs=exact(grid,.2)[0]
        for ax,(key,field) in zip(axes.flat,[('density','densities'),('pressure','pressures'),('velocity','velocities'),('energy','internalEnergies')]):
            ax.plot(grid,refs[key],color='black',lw=1.5,label='Exact Riemann')
            for nx in (200,400):
                for solver in ('spheral','warp'):
                    name=f'n{nx}-{support}-{solver}'
                    if name not in finals:continue
                    b=finals[name];mask=abs(b['positions'][:,0])<.5;x=b['positions'][mask,0];o=np.argsort(x);y=b[field][mask]
                    ax.plot(x[o],y[o,0] if y.ndim==2 else y[o],label=f'{solver} n={nx}',lw=.9,ls='-' if solver=='spheral' else '--',alpha=.85)
            ax.set(xlim=(-.5,.5),xlabel='x',ylabel=key);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);fig.suptitle('Single-material Sod at t=.2; '+support+' supports')
        fig.savefig(report/(support+'-profiles.png'),dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for name,h in histories.items():
        for ax,key in zip(axes,('densityL1','velocityL1','relativeEnergyDrift')):ax.plot([r['time'] for r in h],[r[key] for r in h],label=name,lw=.9)
    for ax,key in zip(axes,('Density L1','Velocity L1','Relative energy drift')):ax.set(xlabel='t',ylabel=key);ax.grid(alpha=.2)
    axes[0].legend(fontsize=6);fig.savefig(report/'diagnostics.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(rows={n:dict(final=r['final'],wallSeconds=r['wallSeconds'],msPerStep=r['meanStepMilliseconds']) for n,r in rows.items()},contrasts=contrasts),indent=2))


def verify(root,report):
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(request,timeout=15)as response:return json.load(response)
    remote={r['id']:r for r in get('/v1/runs')['runs']};rows={}
    for registration in root.glob('*/*/registration.json'):
        if registration.parent.name not in ('spheral','warp','spheral-extraTerminalStep'):continue
        folder=registration.parent;run=json.loads((folder/'comparisonRun.json').read_text());path=folder if folder.name.startswith('spheral') else Path(run['exportPath'])
        if folder.name=='spheral-extraTerminalStep':
            rid=json.loads(registration.read_text())['remoteRunID']
            # Preserve playback after moving this superseded diagnostic into
            # its archive directory; never point two run IDs at the rerun file.
            body=dict(recording=dict(path=str((path/'trajectory.h5').resolve()),mode='trajectory',name=remote[rid]['name'],
                domain=dict(min=[-1.5],max=[1.5],periodic=[True])),message='Extra terminal roundoff step retained as a diagnostic; use the separately recorded matched-step reference for comparison.')
            request=urllib.request.Request(info['endpoint']+'/producer/'+rid+'/status',data=json.dumps(body).encode(),
                headers={'Authorization':'Bearer '+info['producerToken'],'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(request,timeout=15)as response:json.load(response)
        rid=json.loads(registration.read_text())['remoteRunID'];source=HDF5Source(path/'trajectory.h5');history=get('/v1/runs/'+rid+'/history')
        np.testing.assert_allclose(history['times'],source.times,rtol=0,atol=1.e-12)
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(path/'output.mp4')]))
        count=int(video['streams'][0]['nb_frames'])
        if count!=len(source.times) or remote[rid]['status']!='completed':raise ValueError((folder,count,len(source.times),remote[rid]['status']))
        rows[folder.parent.name+'-'+folder.name]=dict(id=rid,status=remote[rid]['status'],historyFrames=len(source.times),videoFrames=count,historyMatches=True,
            selected=folder.name!='spheral-extraTerminalStep')
    if sum(r['selected'] for r in rows.values())!=8:raise ValueError(('Expected 8 selected runs',len(rows)))
    report.mkdir(exist_ok=True,parents=True);save(report/'recordingVerification.json',rows);print('Verified',len(rows),'histories/videos, eight selected matched-step runs')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheral/singleSod1d'));a=p.parse_args()
    if a.mode=='analyze':analyze(a.input.resolve(),a.report.resolve())
    else:verify(a.input.resolve(),a.report.resolve())
