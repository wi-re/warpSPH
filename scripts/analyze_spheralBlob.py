"""Matched-input compiled Spheral blob evidence, including failed trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compare_spheralMaterials import save,FIELDS,PIN


def pressure(a):
    if 'pressures' not in a:a['pressures']=(5/3-1)*a['densities']*a['internalEnergies']
    return a


def interface_metrics(a,initial):
    """Descriptive packing/thermodynamics, not a surface tension measurement."""
    order=np.argsort(a['UIDs']);a={k:v[order] for k,v in a.items()}
    initial={k:v[np.argsort(initial['UIDs'])] for k,v in initial.items()}
    cloud=a['materials']==1;x=a['positions'];m=a['masses'];vol=m/a['densities']
    r0=np.linalg.norm(initial['positions']-[5.,5.],axis=1);out={}
    for name,mask in [('cloud',cloud),('ambient',~cloud)]:
        shell=mask&((r0>.8) if name=='cloud' else (r0<1.2))
        other=np.flatnonzero(~mask);distance,index=cKDTree(x[other]).query(x[shell])
        separation=distance/(.5*(np.sqrt(vol[shell])+np.sqrt(vol[other[index]])))
        out[name]=dict(shellParticles=int(shell.sum()),shellPressureMedian=float(np.median(a['pressures'][shell])),
            shellPressureRange=[float(a['pressures'][shell].min()),float(a['pressures'][shell].max())],
            crossMaterialSeparationP10=float(np.quantile(separation,.1)),crossMaterialSeparationMedian=float(np.median(separation)),
            minimumEnergy=float(a['internalEnergies'][mask].min()))
    center=np.average(x[cloud],axis=0,weights=m[cloud]);cov=np.einsum('i,ij,ik->jk',m[cloud],x[cloud]-center,x[cloud]-center)/m[cloud].sum()
    out['cloudShape']=dict(center=center.tolist(),rmsAxes=np.sqrt(np.linalg.eigvalsh(cov)).tolist())
    return out


def analyze(root,warp_root,output):
    from warpsph_remote.sources import HDF5Source
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    initial=dict(np.load(warp_root/'n256/entropy/input.npz'));records={};histories={}
    sourceSHA=hashlib.sha256((warp_root/'n256/entropy/input.npz').read_bytes()).hexdigest()
    for folder in sorted(root.iterdir()):
        path=folder/'spheral/comparisonRun.json'
        if not path.exists():continue
        result=json.loads(path.read_text());settings=json.loads((folder/'settings.json').read_text())
        assert hashlib.sha256((folder/'input.npz').read_bytes()).hexdigest()==sourceSHA
        loaded=dict(np.load(folder/'spheral/initialState.npz'))
        for k in FIELDS:np.testing.assert_allclose(loaded[k],initial[k],rtol=2e-7 if k=='pressures' else 0,atol=2e-7 if k=='pressures' else 1e-14,err_msg=k)
        failed=dict(np.load(folder/'spheral/failedState.npz'));idx=int(failed['internalEnergies'].argmin())
        result.update(settings=settings,minimumEnergyParticle={k:failed[k][idx].tolist() for k in ('UIDs','materials','positions','densities','internalEnergies','pressures')},
            recordedTimes=list(map(float,HDF5Source(folder/'spheral/trajectory.h5').times)))
        records[folder.name]=result;histories['Spheral '+folder.name]=json.loads((folder/'spheral/diagnosticHistory.json').read_text())
    for name in ('entropy','dudt','dudt-full-policy'):
        histories['Warp '+name]=json.loads((warp_root/'early'/name/'warp/diagnosticHistory.json').read_text())
    # Cadence replay must reproduce the original run, independently of plotting.
    for k in FIELDS:np.testing.assert_array_equal(np.load(root/'n256/spheral/failedState.npz')[k],np.load(root/'replay/spheral/failedState.npz')[k])
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,h in histories.items():
        if name=='Spheral n256':continue
        t=np.array([r['time'] for r in h]);u=np.array([np.nan if r['minEnergy'] is None else r['minEnergy'] for r in h])
        for ax in axes:
            line,=ax.plot(t,u,label=name,ls='--' if name.startswith('Spheral') else '-')
            if np.isfinite(u[-1]) and u[-1]<=0:ax.plot(t[-1],u[-1],'x',color=line.get_color(),ms=7)
    for ax,limit in zip(axes,(.02,.105)):
        ax.axhline(0,color='k',lw=.7);ax.set(xlim=(0,limit),xlabel='Physical time',ylabel='Minimum specific internal energy');ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Actual compiled Spheral and Warp; crosses mark negative-energy watchdog stops')
    fig.savefig(output/'energy.png',dpi=180);plt.close(fig)
    source=HDF5Source(root/'replay/spheral/trajectory.h5');common=float(source.times[-1])
    sources={'Spheral':source}
    for name in ('entropy','dudt'):
        result=json.loads((warp_root/'early'/name/'warp/comparisonRun.json').read_text())
        sources['Warp '+name]=HDF5Source(Path(result['exportPath'])/'trajectory.h5')
    fig,axes=plt.subplots(3,2,figsize=(9,10),layout='constrained');metrics={}
    for row,(name,source) in enumerate(sources.items()):
        idx=int(np.argmin(abs(np.asarray(source.times)-common)));t=float(source.times[idx]);assert abs(t-common)<1e-8
        a=pressure(source.read(idx));x=a['positions'];zoom=(x[:,0]>3.7)&(x[:,0]<6.3)&(x[:,1]>3.7)&(x[:,1]<6.3)
        for ax,field,cmap,vmax in zip(axes[row],('materials','pressures'),('coolwarm','viridis'),(1,18)):
            artist=ax.scatter(*x[zoom].T,c=a[field][zoom],s=5,cmap=cmap,vmin=0,vmax=vmax)
            ax.set(xlim=(3.7,6.3),ylim=(3.7,6.3),aspect='equal',title=f'{name}: {field}, t={t:.8f}')
            if field=='pressures':fig.colorbar(artist,ax=ax,shrink=.8)
        metrics[name]=dict(time=t,**interface_metrics(a,initial))
    fig.suptitle('Last common valid time; early deformation only, no late-time interface comparison')
    fig.savefig(output/'earlyInterface.png',dpi=180);plt.close(fig)
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('scripts/compare_spheralBlob.py'),Path('scripts/compare_spheralMaterials.py'))}
    save(output/'comparison.json',dict(referencePin=PIN,inputNPZSHA256=sourceSHA,sourceSHA256=provenance,runs=records,
        commonTime=common,interfaceMetrics=metrics,scope='Matched saved inputs; native compiled Spheral policies and explicitly noted coefficient controls. No completed reference or late-time surface-tension qualification.'))
    print('Analyzed',len(records),'actual Spheral runs; identical input NPZ; replay failure bitwise identical. Common time',common,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--warp-input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-daemon',action='store_true');a=p.parse_args()
    analyze(a.input,a.warp_input,a.output)
    if a.verify_daemon:
        from analyze_spheralShockBubble import verify_recordings
        verify_recordings(a.input,a.output)
