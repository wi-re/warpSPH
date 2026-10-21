"""Offline matched shock-bubble geometry and common MLS circulation diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compare_spheralMaterials import save
from compare_spheralShockBubble import primary, Measures


def reflection_error(a, count):
    """Audit freely evolved tiles; no averaging or symmetry repair."""
    order=np.argsort(a['UIDs']);x=a['positions'][order].astype(float);v=a['velocities'][order].astype(float)
    np.testing.assert_array_equal(a['UIDs'][order],np.arange(4*count))
    L=np.array([1.1,.4]);rows=[]
    for tile,signs in enumerate(((1,-1),(-1,1),(-1,-1)),1):
        signs=np.array(signs);p=x[tile*count:(tile+1)*count].copy()
        p=np.where(signs<0,2*L-p,p);d=p-x[:count];d-=np.round(d/(2*L))*(2*L)
        dv=v[tile*count:(tile+1)*count]*signs-v[:count]
        rows.append(dict(tile=tile,maxPositionDifference=float(np.linalg.norm(d,axis=1).max()),
            rmsPositionDifference=float(np.sqrt(np.mean(np.sum(d*d,axis=1)))),
            maxVelocityDifference=float(np.linalg.norm(dv,axis=1).max())))
    p=x[:count]
    return dict(tiles=rows,primaryParticlesOutside=int(np.any((p<0)|(p>L),axis=1).sum()))


def circulation(a):
    """Volume-weighted local linear least squares, with slip-reflected images.

    Common diagnostic only, not either solver's native CRK velocity gradient.
    Cubic gather weights and native compact supports; no force computation.
    """
    x=a['positions'].astype(float);v=a['velocities'].astype(float);vol=a['masses']/a['densities'];R=a['supports']
    points=[];values=[];volumes=[]
    for ix in (0,-1,1):
        for iy in (0,-1,1):
            p=x.copy();u=v.copy()
            for axis,flag,L in ((0,ix,1.1),(1,iy,.4)):
                if flag:
                    p[:,axis]=(-x[:,axis] if flag<0 else 2*L-x[:,axis]);u[:,axis]*=-1
            points.append(p);values.append(u);volumes.append(vol)
    points=np.concatenate(points);values=np.concatenate(values);volumes=np.concatenate(volumes)
    neighbors=cKDTree(points).query_ball_point(x,R)
    owner=np.repeat(np.arange(len(x)),[len(ids) for ids in neighbors]);other=np.concatenate(neighbors).astype(int)
    r=points[other]-x[owner];dv=values[other]-v[owner];q=np.linalg.norm(r,axis=1)/R[owner]
    w=volumes[other]/R[owner]**2*np.where(q<.5,1-6*q*q+6*q**3,np.where(q<1,2*(1-q)**3,0))
    matrix=np.zeros((len(x),2,2));cross=np.zeros_like(matrix)
    np.add.at(matrix,owner,w[:,None,None]*r[:,:,None]*r[:,None,:])
    np.add.at(cross,owner,w[:,None,None]*dv[:,:,None]*r[:,None,:])
    J=np.linalg.solve(matrix,cross.transpose(0,2,1)).transpose(0,2,1)
    omega=J[:,1,0]-J[:,0,1];integral=vol*omega
    return dict(upper=float(integral[x[:,1]>=.2].sum()),lower=float(integral[x[:,1]<.2].sum()),
        total=float(integral.sum()),enstrophy=float(.5*np.dot(vol,omega**2)),
        maxMomentCondition=float(np.linalg.cond(matrix).max())),omega


def analyze(root,output):
    from warpsph_remote.sources import HDF5Source
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True);records=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not all(any((directory/s/f).exists() for f in ('comparisonRun.json','failure.json')) for s in ('spheral','warp')):continue
        settings=json.loads((directory/'settings.json').read_text());N=settings['primaryParticles']
        runs={};sources={};native_history={}
        for s in ('spheral','warp'):
            summary=directory/s/'comparisonRun.json'
            history=json.loads((directory/s/'diagnosticHistory.json').read_text());native_history[s]=history
            runs[s]=json.loads(summary.read_text()) if summary.exists() else dict(
                solver=s,passed=False,termination='failed',final=history[-1],time=history[-1]['time'],
                failure=json.loads((directory/s/'failure.json').read_text()),
                stateProvenance='Stop metrics retained; exact failed particle state unavailable in this earlier recording.')
            path=Path(runs[s].get('exportPath',str(directory/s)))
            sources[s]=HDF5Source(path/'trajectory.h5')
        initial={s:primary(dict(np.load(directory/s/'initialState.npz' if (directory/s/'initialState.npz').exists()
                                        else directory/('warpInput.npz' if s=='warp' else 'input.npz'))),N) for s in runs}
        for key in ('positions','masses','materials','internalEnergies','UIDs'):
            np.testing.assert_array_equal(initial['warp'][key],initial['spheral'][key],err_msg=key)
        # Compare stored states at a common valid time, even if a solver stopped
        # later between recording frames. Never substitute its failure metrics
        # for the state at the earlier recorded time.
        common_end=min(source.times[-1] for source in sources.values())
        indices={s:int(np.argmin(abs(np.asarray(source.times)-common_end))) for s,source in sources.items()}
        times={s:float(source.times[indices[s]]) for s,source in sources.items()}
        # Warp's time-based store cadence starts at the first completed step;
        # reference cadence starts at zero. Keep the actual one-step offset
        # explicit rather than inventing interpolated particle states.
        if max(times.values())-min(times.values())>1.01*settings['dt']:
            raise ValueError('No synchronized common terminal recording: '+str(times))
        raw={s:sources[s].read(indices[s]) for s in runs}
        final={s:primary(raw[s],N) for s in runs}
        for a in final.values():
            if 'pressures' not in a:
                a['pressures']=(np.array(settings['gamma'])[a['materials']]-1)*a['densities']*a['internalEnergies']
        series={};final_curl={};common_metrics={};reflection_series=[]
        for s in runs:
            for key in ('UIDs','masses','materials'):np.testing.assert_array_equal(initial[s][key],final[s][key])
            source=sources[s]
            zero,_=circulation(initial[s]);zero['time']=0.;rows=[zero]
            for target in np.linspace(0,common_end,13)[1:]:
                idx=int(np.argmin(abs(np.asarray(source.times)-target)));frame=source.read(idx);a=primary(frame,N)
                result,_=circulation(a);result['time']=float(source.times[idx]);rows.append(result)
                if s=='warp':reflection_series.append(dict(time=float(source.times[idx]),**reflection_error(frame,N)))
            series[s]=rows;final_curl[s],omega=circulation(final[s])
            common_metrics[s]=Measures(settings,initial[s])(final[s],times[s])
        differences={}
        for key in ('bubbleCenterX','bubbleWidth','bubbleHeight','bubbleVolume','bubbleMeanPressure'):
            values={s:common_metrics[s][key] for s in runs};differences[key]=dict(**values,
                absoluteDifference=float(abs(values['warp']-values['spheral'])),
                relativeDifference=float(abs(values['warp']-values['spheral'])/max(abs(values['spheral']),1.e-12)))
        differences['circulationUpper']=dict(**{s:final_curl[s]['upper'] for s in runs},
            relativeDifference=float(abs(final_curl['warp']['upper']-final_curl['spheral']['upper'])/max(abs(final_curl['spheral']['upper']),1.e-12)))
        packing={}
        for s,a in final.items():
            distance=cKDTree(a['positions']).query(a['positions'],k=2)[0][:,1]/np.sqrt(a['masses']/a['densities'])
            packing[s]=dict(min=float(distance.min()),p01=float(np.quantile(distance,.01)),median=float(np.median(distance)),
                belowQuarterFraction=float(np.mean(distance<.25)),belowTenthFraction=float(np.mean(distance<.1)))
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        for ax,s in zip(axes[0,:2],('spheral','warp')):
            a=final[s];ax.scatter(*a['positions'].T,c=a['materials'],s=2*(80/settings['nx'])**2,cmap='coolwarm',vmin=0,vmax=1)
            ax.set(xlim=(.15,1.1),ylim=(0,.4),aspect='equal',title=s+' material at t='+format(times[s],'.4g'))
        for s in runs:
            h=native_history[s];t=[r['time'] for r in h]
            for ax,key in zip([axes[0,2],axes[1,0],axes[1,1]],['bubbleCenterX','bubbleWidth','bubbleHeight']):
                ax.plot(t,[r[key] for r in h],label=s);ax.set(title=key,xlabel='Time');ax.grid(alpha=.2)
            axes[1,2].plot([r['time'] for r in series[s]],[r['upper'] for r in series[s]],label=s)
        axes[1,2].set(title='Common MLS upper-half circulation',xlabel='Time');axes[0,2].legend()
        fig.suptitle(directory.name+'; physical air/helium labels, cubic, native support/AV/work differences')
        fig.savefig(output/(directory.name+'.png'),dpi=180);plt.close(fig)
        row=dict(case=directory.name,settings=settings,runs=runs,differences=differences,
            comparisonTimes=times,commonMetrics=common_metrics,
            commonCirculation=series,finalCirculation=final_curl,packing=packing,
            warpReflectionAtComparison=reflection_error(raw['warp'],N),
            warpReflectionHistory=reflection_series,
            snapshotTimeOffset=max(times.values())-min(times.values()))
        # Same nominal physical time across resolutions, independent of their
        # different stop times. Retain one-step recording offsets explicitly.
        fixed={}
        if common_end>=.2:
            for s,source in sources.items():
                idx=int(np.argmin(abs(np.asarray(source.times)-.2)));frame=source.read(idx);a=primary(frame,N)
                if 'pressures' not in a:
                    a['pressures']=(np.array(settings['gamma'])[a['materials']]-1)*a['densities']*a['internalEnergies']
                curl,_=circulation(a)
                fixed[s]=dict(metrics=Measures(settings,initial[s])(a,source.times[idx]),circulation=curl)
                if s=='warp':fixed[s]['reflection']=reflection_error(frame,N)
        row['fixedTimePoint2']=fixed
        records.append(row);print(directory.name,json.dumps(differences),flush=True)
    save(output/'comparison.json',dict(comparisons=records,
        limits='Numerical cross-code reference, not exact shock-bubble solution. Common saved primary lattice; explicit reflecting Spheral walls versus freely evolved reflected-periodic Warp copies, whose symmetry is audited rather than assumed. Two physical materials merge upstream air NodeLists; XSPH off. Circulation uses a common diagnostic MLS operator, not native solver gradients. Nearest recorded times can differ by one fixed step; actual timestamps retained. No force tuning.'))


def verify_recordings(root,output):
    import subprocess
    import urllib.request
    from warpsph_remote.daemon import descriptor
    from warpsph_remote.sources import HDF5Source
    info=descriptor()
    def get(path):
        request=urllib.request.Request(info['endpoint']+path,headers={'Authorization':'Bearer '+info['token']})
        with urllib.request.urlopen(request,timeout=20) as response:return json.load(response)
    runs={run['id']:run for run in get('/v1/runs')['runs']};checks=[];startup=[]
    for path in sorted(root.glob('**/registration.json')):
        folder=path.parent;rid=json.loads(path.read_text())['remoteRunID'];run=runs[rid]
        summary=folder/'comparisonRun.json'
        result=json.loads(summary.read_text()) if summary.exists() else {'passed':False}
        diagnostic=folder/'diagnosticHistory.json'
        if diagnostic.exists() and not json.loads(diagnostic.read_text()):
            assert run['status']=='failed'
            startup.append(dict(folder=str(folder),id=rid,status=run['status'],
                reason=json.loads((folder/'failure.json').read_text())['error'],
                classification='Startup rejection with no diagnostic timestep; no physical video required'))
            continue
        source=Path(result['exportPath']) if 'exportPath' in result else next(folder.glob('*/trajectory.h5')).parent if folder.name=='warp' else folder
        frames=HDF5Source(source/'trajectory.h5')
        history=get('/v1/runs/'+rid+'/history')
        video=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_frames','-of','json',str(source/'output.mp4')]))['streams'][0]
        images=len(list((source/'images').glob('frame_*.png')))
        assert run['status']==('completed' if result['passed'] else 'failed'),(rid,run['status'])
        assert history['frames']==len(frames.times) and history['dimension']==2
        np.testing.assert_allclose(history['times'],frames.times,rtol=0,atol=1.e-12)
        assert int(video['nb_frames'])==images and images>0
        if result['passed']:assert images>1
        if folder.name=='spheral':assert images==len(frames.times)
        checks.append(dict(case=str(folder.parent.relative_to(root)),solver=folder.name,id=rid,status=run['status'],
            recordedFrames=len(frames.times),plottedFrames=images,lastRecordedTime=frames.times[-1],
            stopTime=result.get('time',json.loads((folder/'diagnosticHistory.json').read_text())[-1]['time']),video=video))
    save(output/'daemonVerification.json',checks);print('Verified',len(checks),'daemon histories and videos',flush=True)
    if startup:save(output/'startupRejections.json',startup)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-daemon',action='store_true')
    a=p.parse_args();analyze(a.input,a.output)
    if a.verify_daemon:verify_recordings(a.input,a.output)
