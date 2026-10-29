#!/usr/bin/env python3
"""Render retained Fourier forcing as an acceleration field; no solver/CUDA.

Uses actual saved particle masses/positions for the applied mean subtraction,
and actual accepted times. Arrow/color scales are fixed across all frames.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
import numpy as np
import torch
from audit_drivenForcing import independent,evaluate


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=40.);p.add_argument('--duration',type=float,default=4.);p.add_argument('--cadence',type=float,default=.025)
    a=p.parse_args();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cfg=json.loads((run/'input.json').read_text());state=torch.load(run/'forcing.pt',map_location='cpu',weights_only=True);sampling=json.loads((run/'sampling.json').read_text());times=np.asarray(sampling['times']);paths=sorted((run/'samples').glob('*.npz'));length=cfg['simulation']['L']
    assert len(paths)==len(times) and times[0]<=a.start<a.start+a.duration<=times[-1]
    targets=np.linspace(a.start,a.start+a.duration,int(round(a.duration/a.cadence))+1)
    indices=[int(np.argmin(abs(times-t))) for t in targets];assert len(set(indices))==len(indices)
    n=65;axis=np.linspace(-length/2,length/2,n);xx,yy=np.meshgrid(axis,axis,indexing='xy');positions=np.stack((xx.ravel(),yy.ravel()),axis=-1)
    fields=[];shifts=[];evidence={};maxDivergence=0.;maxMassMean=0.
    for j,i in enumerate(indices):
        with np.load(paths[i]) as s:
            t=float(s['t']);m=s['masses'].astype(float);x=s['positions'].astype(float)
        assert t==times[i]
        modes,c=independent(state,t);acc,jac=evaluate(positions,modes,c,length);particleAcc,_=evaluate(x,modes,c,length)
        shift=np.sum(m[:,None]*particleAcc,axis=0)/m.sum() if cfg['params']['removeMeanAcceleration'] else np.zeros(2)
        fields.append((acc-shift).reshape(n,n,2));shifts.append(shift);evidence[paths[i].name]=digest(paths[i])
        maxDivergence=max(maxDivergence,float(np.abs(np.trace(jac,axis1=1,axis2=2)).max()));maxMassMean=max(maxMassMean,float(np.abs(np.sum(m[:,None]*(particleAcc-shift),axis=0)/m.sum()).max()))
        if j%40==0:print(f'forcing frame {j+1}/{len(indices)}, t={t:.6f}',flush=True)
    fields=np.asarray(fields);actual=times[indices];speed=np.linalg.norm(fields,axis=-1);limit=float(speed.max());assert np.isfinite(fields).all() and maxDivergence<1e-12
    np.savez_compressed(out/'fields.npz',t=actual,axis=axis,acceleration=fields,massMeanSubtraction=shifts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    def panel(ax,index):
        image=ax.imshow(speed[index],origin='lower',extent=[-length/2,length/2]*2,vmin=0,vmax=limit,cmap='viridis',interpolation='bilinear')
        sl=(slice(None,None,5),slice(None,None,5));vec=fields[index]
        arrows=ax.quiver(xx[sl],yy[sl],vec[sl][...,0],vec[sl][...,1],angles='xy',scale_units='xy',scale=5,color='white',width=.004,pivot='mid')
        ax.set(xlabel='x',ylabel='y',aspect='equal',xlim=(-length/2,length/2),ylim=(-length/2,length/2),title=f't={actual[index]:.4f}')
        return image,arrows,sl
    picks=[int(np.argmin(abs(actual-t))) for t in (a.start,a.start+.5,a.start+1.)]
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,index in zip(axes,picks):image,_,_=panel(ax,index)
    fig.colorbar(image,ax=axes,label='Acceleration magnitude |a|',shrink=.8)
    fig.suptitle(f"Applied forcing, seed{cfg['params']['seed']}: periodic large-scale stirring\nArrows: acceleration direction/strength; fixed scales; actual mass-mean subtraction")
    for suffix in ('png','pdf'):fig.savefig(out/f'forcing-snapshots.{suffix}',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');image,arrows,sl=panel(ax,0);fig.colorbar(image,ax=ax,label='Acceleration magnitude |a|')
    fig.suptitle(f"Applied forcing, seed{cfg['params']['seed']}\nAcceleration field only; fixed scales; ≈1 simulation time unit per video second")
    writer=FFMpegWriter(fps=int(round(1/a.cadence)),codec='libx264',bitrate=2500,extra_args=['-pix_fmt','yuv420p'])
    with writer.saving(fig,str(out/'forcing.mp4'),dpi=120):
        for j in range(len(actual)):
            image.set_data(speed[j]);vec=fields[j];arrows.set_UVC(vec[sl][...,0],vec[sl][...,1]);ax.set_title(f't={actual[j]:.4f}');writer.grab_frame()
    plt.close(fig)
    info=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',str(out/'forcing.mp4')],capture_output=True,text=True,check=True).stdout)
    assert int(info['streams'][0]['nb_frames'])==len(actual)
    meta=dict(run=str(run),parameters=state['parameters'],times=actual.tolist(),modes=modes.tolist(),frames=len(actual),arrowScale=5,colorLimit=limit,maxDivergence=maxDivergence,maxResidualMassMeanAcceleration=maxMassMean,forcingSHA256=digest(run/'forcing.pt'),inputSHA256=digest(run/'input.json'),snapshotsSHA256=evidence,driverSHA256=digest(Path(__file__)),video=info,wallSeconds=time.perf_counter()-start,policy='CPU independent Fourier reconstruction at actual accepted times, with actual particle mass-mean subtraction. No hydro evolution, CUDA allocation or per-frame normalization.')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k not in ('snapshotsSHA256','times')},indent=2))


if __name__=='__main__':main()
