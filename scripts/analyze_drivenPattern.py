#!/usr/bin/env python3
"""Separate rigid translation/amplitude change from deformation in saved flow.

CPU-only periodic Fourier alignment on a declared conservative observation map.
This describes filtered Eulerian patterns, not a Lagrangian chaos diagnostic.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
from scipy.optimize import minimize


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def align(reference,target):
    n=reference.shape[0];axes=(0,1)
    r=reference-reference.mean(axis=axes);v=target-target.mean(axis=axes)
    rh=np.fft.fftn(r,axes=axes)/n**2;vh=np.fft.fftn(v,axes=axes)/n**2
    cross=np.sum(rh.conj()*vh,axis=-1);rr=np.sum(abs(rh)**2);vv=np.sum(abs(vh)**2)
    if min(rr,vv)<=0:raise ValueError('need nonzero mean-subtracted fields')
    k=np.stack(np.meshgrid(np.fft.fftfreq(n)*n,np.fft.fftfreq(n)*n,indexing='ij'),axis=-1)
    integer=np.unravel_index(np.argmax(np.fft.ifftn(cross).real),cross.shape);initial=np.array(integer)/n
    def objective(s):
        z=cross*np.exp(2j*np.pi*np.sum(k*s,axis=-1))
        return -float(z.real.sum()),-np.sum((z[...,None]*2j*np.pi*k).real,axis=(0,1))
    fit=minimize(objective,initial,jac=True,method='BFGS',options=dict(gtol=1e-11))
    c=-fit.fun;corr=float(np.clip(c/np.sqrt(rr*vv),-1,1));raw=float(cross.real.sum()/np.sqrt(rr*vv))
    return dict(rawCorrelation=raw,alignedCorrelation=corr,residualEnergyFractionAfterShiftAndGain=max(0.,1-corr*corr),fittedGain=float(c/rr),shiftBoxLengths=((fit.x+.5)%1-.5).tolist(),optimizerSuccess=bool(fit.success),gradientNorm=float(np.linalg.norm(fit.jac)))


def controls():
    n=32;x,y=np.meshgrid(np.arange(n)/n,np.arange(n)/n,indexing='ij');reference=np.stack((np.sin(2*np.pi*y)+.2*np.cos(2*np.pi*(2*x+y)),np.cos(2*np.pi*x)+.1*np.sin(4*np.pi*y)),axis=-1)
    shift=np.array([.137,-.211]);kx,ky=np.meshgrid(np.fft.fftfreq(n)*n,np.fft.fftfreq(n)*n,indexing='ij')
    translated=np.fft.ifftn(np.fft.fftn(reference,axes=(0,1))*np.exp(2j*np.pi*(kx*shift[0]+ky*shift[1]))[...,None],axes=(0,1)).real*1.7
    exact=align(reference,translated);assert exact['alignedCorrelation']>1-1e-12 and abs(exact['fittedGain']-1.7)<1e-11
    assert np.max(np.abs((np.asarray(exact['shiftBoxLengths'])+shift+.5)%1-.5))<1e-10
    perturbed=translated.copy();perturbed[...,0]+=.3*np.sin(6*np.pi*x)
    deformation=align(reference,perturbed);expected=.3**2/2/(np.mean(np.sum(translated**2,axis=-1))+.3**2/2)
    assert abs(deformation['residualEnergyFractionAfterShiftAndGain']-expected)<1e-12
    return dict(knownSubgridShiftAndGain=exact,knownOrthogonalDeformation=deformation,expectedDeformationFraction=expected)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=120);p.add_argument('--end',type=float,default=160);p.add_argument('--grid',type=int,default=32);p.add_argument('--radius',type=float,default=.0625)
    a=p.parse_args();control=controls();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);run=a.run.resolve();start=time.perf_counter();repo=Path(__file__).resolve().parents[1]
    utility=load(repo/'src/warpSPH/utils/turbulenceBudget.py','pattern_map');clock=utility.AcceptedSampleClock(.1)
    times=np.array(json.loads((run/'sampling.json').read_text())['times']);paths=sorted((run/'samples').glob('*.npz'));fields=[];densities=[];actual=[];evidence={};massBulk=[]
    n=a.grid;eng=utility.PeriodicBudget((n,n));k=np.fft.fftfreq(n)*n;k2=k[:,None]**2+k[None,:]**2;low=[];residual=[];vorticity=[]
    for i,t in enumerate(times):
        if not clock.due(t) or t<a.start-.01 or t>a.end:continue
        with np.load(paths[i]) as z:s={k:z[k] for k in z.files}
        m=s['masses'];v=s['velocities'];massBulk.append(np.linalg.norm(np.sum(m[:,None]*v.astype(float),axis=0)/m.sum()))
        g=utility.depositConservative(s['positions'],m,v,s['densities'],s['pressures'],s['internalEnergies'],np.zeros_like(v),gridSize=n,radius=a.radius)
        f=g['velocity'];assert np.isfinite(f).all();f=f-f.mean(axis=(0,1));fourier=np.fft.fftn(f,axes=(0,1));box=np.fft.ifftn(fourier*(k2==1)[...,None],axes=(0,1)).real
        fields.append(f);densities.append(g['density']);low.append(box);residual.append(f-box);vorticity.append(eng.derivative(f[...,1],0)-eng.derivative(f[...,0],1));actual.append(t);evidence[paths[i].name]=hashlib.sha256(paths[i].read_bytes()).hexdigest()
    actual=np.array(actual);fields=np.asarray(fields);low=np.asarray(low);residual=np.asarray(residual);vorticity=np.asarray(vorticity)
    rows=[]
    for j,f in enumerate(fields):
        row=align(fields[0],f);row.update(t=float(actual[j]),boxFraction=float(np.sum(low[j]**2)/np.sum(f**2)),residualVelocityRMS=float(np.sqrt(np.mean(np.sum(residual[j]**2,axis=-1)))),vorticityRMS=float(np.sqrt(np.mean(vorticity[j]**2))));rows.append(row)
    np.savez_compressed(out/'fields.npz',t=actual,velocity=fields,boxVelocity=low,residualVelocity=residual,vorticity=vorticity,density=densities)
    picks=[int(np.argmin(abs(actual-t))) for t in np.linspace(a.start,a.end,5)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,5,figsize=(15,9),layout='constrained');centers=(np.arange(n)+.5)/n;xx,yy=np.meshgrid(centers,centers,indexing='ij');sl=(slice(None,None,3),slice(None,None,3))
    maxResidual=max(float(np.linalg.norm(residual[j],axis=-1).max()) for j in picks);maxOmega=max(float(abs(vorticity[j]).max()) for j in picks);maxSpeed=max(float(np.linalg.norm(fields[j],axis=-1).max()) for j in picks)
    for col,j in enumerate(picks):
        ax=axes[0,col];im0=ax.imshow(np.linalg.norm(fields[j],axis=-1).T,origin='lower',extent=(0,1,0,1),vmin=0,vmax=maxSpeed,cmap='viridis');ax.quiver(xx[sl],yy[sl],low[j][sl][...,0],low[j][sl][...,1],color='white',scale_units='xy',scale=8,angles='xy');ax.set_title(f't={actual[j]:.3f}')
        im1=axes[1,col].imshow(np.linalg.norm(residual[j],axis=-1).T,origin='lower',extent=(0,1,0,1),vmin=0,vmax=maxResidual,cmap='magma')
        im2=axes[2,col].imshow(vorticity[j].T,origin='lower',extent=(0,1,0,1),vmin=-maxOmega,vmax=maxOmega,cmap='RdBu_r')
        for row in range(3):axes[row,col].set(xlabel='x (periodic box)',aspect='equal')
    for ax,label in zip(axes[:,0],('Full speed + box-mode arrows','Speed after removing box modes','Filtered vorticity')):ax.set_ylabel(label)
    for im,row,label in zip((im0,im1,im2),range(3),('Speed','Residual speed','Vorticity')):fig.colorbar(im,ax=list(axes[row]),shrink=.65,label=label)
    fig.suptitle('Seed1234 late flow: fixed scales; conservative grid32 / radius0.0625\nCoordinates wrapped into [0,1) for display; no temporal interpolation or new hydro')
    for ext in ('png','pdf'):fig.savefig(out/f'late-flow.{ext}',dpi=150)
    fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained');axes[0].plot(actual,[r['rawCorrelation'] for r in rows],label='Fixed coordinates');axes[0].plot(actual,[r['alignedCorrelation'] for r in rows],label='Best periodic translation');axes[0].set(ylabel='Velocity correlation with t≈120');axes[0].legend()
    axes[1].plot(actual,[r['residualEnergyFractionAfterShiftAndGain'] for r in rows]);axes[1].set(xlabel='Time',ylabel='Energy fraction unexplained\nby translation + uniform gain')
    for ax in axes:ax.grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(out/f'pattern-alignment.{ext}',dpi=160)
    data=dict(run=str(run),window=[float(actual[0]),float(actual[-1])],grid=n,radius=a.radius,controls=control,rows=rows,massBulkVelocityMax=float(max(massBulk)),evidence=evidence,wallSeconds=time.perf_counter()-start,driverSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),utilitySHA256=hashlib.sha256((repo/'src/warpSPH/utils/turbulenceBudget.py').read_bytes()).hexdigest(),limitations='Observed vector-field similarity with one fixed reference, allowing periodic translation and one scalar gain. High correlation is not a proof of laminar flow; low correlation is not proof of chaos. Fixed observation filter; vorticity is spectrally differentiated observation data, not native particle vorticity.')
    (out/'summary.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:v for k,v in data.items() if k not in ('rows','evidence')},indent=2));print('Selected frames',json.dumps([rows[j] for j in picks],indent=2))


if __name__=='__main__':main()
