"""Collect the 2D research evidence and produce a static scientific figure."""
import argparse
import json
from pathlib import Path
import numpy as np
from crkClosure2D import pressure_rhs


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    report.mkdir(parents=True,exist_ok=True)
    static=json.loads((root/'static-final/static.json').read_text())
    modes=[]
    for n in (24,32,48):
        d=np.load(root/f'static-final/n{n}-unequal-jitter0-planar.npz');V=d['volume'];x=d['x']
        p=np.sin(2*np.pi*x.sum(axis=1))
        a,_,_=pressure_rhs(V,V,p,np.zeros_like(x),d['i'],d['j'],d['q'])
        _,dVa,_=pressure_rhs(V,V,np.ones_like(V),a,d['i'],d['j'],d['q'])
        omega2=-np.dot(V*p,-1.4*dVa/V)/np.dot(V,p*p);exact2=1.4*8*np.pi**2
        modes.append(dict(n=n,modeRayleighOmegaSquared=float(omega2),continuumOmegaSquared=float(exact2),
            relativeFrequency=float(np.sqrt(omega2/exact2)),note='Static linearized operator Rayleigh diagnostic, not an eigenmode or evolved trajectory.'))
    (report/'modeDispersion.json').write_text(json.dumps(modes,indent=2)+'\n')
    runs={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/researchRun.json'))}
    def compare(finite_names):
        convergence={}
        if not all(k in runs for k in finite_names):return convergence
        data=[np.load(root/name/'finalState.npz') for name in finite_names]
        for field in ('positions','velocities','volumes','internalEnergies','pressures'):
            delta=[data[k][field]-data[k+1][field] for k in range(2)]
            if field=='positions':delta=[(a+.5)%1-.5 for a in delta]
            norms=[np.linalg.norm(a) for a in delta];ratio=float(norms[0]/norms[1])
            convergence[field]=dict(differenceNorms=norms,ratio=ratio,observedOrder=float(np.log2(ratio)))
        return convergence
    convergence=compare(('finite-short-coarse','finite-short-medium','finite-short-fine'))
    tapered_convergence=compare(('taper-finite-coarse','taper-finite-medium','taper-finite-fine'))
    result=dict(static=static,runs=runs,timeConvergence=convergence,
        timeConvergencePassed=bool(convergence) and all(1.8<v['observedOrder']<2.2 for v in convergence.values()),
        taperedTimeConvergence=tapered_convergence,
        taperedTimeConvergencePassed=bool(tapered_convergence) and all(1.8<v['observedOrder']<2.2 for v in tapered_convergence.values()))
    (report/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    data=np.load(root/'static-final/n24-unequal-jitter0.1-circle.npz');V=data['volume'];i=data['i'];j=data['j'];x=data['x']
    for ax,key,title in zip(axes[0],('q0','q'),('Original CRK pair force','Constrained pair force')):
        force=np.zeros_like(x);np.add.at(force,i,-2*data[key]);np.add.at(force,j,2*data[key]);mag=np.linalg.norm(force/V[:,None],axis=1)
        im=ax.scatter(*x.T,c=np.maximum(mag,1.e-12),s=15,cmap='viridis',norm=LogNorm(vmin=1.e-12,vmax=10))
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='x',ylabel='y',title=title+'; uniform pressure / density');ax.set_aspect('equal')
    fig.colorbar(im,ax=axes[0],label='Spurious acceleration magnitude (shared log scale)',shrink=.85)
    acoustic=[runs[k] for k in ('acoustic-n24','acoustic-n32','acoustic-n48') if k in runs]
    if acoustic:
        ns=[r['n'] for r in acoustic]
        for field,label in [('pressureWaveL1','Pressure'),('velocityWaveL1','Velocity')]:
            axes[1,0].loglog(ns,[r['final'][field]/r['amplitude'] for r in acoustic],'o-',label=label)
        axes[1,0].set(xlabel='Fine lattice resolution n',ylabel='Final volume-weighted L1 error / amplitude',title='Oblique wave at t=0.1; 4:1 area jump');axes[1,0].legend();axes[1,0].grid(alpha=.2)
        axes[1,1].plot([r['particles'] for r in acoustic],[r['meanStepMilliseconds'] for r in acoustic],'o-',label='Full step')
        axes[1,1].plot([r['particles'] for r in acoustic],[1000*r['closureSeconds']/r['steps'] for r in acoustic],'o-',label='Two closure solves / step')
        axes[1,1].set(xlabel='Particles',ylabel='CPU milliseconds',title='Prototype cost (video excluded)');axes[1,1].legend();axes[1,1].grid(alpha=.2)
    fig.savefig(report/'closure2D.png',dpi=180);plt.close(fig)
    if tapered_convergence:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        continuity=json.loads((root/'cutoffContinuity.json').read_text())
        for name in ('rawCRK','untapered','tapered'):
            axes[0].loglog([r['displacementScale'] for r in continuity],
                [r['relativeCoefficientChanges'][name] for r in continuity],'o-',label=name)
        axes[0].set(xlabel='Position perturbation scale',ylabel='Relative edge-coefficient change',title='Compact-edge continuity');axes[0].legend();axes[0].grid(alpha=.2)
        fields=list(convergence);at=np.arange(len(fields));width=.38
        axes[1].bar(at-width/2,[convergence[k]['observedOrder'] for k in fields],width,label='Untapered')
        axes[1].bar(at+width/2,[tapered_convergence[k]['observedOrder'] for k in fields],width,label='Tapered')
        axes[1].axhline(2,color='black',ls=':',lw=1,label='Midpoint target')
        axes[1].set_xticks(at,['Position','Velocity','Volume','Energy','Pressure'],rotation=25)
        axes[1].set(ylabel='Observed time order',ylim=(0,2.3),title='A=.05; t=.1; dt=.004/.002/.001');axes[1].legend()
        fig.savefig(report/'cutoffTaper.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(timeConvergence=convergence,taperedTimeConvergence=tapered_convergence,
        runs={k:dict(passed=v['passed'],msPerStep=v['meanStepMilliseconds']) for k,v in runs.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();analyze(args.input,args.output)
