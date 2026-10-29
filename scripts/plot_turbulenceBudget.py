#!/usr/bin/env python3
"""Plot retained scale-budget summaries without rereading particle trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',nargs='+')
    args=p.parse_args();rows=json.loads((args.analysis/'summary.json').read_text())
    if args.labels is not None and len(args.labels)!=len(rows):p.error('one label per run is required')
    labels=args.labels or [Path(r['path']).name for r in rows]
    args.output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    keys=[('deformationTransfer','Deformation transfer Π'),('baropycnalTransfer','Baropycnal transfer Λ'),
          ('totalTransfer','Total transfer Π+Λ: positive downscale'),('pressureDilatation','Pressure dilatation: positive feeds KE'),
          ('forcingWork','Resolved forcing work'),('unclosedResidual','Unclosed residual: no viscosity attribution')]
    for i,(row,label) in enumerate(zip(rows,labels)):
        grids=sorted({o['gridSize'] for o in row['observations']});radii=sorted({o['observationRadius'] for o in row['observations']})
        for o in row['observations']:
            linestyle=['-','--',':','-.'][radii.index(o['observationRadius'])%4]
            x=[s['ell']/row['length'] for s in o['scales']]
            for ax,(key,title) in zip(axes.flat,keys):
                y=[s['means'][key] for s in o['scales']]
                if o['gridSize']==grids[-1]:
                    ax.plot(x,y,linestyle,color=colors[i%10],marker='o',label=f"{label}, H={o['observationRadius']:g}")
                else:ax.plot(x,y,'x',color=colors[i%10],ms=5)
    for ax,(_,title) in zip(axes.flat,keys):
        ax.axhline(0,color='.5',lw=.8);ax.set_xscale('log',base=2);ax.grid(alpha=.2)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Gaussian standard deviation ℓ / box length')
        ax.set_ylabel('Mean energy rate / volume')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Compressible Favre budgets, t=10–40; lines: finer grid, ×: coarser grid\nH is the independent compact observation radius; these are filtered-grid diagnostics')
    fig.savefig(args.output/'rates.png',dpi=160);fig.savefig(args.output/'rates.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,len(rows),figsize=(6*len(rows),4),layout='constrained',squeeze=False)
    for row,label,ax in zip(rows,labels,axes[0]):
        n=max(o['gridSize'] for o in row['observations']);h=min(o['observationRadius'] for o in row['observations'])
        o=next(o for o in row['observations'] if o['gridSize']==n and o['observationRadius']==h)
        selected=o['scales'][min(1,len(o['scales'])-1)]['ell']
        source=args.analysis/Path(row['path']).name/f'grid{n}-radius{h:g}.npz'
        with np.load(source) as data:
            t=data['t'];j=int(np.argmin(abs(data['ell'][0]-selected)));k=data['kineticEnergyDensity'][:,j];rhs=data['predictedRate'][:,j]
        lo,hi=row['window'];clock=np.r_[lo,t[(t>lo)&(t<hi)],hi]
        kval=np.interp(clock,t,k);rval=np.interp(clock,t,rhs)
        integral=np.r_[0,np.cumsum(.5*(rval[1:]+rval[:-1])*np.diff(clock))]
        ax.plot(clock,kval-kval[0],label='Observed resolved ΔK')
        ax.plot(clock,integral,label='Integrated inviscid budget terms')
        ax.plot(clock,kval-kval[0]-integral,ls=':',label='Cumulative unclosed residual')
        ax.set_title(f'{label}: ℓ={selected:g}, H={h:g}');ax.set_xlabel('Physical time');ax.set_ylabel('Energy change / volume');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Resolved energy ledger: gap includes observation/sampling/native effects')
    fig.savefig(args.output/'ledger.png',dpi=160);fig.savefig(args.output/'ledger.pdf');plt.close(fig)


if __name__=='__main__':main()
