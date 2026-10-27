"""Saved-state interface audit; no physical trajectory or force modification.

Track one initial bubble-interface layer by UID. Normalize nearest unlike
spacing by the pair mean sqrt(m/rho), and like spacing by own sqrt(m/rho).
Nearest-pair pressure differences include physical gradients; this diagnostic
cannot establish a capillary force or identify a viscosity cause.
"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_spheralShockBubble import primary
from compare_spheralMaterials import save
from warpsph_remote.sources import HDF5Source
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');args=p.parse_args()
root=args.input;rows=[]
for d in sorted(root.glob('n*')):
 s=json.loads((d/'settings.json').read_text());N=s['primaryParticles'];ini=dict(np.load(d/'input.npz'));bm=ini['materials']==1
 ib=np.flatnonzero(bm);air=np.flatnonzero(~bm)
 initialDist=cKDTree(ini['positions'][air]).query(ini['positions'][ib])[0]
 trackedUID=ini['UIDs'][ib[initialDist<1.5/s['nx']]]
 for solver in ('spheral','warp'):
  p=d/solver;summary=p/'comparisonRun.json';meta=json.loads(summary.read_text()) if summary.exists() else {}
  source=HDF5Source(Path(meta.get('exportPath',str(p)))/'trajectory.h5')
  for target in [0.,.05,.1,.15,.2,.215,.28]:
   if target>source.times[-1]+1e-6:continue
   idx=int(np.argmin(abs(np.asarray(source.times)-target)));a=primary(source.read(idx),N) if target else ini.copy()
   t=source.times[idx] if target else 0.;b=np.flatnonzero(a['materials']==1);o=np.flatnonzero(a['materials']==0)
   dist,nn=cKDTree(a['positions'][o]).query(a['positions'][b]);j=o[nn]
   dsame=cKDTree(a['positions'][b]).query(a['positions'][b],k=2)[0][:,1]
   ell=np.sqrt(a['masses']/a['densities']);length=.5*(ell[b]+ell[j]);tracked=np.isin(a['UIDs'][b],trackedUID)
   P=(np.array(s['gamma'])[a['materials']]-1)*a['densities']*a['internalEnergies']
   delta=P[b]-P[j];relative=delta/(.5*(abs(P[b])+abs(P[j])))
   row=dict(resolution=d.name,solver=solver,target=target,time=float(t),trackedBoundaryParticles=int(tracked.sum()),
    medianCrossSpacing=float(np.median(dist[tracked]/length[tracked])),medianSameSpacing=float(np.median(dsame[tracked]/ell[b][tracked])),
    medianCrossDistance=float(np.median(dist[tracked])),medianCrossToSame=float(np.median(dist[tracked]/dsame[tracked])),
    fractionVeryCloseForeign=float(np.mean(dist[tracked]/length[tracked]<.25)),
    medianSignedPressureJump=float(np.median(relative[tracked])),medianAbsolutePressureJump=float(np.median(abs(relative[tracked]))),
    p90AbsolutePressureJump=float(np.quantile(abs(relative[tracked]),.9)))
   rows.append(row);print(d.name,solver,'t',round(t,5),'cross',round(row['medianCrossSpacing'],3),'same',round(row['medianSameSpacing'],3),'ratio',round(row['medianCrossToSame'],3),'dp',round(row['medianAbsolutePressureJump'],3),flush=True)
args.output.parent.mkdir(parents=True,exist_ok=True)
save(args.output,dict(rows=rows,definition='Track helium particles whose nearest air neighbour is initially within 1.5 dx. Cross spacing divided by mean sqrt(m/rho) of nearest unlike pair; same spacing by own sqrt(m/rho). Nearest-pair pressure jumps include physical gradients; not a surface-tension diagnosis. Primary UID selection, actual recording times, no physical integration.'))

if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze_spheralShockBubble import primary
    from warpsph_remote.sources import HDF5Source
    root=args.input/'n80';s=json.loads((root/'settings.json').read_text())
    fig,axs=plt.subplots(2,2,figsize=(11,6),layout='constrained')
    for col,solver in enumerate(('spheral','warp')):
     folder=root/solver;meta=folder/'comparisonRun.json';run=json.loads(meta.read_text()) if meta.exists() else {}
     source=HDF5Source(Path(run.get('exportPath',str(folder)))/'trajectory.h5');idx=np.argmin(abs(np.asarray(source.times)-.15));a=primary(source.read(idx),s['primaryParticles'])
     P=(np.array(s['gamma'])[a['materials']]-1)*a['densities']*a['internalEnergies'];x=a['positions'];show=(x[:,0]>.4)&(x[:,0]<.85)
     axs[0,col].scatter(*x[show].T,c=a['materials'][show],s=5,cmap='coolwarm',vmin=0,vmax=1)
     plot=axs[1,col].scatter(*x[show].T,c=P[show],s=5,cmap='viridis',vmin=0,vmax=4)
     for ax in axs[:,col]:ax.set(xlim=(.4,.85),ylim=(.02,.38),aspect='equal')
     axs[0,col].set_title(solver+' labels; t='+format(source.times[idx],'.4g'))
     axs[1,col].set_title(solver+' pressure from material EOS')
    fig.colorbar(plot,ax=axs[1,:],label='Pressure',shrink=.8)
    fig.suptitle('Coarse interface before first nominal-wall crossing; air blue / helium red')
    fig.savefig(args.output.parent/'interfaceEarly.png',dpi=160)
