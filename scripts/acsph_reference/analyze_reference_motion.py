#!/usr/bin/env python3
"""Describe the original rejected trial, including its BDF/shift kinematics."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','capture','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    runtime=args.runtime.resolve();sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts/acsph_reference')]
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    import torch,numpy as np
    from scipy.spatial import cKDTree
    from warpSPH.modules.artificialCompressible.dualTime import bdfSource
    from warpSPH.modules.artificialCompressible.wallClosure import bounceVelocities
    entry=torch.load(args.capture/'entry.pt',map_location='cpu',weights_only=False)
    trial=torch.load(args.capture/'trial.pt',map_location='cpu',weights_only=False)
    summary=json.loads((args.capture/'trial-summary.json').read_text())
    pr=entry['problem'];st=pr.state;R=float(st.supports[0])
    low=pr.positionLow if pr.positionLow is not None else torch.zeros_like(entry['x0'])
    previous=pr.previousPositionStep if pr.previousPositionStep is not None else pr.history[0]-pr.history[1]
    if pr.schemeConfig.acParams.incrementalPositions:
        pr=replace(pr,positionOrigin=entry['x0'],positionLow=low,previousPositionStep=previous)
    physical=st.kinds!=2
    move=trial['coordinates']-trial['cells'];distance=move.norm(dim=-1)
    order=distance.argsort(descending=True)[:20]
    prior_shift=getattr(pr.schemeConfig,'_dyngpusphShiftBDF',torch.zeros_like(entry['v0']))
    rate,_=bdfSource(pr,trial['x'],trial['v'])
    bounced=bounceVelocities(pr.bounce,trial['coordinates'],trial['v'])
    a,b,g=pr.bdf
    cutoff=float(trial['adjacency'].querySupports.min())
    box=float((move[physical].max(0).values-move[physical].min(0).values).norm())
    report={'original':summary,'support':R,'actualCandidateCutoff':cutoff,'candidateCutoffOverR':cutoff/R,
        'verletScale':float(pr.config.verletScale),'maxMotionOverR':float(distance.max())/R,
        'finalPairTriangleCertificate':R+2*float(distance[physical].max())<cutoff,
        'finalRelativeBoxCertificate':R+box<cutoff,
        'relativeMotionBoundingBoxDiameter':box,
        'finite':all(bool(torch.isfinite(trial[k]).all()) for k in ('coordinates','v','p')),
        'entryCoordinatesMatchCandidateOrigin':torch.equal(entry['x0'],trial['cells']),
        'top':[]}
    indices=torch.nonzero(physical).flatten().numpy();positions=trial['coordinates'][physical].numpy()
    tree=cKDTree(positions)
    for i in order.tolist():
        near=indices[tree.query_ball_point(trial['coordinates'][i].numpy(),R)]
        q=(trial['coordinates'][near]-trial['coordinates'][i]).norm(dim=-1)/R
        weights=(1-q).clamp_min(0).pow(4)*(1+4*q)*7/(np.pi*R*R)
        sample=float((weights*st.masses[near]/st.densities[near]).sum())
        report['top'].append({'row':i,'UID':int(st.UIDs[i]),'kind':int(st.kinds[i]),
            'start':entry['x0'][i].tolist(),'end':trial['coordinates'][i].tolist(),
            'displacement':move[i].tolist(),'distance':float(distance[i]),
            'velocity':trial['v'][i].tolist(),'shiftBdfRate':prior_shift[i].tolist(),
            'previousStep':previous[i].tolist(),'bdfRate':rate[i].tolist(),
            'positionEquationResidual':(bounced+prior_shift-rate)[i].tolist(),
            'velocityContributionToStep':(bounced[i]/a).tolist(),
            'shiftContributionToStep':(prior_shift[i]/a).tolist(),
            'historyContributionToStep':(g*previous[i]/a).tolist(),
            'pressure':float(trial['p'][i]),'lambda':float(trial['lam'][i]),
            'physicalNeighborsIncludingSelf':len(near),'samplingDensityRatio':sample})
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=trial['coordinates'].numpy();start=trial['cells'].numpy();p=trial['p'].numpy()
    fluid=st.kinds.numpy()==0;i=order[0].item()
    axs[0].scatter(x[fluid,0],x[fluid,1],s=.15,c='0.4',rasterized=True)
    axs[0].scatter(x[i,0],x[i,1],s=40,c='red',label='Maximum rejected motion');axs[0].legend()
    mask=fluid & (np.linalg.norm(x-x[i],axis=1)<4*R)
    colors=axs[1].scatter(x[mask,0],x[mask,1],s=12,c=p[mask],cmap='coolwarm');fig.colorbar(colors,ax=axs[1],label='Trial pressure [Pa]')
    axs[1].plot([start[i,0],x[i,0]],[start[i,1],x[i,1]],'-o',color='black',ms=4)
    from matplotlib.patches import Circle
    axs[1].add_patch(Circle(start[i],.5*R,fill=False,color='red',label='Old0.5R guard'))
    axs[1].add_patch(Circle(start[i],cutoff,fill=False,color='green',label='Actual candidate cutoff'))
    for ax in axs:ax.set(xlabel='x [m]',ylabel='y [m]',aspect='equal');ax.grid(alpha=.15)
    axs[1].legend(fontsize=8);fig.suptitle('Rejected trial only: no physical step accepted')
    fig.savefig(args.out/'rejected-motion.png',dpi=180)
    print('ORIGINAL TRIAL',json.dumps({key:report[key] for key in ('maxMotionOverR','candidateCutoffOverR','finalPairTriangleCertificate','finite')}),flush=True)
    print('TOP PARTICLE',json.dumps(report['top'][0]),flush=True)


if __name__=='__main__':main()
