#!/usr/bin/env python3
"""Bounded Lobovsky impact comparison on explicitly matched post-shift clocks.

Local every-step traces resolve ringing; the saved dynGPUSPH frames only
resolve its impact envelope. Never substitute its earlier-RK probe.plt.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def read_progress(path):
    rows=[json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
    t=np.array([r['t'] for r in rows]);p=np.array([r['pressureProbes'] for r in rows])
    assert np.all(np.diff(t)>0) and np.isfinite(p).all()
    return dict(t=t,p=p,ke=np.array([r['kineticEnergy'] for r in rows]),
        vmax=np.array([r['maxVelocity'] for r in rows]),
        iterations=np.array([r['pseudoIterations'] for r in rows]),rows=rows)


def stats(t,p,lo,hi):
    mask=(t>=lo)&(t<=hi)
    if not mask.any():return None
    u=t[mask];q=p[mask];i=q.argmax();j=q.argmin()
    return dict(peakPa=float(q[i]),peakTime=float(u[i]),minimumPa=float(q[j]),
        minimumTime=float(u[j]),meanPa=float(q.mean()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','reference','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--partial',action='store_true')
    ap.add_argument('--expected-steps',type=int,default=3579)
    ap.add_argument('--before-label',default='warpSPH FP32 — previous Michel shift')
    ap.add_argument('--after-label',default='warpSPH FP32 — dynGPUSPH shift/remap')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    before,after=read_progress(args.before),read_progress(args.after)
    end=after['t'][-1]
    if not args.partial:
        result=json.loads((args.after/'result.json').read_text())
        assert result['nSteps']==args.expected_steps and not result['diverged'] and not result['stop_reason']
    with h5py.File(args.reference) as h:
        rt=h['time'][:];keep=rt<=end+1e-10
        ref=dict(t=rt[keep],p=h['probes/pressure'][:][keep],ke=h['diagnostics/kineticEnergy'][:][keep],vmax=h['diagnostics/maxSpeed'][:][keep])
    nt=len(after['t']);np.testing.assert_allclose(before['t'][:nt],after['t'],atol=1e-12,rtol=0)
    labels={'before':args.before_label,
        'after':args.after_label,
        'reference':'dynGPUSPH FP64 — saved post-shift frames'}
    colors={'before':'#0072b2','after':'#d55e00','reference':'#555555'}
    series={'before':before,'after':after,'reference':ref}
    report=dict(scope=f'Original-IC FP32, {len(after["t"])-1} exact reference timesteps, bounded first impact/ring-down',
        endpoint=float(end),highpassMetric='Uniform 0.2ms resampling, subtract Gaussian sigma5ms; discard10ms at each window edge. This measures fast variation, not error against experiment.',referenceCadenceLimitation='Saved post-shift frames ~10ms; cannot qualify sub-frame native ringing. Earlier-RK probe.plt excluded.',
        windows={'impact':[.38,.46],'ringDown':[.46,min(.65,end)]},methods={})
    for name,s in series.items():
        mask=s['t']<=end+1e-10
        d=dict(pressureImpact=[stats(s['t'],s['p'][:,k],.38,.46) for k in range(4)],
            pressureRingDown=[stats(s['t'],s['p'][:,k],.46,min(.65,end)) for k in range(4)],
            pressureImmediateRing=[stats(s['t'],s['p'][:,k],.417,min(.44,end)) for k in range(4)],
            finalKE=float(np.interp(end,s['t'],s['ke'])),finalMaxSpeed=float(np.interp(end,s['t'],s['vmax'])))
        if name!='reference':
            # Uniform physical-time resampling avoids variable-dt weighting.
            # Gaussian sigma5ms removes the slower impact envelope; this is
            # a declared high-frequency metric, not an experimental noise floor.
            lo,hi=.46,min(.65,end)
            if hi>lo:
                clock=np.arange(lo,hi,2e-4)
                pp=np.stack([np.interp(clock,s['t'],s['p'][:,k]) for k in range(4)],axis=1)
                residual=pp-gaussian_filter1d(pp,25,axis=0,mode='nearest')
                d['ringDownHighpassRMSPa']=np.sqrt(np.mean(residual[50:-50]**2,axis=0)).tolist() if len(residual)>100 else None
            lo,hi=.40,min(.46,end)
            if hi>lo:
                clock=np.arange(lo,hi,2e-4)
                pp=np.stack([np.interp(clock,s['t'],s['p'][:,k]) for k in range(4)],axis=1)
                residual=pp-gaussian_filter1d(pp,25,axis=0,mode='nearest')
                d['impactHighpassRMSPa']=np.sqrt(np.mean(residual[50:-50]**2,axis=0)).tolist() if len(residual)>100 else None
            shared=np.stack([np.interp(ref['t'],s['t'],s['p'][:,k]) for k in range(4)],axis=1)
            use=(ref['t']>=.38)&(ref['t']<=end)
            d['sharedPressureRMSEPa']=np.sqrt(np.mean((shared[use]-ref['p'][use])**2,axis=0)).tolist() if use.any() else None
            ke=np.interp(ref['t'],s['t'],s['ke']);d['sharedKERMSE']=float(np.sqrt(np.mean((ke-ref['ke'])**2)))
            d['capSteps']=int((s['iterations'][1:nt]>=200).sum())
        active=s['t']<=end+1e-10
        d['firstNonzeroProbeTime']=[float(s['t'][np.flatnonzero(active&(s['p'][:,k]>1))[0]])
            if (active&(s['p'][:,k]>1)).any() else None for k in range(4)]
        report['methods'][name]=d
    resultfile=args.after/'result.json'
    if resultfile.exists():
        r=json.loads(resultfile.read_text());report['cost']=dict(wallSeconds=r['wall_seconds'],msPerStep=1000*r['wall_seconds']/r['nSteps'],graphDisabled=r['graph_disabled'])
    solver_cost={}
    for name,run in [('before',args.before),('after',args.after)]:
        timing=[json.loads(l) for l in (run/'solve-timing.jsonl').read_text().splitlines()][:nt-1]
        solver_cost[name]=dict(steps=len(timing),meanSolveMs=float(np.mean([r['solve_wall_ms'] for r in timing])),
            totalPseudoIterations=sum(r['iterations'] for r in timing),graphDisabledSteps=sum(r['disabled'] is not None for r in timing),
            capSteps=sum(r['iterations']>=200 for r in timing))
    report['matchedPrefixSolveCost']=solver_cost
    remap_path=args.after/'reference-shift.jsonl'
    if remap_path.exists():
        remaps=[json.loads(l) for l in (args.after/'reference-shift.jsonl').read_text().splitlines()]
        report['remap']=dict(cumulativeKEChange=float(sum(r['remapKineticEnergyChange'] for r in remaps)),
            cumulativeMomentumChange=np.sum([r['remapMomentumChange'] for r in remaps],axis=0).tolist(),
            largestShiftDx=max(r['maxShiftDx'] for r in remaps),largestPressureRemapPa=max(r['maxPressureRemapPa'] for r in remaps),
            largestVelocityRemap=max(r['maxVelocityRemap'] for r in remaps))
    else:
        report['remap']={'available':False,'reason':'Dedicated reference solver does not record isolated per-shift budgets; total accepted-state dynamics are measured.'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for k,ax in enumerate(axes.flat):
        for name in ('reference','before','after'):
            s=series[name];keep=s['t']<=end+1e-10
            ax.plot(s['t'][keep],s['p'][keep,k]/1000,color=colors[name],label=labels[name],
                lw=1.0 if name!='reference' else 1.2,alpha=.85,
                marker='o' if name=='reference' else None,markersize=2.5)
        ax.set_title(f'P{k+1}, y={[3,15,30,80][k]} mm');ax.set_ylabel('Pressure [kPa]');ax.set_xlabel('Time [s]')
        ax.grid(alpha=.25);ax.set_xlim(.36,max(.37,end));ax.axhline(0,color='#999999',lw=.5)
    fig.suptitle('Lobovsky: first impact and ring-down\nLocal every-step post-shift pressure; dynGPUSPH saved frames (~10 ms)',y=.995)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=1,frameon=False)
    fig.subplots_adjust(left=.07,right=.98,bottom=.075,top=.77,hspace=.32,wspace=.20)
    for ext in ('png','svg'):fig.savefig(args.out/('pressure-first-ring.'+ext),dpi=170)
    plt.close(fig)
    if end>.40:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for ax,k in zip(axes,[0,1]):
            for name in ('reference','before','after'):
                ss=series[name];zoom=(ss['t']>=.395)&(ss['t']<=min(.46,end))
                ax.plot(ss['t'][zoom],ss['p'][zoom,k]/1000,color=colors[name],label=labels[name],
                    lw=1,marker='o' if name=='reference' else None,markersize=3)
            ax.set_xlim(.395,min(.46,end));ax.set_title(f'P{k+1}: impact detail')
            ax.set_xlabel('Time [s]');ax.set_ylabel('Pressure [kPa]');ax.grid(alpha=.25)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=1,frameon=False)
        fig.subplots_adjust(left=.07,right=.98,bottom=.14,top=.74,wspace=.23)
        for ext in ('png','svg'):fig.savefig(args.out/('pressure-impact-detail.'+ext),dpi=170)
        plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for name in ('reference','before','after'):
        s=series[name];keep=s['t']<=end+1e-10
        axes[0].plot(s['t'][keep],s['ke'][keep],label=labels[name],color=colors[name])
        axes[1].plot(s['t'][keep],s['vmax'][keep],color=colors[name])
    axes[0].set_ylabel('Kinetic energy [J/m]');axes[1].set_ylabel('Maximum speed [m/s]');axes[1].set_xlabel('Time [s]')
    for ax in axes:ax.grid(alpha=.25);ax.set_xlim(0,end)
    axes[0].legend();fig.suptitle('Lobovsky dynamics through the first ring');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(args.out/('dynamics-first-ring.'+ext),dpi=170)
    plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
