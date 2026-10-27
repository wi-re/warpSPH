"""Density timing A/B: stage/reset diagnostics, exact controls and short blob."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_spheralMaterials import save
from analyze_blobThermal import material_contrast,local_geometry
from analyze_crkEnergySuite import contrast,verify
from study_blobThermal import Measures as BlobMeasures


def endpoints(root,report):
    """Read-only Eq75/76 evaluation at accepted endpoint x and stored H.

    No support solve, integration or modification of the physical outputs.
    This separates diagnostic time centering from the evolved trajectories.
    """
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from study_densityEvolution import context,weighted_change
    from study_crkEnergySuite import Measures as SimpleMeasures
    from warpSPH.modules.crk.terms import crkFactors
    report.mkdir(parents=True,exist_ok=True);out={}
    for folder in sorted(root.iterdir()):
        file=folder/'warp/finalState.npz'
        if not file.exists():continue
        s=json.loads((folder/'settings.json').read_text());a=dict(np.load(file));initial=dict(np.load(folder/'input.npz'))
        _,ctx,*_=context(s)
        state=ctx.SimulationState(UIDcounter=len(a['masses']),**{k:torch.as_tensor(v,device=ctx.device).clone() for k,v in a.items()})
        volume,rho,_=crkFactors(state,ctx.config.domain,ctx.config.kernel,densityPolicy='massReplacement')
        projected={**a,'densities':rho.detach().cpu().numpy()};projected['pressures']=(s['thermalGamma']-1)*projected['densities']*a['internalEnergies']
        measure=(BlobMeasures if s['case']=='blob' else SimpleMeasures)(s,initial)
        out[folder.name]=dict(densityChange=weighted_change(projected['densities'],a['densities'],a['masses']),
            endpointEq76=measure(projected,s['end']),minimumGeometricVolume=float(volume.min()))
        np.savez_compressed(report/(folder.name+'-endpointEq76.npz'),**projected)
        print('Static endpoint Eq76',folder.name,out[folder.name]['densityChange'],flush=True)
    save(report/'endpointEvaluation.json',dict(scope='Read-only geometry at accepted final x and stored last-stage H; no H solve, dynamics or energy adjustment. Physical outputs remain unchanged.',rows=out))


def native_continuity(root,report):
    """Replay the actual Warp continuity operator on exported native RHS fields."""
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPHCore import (ParticleState,DomainDescription,SupportScheme,buildVerletList,
        CRKState,warpOperation,OperationProperties,WarpOperation,GradientScheme,KernelFunctions)
    from warpSPH.modules.crk.materialDensity import materialDensity
    from shockBubbleForceAudit import coefficients
    provenance=json.loads((root/'provenance.json').read_text());folder=Path(provenance['references']['blobHalf'])
    nodes=np.load(folder/'heating_nodeAudit.npz')['rows'];out={};T=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float64).contiguous()
    for phase in np.unique(nodes[:,0]):
        a=nodes[nodes[:,0]==phase];real=a[:,3]>0
        s=ParticleState(T(a[:,5:7]),T(4/a[:,14]),T(a[:,4]),torch.zeros(len(a),device='cuda',dtype=torch.int32),T(a[:,8]))
        s.materials=torch.as_tensor(a[:,1],device='cuda',dtype=torch.int32);s.velocities=T(a[:,11:13])
        assert np.max(abs(a[:,15]))<1.e-12 and np.max(abs(a[:,14]-a[:,16]))<1.e-12
        domain=DomainDescription(T(a[:,5:7].min(0)-10),T(a[:,5:7].max(0)+10),torch.tensor([False,False],device='cuda'),2)
        adj=buildVerletList(s,domain,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
        crk=CRKState(**{k:T(v) for k,v in coefficients(dict(coefficients=a[:,17:26])).items()})
        g=warpOperation(s,OperationProperties(kernel=KernelFunctions.B8,operation=WarpOperation.Gradient,
            supportMode=SupportScheme.Scatter,gradientMode=GradientScheme.Difference),queryValues=s.velocities,
            domain=domain,adjacency=adj,queryVolumes=T(a[:,7]),crkState=crk).mT
        rate=(-s.densities*torch.einsum('...ii->...',g)).cpu().numpy();target=-a[:,8]*(a[:,26]+a[:,29])
        rho=materialDensity(s,domain,KernelFunctions.B8,T(a[:,7]),crk,adj,'massReplacement').cpu().numpy()
        out[str(int(phase))]=dict(continuityAgainstNativeRecordedGradientRelativeL2=float(np.linalg.norm(rate[real]-target[real])/max(np.linalg.norm(target[real]),1.e-100)),
            continuityScaledMaximum=float(np.max(abs(rate[real]-target[real])/(1+abs(target[real])))),
            summedVsEvolvedDensityRelativeL1=float(np.dot(a[real,4],abs(rho[real]-a[real,8]))/np.dot(a[real,4],a[real,8])))
        assert out[str(int(phase))]['continuityScaledMaximum']<2.e-5
        print('Native common-state continuity',phase,out[str(int(phase))],flush=True)
    report.mkdir(parents=True,exist_ok=True);save(report/'nativeContinuity.json',dict(reference=str(folder),phases=out,
        scope='Warp scatter CRK velocity gradient and Eq79 on identical exported compiled-native beginning/midpoint states, native volumes/corrections and explicit ghosts. Target is -rho times trace of native fresh CRK gradient, not an independently exported native DrhoDt. Spheral uses tabulated kernel; Warp direct spline.'))


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((root/'provenance.json').read_text());rows={};histories={};states={};contrasts={};timesteps={};refs={}
    for name in provenance['rows']:
        folder=root/name;data=folder/'warp';failure=data/'failure.json';success=data/'comparisonRun.json'
        if not success.exists() and not failure.exists():continue
        r=json.loads((failure if failure.exists() else success).read_text());s=json.loads((folder/'settings.json').read_text())
        h=json.loads((data/'diagnosticHistory.json').read_text());audit=json.loads((folder/'densityAudit.json').read_text())
        a=dict(np.load(data/('failedState.npz' if failure.exists() else 'finalState.npz')));initial=dict(np.load(folder/'input.npz'))
        for k,v in dict(np.load(data/'initialState.npz')).items():np.testing.assert_array_equal(v,initial[k])
        np.testing.assert_array_equal(a['UIDs'],initial['UIDs'])
        np.testing.assert_allclose([v['time'] for v in h],np.r_[0,np.cumsum(s['timeGrid'])][:len(h)],atol=1.e-11,rtol=0)
        complete=bool(not failure.exists() and r.get('passed'))
        summary=dict(rhs=audit['rhs'],resets=len(audit['resets']),continuityUpdates=audit['continuityUpdates'],
            maximumStageDensityVsSumL1=max(v['relativeL1'] for v in audit['stageComparisons']),
            maximumStageDensityVsSumPointwise=max(v['relativeMax'] for v in audit['stageComparisons']),
            maxContinuityUpdateError=audit['maxContinuityUpdateError'],maxDensityAliasError=audit['maxDensityAliasError'],negativeEOSCalls=audit['negativeEOSCalls'])
        if audit['resets']:
            summary.update(maximumResetL1=max(v['relativeL1'] for v in audit['resets']),
                maximumResetPointwise=max(v['relativeMax'] for v in audit['resets']),
                sumSignedEntropyResetJumps=sum(v['signedEntropyJump'] for v in audit['resets']),
                sumAbsoluteEntropyResetJumps=sum(v['entropyJumpL1'] for v in audit['resets']))
            summary['initialReset']=audit['resets'][0]
            summary['maximumResetL1AfterInitial']=max((v['relativeL1'] for v in audit['resets'][1:]),default=0.)
            summary['maximumSampledStageL1AfterFirstStep']=max((v['relativeL1'] for v in audit['stageComparisons'] if v['rhs']>2),default=0.)
        if complete:
            assert audit['rhs']==2*s['steps'] and audit['negativeEOSCalls']==0
            assert len(audit['resets'])==(s['steps'] if s['schedule']=='paper' else 0)
            assert audit['continuityUpdates']==(2*s['steps'] if s['schedule']=='paper' else 0)
            assert audit['maxContinuityUpdateError']<1.e-12 and audit['maxDensityAliasError']==0.
        r.update(completed=complete,settings=s,last=h[-1],densityAudit=summary)
        if s['case']=='blob' and complete:r['localGeometry']=local_geometry(a)
        rows[name]=r;histories[name]=h;states[name]=a
    for case in ('wave','sod','blob'):
        for split in (1,2):
            names=[f'{case}-dt{split}-{v}' for v in ('paper','summation')]
            if all(n in rows and rows[n]['completed'] for n in names):
                a,b=names
                contrasts[f'{case}-dt{split}']=material_contrast(states[a],states[b]) if case=='blob' else contrast(states[a],states[b],rows[a]['settings']['domain'])
                for k in dict(np.load(root/a/'input.npz')):
                    np.testing.assert_array_equal(np.load(root/a/'input.npz')[k],np.load(root/b/'input.npz')[k])
        for schedule in ('paper','summation'):
            names=[f'{case}-dt{i}-{schedule}' for i in (1,2)]
            if all(n in rows and rows[n]['completed'] for n in names):
                a,b=names;timesteps[case+'-'+schedule]=dict(finalDiagnosticDifferences={k:rows[b]['last'][k]-rows[a]['last'][k] for k in rows[a]['last'] if rows[a]['last'][k] is not None and rows[b]['last'][k] is not None},
                    state=material_contrast(states[a],states[b]) if case=='blob' else contrast(states[a],states[b],rows[a]['settings']['domain']))
    for grid,name in ((1,'blobGrid'),(2,'blobHalf')):
        folder=Path(provenance['references'][name]);a=dict(np.load(folder/'finalState.npz'));s=json.loads((folder/'settings.json').read_text());s['thermalGamma']=5/3
        initial=dict(np.load(folder/'initializedState.npz'));native_h=json.loads((folder/'diagnosticHistory.json').read_text());measure=BlobMeasures(s,initial)
        refs[name]=dict(path=str(folder),final=measure(a,native_h[-1]['time']),localGeometry=local_geometry(a));states[name]=a
        for schedule in ('summation','paper'):
            n=f'blob-dt{grid}-{schedule}'
            if n in rows and rows[n]['completed']:rows[n]['nativeStateContrast']=material_contrast(states[n],a)
    parity={}
    if 'blob-dt1-summation' in rows and rows['blob-dt1-summation']['completed']:
        old=dict(np.load(Path(provenance['baseline'])/'warp/finalState.npz'))
        parity={k:float(abs(v-states['blob-dt1-summation'][k]).max()) for k,v in old.items()}
        assert max(parity.values())==0.,parity
    save(report/'comparison.json',dict(provenance=provenance,rows=rows,policyComparisons=contrasts,timestep=timesteps,references=refs,baselineReplayMaxDifferences=parity))
    lines=['# Density timing controls','',
        '| Case / grid / density | Complete | Density L1 | Pressure L1 | Cold mass | Entropy L1 | Max energy drift | Wall s / ms per step |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for n,r in rows.items():
        f=r['last'];v=lambda k:f'{f[k]:.6g}' if k in f and f[k] is not None and r['completed'] else '—'
        lines.append('| '+' | '.join([n,str(r['completed']),v('densityL1'),v('pressureL1'),v('cloudMassFraction'),v('entropyL1PerMass'),f"{max(abs(h['relativeEnergyDrift']) for h in histories[n]):.4g}",f"{r['wallSeconds']:.3f} / {1000*r['wallSeconds']/r['steps']:.3f}"])+' |')
    (report/'tables.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
    for i,(case,keys) in enumerate((('wave',('waveVelocityAmplitude','waveVelocityPhase','entropyL1PerMass')),('sod',('densityL1','pressureL1','entropyL1PerMass')),('blob',('cloudMassFraction','cloudMeanEnergy','cloudX95')))):
        for split in (1,2):
            for schedule,color in (('summation','C0'),('paper','C1')):
                name=f'{case}-dt{split}-{schedule}'
                if name not in histories:continue
                h=histories[name]
                for ax,key in zip(axes[i],keys):ax.plot([v['time'] for v in h],[v[key] for v in h],color=color,ls='--' if split==1 else '-',label=f'{schedule} dt/{split}')
        for ax,key in zip(axes[i],keys):ax.set(title=case+' / '+key,xlabel='time');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Density timing only; same geometric volume, support, AV and energy');fig.savefig(report/'diagnostics.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for i,case in enumerate(('wave','sod','blob')):
        for split in (1,2):
            name=f'{case}-dt{split}-paper'
            if name not in rows:continue
            audit=json.loads((root/name/'densityAudit.json').read_text());r=audit['resets']
            axes[i,0].plot([v['time'] for v in r],[v['relativeL1'] for v in r],label=f'dt/{split}')
            axes[i,1].plot([v['time'] for v in r],np.cumsum([v['signedEntropyJump'] for v in r]),label=f'dt/{split}')
        axes[i,0].set(title=case+' / density reset relative L1',xlabel='time');axes[i,0].set_yscale('symlog',linthresh=1.e-12)
        axes[i,1].set(title=case+' / cumulative signed entropy reset jumps',xlabel='time')
        for ax in axes[i]:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Reset changes rho and P at fixed u; this is not a total-energy error');fig.savefig(report/'resets.png',dpi=170);plt.close(fig)
    labels=['blobHalf']+[f'blob-dt2-{v}' for v in ('summation','paper') if f'blob-dt2-{v}' in rows and rows[f'blob-dt2-{v}']['completed']]
    fig,axes=plt.subplots(1,len(labels),figsize=(5*len(labels),4),layout='constrained',squeeze=False)
    for ax,n in zip(axes.flat,labels):
        a=states[n];x=a['positions']%np.array([40.,10.]);order=np.argsort(a['materials'],kind='stable')
        ax.scatter(*x[order].T,c=a['materials'][order],s=5,cmap='coolwarm',vmin=0,vmax=1,linewidths=0)
        ax.set(xlim=(5,11),ylim=(2.5,7.5),aspect='equal',title=n,xlabel='x',ylabel='y')
    fig.suptitle('Short blob at 1 KH; stored native versus density-only Warp controls');fig.savefig(report/'interfaces.png',dpi=180);plt.close(fig)
    print('Analyzed',len(rows),'density rows; baseline replay bitwise',bool(parity and max(parity.values())==0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify','endpoints','native'));p.add_argument('--input',required=True,type=Path)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/densityEvolution'));args=p.parse_args()
    dict(analyze=analyze,verify=verify,endpoints=endpoints,native=native_continuity)[args.mode](args.input.resolve(),args.report.resolve())
