"""Compare isolated support-policy links, preserved failures and native blob."""
import argparse
import json
from pathlib import Path
import numpy as np

from compare_spheralMaterials import save
from study_idealH import POLICIES
from study_blobThermal import Measures
from analyze_blobThermal import material_contrast,local_geometry
from analyze_crkEnergySuite import verify


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True);pr=json.loads((root/'provenance.json').read_text());rows={};states={};histories={};refs={};pairs={};temporal={};parity={}
    for name in pr['rows']:
        folder=root/name;data=folder/'warp';success=data/'comparisonRun.json';failure=data/'failure.json'
        if not success.exists() and not failure.exists():continue
        r=json.loads((failure if failure.exists() else success).read_text());s=json.loads((folder/'settings.json').read_text());history=json.loads((data/'diagnosticHistory.json').read_text())
        a=dict(np.load(data/('failedState.npz' if failure.exists() else 'finalState.npz')));initial=dict(np.load(folder/'input.npz'))
        for k,v in dict(np.load(data/'initialState.npz')).items():np.testing.assert_array_equal(v,initial[k])
        np.testing.assert_array_equal(a['UIDs'],initial['UIDs']);np.testing.assert_allclose([h['time'] for h in history],np.r_[0,np.cumsum(s['timeGrid'])][:len(history)],atol=1.e-11,rtol=0)
        audit=json.loads((folder/'supportAudit.json').read_text());density=json.loads((folder/'densityAudit.json').read_text());complete=bool(not failure.exists() and r.get('passed'))
        if complete:
            assert audit['rhs']==density['rhs']==2*s['steps'] and len(density['resets'])==s['steps'] and density['negativeEOSCalls']==0
            assert density['continuityUpdates']==2*s['steps'] and density['maxContinuityUpdateError']==density['maxDensityAliasError']==0.
            assert audit['replacements']==(2*s['steps'] if s['supportPolicy']=='replacement' else 0)
            assert audit['finalReplacements']==(s['steps'] if s['supportPolicy']=='replacement' else 0)
            if s['supportPolicy']!='warp':assert audit['minimumSupport']>=4.e-5 and audit['maximumSupport']<=2.+1.e-13
        r.update(completed=complete,settings=s,last=history[-1],supportAudit=audit,densitySummary=dict(resets=len(density['resets']),negativeEOSCalls=density['negativeEOSCalls']))
        if complete:r['localGeometry']=local_geometry(a)
        rows[name]=r;states[name]=a;histories[name]=history
    for split in (1,2):
        for first,second in zip(POLICIES[:-1],POLICIES[1:]):
            a,b=f'blob-dt{split}-{second}',f'blob-dt{split}-{first}'
            if all(k in rows and rows[k]['completed'] for k in (a,b)):pairs[f'dt{split}-{second}_vs_{first}']=material_contrast(states[a],states[b])
        name=f'blob-dt{split}-warp'
        if name in rows and rows[name]['completed']:
            old=dict(np.load(Path(pr['parent'])/f'blob-dt{split}-paper/warp/finalState.npz'))
            parity[name]={k:float(abs(v-states[name][k]).max()) for k,v in old.items()};assert max(parity[name].values())==0.
        folder=Path(pr['references']['blobGrid' if split==1 else 'blobHalf']);s=json.loads((folder/'settings.json').read_text());s['thermalGamma']=5/3
        ini=dict(np.load(folder/'initializedState.npz'));a=dict(np.load(folder/'finalState.npz'));m=Measures(s,ini);last=json.loads((folder/'diagnosticHistory.json').read_text())[-1]
        refs[f'dt{split}']=dict(final=m(a,last['time']),geometry=local_geometry(a),path=str(folder));states[f'native-dt{split}']=a
        for p in POLICIES:
            key=f'blob-dt{split}-{p}'
            if key in rows and rows[key]['completed']:rows[key]['nativeContrast']=material_contrast(states[key],a)
    for p in POLICIES:
        a,b=f'blob-dt1-{p}',f'blob-dt2-{p}'
        if all(k in rows and rows[k]['completed'] for k in (a,b)):temporal[p]=dict(stateContrast=material_contrast(states[a],states[b]),coldMassDifference=rows[b]['last']['cloudMassFraction']-rows[a]['last']['cloudMassFraction'])
    save(report/'comparison.json',dict(provenance=pr,rows=rows,adjacentPolicyComparisons=pairs,timestep=temporal,references=refs,baselineReplayMaxDifferences=parity))
    lines=['# Support-policy chain at one KH','',
        '| Grid / policy | Complete | Cold mass | Cloud mean u | Cloud x95 | Cloud y-span90 | Ambient neighbor fraction | Maximum support R | Max energy drift | Wall s / ms per step |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in rows.items():
        f=r['last'];v=lambda k:f'{f[k]:.6g}' if r['completed'] else '—';cost=r.get('failure',r)
        lines.append('| '+' | '.join([n,str(r['completed']),*[v(k) for k in ('cloudMassFraction','cloudMeanEnergy','cloudX95','cloudYspan90')],
            f"{r['localGeometry']['ambientFractionIn12NearestCloudNeighbours']:.6g}" if r['completed'] else '—',
            f"{r['supportAudit']['maximumSupport']:.6g}",f"{max(abs(h['relativeEnergyDrift']) for h in histories[n]):.4g}",f"{cost['wallSeconds']:.3f} / {1000*cost['wallSeconds']/max(cost['steps'],1):.3f}"])+' |')
    for n,r in refs.items():
        f=r['final'];lines.append('| '+' | '.join(['stored native '+n,'True',*[f'{f[k]:.6g}' for k in ('cloudMassFraction','cloudMeanEnergy','cloudX95','cloudYspan90')],f"{r['geometry']['ambientFractionIn12NearestCloudNeighbours']:.6g}",'2.0',f"{abs(f['relativeEnergyDrift']):.4g}",'reused'])+' |')
    (report/'tables.md').write_text('\n'.join(lines)+'\n')
    colors={p:f'C{i}' for i,p in enumerate(POLICIES)};fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');keys=('cloudMassFraction','cloudMeanEnergy','cloudX95','cloudYspan90','cloudEntropyChange','relativeEnergyDrift')
    for split in (1,2):
        for p in POLICIES:
            n=f'blob-dt{split}-{p}'
            if n not in histories:continue
            h=histories[n]
            for ax,k in zip(axes.flat,keys):ax.plot([v['tKH'] for v in h],[v[k] for v in h],color=colors[p],ls='--' if split==1 else '-',lw=1,label=f'{p} dt/{split}')
    ref=json.loads((Path(refs['dt2']['path'])/'diagnosticHistory.json').read_text())
    for ax,k in zip(axes.flat,keys):
        if k in ref[0]:ax.plot([v['tKH'] for v in ref],[v[k] for v in ref],color='k',label='Stored native',lw=1.)
        ax.set(xlabel='t / KH',title=k);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=6,ncol=2);axes[1,2].set_yscale('symlog',linthresh=1.e-12);fig.savefig(report/'diagnostics.png',dpi=170);plt.close(fig)
    labels=['native-dt2']+[f'blob-dt2-{p}' for p in ('warp','replacement') if f'blob-dt2-{p}' in rows and rows[f'blob-dt2-{p}']['completed']]
    panel_titles={
        'native':'Spheral reference\nWeighted IdealH\nReplaced at RK state updates',
        'warp':'Existing WarpSPH support\nUnweighted Owen, lattice lookup\nUpdated before each RHS',
        'bounded':'WarpSPH: add native case bounds\nUnweighted Owen, lattice lookup\nUpdated before each RHS',
        'shell':'WarpSPH: switch to shell lookup\nUnweighted Owen, native case bounds\nUpdated before each RHS',
        'nativeLookup':'WarpSPH: exact native lookup\nUnweighted, native case bounds\nUpdated before each RHS',
        'weighted':'WarpSPH: weighted support target\nNative lookup and case bounds\nUpdated before each RHS',
        'replacement':'Native-style support in WarpSPH\nWeighted IdealH, native lookup/bounds\nReplaced at RK state updates',
    }
    def plot(ax,n):
        a=states[n];x=a['positions']%np.array([40.,10.]);order=np.argsort(a['materials'],kind='stable')
        ax.scatter(*x[order].T,c=a['materials'][order],s=5,cmap='coolwarm',vmin=0,vmax=1,linewidths=0)
        key='native' if n.startswith('native-') else n.rsplit('-',1)[-1]
        ax.set(xlim=(5,11),ylim=(2.5,7.5),aspect='equal',xlabel='x',ylabel='y')
        ax.set_title(panel_titles[key],fontsize=10)
    fig,axes=plt.subplots(1,len(labels),figsize=(5*len(labels),4),layout='constrained',squeeze=False)
    for ax,n in zip(axes.flat,labels):plot(ax,n)
    fig.suptitle('t = 1 KH; 1050 shared timesteps; red: original cloud, blue: ambient gas',fontsize=12);fig.savefig(report/'interfaces.png',dpi=180);plt.close(fig)
    timing_labels=['native-dt2']+[f'blob-dt2-{p}' for p in ('weighted','replacement') if f'blob-dt2-{p}' in rows and rows[f'blob-dt2-{p}']['completed']]
    fig,axes=plt.subplots(1,len(timing_labels),figsize=(5*len(timing_labels),4),layout='constrained',squeeze=False)
    for ax,n in zip(axes.flat,timing_labels):plot(ax,n)
    fig.suptitle('Support timing comparison at t = 1 KH; same target, bounds and weights in both Warp runs',fontsize=12)
    fig.savefig(report/'timingComparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,p in zip(axes.flat,POLICIES):
        n=f'blob-dt2-{p}'
        if n in rows and rows[n]['completed']:plot(ax,n)
        else:ax.set_visible(False)
    fig.suptitle('Support choices only; half-dt trajectories at 1 KH');fig.savefig(report/'policyGallery.png',dpi=170);plt.close(fig)
    initial=dict(np.load(Path(pr['parent'])/'blob-dt1-paper/input.npz'));opfile=report/'operators/blobInitial.npz'
    if opfile.exists():
        op=np.load(opfile);x=initial['positions'];keep=(abs(x[:,1]-5)<.2)&(x[:,0]>3)&(x[:,0]<7)
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for label,R,color in [('Native relaxed input',initial['supports'],'k'),('Warp lattice step',op['latticeTarget'],'C0'),('Native lookup, unweighted',op['unweightedTarget'],'C1'),('Native weighted IdealH',op['native'][:,0],'C3')]:
            axes[0].scatter(x[keep,0],R[keep],s=12,label=label,color=color)
            axes[1].scatter(x[keep,0],R[keep]/initial['supports'][keep],s=12,label=label,color=color)
        for ax in axes:ax.set(xlabel='x, |y-5|<0.2');ax.grid(alpha=.2)
        axes[0].set(ylabel='Compact support R');axes[1].set(ylabel='One-step R / native relaxed input R');axes[0].legend(fontsize=8)
        fig.suptitle('Same startup geometry, mass and density; support operator only');fig.savefig(report/'startupSupports.png',dpi=180);plt.close(fig)
    print('Analyzed',len(rows),'support rows; bitwise baseline checks',len(parity),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/idealH'));a=p.parse_args()
    (analyze if a.mode=='analyze' else verify)(a.input.resolve(),a.report.resolve())
