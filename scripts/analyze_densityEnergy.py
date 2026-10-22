"""Matched energy trajectories and same-state compatible/raw increment comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save
from compare_spheralSod1d import exact
from study_densityEnergy import hashes

LABELS={'compatible':'Interim du/dt + compatible','raw':'Interim du/dt alone',
    'entropy':'Entropy-work du/dt alone','entropyCompatible':'Entropy-work du/dt + compatible'}


def contrast(a,b):
    m=a['masses'];result={}
    for k in ('densities','pressures','internalEnergies','supports'):
        result[k+'RelativeL1']=float(np.dot(m,abs(a[k]-b[k]))/np.dot(m,abs(b[k])))
    result['positionRMS']=float(np.sqrt(np.dot(m,np.sum((a['positions']-b['positions'])**2,axis=1))/m.sum()))
    return result


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True);p=json.loads((root/'provenance.json').read_text());assert hashes()==p['sources']
    rows={};states={};comparisons={};baselines={}
    for name in p['rows']:
        folder=root/name;r=json.loads((folder/'warp/comparisonRun.json').read_text());s=r['settings']
        assert r['passed'] and not (folder/'warp/failure.json').exists(),name
        a=dict(np.load(folder/'warp/finalState.npz'));initial=dict(np.load(folder/'input.npz'))
        for k in initial:np.testing.assert_array_equal(np.load(folder/'warp/initialState.npz')[k],initial[k])
        for k in ('masses','materials','UIDs'):np.testing.assert_array_equal(a[k],initial[k])
        history=json.loads((folder/'warp/diagnosticHistory.json').read_text());summary=json.loads((folder/'entropySummary.json').read_text())
        np.testing.assert_allclose([v['time'] for v in history],np.r_[0,np.cumsum(s['timeGrid'])],rtol=0,atol=1.e-12)
        expected=dict(resets=s['steps'],rhs=2*s['steps'],updates=2*s['steps'],finalizations=s['steps'],negativeEOSCalls=0,supportCalls=2*s['steps'])
        assert summary['counters']==expected
        shadow=json.loads((folder/'shadowEnergy.json').read_text());l=dict(np.load(folder/'entropyLedger.npz'));m=a['masses']
        max_difference=float(abs(l['shadowCompatibleEntropy']-l['compatible']).max())
        if s['choice']['compatible']:assert max_difference<2.e-15
        else:assert abs(l['compatible']).max()==0.
        r.update(minimumMeanEntropy=min(v['entropyChangePerMass'] for v in history),
            shadowSummary=dict(maximumEnergyRelativeDifference=max(v['maximumEnergyRelativeDifference'] for v in shadow),
                meanAbsoluteRateDifference=float(np.mean(abs(l['shadowEnergyDifference'])@m/m.sum())/s['dt']),
                maximumCompatibleEnergyClosure=max(abs(v['shadowEnergyRelativeClosure']) for v in shadow),
                sumMeanEntropyDifference=sum(v['meanEntropyDifference'] for v in shadow),
                shadowVsAppliedEntropyDifference=max_difference),
            maxima={k:max(abs(v[k]) for v in history) for k in history[0] if k!='time'})
        if s['case']=='sod':
            q,star=exact(a['positions'][:,0],s['end']);mask=abs(a['positions'][:,0]-star['velocity']*s['end'])<.05
            r['contact']=dict(maximumPressure=float(a['pressures'][mask].max()),minimumPressure=float(a['pressures'][mask].min()),
                exactPressure=star['pressure'],
                UID149={k:float(a[k][np.where(a['UIDs']==149)[0][0]]) for k in ('densities','pressures','internalEnergies')},
                UID150={k:float(a[k][np.where(a['UIDs']==150)[0][0]]) for k in ('densities','pressures','internalEnergies')})
        if s['policy']=='compatible':
            base=Path(p['parent'])/f'{s["case"]}-dt{1 if s["dt"]==.001 else 2}/warp/finalState.npz'
            b=dict(np.load(base));equality={k:bool(np.array_equal(a[k],b[k])) for k in a};assert all(equality.values()),(name,equality);baselines[name]=equality
        rows[name]=r;states[name]=a
    for case in ('sod','contact'):
        for split in (1,2):
            ref=states[f'{case}-dt{split}-compatible']
            for policy in LABELS:comparisons[f'{case}-dt{split}-{policy}']=contrast(states[f'{case}-dt{split}-{policy}'],ref)
    save(report/'energyComparison.json',dict(provenance=p,rows=rows,contrastsVsPreferred=comparisons,
        preferredControlsBitwiseUnchanged=baselines,labels=LABELS))
    lines=['# Density/energy controls','','| Run | Max relative E drift | Final mean log-entropy | Contact Pmax | u(UID150) | Wall s / ms per step |','|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        c=r.get('contact',{});pmax=c.get('maximumPressure',float(states[name]['pressures'].max()))
        u=c.get('UID150',{}).get('internalEnergies');u_text='—' if u is None else f'{u:.6g}'
        lines.append(f"| {name} | {r['maxAbsoluteEnergyDrift']:.5g} | {r['final']['entropyChangePerMass']:.6g} | {pmax:.6g} | {u_text} | {r['wallSeconds']:.2f} / {r['meanStepMilliseconds']:.2f} |")
    (report/'energyTables.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');x=np.linspace(-.5,.5,1600);q,_=exact(x,.2)
    for row,split in enumerate((1,2)):
        for ax,key,field in zip(axes[row],('densities','pressures','internalEnergies'),('density','pressure','energy')):
            ax.plot(x,q[field],color='black',lw=1,label='Exact Sod')
            for policy,marker in zip(LABELS,('o','x','+','.')):
                a=states[f'sod-dt{split}-{policy}'];order=np.argsort(a['positions'][:,0]);mask=abs(a['positions'][order,0])<.5;order=order[mask]
                ax.plot(a['positions'][order,0],a[key][order],marker,ms=3,label=LABELS[policy])
            ax.set(title=f'{field}, dt/{split}',xlabel='x',xlim=(-.5,.5))
        axes[row,0].legend(fontsize=7)
    fig.savefig(report/'energyProfiles.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for policy in LABELS:
        for split,ls in ((1,'-'),(2,'--')):
            r=rows[f'sod-dt{split}-{policy}'];sh=json.loads((root/f'sod-dt{split}-{policy}/shadowEnergy.json').read_text())
            axes[0].plot([v['time'] for v in sh],[v['maximumEnergyRelativeDifference'] for v in sh],ls,label=LABELS[policy]+f' dt/{split}')
        pair=[rows[f'sod-dt{split}-{policy}']['shadowSummary']['meanAbsoluteRateDifference'] for split in (1,2)]
        axes[1].plot([.001,.0005],pair,'o-',label=LABELS[policy])
    axes[0].set(title='Same-state full-step compatible vs integrated thermal candidate',xlabel='time',ylabel='Max particle relative energy difference',yscale='log')
    axes[1].set(title='Difference per unit dt on each evolving trajectory',xlabel='dt',ylabel='Time-mean mass-weighted |delta u| / dt',xscale='log',yscale='log')
    for ax in axes:ax.legend(fontsize=7)
    fig.savefig(report/'sameStateEnergy.png',dpi=170);plt.close(fig)
    print('ANALYZED',len(rows),'completed energy trajectories; four preferred controls bitwise unchanged',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/densityEnergy'));a=p.parse_args()
    analyze(a.input.resolve(),a.report.resolve())
