"""Accepted-path entropy budgets, saved reset evidence and contact localization."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save
from compare_spheralSod1d import exact
from study_crkEntropy import hashes

TERMS=('reset','density','pressureHeating','viscousHeating','compatible','finalizationDensity')


def retained(report):
    result={}
    root=Path('results/materialEOS/owen_dimensions_2026-10-21')
    for folder in sorted(root.glob('sod-*')):
        path=folder/'densityAudit.json'
        if not path.exists():continue
        audit=json.loads(path.read_text());h=json.loads((folder/'warp/diagnosticHistory.json').read_text())
        resets=np.r_[0.,np.cumsum([r['signedEntropyJump'] for r in audit['resets']])]
        net=np.array([r['entropyChangePerMass'] for r in h]);np.testing.assert_equal(len(resets),len(net))
        index=int(net.argmin())
        result[folder.name]=dict(finalEntropy=float(net[-1]),resetContribution=float(resets[-1]),
            remainingContribution=float(net[-1]-resets[-1]),initialReset=float(resets[1]),
            minimumMean=dict(time=h[index]['time'],net=float(net[index]),reset=float(resets[index]),
                remainder=float(net[index]-resets[index])),
            source=str(folder),scope='Saved accepted-state/reset scalar evidence; remainder was not separately instrumented')
    save(report/'retainedBudgets.json',result)
    return result


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True);old=retained(report)
    pr=json.loads((root/'provenance.json').read_text());assert hashes()==pr['sources']
    rows={};ledgers={};histories={};hotspots={}
    for name in pr['rows']:
        folder=root/name;r=json.loads((folder/'warp/comparisonRun.json').read_text());s=r['settings']
        assert r['passed'] and not (folder/'warp/failure.json').exists()
        a=dict(np.load(folder/'warp/finalState.npz'));initial=dict(np.load(folder/'input.npz'))
        for key in initial:np.testing.assert_array_equal(np.load(folder/'warp/initialState.npz')[key],initial[key])
        for key in ('masses','materials','UIDs'):np.testing.assert_array_equal(a[key],initial[key])
        l=dict(np.load(folder/'entropyLedger.npz'));summary=json.loads((folder/'entropySummary.json').read_text())
        h=json.loads((folder/'warp/diagnosticHistory.json').read_text())
        np.testing.assert_allclose([v['time'] for v in h],np.r_[0,np.cumsum(s['timeGrid'])],rtol=0,atol=1.e-12)
        np.testing.assert_allclose(l['net'],sum(l[t] for t in TERMS),rtol=0,atol=1.e-14)
        np.testing.assert_allclose(summary['groups']['globalBox']['net'],h[-1]['entropyChangePerMass'],rtol=0,atol=1.e-12)
        totals={k:v.sum(axis=0) for k,v in l.items() if k in (*TERMS,'net')}
        masks=dict(globalBox=np.ones(len(a['masses']),bool),interfaceInitial=abs(initial['positions'][:,0])<.05)
        if s['case']=='contact':masks['interfaceInitial']=abs(abs(initial['positions'][:,0])-.25)<.05
        r.update(entropySummary=summary,minimumMeanEntropy=min(v['entropyChangePerMass'] for v in h),
            maxima={k:max(abs(v[k]) for v in h) for k in h[0] if k!='time'})
        rhs=json.loads((folder/'entropyRHS.json').read_text())
        r['integratedMidpointRates']={group:{key:float(s['dt']*sum(v[group][key] for v in rhs if v['stage']==1))
            for key in ('pressureEntropyRate','viscousEntropyRate')} for group in masks}
        if s['case']=='sod':
            q,star=exact(a['positions'][:,0],s['end']);contact=star['velocity']*s['end'];near=abs(a['positions'][:,0]-contact)<.05
            index=np.where(near)[0][np.argmax(a['pressures'][near]-q['pressure'][near])]
            hotspots[name]=dict(UID=int(a['UIDs'][index]),initialX=float(initial['positions'][index,0]),
                finalX=float(a['positions'][index,0]),density=float(a['densities'][index]),
                pressure=float(a['pressures'][index]),exactPressure=float(q['pressure'][index]),
                exactContactPosition=float(contact),terms={k:float(v[index]) for k,v in totals.items()},
                localNegativeCompatibleMassFraction=float(a['masses'][totals['compatible']<-1.e-8].sum()/a['masses'].sum()))
        rows[name]=r;ledgers[name]=l;histories[name]=h
    baseline=Path(pr['baseline'])/'warp/finalState.npz';a=dict(np.load(root/'sod-dt1/warp/finalState.npz'));b=dict(np.load(baseline))
    equality={k:bool(np.array_equal(a[k],b[k])) for k in a};assert all(equality.values())
    save(report/'comparison.json',dict(provenance=pr,rows=rows,hotspots=hotspots,
        baselineBitwiseFields=equality,retainedBudgets=old))
    lines=['# Accepted-path entropy accounting','','| Case/grid | Reset | Continuity + pressure | AV thermal residual | Compatible correction | Net | Wall s / ms per step |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        b=r['entropySummary']['groups']['globalBox']
        lines.append(f"| {name} | {b['reset']:.8g} | {b['density']+b['pressureHeating']:.8g} | {b['viscousHeating']:.8g} | {b['compatible']:.8g} | {b['net']:.8g} | {r['wallSeconds']:.2f} / {r['meanStepMilliseconds']:.2f} |")
    (report/'tables.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for col,group in enumerate(('globalBox','interfaceInitial')):
        for split,style in ((1,'-'),(2,'--')):
            name=f'sod-dt{split}';r=rows[name];l=ledgers[name];m=l['masses'];mask=np.ones(len(m),bool) if group=='globalBox' else abs(l['initialPositions'][:,0])<.05
            def curve(k):return np.r_[0,np.cumsum(l[k][:,mask]@m[mask]/m[mask].sum())]
            t=np.r_[0,np.cumsum(r['settings']['timeGrid'])]
            for key,color in [('net','black'),('reset','tab:red'),('compatible','tab:green')]:
                axes[0,col].plot(t,curve(key),style,color=color,label=f'{key} dt/{split}')
            axes[0,col].plot(t,curve('density')+curve('pressureHeating')+curve('viscousHeating'),style,color='tab:blue',label=f'raw full step dt/{split}')
            axes[1,col].plot(t,curve('density')+curve('pressureHeating'),style,color='tab:orange',label=f'continuity + pressure dt/{split}')
            axes[1,col].plot(t,curve('viscousHeating'),style,color='tab:purple',label=f'AV thermal residual dt/{split}')
        for ax in axes[:,col]:ax.axhline(0,color='grey',lw=.5);ax.set(xlabel='time',ylabel='Mass-mean change in log(P/rho^gamma)');ax.legend(fontsize=8)
        axes[0,col].set_title('Sod: '+('whole periodic box' if group=='globalBox' else 'particles initially |x|<0.05'))
    fig.savefig(report/'budgets.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for split,marker in ((1,'o'),(2,'x')):
        name=f'sod-dt{split}';a=dict(np.load(root/name/'warp/finalState.npz'));l=ledgers[name];x=a['positions'][:,0];near=abs(x-.18549)<.09
        axes[0].plot(x[near],a['pressures'][near],marker+'-',ms=4,label=f'dt/{split}')
        for key,color in [('reset','tab:red'),('compatible','tab:green'),('net','black')]:
            axes[1].plot(x[near],l[key].sum(axis=0)[near],marker,color=color,label=f'{key} dt/{split}')
    axes[0].axhline(hotspots['sod-dt1']['exactPressure'],color='black',lw=1,label='Exact star pressure')
    for ax in axes:ax.axvline(hotspots['sod-dt1']['exactContactPosition'],color='grey',ls=':');ax.set(xlabel='Final x');ax.legend(fontsize=8)
    axes[0].set(title='Contact pressure overshoot persists on dt bisection',ylabel='Pressure')
    axes[1].set(title='Per-particle cumulative entropy budget',ylabel='Change in log(P/rho^gamma)')
    fig.savefig(report/'contactLocalization.png',dpi=170);plt.close(fig)
    print('ANALYZED',len(rows),'closed entropy ledgers; all coarse Sod physical fields bitwise match retained baseline',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/entropyAccounting'));a=p.parse_args()
    analyze(a.input.resolve(),a.report.resolve())
