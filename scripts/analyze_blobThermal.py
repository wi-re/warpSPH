"""Blob thermal-policy comparison with reused compiled-Spheral reference."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from compare_spheralMaterials import save
from study_blobThermal import Measures
from study_crkEnergySuite import POLICIES
from analyze_crkEnergySuite import contrast,verify


def local_geometry(a):
    box=np.array([40.,10.]);x=a['positions']%box;cloud=a['materials']==1
    tree=cKDTree(x,boxsize=box);_,indices=tree.query(x[cloud],k=13)
    return dict(ambientFractionIn12NearestCloudNeighbours=float(np.mean(a['materials'][indices[:,1:]]==0)),
        caveat='Geometric interpenetration/neighbor diagnostic, not physical scalar mixing or a convergence oracle.')


def read(root):
    provenance=json.loads((root/'provenance.json').read_text());rows={};states={};histories={}
    common=dict(np.load(root/provenance['rows'][0]/'input.npz'))
    for name in provenance['rows']:
        folder=root/name;data=folder/'warp';file=data/'comparisonRun.json';failure=data/'failure.json'
        if not file.exists() and not failure.exists():continue
        s=json.loads((folder/'settings.json').read_text());r=json.loads((file if file.exists() else failure).read_text());h=json.loads((data/'diagnosticHistory.json').read_text())
        r.update(settings=s,numericalFailure=failure.exists(),completed=bool(not failure.exists() and r.get('passed')))
        if failure.exists():r['failure']=json.loads(failure.read_text())
        for field,value in dict(np.load(data/'initialState.npz')).items():np.testing.assert_array_equal(value,common[field])
        np.testing.assert_allclose([v['time'] for v in h],np.r_[0,np.cumsum(s['timeGrid'])][:len(h)],atol=1.e-11,rtol=0)
        a=dict(np.load(data/('failedState.npz' if failure.exists() else 'finalState.npz')))
        np.testing.assert_array_equal(a['UIDs'],common['UIDs']);states[name]=a;histories[name]=h;r['final']=h[-1]
        audit=json.loads((folder/'energyAudit.json').read_text());r['audit']={k:v for k,v in audit.items() if k!='stagePower'}
        p=np.array([[np.nan if v is None else v for v in row] for row in audit['stagePower']]);finite=np.isfinite(p[:,0])
        r['audit']['maximumScaledPowerResidual']=float(np.max(abs(p[finite,0])/np.maximum(p[finite,1],1.e-100)))
        if not s['choice']['compatible']:assert audit['compatibleAccumulations']==audit['compatibleFinalizations']==0
        if r['completed']:
            assert len(h)==s['steps']+1 and audit['negativeEOSCalls']==0
            r['localGeometry']=local_geometry(a)
        rows[name]=r
    reference={}
    for grid,path in provenance['references'].items():
        folder=Path(path);a=dict(np.load(folder/'finalState.npz'));h=json.loads((folder/'diagnosticHistory.json').read_text())
        ref_settings=json.loads((folder/'settings.json').read_text());ref_settings['thermalGamma']=5/3
        ini=dict(np.load(folder/'initializedState.npz'))
        for k,v in common.items():np.testing.assert_array_equal(ini[k],v)
        reference[grid]=dict(path=path,summary=json.loads((folder/'summary.json').read_text()),final=Measures(ref_settings,common)(a,h[-1]['time']),localGeometry=local_geometry(a),history=h)
        states['spheral-'+grid]=a
    return provenance,rows,states,histories,reference


def material_contrast(a,b):
    answer={}
    for tag,label in ((None,'all'),(1,'cloud')):
        keep=np.ones(len(b['masses']),dtype=bool) if tag is None else b['materials']==tag
        x,y=a['positions'][keep],b['positions'][keep];delta=x-y;L=np.array([40.,10.]);delta-=L*np.round(delta/L)
        m=b['masses'][keep];out=dict(positionRMS=float(np.sqrt(np.dot(m,np.sum(delta*delta,axis=1))/m.sum())))
        for field in ('densities','pressures','internalEnergies','velocities'):
            error=abs(a[field][keep]-b[field][keep]);norm=abs(b[field][keep])
            if error.ndim>1:error=np.linalg.norm(error,axis=1);norm=np.linalg.norm(norm,axis=1)
            out[field+'RelativeL1']=float(np.dot(m,error)/max(np.dot(m,norm),1.e-100))
        answer[label]=out
    return answer


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True);provenance,rows,states,histories,references=read(root)
    comparisons={};ref_metrics={};time_metrics={}
    for split,grid in ((1,'grid'),(2,'halfGrid')):
        for policy in POLICIES:
            name=f'dt{split}-{policy}'
            if name not in rows or not rows[name]['completed']:continue
            ref_metrics[name]=material_contrast(states[name],states['spheral-'+grid])
            h=histories[name];ref=references[grid]['history'];t=np.array([r['time'] for r in h]);mask=t>.01*rows[name]['settings']['tKH']
            delta=np.array([r['cloudMassFraction'] for r in h])-np.interp(t,[r['time'] for r in ref],[r['cloudMassFraction'] for r in ref])
            ref_metrics[name]['coldMassCurveRMSAfter001KH']=float(np.sqrt(np.mean(delta[mask]**2)))
        for first,second in [('raw','compatible'),('entropy','compatible'),('entropyCompatible','compatible'),('entropy','entropyCompatible')]:
            a,b=[f'dt{split}-{p}' for p in (first,second)]
            if all(k in rows and rows[k]['completed'] for k in (a,b)):comparisons[f'dt{split}-{first}_vs_{second}']=material_contrast(states[a],states[b])
    for policy in POLICIES:
        a,b='dt1-'+policy,'dt2-'+policy
        if all(k in rows and rows[k]['completed'] for k in (a,b)):
            ea,eb=[abs(rows[k]['final']['relativeEnergyDrift']) for k in (a,b)]
            time_metrics[policy]=dict(energyReductionFactor=ea/max(eb,1.e-100),stateContrast=material_contrast(states[a],states[b]),
                coldMassDifference=rows[b]['final']['cloudMassFraction']-rows[a]['final']['cloudMassFraction'])
    # The unchanged rerun must reproduce the established matched baseline.
    parity={}
    if 'dt1-compatible' in rows and rows['dt1-compatible']['completed']:
        old=dict(np.load(Path(provenance['baseline'])/'warp/finalState.npz'))
        parity={k:float(abs(old[k]-states['dt1-compatible'][k]).max()) for k in old}
        assert max(parity.values())==0.,parity
    save(report/'comparison.json',dict(provenance=provenance,rows=rows,references=references,referenceComparisons=ref_metrics,policyComparisons=comparisons,timestep=time_metrics,baselineReplayMaximumDifferences=parity))
    text=['# Short blob thermal-policy results','',
        '| Grid/policy | Complete | Cold mass | Original-cloud cold mass | Cloud mean u | Cloud entropy change | Relative energy drift | Cloud x95 | Cloud y-span90 | Wall s / ms per step |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        f=r['final'];cost=r.get('failure',r);wall=cost['wallSeconds'];steps=cost['steps']
        v=lambda key:'—' if not r['completed'] or f.get(key) is None else f'{f[key]:.6g}'
        status='yes' if r['completed'] else f"no, t/KH={f['tKH']:.5g}"
        text.append('| '+' | '.join([name,status,*[v(k) for k in ('cloudMassFraction','originalCloudMassFraction','cloudMeanEnergy','cloudEntropyChange','relativeEnergyDrift','cloudX95','cloudYspan90')],f'{wall:.3f} / {1000*wall/max(steps,1):.3f}'])+' |')
    for grid,r in references.items():
        f=r['final'];summary=r['summary'];text.append('| '+' | '.join(['stored Spheral '+grid,'yes',*[f'{f[k]:.6g}' for k in ('cloudMassFraction','originalCloudMassFraction','cloudMeanEnergy','cloudEntropyChange','relativeEnergyDrift','cloudX95','cloudYspan90')],f"{summary['wallSeconds']:.3f} / {1000*summary['wallSeconds']/summary['steps']:.3f}"])+' |')
    (report/'tables.md').write_text('\n'.join(text)+'\n')
    colors=dict(zip(POLICIES,['C0','C1','C2','C3']));fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    keys=('cloudMassFraction','cloudMeanEnergy','cloudX95','cloudEntropyChange','cloudYspan90','relativeEnergyDrift')
    for policy in POLICIES:
        for split in (1,2):
            name=f'dt{split}-{policy}'
            if name not in histories:continue
            h=histories[name]
            for ax,key in zip(axes.flat,keys):ax.plot([r['tKH'] for r in h],[r[key] for r in h],label=name,color=colors[policy],ls='--' if split==1 else '-',lw=1.)
    ref=references['halfGrid']['history']
    for ax,key in zip(axes.flat,keys):
        if key in ref[0]:ax.plot([r['tKH'] for r in ref],[r[key] for r in ref],color='k',label='Stored Spheral fresh CRK',lw=1.)
        ax.set(xlabel='t/KH',ylabel=key);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2);axes[1,2].set_yscale('symlog',linthresh=1.e-12)
    fig.suptitle('Short blob — only thermal RHS and compatible correction vary');fig.savefig(report/'diagnostics.png',dpi=170);plt.close(fig)
    labels=['spheral-halfGrid']+[f'dt2-{p}' for p in POLICIES if f'dt2-{p}' in rows and rows[f'dt2-{p}']['completed']]
    fig,axes=plt.subplots(len(labels),2,figsize=(12,3*len(labels)),layout='constrained')
    for i,name in enumerate(labels):
        if name not in states:continue
        if name in rows and not rows[name]['completed']:continue
        a=states[name];x=a['positions']%np.array([40.,10.]);order=np.argsort(a['materials'],kind='stable')
        for j,key in enumerate(('densities','materials')):
            ax=axes[i,j];im=ax.scatter(*x[order].T,c=a[key][order],s=3.,cmap='turbo' if j==0 else 'coolwarm',vmin=.75 if j==0 else 0.,vmax=10 if j==0 else 1.,linewidths=0)
            ax.set(xlim=(3,13),ylim=(1,9),aspect='equal',title=name+' / '+key,xlabel='x',ylabel='y')
    fig.suptitle('t=1 KH; stored Spheral versus half-step Warp variants, identical initial samples');fig.savefig(report/'interfaces.png',dpi=180);plt.close(fig)
    print('Analyzed',len(rows),'blob rows; bitwise baseline replay:',bool(parity and max(parity.values())==0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/blobThermal'));args=p.parse_args()
    (analyze if args.mode=='analyze' else verify)(args.input.resolve(),args.report.resolve())
