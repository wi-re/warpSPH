"""Compare generated-table blob controls against the retained imported control."""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save
from analyze_blobThermal import material_contrast
from analyze_crkEnergySuite import verify


def analyze(root,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report.mkdir(parents=True,exist_ok=True);pr=json.loads((root/'provenance.json').read_text());rows={};states={}
    for name in pr['rows']:
        folder=root/name;data=folder/'warp'
        if not (data/'comparisonRun.json').exists():continue
        r=json.loads((data/'comparisonRun.json').read_text());s=json.loads((folder/'settings.json').read_text())
        a=dict(np.load(data/'finalState.npz'));initial=dict(np.load(folder/'input.npz'));h=json.loads((data/'diagnosticHistory.json').read_text())
        assert r['passed'] and not (data/'failure.json').exists()
        for k in ('masses','UIDs','materials'):np.testing.assert_array_equal(a[k],initial[k])
        for k,v in dict(np.load(data/'initialState.npz')).items():np.testing.assert_array_equal(v,initial[k])
        np.testing.assert_allclose([v['time'] for v in h],np.r_[0,np.cumsum(s['timeGrid'])],rtol=0,atol=1.e-11)
        support=json.loads((folder/'supportAudit.json').read_text());density=json.loads((folder/'densityAudit.json').read_text())
        assert support['rhs']==density['rhs']==2*s['steps'] and len(density['resets'])==s['steps'] and density['negativeEOSCalls']==0
        assert support['maximumSupport']<=2.+1.e-12
        split=1 if '-dt1-' in name else 2
        imported=Path(pr['parent'])/f'blob-dt{split}-weighted/warp'
        reference=Path(f'results/materialEOS/crk_heating_v2_2026-10-20/spheral-{"grid" if split==1 else "halfGrid"}-averaged')
        r.update(vsImported=material_contrast(a,dict(np.load(imported/'finalState.npz'))),vsSpheral=material_contrast(a,dict(np.load(reference/'finalState.npz'))),
            maxMomentumDiagnostic=max(abs(v['relativeMomentumError']) for v in h),supportAudit=support)
        rows[name]=r;states[name]=a
    save(report/'comparison.json',dict(provenance=pr,rows=rows))
    lines=['# Generated table controls','',
        '| Grid/preset | Cloud position RMS vs imported | Cloud P relative L1 vs imported | Cloud position RMS vs Spheral | Cold mass | Max energy drift | Wall s / ms per step |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        c=r['vsImported']['cloud'];lines.append(f"| {name} | {c['positionRMS']:.6g} | {c['pressuresRelativeL1']:.6g} | {r['vsSpheral']['cloud']['positionRMS']:.6g} | {r['final']['cloudMassFraction']:.6g} | {r['maxAbsoluteEnergyDrift']:.3g} | {r['wallSeconds']:.2f} / {r['meanStepMilliseconds']:.2f} |")
    (report/'tables.md').write_text('\n'.join(lines)+'\n')
    if 'blob-dt2-generated' in states and 'blob-dt2-generatedNative' in states:
        native=dict(np.load('results/materialEOS/crk_heating_v2_2026-10-20/spheral-halfGrid-averaged/finalState.npz'))
        imported=dict(np.load(Path(pr['parent'])/'blob-dt2-weighted/warp/finalState.npz'))
        panels=[('Spheral reference',native),('Imported Spheral tables\nWarp before-RHS timing',imported),
            ('Locally generated tables\n1000-point linear inverse',states['blob-dt2-generatedNative']),
            ('Locally generated tables\n4096-point log inverse',states['blob-dt2-generated'])]
        fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
        for ax,(label,a) in zip(axes,panels):
            order=np.argsort(a['materials'],kind='stable');x=a['positions']%np.array([40.,10.])
            ax.scatter(*x[order].T,c=a['materials'][order],s=4,cmap='coolwarm',vmin=0,vmax=1,linewidths=0)
            ax.set(title=label,xlim=(5,11),ylim=(2.5,7.5),aspect='equal')
        fig.suptitle('Exact shared blob input, 1050 shared timesteps, t=1 KH')
        fig.savefig(report/'blobComparison.png',dpi=170);plt.close(fig)
    print('ANALYZED',len(rows),'completed generated-table runs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',required=True,type=Path)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/generatedOwen'));a=p.parse_args()
    (analyze if a.mode=='analyze' else verify)(a.input.resolve(),a.report.resolve())
