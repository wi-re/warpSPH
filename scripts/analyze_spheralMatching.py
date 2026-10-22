"""Compare full matched trajectories and the earliest retained native updates."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from compare_spheralMaterials import save, FIELDS
from analyze_blobThermal import material_contrast
from analyze_crkEnergySuite import verify


def metrics(a,b,m):
    delta=np.abs(a-b);scale=np.abs(b)
    if delta.ndim>1:
        delta=np.linalg.norm(delta.reshape(len(m),-1),axis=1)
        scale=np.linalg.norm(scale.reshape(len(m),-1),axis=1)
    return dict(weightedAbsoluteL1=float(np.dot(m,delta)/m.sum()),
        weightedRelativeL1=float(np.dot(m,delta)/max(np.dot(m,scale),1.e-100)),maximumAbsolute=float(delta.max()))


def fields(a,b):
    out={k:metrics(a[k],b[k],b['masses']) for k in FIELDS if k not in ('positions','UIDs','kinds','materials','masses')}
    dx=a['positions']-b['positions'];length=np.array([40.,10.]);dx-=length*np.round(dx/length)
    out['positions']=metrics(dx,np.zeros_like(dx),b['masses'])
    # Position uses an absolute norm; its reference-zero relative value is meaningless.
    out['positions'].pop('weightedRelativeL1')
    return out


def native_rhs(rows,index):
    a=rows[(rows[:,0]==index)&(rows[:,3]>0)]
    return dict(masses=a[:,4],positions=a[:,5:7],volumes=a[:,7],densities=a[:,8],pressures=a[:,9],
        internalEnergies=a[:,10],velocities=a[:,11:13],supports=4./a[:,14],A=a[:,17],AB=a[:,18:20],
        velocityGradient=a[:,26:30].reshape(-1,2,2),dudt=a[:,31],acceleration=a[:,34:36])


def align_native_rhs(rows,index,positions):
    # Native State FieldLists use a different NodeList ordering from the
    # example's nodeSet export. Match the early near-identical geometries.
    from scipy.spatial import cKDTree
    b=native_rhs(rows,index);distance,order=cKDTree(b['positions']).query(positions)
    assert len(np.unique(order))==len(order) and distance.max()<1.e-6
    return {k:v[order] for k,v in b.items()},dict(maximumPositionDistance=float(distance.max()),bijective=True)


def analyze(root,report):
    report.mkdir(parents=True,exist_ok=True);pr=json.loads((root/'provenance.json').read_text());rows={};states={};refs={};early={};parity={}
    for split,path in pr['references'].items():
        folder=Path(path);refs[split]=dict(np.load(folder/'finalState.npz'))
        assert json.loads((folder/'summary.json').read_text())['status']=='completed'
        expected=pr['referenceHashes'][split]
        assert all(hashlib.sha256((folder/k).read_bytes()).hexdigest()==v for k,v in expected.items())
    for name in pr['rows']:
        folder=root/name;s=json.loads((folder/'settings.json').read_text());split=name.split('-')[1][2:];data=folder/'warp'
        r=json.loads((data/'comparisonRun.json').read_text());assert r['passed']
        a=dict(np.load(data/'finalState.npz'));initial=dict(np.load(folder/'input.npz'));states[name]=a
        reference=Path(pr['references'][split]);ini=dict(np.load(reference/'initializedState.npz'))
        for k in FIELDS:np.testing.assert_array_equal(initial[k],ini[k]);np.testing.assert_array_equal(np.load(data/'initialState.npz')[k],ini[k])
        for k in ('masses','materials','UIDs','kinds'):np.testing.assert_array_equal(a[k],initial[k])
        density=json.loads((folder/'densityAudit.json').read_text());support=json.loads((folder/'supportAudit.json').read_text());obs=json.loads((folder/'observerAudit.json').read_text())
        assert density['rhs']==obs['rhs']==2*s['steps'] and len(density['resets'])==s['steps']
        assert support['replacements']==obs['updates']==2*s['steps'] and support['finalReplacements']==obs['accepted']==s['steps']
        assert density['negativeEOSCalls']==0 and density['maxContinuityUpdateError']==density['maxDensityAliasError']==0
        contrast=material_contrast(a,refs[split]);contrast['support']=metrics(a['supports'],refs[split]['supports'],a['masses'])
        history=json.loads((data/'diagnosticHistory.json').read_text())
        r.update(nativeContrast=contrast,maximumMomentumError=max(abs(h['relativeMomentumError']) for h in history),
            minimumEnergy=min(h['minEnergy'] for h in history),minimumDensity=min(h['minDensity'] for h in history),
            maximumMassError=max(abs(h['materialMassError']) for h in history))
        rows[name]=r
        if s['matchingPolicy']=='native':
            old=dict(np.load(Path(pr['parent'])/f'blob-dt{split}-replacement/warp/finalState.npz'))
            parity[name]={k:bool(np.array_equal(a[k],old[k])) for k in FIELDS};assert all(parity[name].values()),parity[name]
        rhs=np.load(reference/'heating_nodeAudit.npz')['rows'];checks={}
        reset=np.load(reference/'firstDensityAudit.npz')['rows'];reset=reset[reset[:,12]>0]
        from scipy.spatial import cKDTree
        w=dict(np.load(folder/'earlyStates/rhs_01.npz'));dist,order=cKDTree(reset[:,:2]).query(w['positions'])
        assert dist.max()<1.e-12 and len(np.unique(order))==len(order)
        reset=reset[order];np.testing.assert_array_equal(w['masses'],reset[:,2]);w['AB']=w['A'][:,None]*w['B']
        b=dict(volumes=reset[:,3],supports=4./reset[:,4],densities=reset[:,7],A=reset[:,8],AB=reset[:,9:11],pressures=reset[:,13],internalEnergies=reset[:,14])
        checks['initialReset']={k:metrics(w[k],v,w['masses']) for k,v in b.items()}
        with h5py.File(reference/'trajectory.h5') as f:
            for step in (1,2):
                g=f['states'][f'frame_{step:07d}'];b={k:g[k][()] for k in FIELDS};w=dict(np.load(folder/f'earlyStates/accepted_{step:02d}.npz'))
                checks[f'accepted{step}']=fields(w,b)
        for index in (1,2):
            w=dict(np.load(folder/f'earlyStates/rhs_{index+2:02d}.npz'));w['AB']=w['A'][:,None]*w['B']
            b,mapping=align_native_rhs(rhs,index,w['positions']);np.testing.assert_array_equal(w['masses'],b['masses'])
            checks[f'step2RHS{index}']={k:metrics(w[k],b[k],b['masses']) for k in b if k!='masses'}
            checks[f'step2RHS{index}']['velocityGradientUntransposed']=checks[f'step2RHS{index}']['velocityGradient']
            checks[f'step2RHS{index}']['velocityGradient']=metrics(w['velocityGradient'].transpose(0,2,1),b['velocityGradient'],b['masses'])
            mapping['gradientConvention']='Warp saved stage tensor transposed to native v dyad grad. Raw-layout discrepancy retained; trace and longitudinal Q contractions are transpose invariant.'
            checks[f'step2RHS{index}Mapping']=mapping
        early[name]=checks
    source={k:hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in pr['sources'].items()};assert all(source.values())
    comparisons={}
    for split in (1,2):
        comparisons[str(split)]=material_contrast(states[f'blob-dt{split}-reference'],states[f'blob-dt{split}-native'])
    save(report/'comparison.json',dict(provenance=pr,rows=rows,earlyStates=early,policyContrasts=comparisons,
        baselineReplayBitwise=parity,physicalSourcesUnchanged=source,
        scope='Matched full trajectories, plus evolved early-state comparisons. Native RHS capture is step2, not the first initial RHS. Reference fraction changes final reconstruction only; raw midpoint du/dt remains standard.'))
    lines=['# Matched blob compatible partition controls','',
        '| Steps / fraction | Cloud position RMS vs native | Cloud pressure relative L1 | Cloud energy relative L1 | Cloud velocity relative L1 | Cold mass | Maximum relative E drift | Wall seconds / ms per step |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        c=r['nativeContrast']['cloud'];f=r['final']
        lines.append('| '+' | '.join([f'{r["steps"]} / '+name.rsplit('-',1)[-1],f'{c["positionRMS"]:.8g}',
            *[f'{c[k+"RelativeL1"]:.8g}' for k in ('pressures','internalEnergies','velocities')],
            f'{f["cloudMassFraction"]:.8g}',f'{r["maxAbsoluteEnergyDrift"]:.4g}',f'{r["wallSeconds"]:.2f} / {r["meanStepMilliseconds"]:.2f}'])+' |')
    (report/'tables.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,(title,a) in zip(axes,[('Compiled Spheral reference',refs['2']),('Warp: RHS estimated fraction',states['blob-dt2-native']),('Warp: native final work fraction',states['blob-dt2-reference'])]):
        x=a['positions']%np.array([40.,10.]);order=np.argsort(a['materials'],kind='stable')
        ax.scatter(*x[order].T,c=a['materials'][order],s=5,cmap='coolwarm',vmin=0,vmax=1,linewidths=0)
        ax.set(xlim=(5,11),ylim=(2.5,7.5),aspect='equal',xlabel='x',ylabel='y',title=title)
    fig.suptitle('t = 1 KH, 1050 shared steps; native target and support replacement timing fixed')
    fig.savefig(report/'interfaces.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    keys=('cloudMassFraction','cloudMeanEnergy','relativeEnergyDrift')
    native=json.loads((Path(pr['references']['2'])/'diagnosticHistory.json').read_text())
    histories=[('Spheral',native)]+[(n,json.loads((root/n/'warp/diagnosticHistory.json').read_text())) for n in pr['rows'] if '-dt2-' in n]
    for label,h in histories:
        for ax,k in zip(axes,keys):ax.plot([v['tKH'] for v in h],[v[k] for v in h],label=label)
    for ax,k in zip(axes,keys):ax.set(xlabel='t / KH',ylabel=k);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[-1].set_yscale('symlog',linthresh=1e-14)
    fig.savefig(report/'diagnostics.png',dpi=170);plt.close(fig)
    print((report/'tables.md').read_text(),flush=True)
    for n,c in early.items():
        print(n,{k:{f:round(v[f]['weightedAbsoluteL1' if f=='positions' else 'weightedRelativeL1'],12) for f in ('positions','internalEnergies','supports')} for k,v in c.items() if k.startswith('accepted')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('analyze','verify'));p.add_argument('--input',type=Path,required=True)
    p.add_argument('--report',type=Path,default=Path('reports/materialEOS/spheralMatching'));a=p.parse_args()
    (analyze if a.mode=='analyze' else verify)(a.input.resolve(),a.report.resolve())
