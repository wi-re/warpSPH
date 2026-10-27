"""Offline evidence for the matched Warp blob energy-partition controls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save


def analyze(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from warpsph_remote.sources import HDF5Source
    output.mkdir(parents=True, exist_ok=True)
    runs={};histories={};inputs=[]
    for path in sorted(root.glob('**/diagnosticHistory.json')):
        name=str(path.parent.parent.relative_to(root))
        summary=path.parent/'comparisonRun.json'
        if summary.exists():result=json.loads(summary.read_text())
        else:
            failure=json.loads((path.parent/'failure.json').read_text())
            result=dict(failure,passed=False,termination='failed',settings=json.loads((path.parent.parent/'settings.json').read_text()),
                exportPath=str(next(path.parent.glob('*/trajectory.h5')).parent),
                diagnosticScope='Non-finite terminal state; null diagnostic values retained, no fabricated completion summary.')
        settings=json.loads((path.parent.parent/'settings.json').read_text())
        a=dict(np.load(path.parent.parent/'input.npz'))
        inputs.append(a)
        h=json.loads(path.read_text());histories[name]=h
        thresholds={}
        for level in (.9,.5,.1):
            times=[r['tKH'] for r in h if r['cloudMassFraction']<level]
            thresholds[str(level)]=times[0] if times else None
        run=dict(result,cloudThresholdTimes=thresholds)
        capture=path.parent.parent/'negativeUpdate.json'
        if capture.exists():run['negativeUpdate']=json.loads(capture.read_text())
        resolved=json.loads((Path(result['exportPath'])/'resolvedConfiguration.json').read_text())
        spec=resolved['spec'];solver=resolved['solverConfig']
        assert solver['crkEnergyPartition']==settings['energyPartition']
        assert solver['crkEnergyPredictor']==settings['energyPredictor']
        assert solver['crkViscosityParams']['oneSidedWork']==settings['oneSidedWork']
        assert solver['crkDensityPolicy']=='sameMaterial' and solver['crkEvolution']=='spheral'
        assert spec['kernel']=='QuinticSpline' and spec['n_h']==4.05
        assert spec['params']['viscositySwitch']=='NoneSwitch' and spec['integrationScheme']=='rungeKutta2'
        assert solver['diffusionParams']['C_l']==solver['diffusionParams']['C_q']==1.
        run['resolvedConfigurationSHA256']=hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()
        runs[name]=run
    for a in inputs[1:]:
        for key in inputs[0]:np.testing.assert_array_equal(a[key],inputs[0][key],err_msg=key)
    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in inputs[0].items()}
    colors={'entropy':'tab:blue','dudt':'tab:red','entropy-common-work':'tab:green'}
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for key,h in histories.items():
        if not key.startswith('n256/'):continue
        label=key.split('/')[-1];t=[r['tKH'] for r in h]
        for ax,field in zip(axes,('cloudMassFraction','minEnergy','relativeEnergyDrift')):
            ax.plot(t,[r[field] for r in h],label=label,color=colors[label])
            if not runs[key]['passed']:ax.plot(t[-1],h[-1][field],'x',color=colors[label],ms=9)
    for ax,title in zip(axes,('Original cold cloud mass / initial cloud mass','Minimum specific internal energy','Relative total energy change')):
        ax.set(xlabel=r'$t/\tau_{KH}$',title=title);ax.grid(alpha=.2)
    axes[1].axhline(0,color='k',lw=.6);axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[0].legend()
    fig.savefig(output/'diagnostics.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for key,h in histories.items():
        if not key.startswith('early/'):continue
        line,=ax.plot([r['time'] for r in h],[r['minEnergy'] for r in h],label=key.split('/')[-1])
        if not runs[key]['passed']:
            if h[-1]['minEnergy'] is None:ax.axvline(h[-1]['time'],color='k',ls=':',lw=.8,label='full-policy non-finite stop')
            else:ax.plot(h[-1]['time'],h[-1]['minEnergy'],'x',ms=8,color=line.get_color())
    ax.axhline(0,color='k',lw=.7);ax.set(xlabel='Physical time',ylabel='Minimum specific internal energy',title='Early thermodynamic controls; crosses mark watchdog stops');ax.legend();ax.grid(alpha=.2)
    fig.savefig(output/'earlyEnergy.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,4.5),layout='constrained');snapshots=[]
    for row,policy in enumerate(('entropy','entropy-common-work')):
        result=runs['n256/'+policy];source=HDF5Source(Path(result['exportPath'])/'trajectory.h5')
        for ax,target in zip(axes[row],(1,4,8)):
            idx=int(np.argmin(abs(np.asarray(source.times)/result['settings']['tKH']-target)))
            a=source.read(idx);x=a['positions'];mask=np.ones(len(x),dtype=bool)
            ax.scatter(x[mask,0],x[mask,1],c=a['materials'][mask],s=1,cmap='coolwarm',vmin=0,vmax=1,rasterized=True)
            actual=float(source.times[idx]/result['settings']['tKH']);snapshots.append(dict(policy=policy,targetTKH=target,actualTKH=actual))
            ax.set(xlim=(0,40),ylim=(0,10),aspect='equal',title=f'{policy}, t/KH={actual:.3f}')
    fig.suptitle('Original material labels; ambient blue, cloud red. dudt run failed before these times.')
    fig.savefig(output/'materials.png',dpi=190);plt.close(fig)
    for key in ('positions','velocities','internalEnergies','densities','materials','UIDs'):
        np.testing.assert_array_equal(np.load(root/'n256/dudt/warp/failedState.npz')[key],
                                      np.load(root/'early/dudt/warp/failedState.npz')[key],err_msg=key)
    sources=['src/warpSPH/modules/crk/energyControl.py','src/warpSPH/caseUtils/compressible/blob.py',
             'scripts/study_blobPartition.py','scripts/capture_blobPartitionFailure.py',
             'results/materialEOS/spheral-build/source/src/Hydro/SpecificThermalEnergyPolicy.cc']
    sourceHashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    save(output/'comparison.json',dict(runs=runs,inputArraysSHA256=hashes,sourceSHA256=sourceHashes,snapshots=snapshots,
        scope='Two energy partition formulas inside Warp CRKSPH, plus controls; not a full compiled Spheral trajectory comparison.'))
    print('Analyzed',len(runs),'runs; all initial arrays identical',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-daemon',action='store_true');a=p.parse_args()
    analyze(a.input,a.output)
    if a.verify_daemon:
        from analyze_spheralShockBubble import verify_recordings
        verify_recordings(a.input,a.output)
