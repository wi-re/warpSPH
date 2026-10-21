"""Common-time geometry, pressure and circulation for the AV 2x2 controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from compare_spheralMaterials import save
from compare_spheralShockBubble import primary, Measures
from analyze_spheralShockBubble import circulation, reflection_error
from study_shockBubbleViscosity import VARIANTS
from warpsph_remote.sources import HDF5Source


def interface(initial,a,dx):
    ib=np.flatnonzero(initial['materials']==1);ia=np.flatnonzero(initial['materials']==0)
    distance=cKDTree(initial['positions'][ia]).query(initial['positions'][ib])[0]
    ids=initial['UIDs'][ib[distance<1.5*dx]]
    ib=np.flatnonzero(a['materials']==1);ia=np.flatnonzero(a['materials']==0)
    d,idx=cKDTree(a['positions'][ia]).query(a['positions'][ib]);other=ia[idx]
    same=cKDTree(a['positions'][ib]).query(a['positions'][ib],k=2)[0][:,1]
    ell=np.sqrt(a['masses']/a['densities']);mask=np.isin(a['UIDs'][ib],ids)
    jump=(a['pressures'][ib]-a['pressures'][other])/(.5*(abs(a['pressures'][ib])+abs(a['pressures'][other])))
    return dict(medianLikeSpacing=float(np.median(same[mask]/ell[ib][mask])),
        medianUnlikeSpacing=float(np.median(d[mask]/(.5*(ell[ib]+ell[other]))[mask])),
        medianAbsoluteNearestPressureJump=float(np.median(abs(jump[mask]))),trackedParticles=int(mask.sum()))


def analyze(root,reference,out,variants=None,times=(.1,.15,.2),title='AV controls',limits=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings=json.loads((reference/'settings.json').read_text());N=settings['primaryParticles']
    initial=dict(np.load(reference/'input.npz'));sources={};runs={};histories={}
    sources['Spheral']=HDF5Source(reference/'spheral/trajectory.h5')
    for variant in (tuple(VARIANTS) if variants is None else variants):
        folder=root/variant/'warp';meta=json.loads((folder/'comparisonRun.json').read_text());runs[variant]=meta
        settings_control=json.loads((root/variant/'settings.json').read_text())
        config=json.loads((Path(meta['exportPath'])/'config.json').read_text())['schemeConfig']
        assert config['crkViscosityGradient']==settings_control['viscosityGradient']
        assert config['crkViscosityParams']['viscosityKernelScale']==settings_control['viscosityKernelScale']
        assert config['diffusionParams']['C_l']==1. and config['diffusionParams']['C_q']==.75
        if 'energyPartition' in settings_control:
            assert config['crkEnergyPartition']==settings_control['energyPartition']
            assert config['crkEnergyPredictor']==settings_control['energyPredictor']
            assert config['crkViscosityParams']['oneSidedWork']==settings_control['oneSidedWork']
        for filename in ('input.npz','warpInput.npz'):
            expected=dict(np.load(reference/filename));actual=dict(np.load(root/variant/filename))
            for key in expected:np.testing.assert_array_equal(actual[key],expected[key])
        sources[variant]=HDF5Source(Path(meta['exportPath'])/'trajectory.h5')
        histories[variant]=json.loads((folder/'diagnosticHistory.json').read_text())
    out.mkdir(parents=True,exist_ok=True);rows=[];frames={}
    for target in times:
        samples={};states={}
        for name,source in sources.items():
            idx=int(np.argmin(abs(np.asarray(source.times)-target)))
            if abs(source.times[idx]-target)>settings['dt']*1.01:raise ValueError((name,target,source.times[idx]))
            raw=source.read(idx);a=primary(raw,N);states[name]=a
            if 'pressures' not in a:
                a['pressures']=(np.array(settings['gamma'])[a['materials']]-1)*a['densities']*a['internalEnergies']
            m=Measures(settings,initial)(a,source.times[idx]);curl,_=circulation(a)
            sample=dict(time=float(source.times[idx]),bubble={k:v for k,v in m.items() if k.startswith('bubble')},
                interface=interface(initial,a,1/settings['nx']),circulation=curl)
            if name!='Spheral':sample['reflection']=reflection_error(raw,N)
            samples[name]=sample
        for name,a in states.items():
            samples[name]['positionRMSFromReference']=float(np.sqrt(np.mean(np.sum((a['positions']-states['Spheral']['positions'])**2,axis=1))))
            samples[name]['positionRMSFromNative']=float(np.sqrt(np.mean(np.sum((a['positions']-states['native-current']['positions'])**2,axis=1))))
        frames[target]=states;rows.append(dict(target=target,samples=samples))
        print('time',target,{name:dict(like=v['interface']['medianLikeSpacing'],curl=v['circulation']['upper'],rms=v['positionRMSFromReference']) for name,v in samples.items()},flush=True)
    plot_times=times[-2:]
    fig,axes=plt.subplots(len(plot_times),len(sources),figsize=(3.4*len(sources),3*len(plot_times)),layout='constrained',squeeze=False)
    for r,target in enumerate(plot_times):
        for c,name in enumerate(sources):
            a=frames[target][name];ax=axes[r,c]
            ax.scatter(*a['positions'].T,c=a['materials'],s=3*(80/settings['nx'])**2,cmap='coolwarm',vmin=0,vmax=1)
            ax.set(xlim=(.35,.85),ylim=(.02,.38),aspect='equal',title=name+'; t≈'+str(target))
    fig.suptitle(title+': pressure-force formula and coefficients fixed; air blue / helium red')
    fig.savefig(out/'interfaces.png',dpi=160);plt.close(fig)
    save(out/'comparison.json',dict(rows=rows,runs=runs,reference=str(reference),
        limits=limits or 'Controlled viscosity ablation, not full Spheral operator parity. Nearest stored times differ by at most one step. Interface layer tracked by original UID. Pressure jumps include physical gradients. Mirror approximation still qualified separately. No force fitting.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',choices=tuple(VARIANTS))
    a=p.parse_args();analyze(a.input,a.reference,a.output,a.variants)
