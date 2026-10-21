"""Analyze native frozen-stage force bridges and controlled Warp replays.

All counterfactuals are static evaluations, never physical trajectories.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from compare_spheralMaterials import save
from shockBubbleForceAudit import (native_split, coefficients, delta,
                                   q_pressures, reduce_pairs)


def doubles(path):
    return {k:v.astype(float) if np.issubdtype(v.dtype,np.floating) else v
            for k,v in dict(np.load(path)).items()}


def scalar_sum(a,wi,wj):
    out=np.zeros(len(a['masses']));np.add.at(out,a['i'],wi)
    if 'coefficients' in a:np.add.at(out,a['j'],wj)
    return out


def rms(x):return float(np.sqrt(np.mean(np.square(x))))


def interface_mask(a):
    x=a['positions'];mask=a['real']&(x[:,0]>.4)&(x[:,0]<.85)&(x[:,1]>.04)&(x[:,1]<.36)
    if 'coefficients' not in a:mask &= a['UIDs']<2816
    return mask


def energy_controls(a,split):
    """Same fields/Q, alternative raw rates and compatible-work partitions.

    dt=0 rates compare spatial partition only, independently of the two
    integrators' finite-step compatible-work accumulation schedules.
    """
    i,j=a['i'],a['j'];m=a['masses'];V=a['volumes'];P=a['pressures']
    qi,qj=q_pressures(a,2. if 'coefficients' in a else 1.778002)
    G=.5*(split['gradientI']-split['gradientJ'])
    s=np.sum((a['velocities'][i]-a['velocities'][j])*G,axis=1);pairV=V[i]*V[j]
    rawS=scalar_sum(a,pairV/m[i]*(P[j]+qj)*s,pairV/m[j]*(P[i]+qi)*s)
    rawW=scalar_sum(a,pairV/m[i]*(P[j]+.5*(qi+qj))*s,pairV/m[j]*(P[i]+.5*(qi+qj))*s)
    pair=split['pressurePair']+split['viscousPair']
    work=-np.sum((a['velocities'][i]-a['velocities'][j])*pair,axis=1)
    eps=np.finfo(float).eps
    def spheral_fraction(work):
        # finalizeDerivatives has applied reflecting ghost boundaries to these
        # node rates. A direct pair sum over ghost rows is not that BC fill.
        weights=a['nativeDudt'] if 'coefficients' in a else rawS
        wi=np.maximum(eps,weights[i]*np.sign(work));wj=np.maximum(eps,weights[j]*np.sign(work))
        return wi/(wi+wj)
    def warp_fraction(work):
        gamma=np.array([1.4,1.648])[a['materials']]
        entropy=P/a['densities']**gamma
        lo=np.minimum(abs(entropy[i]),abs(entropy[j]));hi=np.maximum(abs(entropy[i]),abs(entropy[j]))
        choose_low=((work>=0)&(entropy[i]>=entropy[j]))|((work<0)&(entropy[i]<entropy[j]))
        f=np.where(choose_low,lo/(lo+hi+1e-14),hi/(lo+hi+1e-14))
        return np.where((gamma[i]!=gamma[j])|(entropy[i]==entropy[j]),.5,f)
    fs=spheral_fraction(work);fw=warp_fraction(work)
    hs=scalar_sum(a,fs*work,(1-fs)*work*m[i]/m[j])
    hw=scalar_sum(a,fw*work,(1-fw)*work*m[i]/m[j])
    mask=interface_mask(a);unlike=a['materials'][i]!=a['materials'][j]
    f_cross=np.where(unlike,fw,fs);f_like=np.where(unlike,fs,fw)
    h_cross=scalar_sum(a,f_cross*work,(1-f_cross)*work*m[i]/m[j])
    h_like=scalar_sum(a,f_like*work,(1-f_like)*work*m[i]/m[j])
    active=unlike&mask[i]&(abs(work)>1e-12)
    rel=lambda x,y:float(np.linalg.norm((x-y)[mask])/max(float(np.linalg.norm(y[mask])),1e-300))
    checks={}
    if 'nativeNextEnergyReal' in a:
        multiplier=float(a['nativeNextTime']-a['time'])
        np.testing.assert_allclose(multiplier,float(a['dt']),rtol=0,atol=1e-14)
        vh=a['velocities']+.5*multiplier*a['nativeAcceleration']
        wf=-np.sum((vh[i]-vh[j])*a['nativePairAcceleration'],axis=1)
        f=spheral_fraction(wf)
        rate=scalar_sum(a,f*wf,(1-f)*wf*m[i]/m[j])
        expected=a['internalEnergies'][a['real']]+multiplier*rate[a['real']]
        actual=a['nativeNextEnergyReal']
        checks['compiledHalfStepEnergyMaxAbs']=float(np.max(abs(expected-actual)))
        change=actual-a['internalEnergies'][a['real']]
        checks['compiledHalfStepIncrementRelativeL2']=float(np.linalg.norm(expected-actual)/max(float(np.linalg.norm(change)),1e-300))
    if 'energyFraction' in a:
        # Exactly the finite-half-step velocity passed to Warp's native balance
        # call. Its saved f validates the independent entropy partition bridge.
        # Retain native float32 signs for nearly zero-work pairs; promoting
        # their arithmetic can flip the sign and hence the entropy branch.
        vh=a['velocities'].astype(np.float32)+np.float32(.5*float(a['dt']))*a['nativeAcceleration'].astype(np.float32)
        native_pair=a['pressurePairAcceleration'].astype(np.float32)+a['viscousPairAcceleration'].astype(np.float32)
        wf=-np.sum((vh[i]-vh[j])*native_pair,axis=1).astype(float)
        difference=abs(warp_fraction(wf)-a['energyFraction'])
        meaningful=abs(wf)>max(float(np.max(abs(wf)))*1e-8,1e-20)
        checks['savedWarpFractionMaxAbsNonzeroWork']=float(np.max(difference[meaningful]))
        checks['savedWarpFractionWorkWeightedMeanAbs']=float(np.sum(difference*abs(wf))/max(float(np.sum(abs(wf))),1e-300))
    # Pair-local work closure: use both endpoint contributions, including the
    # counterpart not explicitly represented in a directed Warp list row.
    kinetic=-m[i]*work
    residual=kinetic+m[i]*fs*work+m[j]*(1-fs)*work*m[i]/m[j]
    checks['pairWorkClosureRelativeL1']=float(np.sum(abs(residual))/max(float(np.sum(abs(kinetic))),1e-300))
    data=dict(rawRateRelativeL2=rel(rawW,rawS),compatibleSpatialRateRelativeL2=rel(hw,hs),
        unlikePartitionOnlyRelativeL2=rel(h_cross,hs),likePartitionOnlyRelativeL2=rel(h_like,hs),
        activeUnlikeDirectedOrNativePairs=int(active.sum()),
        unlikeFractionMeanSpheral=float(np.mean(fs[active])) if active.any() else None,
        unlikeFractionMeanWarp=float(np.mean(fw[active])) if active.any() else None,
        unlikeFractionAbsDifferenceMedian=float(np.median(abs(fs[active]-fw[active]))) if active.any() else None,
        checks=checks,limits='Compatible spatial rates use dt=0 pair work on identical fields. This is not either integrator\'s finite-step energy update.')
    for material in (0,1):
        k=mask&(a['materials']==material)
        data['material'+str(material)]=dict(particles=int(k.sum()),
            compatibleSpatialRateRelativeL2=float(np.linalg.norm((hw-hs)[k])/max(float(np.linalg.norm(hs[k])),1e-300)),
            rawSpheralPower=float(np.sum(m[k]*rawS[k])),rawWarpPower=float(np.sum(m[k]*rawW[k])),
            compatibleSpheralPower=float(np.sum(m[k]*hs[k])),compatibleWarpPower=float(np.sum(m[k]*hw[k])))
    return data,dict(rawSpheral=rawS,rawWarp=rawW,compatibleSpheral=hs,compatibleWarp=hw)


def analyze(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True);rows=[];snapshots={}
    for source,folder in [('spheral',root/'spheral/rhs'),('warp',root/'warp-rhs')]:
        for path in sorted(folder.glob('*.npz')):
            a=doubles(path);sp=native_split(a);i,j=a['i'],a['j'];V=a['volumes'];m=a['masses'];mask=interface_mask(a)
            unlike=a['materials'][i]!=a['materials'][j]
            crossP=reduce_pairs(a,sp['pressurePair']*unlike[:,None]);crossQ=reduce_pairs(a,sp['viscousPair']*unlike[:,None])
            G=.5*(sp['gradientI']-sp['gradientJ'])
            e0=reduce_pairs(a,-(2*V[i]*V[j]/m[i])[:,None]*G)
            # Common diagnostic colour-gradient estimate, using each donor's
            # actual native RK gradient. n points from helium to air.
            ci=a['materials'][i];cj=a['materials'][j];colour=np.zeros_like(a['positions'])
            np.add.at(colour,i,V[j,None]*(cj-ci)[:,None]*sp['gradientI'])
            if source=='spheral':np.add.at(colour,j,V[i,None]*(ci-cj)[:,None]*sp['gradientJ'])
            length=np.linalg.norm(colour,axis=1);normal=-colour/np.maximum(length,1e-300)[:,None]
            band=mask&(a['materials']==1)&(length>1e-3)
            components={}
            for name,value in [('pressure',sp['pressure']),('viscous',sp['viscous']),
                               ('crossPressure',crossP),('likePressure',sp['pressure']-crossP),
                               ('crossViscous',crossQ),('unitPressureResidual',e0)]:
                n=np.sum(value*normal,axis=1);t=value[:,0]*normal[:,1]-value[:,1]*normal[:,0]
                components[name]=dict(normalMean=float(np.mean(n[band])),normalRMS=rms(n[band]),tangentRMS=rms(t[band]),vectorRMS=rms(value[band]))
            replay_folder=root/'replay-v2'/source/path.stem
            replays={mode:json.loads((replay_folder/(mode+'Checks.json')).read_text()) for mode in ('imported','rebuilt')}
            coeff=coefficients(a);rebuilt=doubles(replay_folder/'rebuilt.npz')
            coeff_changes={k:rms((rebuilt[k]-v)[mask]*a['supports'][mask].reshape((-1,)+(1,)*(v.ndim-1))**power)
                for k,v,power in [(k,coeff[k],p) for k,p in [('A',0),('B',1),('gradA',1),('gradB',2)]]}
            for mode in replays:
                b=doubles(replay_folder/(mode+'.npz'))
                for field,ref in [('pressure',sp['pressure']),('viscous',sp['viscous']),('acceleration',a['nativeAcceleration'])]:
                    replays[mode][field+'ErrorRMS']=rms((b[field]-ref)[mask])
                    replays[mode][field+'ReferenceRMS']=rms(ref[mask])
            energy,heat=energy_controls(a,sp)
            row=dict(source=source,time=float(a['time']),input=str(path),interfaceWindowParticles=int(mask.sum()),
                heliumBandParticles=int(band.sum()),nativeBridge=sp['checks'],replays=replays,
                scaledCoefficientErrorRMS=coeff_changes,forceComponents=components,energy=energy)
            rows.append(row);snapshots[source,round(float(a['time']),6)]=(a,sp,band,normal,heat)
            print(source,float(a['time']),'pressure rebuild',replays['rebuilt']['pressureRelativeL2'],
                  'raw du',energy['rawRateRelativeL2'],'partition du',energy['compatibleSpatialRateRelativeL2'],flush=True)
    evolved=[]
    for t in (0.,.05,.1,.15):
        donors={}
        for s in ('spheral','warp'):
            a=snapshots[s,t][0];mask=a['real']&(a['UIDs']<2816);order=np.argsort(a['UIDs'][mask]);donors[s]={k:v[mask][order] for k,v in a.items() if v.ndim and len(v)==len(a['masses'])}
        np.testing.assert_array_equal(donors['spheral']['UIDs'],donors['warp']['UIDs'])
        s,w=donors['spheral'],donors['warp'];region=(s['positions'][:,0]>.4)&(s['positions'][:,0]<.85)&(s['positions'][:,1]>.04)&(s['positions'][:,1]<.36)
        fields={k:dict(relativeL2=float(np.linalg.norm((w[k]-s[k])[region])/max(float(np.linalg.norm(s[k][region])),1e-300)),errorRMS=rms((w[k]-s[k])[region]))
                for k in ('positions','velocities','supports','volumes','densities','pressures','internalEnergies')}
        evolved.append(dict(time=t,particles=int(region.sum()),fields=fields))
    save(out/'comparison.json',dict(stages=rows,evolvedFields=evolved,
        limits='Early coarse native stage states. Replays hold donor geometry/support/volume/P/v/cs/Q gradient fixed, source Q length imported. Force bridge is actual native pair data; rebuilt uses actual Warp moments. Different evolving donors are not a same-state operator comparison. Uniform-P and energy-partition alternatives are static diagnostics only.'))
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,t in enumerate((.05,.1,.15)):
        for row,s in enumerate(('spheral','warp')):
            a,sp,band,normal,heat=snapshots[s,t];ax=axes[row,col];k=interface_mask(a)
            sc=ax.scatter(*a['positions'][k].T,c=a['materials'][k],s=5,cmap='coolwarm',vmin=0,vmax=1)
            b=np.flatnonzero(band)[::2];f=sp['pressure'][b]+sp['viscous'][b]
            ax.quiver(*a['positions'][b].T,*f.T,angles='xy',scale_units='xy',scale=1600,width=.003)
            ax.set(xlim=(.4,.85),ylim=(.04,.36),aspect='equal',title=f'{s} native t={t:g}: total acceleration')
    fig.savefig(out/'nativeForces.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for source in ('spheral','warp'):
        r=[row for row in rows if row['source']==source and row['time']>0];times=[v['time'] for v in r]
        axes[0].semilogy(times,[v['replays']['rebuilt']['pressureRelativeL2'] for v in r],'-o',label=source+' donor')
        axes[1].plot(times,[100*v['energy']['rawRateRelativeL2'] for v in r],'-o',label=source+' donor')
        axes[2].plot(times,[100*v['energy']['compatibleSpatialRateRelativeL2'] for v in r],'-o',label=source+' donor')
    for ax,title,ylabel in zip(axes,['Fixed fields: rebuilt Warp pressure','Same fields: raw thermal rate','Same fields: compatible spatial rate'],['Relative L2 error','Difference (%)','Difference (%)']):
        ax.set(title=title,xlabel='Native stage time',ylabel=ylabel);ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'operatorAndWork.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation',type=Path,help='Separate native capture with next-half-step energy exports')
    a=p.parse_args();analyze(a.input,a.output)
    if a.validation:
        checks=[]
        for path in sorted((a.validation/'spheral/rhs').glob('*.npz')):
            native=doubles(path);data,_=energy_controls(native,native_split(native))
            assert data['checks']['compiledHalfStepEnergyMaxAbs']<1e-12
            checks.append(dict(time=float(native['time']),input=str(path),checks=data['checks']))
        assert len(checks)==4
        save(a.output/'compiledEnergyValidation.json',checks)
