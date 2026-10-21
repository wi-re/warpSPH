"""Short current-Warp comparison against the qualified paper-configured Spheral.

Reuses the actual reference startup arrays and completed trajectory. No new
Spheral trajectory or production policy modification is needed. Native Warp
summation/Butcher energy and Owen support evolution remain explicit differences.
"""
import argparse
import hashlib
import json
import math
import copy
from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
from compare_spheralMaterials import FIELDS,save,warp_run
from study_blobPartition import Measures,summary


def prepare(args):
    ref=args.reference.resolve();result=json.loads((ref/'summary.json').read_text())
    assert result['status']=='completed' and result['positiveEnergy']
    settings=json.loads((ref/'settings.json').read_text());assert settings['paperPolicies']
    args.output.mkdir(parents=True,exist_ok=False)
    with np.load(ref/'initializedState.npz') as a:
        np.savez_compressed(args.output/'input.npz',**{k:a[k] for k in FIELDS})
        assert np.max(abs(a['H'][:,0,1]))<1e-14 and np.max(abs(a['H'][:,0,0]-a['H'][:,1,1]))<1e-14
    steps=math.ceil(result['time']/args.dt)
    settings.update(gammaVariant='paper-input-n128-short',reference=str(ref),steps=steps,end=result['time'],dt=result['time']/steps,
        frameEvery=50,energyPartition='native',energyPredictor='raw',volumePolicy='kernelSum',precision='float64',referenceKernel=settings['kernel'],kernel='B8',
        comparisonPolicy='Native Warp summation: geometric gather volume / corrected massReplacement density every RHS; native entropy/Butcher pair work; Owen supports; B8 (degree7), nh4, Cl2/Cq1, Q length R/4.',
        differences=['Spheral Eq76 reset once per full step with continuity density in stages; Warp recalculates Eq76 at every RHS',
            'Spheral CheapRK2 reuses previous midpoint derivatives for predictor; Warp explicit midpoint evaluates a fresh first RHS',
            'Spheral Eq63 partitions final work using latest RHS entropy/work sign; Warp keeps native Butcher-compatible entropy estimator',
            'Native support evolution, neighbor traversal and limiter-gradient scheduling differ; only initial full isotropic H is copied',
            'Spheral adaptive dt; Warp fixed dt <=.001. No claim of identical temporal discretization'],
        source='Qualified reference initialized arrays copied exactly; common geometry/kernel/Q parameters with current Warp evolution, not a complete policy transplant.')
    settings['inputSHA256']=hashlib.sha256((args.output/'input.npz').read_bytes()).hexdigest()
    save(args.output/'settings.json',settings);print('Prepared',args.output,steps,'steps to',settings['end'],flush=True)


def context(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=settings.get('precision','float64'))
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.blob import blobCase
    params=dict(viscositySwitch='NoneSwitch',adaptiveSupportScheme='Owen',massMatch=True,initialDensity='prescribed')
    solver=dict(crkDensityPolicy='massReplacement',crkEvolution='summation',crkEnergyPartition='native',crkEnergyPredictor='raw',
        crkViscosityParams={'oneSidedWork':False,'viscosityKernelScale':4.},diffusionParams={'C_l':2.,'C_q':1.})
    sim=dict(nx=settings['nx'],tLimit=settings['end'],nSteps=settings['steps'],kernel='B8',n_h=4.,dt=settings['dt'],adaptiveDt=False,
        execution='eager',cflFactor=.25,precision=settings.get('precision','float64'))
    def merge(target,changes):
        for key,value in changes.items():
            if isinstance(value,dict):merge(target.setdefault(key,{}),value)
            else:target[key]=value
    controls=settings.get('matrixControls',{})
    merge(params,controls.get('params',{}));merge(solver,controls.get('solver',{}));merge(sim,controls.get('simulation',{}))
    if settings.get('timeGrid'):
        grid=settings['timeGrid'];basePost=blobCase.postStep
        def post(c,s,step):
            basePost(c,s,step);c.scratch['matrixNextStep']=step+1
        def timestep(c,s):return grid[min(c.scratch['matrixNextStep'],len(grid)-1)]
        blobCase=replace(blobCase,postStep=post,timestep=timestep)
    ctx=RunConfiguration('blob',simulation=sim,params=params,solver=solver).prepare(blobCase)
    return blobCase,ctx,sim,params,solver


def initialize(ctx,system,a,settings):
    # Generic older comparison harness seeds m/rho volumes for its Spheral-stage
    # experiment. Current Warp summation must recompute its geometric volumes.
    system.state.crkVolumes=None;system.state.crkMassDensities=None
    ctx.scratch['_titleProvenance']='Current Warp | exact paper-Spheral startup | B8/W7 | nh4 | Cl2 Cq1 | native summation/energy'


class PaperMeasures(Measures):
    def __call__(self,a,t):
        row=super().__call__(a,t);row['originalCloudMassFraction']=row['cloudMassFraction']
        mask=(a['densities']>6.4)&(a['internalEnergies']<1.35)
        row['cloudMassFraction']=float(a['masses'][mask].sum()/a['masses'][a['materials']==1].sum())
        return row


def operators(root,settings):
    _,ctx,*_=context(settings)
    import torch
    from scipy.spatial import cKDTree
    from warpSPHCore import buildVerletList,SupportScheme
    from warpSPH.modules.crk.terms import crkFactors
    a=dict(np.load(root/'input.npz'));s=ctx.SimulationState(UIDcounter=len(a['masses']),**{k:torch.as_tensor(v,device=ctx.device) for k,v in a.items()})
    adjacency=buildVerletList(s,ctx.config.domain,verletScale=1.4,supportMode=SupportScheme.SuperSymmetric)
    v,rho,c=crkFactors(s,ctx.config.domain,ctx.config.kernel,adjacency,densityPolicy='massReplacement')
    ref=np.load(Path(settings['reference'])/'firstDensityAudit.npz')['rows'];ref=ref[ref[:,12]>0]
    distance,indices=cKDTree(ref[:,:2]).query(a['positions']);assert distance.max()<1e-12
    ref=ref[indices];cpu=lambda v:v.detach().cpu().numpy()
    C=np.column_stack((cpu(c.A),cpu(c.A[:,None]*c.B)))
    errors=dict(volumeRelativeMax=float(np.max(abs(cpu(v)/ref[:,3]-1))),densityRelativeMax=float(np.max(abs(cpu(rho)/ref[:,7]-1))),
        correctionPolynomialMax=float(np.max(abs(C-ref[:,8:11]))))
    result=dict(errors=errors,particles=len(v),precision='float64',passed=all(x<2e-5 for x in errors.values()),
        scope='Same initialized real particles/isotropic H; native Warp geometric volume, corrected material density and RK polynomial against compiled Spheral Eq75/76 snapshot. No force/dynamics advance.')
    save(root/'initialOperators.json',result);np.savez_compressed(root/'initialOperators.npz',volume=cpu(v),density=cpu(rho),C=C,reference=ref)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise RuntimeError('Initial operator comparison failed; do not advance blindly')


def analyze(root,output,verify=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    from warpsph_remote.sources import HDF5Source
    settings=json.loads((root/'settings.json').read_text());ref=Path(settings['reference'])
    hS=json.loads((ref/'diagnosticHistory.json').read_text());hW=json.loads((root/'warp/diagnosticHistory.json').read_text())
    result=json.loads((root/'warp/comparisonRun.json').read_text());output.mkdir(parents=True,exist_ok=True)
    resolved=json.loads((Path(result['exportPath'])/'resolvedConfiguration.json').read_text())
    spec=resolved['spec'];solver=resolved['solverConfig']
    assert spec['kernel']=='B8' and spec['n_h']==4. and spec['precision']=='float64'
    assert solver['crkDensityPolicy']=='massReplacement' and solver['crkEvolution']=='summation' and solver['crkEnergyPartition']=='native'
    assert solver['diffusionParams']['C_l']==2. and solver['diffusionParams']['C_q']==1.
    assert solver['crkViscosityParams']['viscosityKernelScale']==4. and not solver['crkViscosityParams']['oneSidedWork']
    initial=np.load(root/'input.npz');native=np.load(ref/'initializedState.npz')
    for k in FIELDS:np.testing.assert_array_equal(initial[k],native[k],err_msg=k)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for name,h in (('paper-configured Spheral',hS),('current Warp',hW)):
        for ax,field in zip(axes,('cloudMassFraction','minEnergy','relativeEnergyDrift')):
            ax.plot([x['tKH'] for x in h],[x[field] for x in h],label=name)
    for ax,title in zip(axes,('Cold-cloud mass / initial mass (all gas)','Minimum internal energy','Relative total-energy change')):
        ax.set(xlabel='t / KH',title=title);ax.grid(alpha=.2)
    axes[0].legend();axes[1].axhline(0,color='k',lw=.7);axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.savefig(output/'diagnostics.png',dpi=180);plt.close(fig)
    sourceS=HDF5Source(ref/'trajectory.h5');sourceW=HDF5Source(Path(result['exportPath'])/'trajectory.h5')
    # Legacy trajectory frame_00000 is after the first step, not t=0. Startup
    # geometry is stored separately in combined* datasets (float32 export).
    import h5py
    initialErrors={}
    with h5py.File(Path(result['exportPath'])/'trajectory.h5') as f:
        for k in ('positions','velocities','densities','masses','supports','materials','UIDs','kinds'):
            actual=f['combined'+k[0].upper()+k[1:]][()]
            np.testing.assert_array_equal(actual,native[k].astype(actual.dtype),err_msg='recorded Warp initial '+k)
            initialErrors[k]=float(np.max(abs(actual-native[k])))
    snapshots=[];fig,axes=plt.subplots(2,3,figsize=(14,5),layout='constrained')
    for col,target in enumerate((.25,.5,1.)):
        for row,(name,source) in enumerate((('Spheral',sourceS),('Warp',sourceW))):
            ax=axes[row,col]
            if target*settings['tKH']>source.times[-1]+1e-8:
                ax.text(.5,.5,f'{name}: target not reached',ha='center',transform=ax.transAxes);ax.set_axis_off();continue
            idx=int(np.argmin(abs(np.asarray(source.times)/settings['tKH']-target)));a=source.read(idx)
            ax.scatter(*a['positions'].T,c=a['densities'],s=1,cmap='turbo',vmin=.75,vmax=10)
            ax.set(xlim=(2,13),ylim=(0,10),aspect='equal',title=f'{name} actual t/KH={source.times[idx]/settings["tKH"]:.4f}')
            snapshots.append(dict(solver=name,target=target,time=float(source.times[idx])))
    fig.suptitle('Same initial arrays / kernel / Q parameters; remaining numerical policies differ')
    fig.savefig(output/'density.png',dpi=180);plt.close(fig)
    # Per-particle error is informative only before nonlinear divergence; report
    # differences without treating Spheral as a pointwise exact solution.
    finalW=np.load(root/'warp'/('finalState.npz' if result.get('termination')!='failed' else 'failedState.npz'))
    finalS=np.load(ref/'finalState.npz');errors={}
    if abs(hW[-1]['time']-hS[-1]['time'])<settings['dt']*1.01:
        for k in ('densities','internalEnergies','velocities','supports'):
            d=finalW[k]-finalS[k];errors[k]=dict(rms=float(np.sqrt(np.mean(d*d))),maximum=float(abs(d).max()))
        d=finalW['positions']-finalS['positions'];L=np.array([40.,10.]);d-=L*np.round(d/L)
        errors['positions']=dict(rms=float(np.sqrt(np.mean(d*d))),maximum=float(abs(d).max()))
    t=np.array([r['time'] for r in hW]);massDifference=np.array([r['cloudMassFraction'] for r in hW])-np.interp(t,[r['time'] for r in hS],[r['cloudMassFraction'] for r in hS])
    observed=t>=.01*settings['tKH']
    massMetrics=dict(endpointDifference=float(massDifference[-1]),rmsAfter001KH=float(np.sqrt(np.mean(massDifference[observed]**2))),
        maximumAfter001KH=float(abs(massDifference[observed]).max()),method='Reference history linearly interpolated onto uniform Warp sample times; initial 0–.01KH density-reset transient excluded from curve RMS/max only.')
    comparison=dict(settings=settings,warp=result,referenceSummary=json.loads((ref/'summary.json').read_text()),
        resolvedWarpConfiguration=resolved,
        initialOperators=json.loads((root/'initialOperators.json').read_text()),actualInitialMaximumDifferences=initialErrors,finalParticleDifferences=errors,snapshots=snapshots,
        coldMassComparison=massMetrics,
        initialRecordingScope='Copied input arrays match reference bitwise. Actual startup combined* HDF5 geometry/identity fields match their expected export casts; floating startup export is float32, while dynamics/input NPZ are float64. Initial u/P have no separate t=0 HDF5 datasets.',
        referenceReuse='Completed native 128x32/1KH paper-configured run; no new native physical trajectory.',
        scope='Matched input/kernel/Q comparison of current Warp against paper-configured Spheral. Remaining recorded policy differences prevent attributing a discrepancy to any single cause.')
    if verify:
        import subprocess,urllib.request
        from warpsph_remote.daemon import descriptor
        d=descriptor()
        def get(path):
            request=urllib.request.Request(d['endpoint']+path,headers={'Authorization':'Bearer '+d['token']})
            with urllib.request.urlopen(request,timeout=15) as response:return json.load(response)
        runs=get('/v1/runs')['runs'];checks={}
        for name,folder,source,rid in (('Spheral',ref,sourceS,json.loads((ref/'registration.json').read_text())['remoteRunID']),
                ('Warp',Path(result['exportPath']),sourceW,result['remoteRunID'])):
            run=next(r for r in runs if r['id']==rid);history=get('/v1/runs/'+rid+'/history')
            np.testing.assert_allclose(history['times'],source.times,rtol=0,atol=1e-12)
            movie=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',str(folder/'output.mp4')]))['streams'][0]
            count=len(list((folder/'images').glob('frame_*.png')))
            assert count==int(movie['nb_frames'])==len(source.times),(name,count,movie,len(source.times))
            assert run['status']==('completed' if name=='Spheral' or result['passed'] else 'failed')
            checks[name]=dict(id=rid,status=run['status'],frames=count,historyMatches=True)
        comparison['daemonVideoVerification']=checks
    save(output/'comparison.json',comparison)
    print('Saved paired comparison',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','operators','run','analyze'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,default=Path('results/materialEOS/blob_paper_2026-10-20/qualify128rhs'))
    p.add_argument('--dt',type=float,default=.001);p.add_argument('--report',type=Path)
    p.add_argument('--verify-daemon',action='store_true')
    args=p.parse_args()
    if args.mode=='prepare':prepare(args)
    elif args.mode=='analyze':analyze(args.output,args.report or args.output/'report',args.verify_daemon)
    else:
        settings=json.loads((args.output/'settings.json').read_text())
        if args.mode=='operators':operators(args.output,settings)
        else:
            assert json.loads((args.output/'initialOperators.json').read_text())['passed']
            raise SystemExit(warp_run(args,settings,context,PaperMeasures,summary,'Short paper input',initialize_factory=initialize))
