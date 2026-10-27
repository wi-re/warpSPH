"""Qualify weighted before-RHS support and accepted-state restart in 2D.

Research overlay only. Shared samples and timestep grids, fresh midpoint,
paper density, geometric volumes and original compatible energy stay fixed.
Every physical execution is recorded and daemon-visible.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from compare_spheralMaterials import save, warp_run, FIELDS
from compare_spheralSod1d import exact as sod_exact
from compare_spheralKH import sample as kh_sample, mixing_mode
from study_crkEnergySuite import context as base_context, summarize
from study_idealH import install_support, study
from study_densityEvolution import install_audit as install_density

CASES = dict(
    hydro=dict(nx=64, dim=2, gamma=5/3, domain=[-.5,.5], end=2., steps=800),
    sod=dict(nx=100, dim=2, gamma=1.4, domain=[-1.5,1.5], end=.2, steps=400),
    kh=dict(nx=64, dim=2, gamma=5/3, domain=[0.,1.], end=2., steps=1000),
)


def sample(case):
    cfg=CASES[case];n=cfg['nx'];g=cfg['gamma']
    if case=='kh':return kh_sample(n)
    dx=1/n
    if case=='hydro':
        axis=-.5+(np.arange(n)+.5)*dx
        x,y=np.meshgrid(axis,axis,indexing='xy');pos=np.column_stack((x.ravel(),y.ravel()))
        tag=(abs(pos)<.25).all(axis=1).astype(np.int32);rho=np.where(tag==1,4.,1.);P=np.ones(len(pos));v=np.zeros_like(pos)
    else:
        axis=-1.5+(np.arange(3*n)+.5)*dx
        transverse=-.1+(np.arange(n//5)+.5)*dx
        x,y=np.meshgrid(axis,transverse,indexing='xy');pos=np.column_stack((x.ravel(),y.ravel()))
        q,_=sod_exact(pos[:,0],0.);rho=q['density'];P=q['pressure'];v=np.column_stack((q['velocity'],np.zeros(len(pos))))
        tag=np.zeros(len(pos),dtype=np.int32)
    return dict(positions=pos,velocities=v,densities=rho,pressures=P,internalEnergies=P/((g-1)*rho),
        supports=np.full(len(pos),4*dx),masses=rho*dx**2,materials=tag,
        UIDs=np.arange(len(pos),dtype=np.int64),kinds=np.zeros(len(pos),dtype=np.int32))


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);rows=[]
    shutil.copyfile('reports/materialEOS/idealH/operators/tables.npz',root/'tables.npz')
    # First answer the user's weights-only blob question using the exact
    # saved inputs/grids of the completed native-lookup comparison.
    parent=Path('results/materialEOS/idealH_2026-10-20').resolve()
    for split in (1,2):
        source=parent/f'blob-dt{split}-weighted';name=f'blob-dt{split}-material';folder=root/name;folder.mkdir()
        shutil.copyfile(source/'input.npz',folder/'input.npz');s=json.loads((source/'settings.json').read_text())
        s.update(gammaVariant=name,supportPolicy='material',inputPath=str((folder/'input.npz').resolve()))
        save(folder/'settings.json',s);rows.append(name)
    for case in ('kh','hydro','sod'):
        cfg=CASES[case];initial=sample(case)
        for split in (1,2):
            steps=cfg['steps']*split;grid=np.full(steps,cfg['end']/steps)
            for policy in ('weighted','material','warp'):
                name=f'{case}-dt{split}-{policy}';folder=root/name;folder.mkdir();np.savez_compressed(folder/'input.npz',**initial)
                s=dict(cfg,case=case,gammaVariant=name,policy=policy,supportPolicy=policy,schedule='paper',
                    choice=dict(compatible=True,thermal='base'),thermalGamma=cfg['gamma'],steps=steps,exactSteps=steps,
                    dt=float(grid[0]),timeGrid=grid.tolist(),frameEvery=max(1,steps//40),
                    idealTable=str((root/'tables.npz').resolve()),inputPath=str((folder/'input.npz').resolve()),
                    comparisonPolicy='Only support policy differs: '+policy+'; fresh midpoint, paper density, geometric volumes, native compatible energy, Cl2/Cq1, averaged Q heating, AV length R/4')
                if case=='kh' and split==1 and policy=='weighted':s['checkpointStep']=steps//2
                save(folder/'settings.json',s);rows.append(name)
                if 'checkpointStep' in s:
                    # This child runs only after its uninterrupted parent has finished.
                    resumed=root/(name+'-resume');resumed.mkdir();np.savez_compressed(resumed/'input.npz',**initial)
                    rs=copy.deepcopy(s);offset=rs.pop('checkpointStep');rs.update(gammaVariant=name+'-resume',
                        inputPath=str((resumed/'input.npz').resolve()),resumeFrom=str((folder/'checkpoint.h5').resolve()),
                        resumeStepOffset=offset,steps=steps-offset,exactSteps=steps-offset,timeGrid=grid[offset:].tolist(),
                        restartParent=name)
                    save(resumed/'settings.json',rs);rows.append(resumed.name)
    sources={};(root/'sources').mkdir()
    for name in (__file__,'scripts/study_idealH.py','scripts/crkIdealH.py','scripts/study_densityEvolution.py',
                 'scripts/compare_spheralMaterials.py','src/warpSPH/io/hdf5.py','src/warpSPH/runner/runner.py',
                 'src/warpSPH/systems/compSPH.py','src/warpSPH/schemes/compressible/crkSPH.py',
                 'src/warpSPH/modules/adaptiveSupport/wp_materialPsi.py','src/warpSPH/modules/adaptiveSupport/optimalSupportOwen.py',
                 'src/warpSPH/configurations/compressibleConfig.py'):
        p=Path(name);h=hashlib.sha256(p.read_bytes()).hexdigest();shutil.copyfile(p,root/'sources'/f'{p.stem}_{h[:12]}{p.suffix}');sources[str(p)]=h
    save(root/'provenance.json',dict(rows=rows,sources=sources,cases=CASES,parent=str(parent),
        tableSHA256=hashlib.sha256((root/'tables.npz').read_bytes()).hexdigest(),
        scope='18 support-only 2D controls on two dt grids, independent-process accepted-state restart, and two exact-input weights-only blob controls. Native operator/source qualification and stored blob references reused; no numerical defaults changed.'))


def context(s):
    from warpSPH.runner import RunConfiguration
    import torch
    case,_,sim,params,solver=base_context(s)
    configure_base=case.configureScheme
    def configure(c):
        configure_base(c)
        if s['case']=='sod':c.config.domain.min[1]=-.1;c.config.domain.max[1]=.1
    original_post=case.postStep
    def post(c,system,step):
        original_post(c,system,step)
        if step+1==s.get('checkpointStep'):
            import h5py
            from warpSPH.io.hdf5 import dumpState,loadState
            path=Path(s['inputPath']).parent/'checkpoint.h5'
            with h5py.File(path,'w') as f:
                f.attrs['time']=float(system.t);f.attrs['step']=step+1
                dumpState(system.state,f.create_group('state'))
            with h5py.File(path) as f:loaded=loadState(f['state'],c.device,type(system.state))
            errors={}
            for key,value in vars(system.state).items():
                if isinstance(value,torch.Tensor):
                    other=getattr(loaded,key);errors[key]=float((value.double()-other.double()).abs().max()) if value.numel() else 0.
                elif value is not None:assert getattr(loaded,key)==value,(key,getattr(loaded,key),value)
            assert max(errors.values())==0.
            save(path.with_suffix('.json'),dict(time=float(system.t),step=step+1,tensorRoundtripMaxErrors=errors,
                endpointSupport='last midpoint RHS support; explicit seed for next step; no extra endpoint relaxation',
                cachePolicy='No adjacency, grids, pairWork or derivative history saved'))
            print('CHECKPOINT',path,step+1,float(system.t),flush=True)
    case=replace(case,name='supportQualification_'+s['case'],configureScheme=configure,postStep=post)
    solver.update(crkEvolution='paper',crkDensityPolicy='massReplacement',
        crkViscosityParams=dict(oneSidedWork=False,viscosityKernelScale=4.),diffusionParams=dict(C_l=2.,C_q=1.))
    if s['case']=='hydro':solver['materials']=[dict(name=name,eos=dict(type='idealGas',gamma=s['gamma'])) for name in ('ambient','square')]
    if 'resumeFrom' in s:sim.update(resumeFrom=s['resumeFrom'],resumeStepOffset=s['resumeStepOffset'])
    return case,RunConfiguration(case.name,simulation=sim,params=params,solver=solver).prepare(case),sim,params,solver


class Measures:
    def __init__(self,s,initial):
        self.s=s;self.initial=initial;self.m=initial['masses'];self.mass=self.m.sum()
        self.E=np.dot(self.m,initial['internalEnergies']+.5*np.sum(initial['velocities']**2,axis=1))
        self.p=np.sum(self.m[:,None]*initial['velocities'],axis=0)
        self.entropy=np.log(initial['pressures']/initial['densities']**s['gamma'])
    def __call__(self,a,t):
        m,rho,u,v,x=(a[k] for k in ('masses','densities','internalEnergies','velocities','positions'));P=a['pressures'];V=m/rho
        ds=np.log(P/rho**self.s['gamma'])-self.entropy;kinetic=.5*np.dot(m,np.sum(v*v,axis=1))
        out=dict(time=float(t),relativeEnergyDrift=float((np.dot(m,u)+kinetic)/self.E-1),
            momentumError=float(np.linalg.norm(np.sum(m[:,None]*v,axis=0)-self.p)/self.mass),
            materialMassError=float(abs(m.sum()/self.mass-1)),minDensity=float(rho.min()),minEnergy=float(u.min()),
            minPressure=float(P.min()),maxVelocity=float(np.linalg.norm(v,axis=1).max()),
            entropyChangePerMass=float(np.dot(m,ds)/self.mass),entropyL1PerMass=float(np.dot(m,abs(ds))/self.mass),
            negativeEntropyFraction=float(m[ds<-1.e-8].sum()/self.mass),kineticEnergy=float(kinetic),
            minSupport=float(a['supports'].min()),maxSupport=float(a['supports'].max()))
        if self.s['case']=='hydro':
            dx=(x-self.initial['positions']+.5)%1-.5
            out.update(pressureError=float(abs(P-1).max()),shapeDisplacement=float(np.linalg.norm(dx,axis=1).max()),
                displacementInSpacings=float(np.linalg.norm(dx,axis=1).max()*self.s['nx']),
                densityRelativeL1=float(np.mean(abs(rho-self.initial['densities'])/self.initial['densities'])))
        elif self.s['case']=='sod':
            q,_=sod_exact(x[:,0],t);mask=abs(x[:,0])<.5
            for name,value in [('density',rho),('pressure',P),('energy',u),('velocity',v[:,0])]:
                out[name+'L1']=float(np.dot(V[mask],abs(value[mask]-q[name][mask]))/V[mask].sum())
            out['transverseVelocityRMS']=float(np.sqrt(np.dot(m,v[:,1]**2)/self.mass))
        else:
            out.update(modeAmplitude=mixing_mode(a),transverseKineticEnergy=float(.5*np.dot(m,v[:,1]**2)),
                pressureRMS=float(np.sqrt(np.dot(V,(P-2.5)**2)/V.sum())))
        return out


def run_row(args):
    sys.path.insert(0,str(args.publisher_path));s=json.loads((args.output/'settings.json').read_text())
    density=install_density(s);support=install_support(s)
    if s['case']=='blob':
        from study_densityEvolution import context as blob_context
        from compare_spheralBlobPaper import initialize as blob_initialize
        from study_blobThermal import Measures as BlobMeasures,summarize as blob_summary
        def init_blob(c,system,a,settings):
            blob_initialize(c,system,a,settings);system.state.crkVolumes=None
            c.scratch['_titleProvenance']='Existing Owen + material weights only | '+s['gammaVariant']
        try:return warp_run(args,s,blob_context,BlobMeasures,blob_summary,'Support qualification',initialize_factory=init_blob)
        finally:save(args.output/'densityAudit.json',density);save(args.output/'supportAudit.json',support)
    def initialize(c,system,a,settings):
        system.state.crkVolumes=None
        c.scratch['_titleProvenance']='Support qualification | '+s['gammaVariant']+' | fresh midpoint / paper density'
        save(args.output/'resolvedPolicy.json',dict(support=s['supportPolicy'],density=c.schemeConfig.crkEvolution,
            partition=c.schemeConfig.crkEnergyPartition,predictor=c.schemeConfig.crkEnergyPredictor,
            gradient=c.schemeConfig.crkViscosityGradient,AVCl=c.schemeConfig.diffusionParams.C_l,
            AVCq=c.schemeConfig.diffusionParams.C_q,AVlength=c.schemeConfig.crkViscosityParams.viscosityKernelScale,
            oneSidedWork=c.schemeConfig.crkViscosityParams.oneSidedWork,compatible=c.schemeConfig.compatibleEnergy,
            integrator=c.config.integrationScheme.name,domainMin=c.config.domain.min.cpu().tolist(),domainMax=c.config.domain.max.cpu().tolist()))
    try:return warp_run(args,s,context,Measures,summarize,'Support qualification',initialize_factory=initialize)
    finally:save(args.output/'densityAudit.json',density);save(args.output/'supportAudit.json',support)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','study','row'))
    p.add_argument('--output',required=True,type=Path);p.add_argument('--publisher-path',type=Path,default=Path('/home/lu26029/dev/warpSPHRemote/python'))
    args=p.parse_args();args.output=args.output.resolve()
    if args.mode=='prepare':prepare(args.output)
    elif args.mode=='study':study(args.output,row_script=__file__)
    else:raise SystemExit(run_row(args))


if __name__=='__main__':main()
