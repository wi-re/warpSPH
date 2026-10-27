"""Native IdealH operator and table audit on static common 2D states."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np


def library(root):
    build=json.loads((root/'build.json').read_text());lib=ctypes.CDLL(build['libraryPath'])
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.spheral_support_table.argtypes=[ctypes.c_int,ptr,ctypes.c_void_p];lib.spheral_support_table.restype=ctypes.c_int
    lib.spheral_support_state.argtypes=[ptr,ctypes.c_int,*[ctypes.c_double]*4,ptr];lib.spheral_support_state.restype=ctypes.c_int
    lib.spheral_support_lookup.argtypes=[ptr,ctypes.c_int,ptr]
    return lib,build


def export(root,output):
    lib,build=library(root);a={}
    for i,name in enumerate(('lookup','grad')):
        meta=np.zeros(4);n=lib.spheral_support_table(i,meta,None);data=np.zeros(n)
        lib.spheral_support_table(i,meta,data.ctypes.data);a[name+'Meta']=meta;a[name+'Data']=data
    np.savez_compressed(output,**a)
    return lib,build


def periodic_rows(a,box):
    """Real particles followed by explicit periodic ghosts within max R."""
    x=a['positions']%box;R=a['supports'];m=a['masses'];rho=a['densities'];tags=a['materials'];n=len(R)
    real=np.column_stack((x,R,m,rho,tags,np.ones(n)));parts=[real];margin=R.max()
    for ix in (-1,0,1):
        for iy in (-1,0,1):
            if ix==iy==0:continue
            shifted=x+box*np.array([ix,iy]);keep=np.all((shifted>=-margin)&(shifted<=box+margin),axis=1)
            ghost=real[keep].copy();ghost[:,:2]=shifted[keep];ghost[:,-1]=0.;parts.append(ghost)
    return np.ascontiguousarray(np.concatenate(parts))


def audit(root,output):
    output.mkdir(parents=True,exist_ok=True);lib,build=export(root,output/'tables.npz')
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPHCore import ParticleState,DomainDescription,buildVerletList,SupportScheme,KernelFunctions
    from warpSPH.modules.adaptiveSupport.optimalSupportOwen import evaluateOptimalSupportOwen
    from warpSPH.configurations import CRKSPHConfig
    from crkIdealH import ideal_support
    n=32;axis=(np.arange(n)+.5)/n;x,y=np.meshgrid(axis,axis,indexing='xy');x=np.column_stack((x.ravel(),y.ravel()));tag=(x[:,0]>.5).astype(np.int32)
    rho=np.where(tag,10.,1.);a=dict(positions=x,supports=np.full(n*n,4/n),masses=rho/n**2,densities=rho,materials=tag)
    cases=dict(uniform=a,volumeJump=dict(a,masses=a['masses']*np.where(tag,2.,1.)),
        uniformSmallH=dict(a,supports=a['supports']*.7),uniformLargeH=dict(a,supports=a['supports']*1.4))
    paths=dict(blobInitial='results/materialEOS/blob_matrix_2026-10-20/matched/input.npz',
        blobWarp='results/materialEOS/density_evolution_2026-10-20/blob-dt2-paper/warp/finalState.npz',
        blobNative='results/materialEOS/crk_heating_v2_2026-10-20/spheral-halfGrid-averaged/finalState.npz')
    cases.update({k:dict(np.load(p)) for k,p in paths.items()});result={};T=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float64).contiguous()
    for name,a in cases.items():
        box=np.array([1.,1.]) if name not in paths else np.array([40.,10.]);a=dict(a,positions=a['positions']%box)
        native_input=periodic_rows(a,box);out=np.full((len(native_input),3),np.nan)
        assert lib.spheral_support_state(native_input,len(native_input),1.,1.,1.e-5,.5,out)==0
        native=out[:len(a['supports'])].copy()
        state=ParticleState(T(a['positions']),T(a['supports']),T(a['masses']),torch.zeros(len(a['supports']),device='cuda',dtype=torch.int32),T(a['densities']))
        state.materials=torch.as_tensor(a['materials'],device='cuda',dtype=torch.int32)
        domain=DomainDescription(T([0.,0.]),T(box),torch.tensor([True,True],device='cuda'),2)
        adj=buildVerletList(state,domain,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
        target,z,current=ideal_support(state,domain,adj,output/'tables.npz');arrays=dict(native=native,gpuTarget=target.cpu().numpy(),gpuRoot=z.cpu().numpy(),gpuEffective=current.cpu().numpy())
        rel=abs(arrays['gpuTarget']/native[:,0]-1)
        record=dict(nativeGpuSupportMaxRelative=float(rel.max()),nativeGpuSupportRMSRelative=float(np.sqrt(np.mean(rel**2))),
            nativeGpuMomentMaxAbsolute=float(abs(arrays['gpuRoot']-native[:,1]).max()),
            nativeGpuEffectiveMaxAbsolute=float(abs(arrays['gpuEffective']-native[:,2]).max()),
            inputSupportMin=float(a['supports'].min()),inputSupportMax=float(a['supports'].max()),
            nativeLowerBoundCount=int(np.count_nonzero(native[:,0]<=4.e-5)),nativeUpperBoundCount=int(np.count_nonzero(native[:,0]>=2.)))
        assert record['nativeGpuSupportMaxRelative']<1.e-10,(name,record)
        unweighted=ideal_support(state,domain,adj,output/'tables.npz',weighted=False)[0].cpu().numpy();arrays['unweightedTarget']=unweighted
        record['crossMaterialWeightEffectMaxRelative']=float(abs(native[:,0]/unweighted-1).max())
        for table in ('lattice','shell'):
            state.supports=T(a['supports'])
            cfg=CRKSPHConfig();cfg.owenTable=table;cfg.adaptiveSupportIterations=1
            config=NS(kernel=KernelFunctions.B8,domain=domain,targetNeighbors=16*np.pi,verletScale=1.4,supportMode=SupportScheme.KernelMeanSymmetric)
            h=evaluateOptimalSupportOwen(state,config,cfg,adjacency=None,validateOnce=True,computeDensity=False)[1].cpu().numpy()
            arrays[table+'Target']=h
            record[table+'VsUnweightedNativeMaxRelative']=float(abs(h/unweighted-1).max())
            record[table+'MedianSupportFactor']=float(np.median(h/a['supports']))
        np.savez_compressed(output/(name+'.npz'),**arrays)
        result[name]=record;print('STATIC IDEALH',name,record,flush=True)
    save=dict(nativeBuild=build,tableSHA256=hashlib.sha256((output/'tables.npz').read_bytes()).hexdigest(),cases=result,
        scope='Actual compiled SPHSmoothingScale::evaluateDerivatives on identical real+periodic-ghost states. GPU exact exported native spline/interpolation tables; isotropic W7 2D, nPerh1 damping1 h bounds1e-5/.5. Warp lattice/shell one-relaxation controls retain own limits/statistic.')
    (output/'audit.json').write_text(json.dumps(save,indent=2)+'\n')


def ranges(output):
    """Separate the old lookup's saturation from absolute support bounds."""
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from warpSPHCore import ParticleState,DomainDescription,buildVerletList,SupportScheme
    from crkIdealH import ideal_support
    paths=dict(blobInitial='results/materialEOS/blob_matrix_2026-10-20/matched/input.npz',
        blobWarp='results/materialEOS/density_evolution_2026-10-20/blob-dt2-paper/warp/finalState.npz',
        blobNative='results/materialEOS/crk_heating_v2_2026-10-20/spheral-halfGrid-averaged/finalState.npz')
    answer={};T=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float64).contiguous()
    for name,path in paths.items():
        a=dict(np.load(path));s=ParticleState(T(a['positions']%np.array([40.,10.])),T(a['supports']),T(a['masses']),torch.zeros(len(a['supports']),device='cuda',dtype=torch.int32),T(a['densities']))
        s.materials=torch.as_tensor(a['materials'],device='cuda',dtype=torch.int32)
        domain=DomainDescription(T([0.,0.]),T([40.,10.]),torch.tensor([True,True],device='cuda'),2)
        adj=buildVerletList(s,domain,supportMode=SupportScheme.SuperSymmetric,verletScale=1.01)
        target,_,current=ideal_support(s,domain,adj,output/'tables.npz',weighted=False,hmax=1.e100)
        target=target.cpu().numpy();nh=4*current.cpu().numpy();inside=(nh>=2)&(nh<=6);old=np.load(output/(name+'.npz'))['shellTarget']
        answer[name]=dict(unweightedEstimatedCompactNhMin=float(nh.min()),unweightedEstimatedCompactNhMax=float(nh.max()),
            belowOldRange=int(np.count_nonzero(nh<2)),aboveOldRange=int(np.count_nonzero(nh>6)),
            oldShellVsUnboundedNativeMaxRelative=float(abs(old/target-1).max()),
            oldShellVsUnboundedNativeMaxRelativeInsideOldRange=float(abs(old[inside]/target[inside]-1).max()),
            unboundedNativeTargetsAboveCaseMaximum=int(np.count_nonzero(target>2.)))
        print('LOOKUP RANGE',name,answer[name],flush=True)
    (output/'rangeAudit.json').write_text(json.dumps(answer,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,default=Path('results/materialEOS/spheral-support-audit'))
    p.add_argument('--output',type=Path,default=Path('reports/materialEOS/idealH/operators'));p.add_argument('--ranges',action='store_true');a=p.parse_args()
    if a.ranges:ranges(a.output.resolve())
    else:audit(a.build.resolve(),a.output.resolve())
