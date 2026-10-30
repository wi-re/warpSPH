#!/usr/bin/env python3
"""Non-evolving shift controls: separate input rounding from arithmetic.

No integration is performed; video is deliberately omitted for these fixed
operator and three-particle controls. The published/source model is unmodified.
"""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--precision',choices=('float32','float64'),required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import numpy as np
    import torch
    from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
    from warpSPH.configurations.simulationConfig import SimulationConfig
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPH.modules.artificialCompressible.dyngpusphShifting import shift_and_remap
    torch.set_num_threads(4);start=time.monotonic();dtype=getattr(torch,args.precision)
    def read(name,dtype='<f8',width=1):
        a=np.fromfile(args.capture/f'i000-s9-{name}.bin',dtype=dtype)
        return a.reshape(-1,width) if width>1 else a
    arrays={k:read(name,width=w) for k,name,w in [('x','x_pre',2),('v','v_pre',2),('p','p_pre',1),('rho','rho_pre',1),('m','mass_pre',1),('lam','lambda_pre',1),('norm','normal_pre',2)]}
    index=read('index','<u4').astype(np.int64)
    kinds=np.where(read('bc_pre','<i4')==1,0,1).astype(np.int32)
    edges=read('edge_pre','<i4');expected=read('shift_post',width=2)[index]
    worst=int(np.argmin(np.linalg.norm(arrays['x']-np.array([.59775,.29325]),axis=1)))
    def evaluate(a,k,e,dt=1.2e-5,device='cuda:0',multiplicity=3):
        t=lambda v:torch.as_tensor(v,dtype=dtype,device=device)
        x=t(a['x']);n=len(x)
        state=ArtificialCompressibleState(positions=x,velocities=t(a['v']),pressures=t(a['p']),densities=t(a['rho']),masses=t(a['m']),supports=torch.full((n,),.003,dtype=dtype,device=device),kinds=torch.as_tensor(k,device=device,dtype=torch.int32),materials=torch.zeros(n,dtype=torch.int32,device=device),UIDs=torch.arange(n,dtype=torch.int32,device=device),UIDcounter=n)
        config=SimulationConfig(device=torch.device(device),dtype=dtype,dim=2,dx=.00075,domain=buildDomainDescription(l=4.,dim=2,periodic=False,device=device,dtype=dtype))
        result=shift_and_remap(state,config,dt=dt,lambdas=t(a['lam']),edges=torch.as_tensor(e,device=device,dtype=torch.int32),normals=t(a['norm']),pair_multiplicity=multiplicity)
        return [v.detach().cpu().numpy() for v in result]
    report=dict(precision=args.precision,scope='Fixed operators only; no physical advance/video',cases={})
    for label,rounded in [('original',set()),('rounded-x',{'x'}),('rounded-lambda',{'lam'}),('rounded-normal',{'norm'}),('rounded-other-fields',set(arrays)-{'x'}),('rounded-all',set(arrays))]:
        a={k:(v.astype(np.float32).astype(np.float64) if k in rounded else v.copy()) for k,v in arrays.items()}
        before=time.monotonic();got=evaluate(a,kinds,edges);err=got[1].astype(np.float64)-expected;f=kinds==0
        bad=np.flatnonzero(f&(np.linalg.norm(err,axis=1)>1e-8))
        row=dict(changedRows=[dict(row=int(j),x=a['x'][j].tolist(),lambda64=float(arrays['lam'][j]),inputLambda=float(a['lam'][j]),inputNormal=a['norm'][j].tolist(),shift=got[1][j].tolist(),expectedShift=expected[j].tolist()) for j in bad],maxShiftError=float(abs(err[f]).max()),componentRMS=float(np.sqrt(np.mean(err[f]**2))),rowsAbove10nm=int((np.linalg.norm(err[f],axis=1)>1e-8).sum()),worstRowShift=got[1][worst].tolist(),wallSeconds=time.monotonic()-before)
        for entry in row['changedRows']:
            j=entry['row'];delta=a['x']-a['x'][j];near=(np.linalg.norm(delta,axis=1)<.006)&(edges==1)&(np.arange(len(delta))!=j)
            cos=(a['norm'][near]*a['norm'][j]).sum(-1);valid=np.abs(cos)<=1
            entry['maxRawAngle']=float(np.arccos(cos[valid]).max()) if valid.any() else 0.
            entry['invalidAntiparallelAngles']=int((cos < -1).sum())
            entry['invalidParallelAngles']=int((cos > 1).sum())
        report['cases'][label]=row
        print('FIXED SHIFT',label,json.dumps(row),flush=True)
    # A three-fluid-particle cloud isolates the discontinuous angular veto:
    # the far raw-surface neighbor changes no gradient (same lambda as i),
    # but its normal alone vetoes the finite cohesion from the nearby interior.
    toy=dict(x=np.array([[0.,0.],[.00075,0.],[0.,.006]]),v=np.zeros((3,2)),p=np.ones(3)*100,rho=np.ones(3)*1000,m=np.ones(3)*1000*.00075**2,lam=np.array([.6,.8,.6]),norm=np.array([[1.,0.],[1.,0.],[0.,1.]]))
    report['cutoffSweep']=[]
    for delta in (-1e-5,-1e-7,-1e-9,-1e-11,0.,1e-11,1e-9,1e-7,1e-5):
        a=copy.deepcopy(toy);a['x'][2,1]=.006+delta;got=evaluate(a,np.zeros(3,np.int32),np.array([1,0,1],np.int32),device='cpu',multiplicity=1)
        row=dict(distanceOffsetMeters=delta,actualInputDistance=float(np.array(a['x'][2,1],dtype=getattr(np,args.precision))),shift=got[0][0].tolist())
        report['cutoffSweep'].append(row)
        print('CUTOFF',json.dumps(row),flush=True)
    report['wallSeconds']=time.monotonic()-start
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
