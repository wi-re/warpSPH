#!/usr/bin/env python3
"""Fixed-field audit of wall inputs to ACSPH shifting and MG viscosity.

No physical evolution, so video intentionally omitted. All controls share
positions, surface fields, masses and timesteps; only operator wall inputs vary.
"""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--trajectory',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import numpy as np,torch,h5py
    from warpSPHCore import buildVerletList,SupportScheme
    from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
    from warpSPH.configurations import ArtificialCompressibleSPHConfig
    from warpSPH.configurations.simulationConfig import SimulationConfig
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPH.modules.artificialCompressible.dyngpusphShifting import shift_and_remap,reference_surface
    from warpSPH.modules.artificialCompressible.residual import fusedResidualTerms
    from audit_local import reference_wall_velocities
    torch.set_num_threads(4);start=time.monotonic();device='cuda:0'
    config=SimulationConfig(device=torch.device(device),dtype=torch.float64,dim=2,dx=.00075,domain=buildDomainDescription(l=4.,dim=2,periodic=False,device=device,dtype=torch.float64))
    scheme=ArtificialCompressibleSPHConfig()
    def tensor(a):return torch.as_tensor(np.array(a,copy=True),device=device,dtype=torch.float64)
    def state(x,v,p,rho,m,k):
        n=len(x)
        return ArtificialCompressibleState(positions=tensor(x),velocities=tensor(v),pressures=tensor(p),densities=tensor(rho),masses=tensor(m),supports=torch.full((n,),.003,device=device,dtype=torch.float64),kinds=torch.as_tensor(np.array(k,copy=True),device=device,dtype=torch.int32),materials=torch.zeros(n,device=device,dtype=torch.int32),UIDs=torch.arange(n,device=device,dtype=torch.int32),UIDcounter=n)
    def read(prefix,name,dtype='<f8',width=1):
        a=np.fromfile(args.capture/f'{prefix}-{name}.bin',dtype=dtype)
        return a.reshape(-1,width) if width>1 else a
    def stats(a):return dict(maxAbs=float(a.abs().max()),componentRMS=float(a.square().mean().sqrt()))
    report={};prefix='i000-s9'
    k=np.where(read(prefix,'bc_pre','<i4')==1,0,1)
    states=[('first-shift',state(read(prefix,'x_pre',width=2),read(prefix,'v_pre',width=2),read(prefix,'p_pre'),read(prefix,'rho_pre'),read(prefix,'mass_pre'),k))]
    with h5py.File(args.trajectory) as h:
        for t in (.420056,.700117):
            j=int(np.argmin(abs(h['time'][:]-t)));states.append((f't{float(h["time"][j]):.9f}',state(h['frames/positions'][j],h['frames/velocities'][j],h['frames/pressures'][j],h['frames/densities'][j],h['particles/masses'][:],h['particles/kinds'][:])))
    for label,view in states:
        adjacency=buildVerletList(view,config.domain,verletScale=config.verletScale,supportMode=SupportScheme.Gather,verbose=False)
        if label=='first-shift':
            lam=tensor(read(prefix,'lambda_pre'));norm=tensor(read(prefix,'normal_pre',width=2));edges=torch.as_tensor(read(prefix,'edge_pre','<i4'),device=device)
        else:lam,norm,edges=reference_surface(view,view,config,stage_adjacency=adjacency,accepted_adjacency=adjacency)
        f=view.kinds==0;wall=view.kinds==1
        mirrored=reference_wall_velocities(view,config,scheme,adjacency)
        base=None;row={}
        for name,zero_p,mirror_v in [('reference-fields',False,False),('zero-wall-p',True,False),('mirror-wall-v',False,True),('current-wrapper-inputs',True,True)]:
            control=copy.copy(view)
            control.pressures=torch.where(wall,torch.zeros_like(view.pressures),view.pressures) if zero_p else view.pressures
            control.velocities=mirrored if mirror_v else view.velocities
            got=shift_and_remap(control,config,dt=1.2e-5,lambdas=lam,edges=edges,normals=norm,pair_multiplicity=3)
            if base is None:base=got
            row[name]=dict(shiftDifference=stats((got[0]-base[0])[f]),pressureDifferencePa=stats((got[2]-base[2])[f]),velocityDifference=stats((got[3]-base[3])[f]),remappedKEChangeVsReference=float((.5*view.masses[f]*(got[3][f].square().sum(-1)-base[3][f].square().sum(-1))).sum()))
            print(label,name,json.dumps(row[name]),flush=True)
        report[label]=row
    # Actual compiled first-step RK residuals provide a separate viscosity oracle.
    report['same-field-viscosity']={}
    for prefix in ('i000-s0','i199-s2'):
        idx=read(prefix,'index','<u4');f=read(prefix,'bc','<i4')==1
        view=state(read(prefix,'x',width=2),read(prefix,'v',width=2),read(prefix,'p'),read(prefix,'rho'),read(prefix,'m'),np.where(f,0,1))
        adjacency=buildVerletList(view,config.domain,verletScale=config.verletScale,supportMode=SupportScheme.Gather,verbose=False)
        view.velocities=reference_wall_velocities(view,config,scheme,adjacency)
        _,_,visc,_=fusedResidualTerms(view,config,scheme,adjacency,torch.full_like(view.pressures,.0001608375),referenceVelocities=view.velocities)
        expected=tensor(read(prefix,'terms',width=10)[idx,2:4]);fluid=view.kinds==0
        row=dict(error=stats((visc-expected)[fluid]),reference=stats(expected[fluid]),warpSPH=stats(visc[fluid]))
        report['same-field-viscosity'][prefix]=row;print('VISCOSITY',prefix,json.dumps(row),flush=True)
    report['wallSeconds']=time.monotonic()-start
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
