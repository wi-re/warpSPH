#!/usr/bin/env python3
"""Non-evolving compiled-dump qualification of the integrated reference operators.

All rows retain physical wall velocities. No trajectory/video is produced.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--precision',choices=['float32','float64'],default='float64')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import numpy as np,torch
    from warpSPHCore import buildVerletList,SupportScheme
    from warpSPH.systems.artificialCompressible import ArtificialCompressibleState
    from warpSPH.configurations.simulationConfig import SimulationConfig
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPH.modules.artificialCompressible import dyngpusphOperators as ops
    from warpSPH.modules.artificialCompressible.surfaceFlags import diffusionTaper
    from warpSPH.modules.artificialCompressible.workingState import workingState
    torch.set_num_threads(4);dtype=getattr(torch,args.precision);device='cuda:0';start=time.monotonic()
    config=SimulationConfig(device=torch.device(device),dtype=dtype,dim=2,dx=.00075,
        domain=buildDomainDescription(l=4.,dim=2,periodic=False,device=device,dtype=dtype))
    def read(m,s,name,width=1,kind='<f8'):
        a=np.fromfile(args.capture/f'i{m:03d}-s{s}-{name}.bin',dtype=kind)
        return a.reshape(-1,width) if width>1 else a
    def t(a):return torch.as_tensor(np.array(a,copy=True),dtype=dtype,device=device)
    def stats(a):return dict(maxAbs=float(a.abs().max()),rms=float(a.square().mean().sqrt()))
    report={'scope':'Fixed captured operators only; no evolution/video','precision':args.precision,'stages':{}}
    initial_index=read(0,0,'index',kind='<u4');base=np.empty_like(read(0,0,'x',2));base[initial_index]=read(0,0,'x',2)
    switch=None
    for m in (0,1,199):
        frozen=None;frozen_diffusion=None
        for s in range(3):
            idx=read(m,s,'index',kind='<u4');k=np.where(read(m,s,'bc',kind='<i4')==1,0,1);n=len(k)
            state=ArtificialCompressibleState(positions=t(read(m,s,'x',2)),velocities=t(read(m,s,'v',2)),
                pressures=t(read(m,s,'p')),supports=torch.full((n,),.003,device=device,dtype=dtype),
                densities=t(read(m,s,'rho')),masses=t(read(m,s,'m')),kinds=torch.as_tensor(k,device=device,dtype=torch.int32),
                materials=torch.zeros(n,device=device,dtype=torch.int32),UIDs=torch.arange(n,device=device,dtype=torch.int32),UIDcounter=n)
            adj=buildVerletList(state,config.domain,verletScale=config.verletScale,supportMode=SupportScheme.Gather,verbose=False)
            fluid=state.kinds==0;wall=state.kinds==1;body=torch.zeros_like(state.velocities);body[:,1]=-9.81
            # Native p dump is already closed. Test the closure separately on
            # wall rows; retained stage lambda/tags need not equal its prior input.
            lam=t(read(m,s,'lambda'));edges=torch.zeros_like(lam)
            closed,normal,average,divF,lapV=ops.closures(state,config,adj,lam,edges,body,cell_positions=t(base[idx]))
            if s==0:
                L,N,G,P=ops.moments(state,config,adj,cell_positions=t(base[idx]))
                raw=ops.raw_surface(state,config,adj,L,N)
                frozen=(L,N,G,P,raw)
            L,N,G,P,raw=frozen
            terms=ops.residual_terms(state,config,adj,normal,average,divF,lapV,G,P,raw,
                frozen_switch=switch,cell_positions=t(base[idx]))
            if m==0 and s==0:switch=terms[8]
            beta=375. if m<2 else 187.5
            jst=ops.momentum_jst(state,terms,lapV,beta)
            expected=t(read(m,s,'terms',10)[idx]);native_rv=t(read(m,s,'rV',2)[idx]);native_rp=t(read(m,s,'rP')[idx])
            rv=terms[1]+.0001608375*terms[2]+body+beta*.0015*terms[3]+jst
            if s==0:frozen_diffusion=(.1*.0015*beta)*terms[7]
            rp=-beta**2*state.densities*terms[0]+frozen_diffusion*diffusionTaper(L)
            row={name:stats((value-target)[fluid]) for name,value,target in [
                ('pressureForce',terms[1],expected[:,:2]),('viscosity',.0001608375*terms[2],expected[:,2:4]),
                ('momentumJST',jst,expected[:,4:6]),('acousticDamper',beta*.0015*terms[3],expected[:,6:8]),
                ('divergence',terms[0],expected[:,8]),('pressureDiffusion',frozen_diffusion,expected[:,9]),
                ('momentumResidual',rv,native_rv),('pressureResidual',rp,native_rp),
                ('lambda',L,lam),('gradient',G,t(read(m,s,'gradP',2))),
                ('fluidDiv',divF,t(read(m,s,'divVfluid'))),('velocityLap',lapV,t(read(m,s,'lapV',2))) ]}
            row['wallPressure']=stats((closed-state.pressures)[wall]);row['wallAverage']=stats((average-t(read(m,s,'wallV',2)))[wall])
            row['wallNormal']=stats((normal-t(read(m,s,'norm',2)))[wall])
            report['stages'][f'i{m:03d}-s{s}']=row
            print('QUALIFY',m,s,json.dumps(row),flush=True)
    # Absolute FP64 gates sit above measured reduction roundoff; report every
    # difference and retain failed controls. Rounded FP32 inputs are a separate
    # accuracy measurement, not an equality-scale FP64 oracle.
    gates={'pressureForce':1e-8,'viscosity':1e-9,'momentumJST':1e-9,
        'acousticDamper':1e-8,'divergence':1e-10,'pressureDiffusion':1e-4,
        'momentumResidual':1e-8,'pressureResidual':1e-3,'lambda':1e-11,
        'gradient':1e-7,'fluidDiv':1e-10,'velocityLap':1e-8,
        'wallAverage':1e-10,'wallNormal':1e-10}
    report['gates']=gates if args.precision=='float64' else None
    report['passed']=all(row[k]['maxAbs']<=tol for row in report['stages'].values() for k,tol in gates.items()) if args.precision=='float64' else None
    report['wallPressureControlLimit']='Prior raw flags/lambda before wall closure are not retained in these dumps; this column is observational and not an exact-input oracle.'
    # Qualify the complete first pseudo-iteration against the next captured
    # stage-zero state, including warmup, Butcher factors and wall carryover.
    from warpSPH.configurations import ArtificialCompressibleSPHConfig
    from warpSPH.modules.artificialCompressible.dualTime import makeDualTimeProblem
    from warpSPH.modules.artificialCompressible.dyngpusphReference import iteration
    from warpSPH.systems.artificialCompressible import bdfCoefficients
    from warpSPH.utils.cudaGraph import GraphedTensorFunction
    idx=read(0,0,'index',kind='<u4');n=len(idx)
    state.positions=t(read(0,0,'x',2));state.velocities=t(read(0,0,'v',2));state.pressures=t(read(0,0,'p'))
    # All dumps retain the same physical sort for this first real step.
    state.kinds=torch.as_tensor(np.where(read(0,0,'bc',kind='<i4')==1,0,1),device=device,dtype=torch.int32)
    adj=buildVerletList(state,config.domain,verletScale=config.verletScale,supportMode=SupportScheme.Gather,verbose=False)
    sc=ArtificialCompressibleSPHConfig();sc.acParams.rkStages=3;sc.acParams.dtOverDtau=2.;sc.acParams.usePointImplicit=False
    sc.acParams.bdfShiftCorrection=False;sc.fluid.restDensity=1000.;sc.fluid.fixedSoundSpeed=85.78
    history=(state.positions,state.positions,state.velocities,state.velocities)
    problem=makeDualTimeProblem(state,config,sc,adj,dtau=6e-6,k1=torch.full_like(state.pressures,187.5**2),
        k2=torch.full_like(state.pressures,.1*.0015*187.5),nu=torch.full_like(state.pressures,.0001608375),
        bdf=bdfCoefficients(1.2e-5,1e-5)[:3],history=history,bodyForce=body,
        wallBodyForce=body,surface=None,contact=None,bounce=None)
    inputs=(state.positions,state.velocities,state.pressures,torch.zeros_like(state.pressures),
        torch.zeros_like(state.velocities),torch.zeros_like(state.pressures),torch.zeros_like(state.pressures))
    fn=lambda *a:iteration(problem,*a,warmup=True,initialize_switch=True,cell_positions=state.positions,adjacency=adj)
    got=fn(*inputs)
    expected=[t(read(1,0,name,width)) for name,width in [('x',2),('v',2),('p',1)]]
    f=state.kinds==0;w=state.kinds==1
    report['firstIteration']={name:stats((value-target)[f]) for name,value,target in zip(['position','velocity','pressure'],got[:3],expected)}
    report['firstIteration']['lastStageWallPressure']=stats((got[2]-t(read(0,2,'p')))[w])
    graph=GraphedTensorFunction(fn,'reference first-iteration qualification',validate=True)
    graph_result=graph(*inputs)
    report['graph']={'disabled':graph.disabled,'captured':graph.graph is not None,
        'allOutputsBitwise':all(torch.equal(a,b) for a,b in zip(got,graph_result))}
    integrated_gates={'position':1e-12,'velocity':1e-10,'pressure':1e-6,'lastStageWallPressure':1e-7}
    report['integratedGates']=integrated_gates if args.precision=='float64' else None
    report['integratedPassed']=(all(report['firstIteration'][k]['maxAbs']<=v for k,v in integrated_gates.items()) if args.precision=='float64' else True) and report['graph']['allOutputsBitwise'] and report['graph']['captured'] and report['graph']['disabled'] is None
    if args.precision=='float64':report['passed']=report['passed'] and report['integratedPassed']
    print('FIRST ITERATION',json.dumps(report['firstIteration']),flush=True)
    print('GRAPH',json.dumps(report['graph']),flush=True)
    report['wallSeconds']=time.monotonic()-start
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['passed'] is False or not report['integratedPassed']:raise SystemExit('Captured operator gate failed; see retained summary')

if __name__=='__main__':main()
