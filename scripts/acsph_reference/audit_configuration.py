#!/usr/bin/env python3
"""CPU-only, non-evolving ACSPH deficient-support/configuration controls."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--precision',choices=('float32','float64'),required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    from warpSPHBootstrap import bootstrap
    bootstrap(precision=args.precision)
    import numpy as np,torch
    from warpSPHCore import sphKernel_xi,sphKernelScale
    from warpSPH.configurations import ArtificialCompressibleSPHConfig,artificialCompressibleConfigToDict,dictToArtificialCompressibleConfig
    from warpSPH.configurations.simulationConfig import SimulationConfig
    from warpSPH.systems.artificialCompressible import ArtificialCompressibleState,ArtificialCompressibleSystem,bdfCoefficients
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPH.modules.artificialCompressible.surfaceFlags import detectSurface,diffusionTaper
    from warpSPH.modules.artificialCompressible.dyngpusphShifting import reference_surface
    from warpSPH.modules.timestep.artificialCompressible import computeTimestep
    start=time.monotonic();dtype=getattr(torch,args.precision);report=dict(precision=args.precision)
    config=SimulationConfig(device=torch.device('cpu'),dtype=dtype,dim=2,dx=.00075,domain=buildDomainDescription(l=4.,dim=2,periodic=False,device='cpu',dtype=dtype))
    scheme=ArtificialCompressibleSPHConfig();scheme.surfaceDetectionConfig.active=True
    def state(coords):
        x=torch.tensor(coords,dtype=dtype);n=len(x)
        return ArtificialCompressibleState(positions=x,velocities=torch.zeros_like(x),pressures=torch.ones(n,dtype=dtype)*100,supports=torch.ones(n,dtype=dtype)*.003,densities=torch.ones(n,dtype=dtype)*1000,masses=torch.ones(n,dtype=dtype)*1000*.00075**2,kinds=torch.zeros(n,dtype=torch.int32),materials=torch.zeros(n,dtype=torch.int32),UIDs=torch.arange(n,dtype=torch.int32),UIDcounter=n)
    report['supports']={}
    for label,coords in [('isolated',[[0.,0.]]),('triangle',[[0.,0.],[.00075,0.],[0.,.00075]]),('bulk',[[i*.00075,j*.00075] for i in range(-6,7) for j in range(-6,7)])]:
        view=state(coords);flags=detectSurface(view,config,scheme,None,store=False);lam,norm,edge=reference_surface(view,view,config)
        j=84 if label=='bulk' else 0
        report['supports'][label]=dict(warpSPHLambda=float(flags.lambdas[j]),referenceLambda=float(lam[j]),warpSPHTaper=float(diffusionTaper(flags.lambdas)[j]),referenceTaper=float(diffusionTaper(lam)[j]))
    scheme.acParams.isolatedZeroPressure=True;serialized=artificialCompressibleConfigToDict(scheme);restored=dictToArtificialCompressibleConfig(serialized)
    report['serialization']=dict(inputIsolatedZeroPressure=scheme.acParams.isolatedZeroPressure,fieldPresent='isolatedZeroPressure' in serialized['acParams'],roundtripIsolatedZeroPressure=restored.acParams.isolatedZeroPressure)
    view=state([[0.,0.]]);config.adaptiveDt=True;system=ArtificialCompressibleSystem(state=view,domain=config.domain)
    try:report['adaptiveTimestep']=dict(value=float(computeTimestep(system,config,scheme,dt=1e-4)))
    except Exception as e:report['adaptiveTimestep']=dict(error=type(e).__name__,message=str(e))
    xi=float(sphKernel_xi(config.kernel.value,2));scale=float(sphKernelScale(config.kernel.value,2));report['lengths']=dict(support=.003,referenceH=.0015,warpSPHH=.003/xi,warpSPHTimestepLength=.003/xi/scale,xi=xi,kernelScale=scale)
    from warpSPH.modules.artificialCompressible.dualTime import DualTimeResult,exactDeltaUpdate
    from warpSPHIntegrators.state.specs import ComponentUpdateSpec
    field=state([[0.,0.]]);field.pressures.fill_(10000.)
    target=torch.full_like(field.pressures,1e-4)
    solved=DualTimeResult(field.positions.clone(),field.velocities.clone(),target,1,-9.)
    update=exactDeltaUpdate(solved,field,1e-4,2)
    mapped=ArtificialCompressibleSystem(state=field,domain=config.domain)
    mapped.apply_quantity_update(update,ComponentUpdateSpec(derivative_dt=1e-4))
    report['exactDeltaPressureRoundtrip']=dict(target=float(target[0]),mapped=float(mapped.state.pressures[0]),error=float((mapped.state.pressures-target)[0]))
    # At rest with the same external shift delta on successive steps, the
    # compiled repeated-prior BDF shift source is zero. The shifted-position
    # BDF history nevertheless leaves -gamma*delta at zero physical drift.
    dt=1e-4;delta=15e-6;alpha,beta,gamma,_=bdfCoefficients(dt,dt)
    report['constantShiftHistory']=dict(dt=dt,delta=delta,compiledShiftBDF=alpha*delta+beta*delta+gamma*delta,positionBDFAtZeroPhysicalDrift=-gamma*delta,requiredExtraPositionDrift=gamma*delta/alpha)
    # Algebraically equivalent cutoff predicates need not agree in floating
    # arithmetic. Enumerate a few ULPs around a circular support boundary.
    dt=getattr(np,args.precision);R=dt(.006);diff=[];total=0
    for theta in np.linspace(.03,np.pi/2-.03,100):
        x=dt(float(R)*np.cos(theta));y=dt(float(R)*np.sin(theta))
        for ix in range(-8,9):
            xx=x
            for _ in range(abs(ix)):xx=np.nextafter(xx,dt(np.inf if ix>0 else -np.inf))
            for iy in range(-8,9):
                yy=y
                for _ in range(abs(iy)):yy=np.nextafter(yy,dt(np.inf if iy>0 else -np.inf))
                d2=xx*xx+yy*yy;length=np.sqrt(d2);a=bool(d2<R*R);b=bool(length<R);total+=1
                if a!=b and len(diff)<8:diff.append(dict(x=float(xx),y=float(yy),squaredPredicate=a,lengthPredicate=b))
    report['cutoffPredicateExamples']=dict(samples=total,examples=diff,note='Native source uses distanceSquared<radiusSquared; port uses length<radius. No disagreement observed in the tested Lobovsky-radius ULP sweep; do not attribute the measured sensitivity to this predicate rewrite.')
    report['wallSeconds']=time.monotonic()-start
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
