"""Copy a fixed dual-time solve for diagnostics; never rebuild its topology.

Construction-only region/body objects contain initializer closures. Exclude
those metadata lists from the diagnostic configuration; the solve uses the
already materialized wall/ghost state, forces and contact/bounce topology.
The immediate reloaded-input control must reproduce the original solve.
"""
import copy
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace


def map_tensors(value, *, device, dtype=None, memo=None):
    import torch
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device=device, dtype=dtype if dtype is not None and value.is_floating_point() else value.dtype).clone()
        memo[id(value)] = result
        return result
    if dtype is not None and isinstance(value, torch.dtype) and value.is_floating_point:
        return dtype
    if isinstance(value, dict):
        result = {}; memo[id(value)] = result
        result.update({k: map_tensors(v, device=device, dtype=dtype, memo=memo) for k,v in value.items()})
        return result
    if isinstance(value, (list, tuple)):
        values = [map_tensors(v, device=device, dtype=dtype, memo=memo) for v in value]
        result = values if isinstance(value,list) else type(value)(*values) if hasattr(value,'_fields') else tuple(values)
        memo[id(value)] = result
        return result
    if is_dataclass(value) and not isinstance(value,type) or isinstance(value,SimpleNamespace):
        result=copy.copy(value);memo[id(value)]=result
        names={f.name for f in fields(value)} if is_dataclass(value) else set()
        names.update(getattr(value,'__dict__',{}))
        for name in names:
            object.__setattr__(result,name,map_tensors(getattr(value,name),device=device,dtype=dtype,memo=memo))
        return result
    return value


def solve_payload(problem, x0, v0, p0):
    config=copy.copy(problem.config)
    scheme=copy.copy(problem.schemeConfig)
    for owner in (config,scheme):
        for name in ('regions','rigidBodies'):
            if hasattr(owner,name):setattr(owner,name,[])
    return dict(problem=replace(problem,config=config,schemeConfig=scheme),
                x0=x0,v0=v0,p0=p0,
                excludedConstructionMetadata=['config/scheme regions and rigidBodies lists'])


def equation_residuals(problem, x0, result):
    """Dimensionful final equation residuals, distinct from the stopping norm."""
    import torch
    from warpSPH.modules.artificialCompressible import dualTime
    from warpSPH.modules.artificialCompressible.workingState import workingState
    from warpSPH.modules.artificialCompressible.pressureSmoothing import computePressureSmoothing
    if problem.schemeConfig.acParams.incrementalPositions:
        previous=problem.previousPositionStep
        if previous is None:previous=problem.history[0]-problem.history[1]
        problem=replace(problem,positionOrigin=x0,
            positionLow=problem.positionLow if problem.positionLow is not None else torch.zeros_like(x0),
            previousPositionStep=previous)
        q=result.positionStep
    else:q=result.positions
    x=dualTime.positionCoordinates(problem,q)
    v,p=result.velocities,result.pressures
    view=workingState(problem.state,x,v,p)
    flags=dualTime.surfaceAt(problem,view)
    view.pressures=dualTime.closedWallPressure(problem,view,flags)
    diffusion=computePressureSmoothing(view,problem.config,problem.schemeConfig,problem.adjacency,
        flags.renormalization,pressures=view.pressures)
    view.velocities=dualTime.computeBoundaryVelocities(view,problem.config,problem.schemeConfig,problem.adjacency)
    rP,rV=dualTime.spatialResidual(view,problem.config,problem.schemeConfig,problem.adjacency,
        problem.k1,problem.k2,diffusion,problem.bodyForce,problem.nu,
        unilateralWall=problem.schemeConfig.acParams.cavitationProjection=='wall',
        taper=flags.taper,preparedViscosity=problem.preparedViscosity)
    bx,bv=dualTime.bdfSource(problem,q,v)
    position=dualTime.bounceVelocities(problem.bounce,x,v)-bx
    residuals={'position_m_per_s':position,'momentum_m_per_s2':rV-bv,'pressure_Pa_per_s':rP}
    report={}
    for name,values in residuals.items():
        fluid=values.index_select(0,problem.fluidIndex)
        squared=fluid.square().sum(-1) if fluid.ndim==2 else fluid.square()
        top_values,order=squared.topk(10);top=problem.fluidIndex[order]
        report[name]=dict(rms=float(squared.mean().sqrt()),maximum=float(squared.max().sqrt()),
            top1SquaredFraction=float(top_values[0]/squared.sum().clamp_min(1e-300)),
            topUIDs=problem.state.UIDs[top].cpu().tolist(),topValues=values[top].cpu().tolist())
    report['metricWithoutBounce']=float(dualTime.convergenceMetricDevice(v-bx,v,problem.fluidIndex,problem.nFluid,problem.schemeConfig))
    report['metricWithBounce']=float(dualTime.convergenceMetricDevice(position,v,problem.fluidIndex,problem.nFluid,problem.schemeConfig))
    return report
