"""Fixture-only Lobovsky hybrids. Production solver/boundary defaults untouched.

The source operator block stays coordinated. The remaining switches are
independent arithmetic, transport, clock or stopping changes around that block.
All-enabled uses the original reference solver without replacing its map.
"""
import copy
from dataclasses import replace
import hashlib
import importlib
import inspect

try:
    from .ablation_matrix import Options,validate
except ImportError:
    from ablation_matrix import Options,validate


def select_remap_outputs(result,state,options):
    if options.pressure_remap and options.velocity_remap:return result
    result=list(result)
    if not options.pressure_remap:result[2]=state.pressures
    if not options.velocity_remap:result[3]=state.velocities
    return tuple(result)


def warm_problem(problem):
    scheme=copy.copy(problem.schemeConfig);scheme.acParams=copy.copy(scheme.acParams)
    scheme.acParams.dtOverDtau*=2
    return replace(problem,schemeConfig=scheme,dtau=problem.dtau/2,
        k1=problem.k1*4,k2=problem.k2*2)


def legacy_warm_solve(problem,x0,v0,p0,verbose=False):
    """Same generic map/metric; first two iterations use the half clock.

    In the Lobovsky fixture audit_local already supplies destination-row RK
    factors, so warm_problem changes dtOverDtau as well as dtau/beta.
    A normal-clock graph is captured only after the two eager warm iterations.
    """
    import torch
    from warpSPH.modules.artificialCompressible import dualTime as dual
    x,v,p=x0,v0,dual.projectPressure(problem.contact,p0)
    ac=problem.schemeConfig.acParams
    if ac.incrementalPositions:
        low=problem.positionLow if problem.positionLow is not None else torch.zeros_like(x0)
        previous=problem.previousPositionStep
        if previous is None:previous=problem.history[0]-problem.history[1]
        problem=replace(problem,positionOrigin=x0,positionLow=low,previousPositionStep=previous)
        x=torch.zeros_like(x0)
    half=warm_problem(problem);graph=dual.graphedIteration(problem);eps_history=[]
    for m in range(ac.maxPseudoIterations):
        if m<2:out=dual.iterate(half,x,v,p)
        elif graph is not None:
            out=graph(x,v,p)
            if graph.disabled is not None:problem.schemeConfig._acsphGraphDisabled=graph.disabled;graph=None
            else:problem.schemeConfig._acsphGraphValidated=True
        else:out=dual.iterate(problem,x,v,p)
        x,v,p,eps=out;value=float(eps);eps_history.append(value)
        if m+1>=ac.minPseudoIterations and value<ac.epsilonV:break
    if problem.positionOrigin is not None:
        coords=dual.positionCoordinates(problem,x)
        low=(problem.positionOrigin-coords)+(problem.positionLow+x)
        return dual.DualTimeResult(coords,v,p,m+1,eps_history[-1],x,low)
    return dual.DualTimeResult(x,v,p,m+1,eps_history[-1])


def accepted_metric(problem,x,v):
    from warpSPH.modules.artificialCompressible import dualTime as dual
    from warpSPH.modules.artificialCompressible.parameters import convergenceMetricDevice
    rate=dual.bdfSource(problem,x,v)[0]
    prior=getattr(problem.schemeConfig,'_dyngpusphShiftBDF',None)
    if prior is not None:rate=rate-prior
    bounced=dual.bounceVelocities(problem.bounce,dual.positionCoordinates(problem,x),v)
    return convergenceMetricDevice(bounced-rate,v,problem.fluidIndex,problem.nFluid,problem.schemeConfig)


def reference_stopping(options):
    """Change only the declared stop test around the qualified reference map.

    Exact source substitutions are asserted and retained in the row receipt.
    Graph validation includes the auxiliary accepted-state metric when used.
    """
    from warpSPH.modules.artificialCompressible import dyngpusphReference as ref
    original=ref.solve
    if options.plateau_trim:return {'solve':'unchanged current reference endpoint'}
    source=inspect.getsource(original);namespace=dict(original.__globals__)
    def change(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    change('        if m>=5:\n','        if False: # ablation: plateau/trim not yet enabled\n')
    if not options.pressure_gate:
        change('checked[0]<ac.epsilonV and checked[1]<ac.epsilonV+1:',
            'checked[0]<ac.epsilonV: # ablation: velocity gate only')
    receipt={'originalSolveSHA256':hashlib.sha256(inspect.getsource(original).encode()).hexdigest()}
    if not options.last_stage_metric:
        iteration=inspect.getsource(ref.iteration)
        marker='        return (*out,motion)\n';assert iteration.count(marker)==1
        iteration=iteration.replace(marker,'        return (*out,motion,_accepted_metric(problem,xN,vN))\n')
        namespace['_accepted_metric']=accepted_metric
        exec(compile(iteration,'ablation-accepted-metric-iteration','exec'),namespace)
        change('x,v,p,lam,norm,edges,switch,metric,last_position,last_pressure,motion=out\n',
            'x,v,p,lam,norm,edges,switch,metric,last_position,last_pressure,motion=out[:11]\n')
        change('torch.cat((metric,motion.reshape(1).to(metric.dtype)))',
            'torch.cat((metric,motion.reshape(1).to(metric.dtype),out[11].reshape(1).to(metric.dtype)))')
        change('        checked=full\n','        checked=[values[3],full[1]] # ablation: old accepted-state velocity norm\n')
        receipt['observedIterationSource']=iteration
    exec(compile(source,'ablation-reference-stop-policy','exec'),namespace)
    ref.solve=namespace['solve'];receipt['solveSource']=source
    return receipt


def install(options:Options):
    """Called after fixture configure, before initializer hooks/performance."""
    validate(options)
    import torch
    from warpSPH.modules.artificialCompressible import dualTime as dual
    from warpSPH.modules.artificialCompressible import dyngpusphShifting as transport
    from warpSPH.systems import artificialCompressible as system
    step=importlib.import_module('warpSPH.schemes.incompressible.artificialCompressible')
    receipt={}
    if options.static_geometry:
        original_body=system.updateBodyParticlesWCSPH;checked=[False]
        def body(state,rigid):
            if not checked[0]:
                assert float(rigid.angularVelocity)==0. and not bool((rigid.linearVelocity!=0).any())
                checked[0]=True
            positions,offsets=state.positions,state.ghostOffsets
            result=original_body(state,rigid)
            # Preserve static geometry while retaining the old velocity/body
            # kinematic reset. This is local to ACSPH finalization, not init.
            result.positions=positions;result.ghostOffsets=offsets
            return result
        system.updateBodyParticlesWCSPH=body
    if options.accepted_fields:
        original_delta=dual.exactDeltaUpdate;original_finalize=system.ArtificialCompressibleSystem.finalize
        def delta(result,*a,**kw):
            update=original_delta(result,*a,**kw)
            if getattr(result,'referenceFields',None) is None:
                update._ablationAcceptedFields=(result.velocities,result.pressures)
            return update
        def finalize(self,initial,dt,returns,updates,weights=...,*a,**kw):
            fields=getattr(updates[-1],'_ablationAcceptedFields',None)
            if fields is not None:self.state.velocities,self.state.pressures=fields
            return original_finalize(self,initial,dt,returns,updates,weights,*a,**kw)
        dual.exactDeltaUpdate=delta;step.exactDeltaUpdate=delta
        system.ArtificialCompressibleSystem.finalize=finalize
    if options.source_shift:
        original_transport=transport.shift_and_remap
        def shifted(state,*a,**kw):
            result=original_transport(state,*a,**kw)
            # Native geometry/history stay identical; suppress only the chosen
            # field assignment, including that field's native safeguards.
            return select_remap_outputs(result,state,options)
        transport.shift_and_remap=shifted
    if options.warmup and not options.source_operators:
        dual.solveDualTime=legacy_warm_solve;step.solveDualTime=legacy_warm_solve
    if options.source_operators:receipt['stopping']=reference_stopping(options)
    return receipt
