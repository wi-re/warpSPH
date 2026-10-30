"""Explicit static-2D dynGPUSPH AC-2L solve and external transport.

Keeps the compiled model separate from the published/default path. The
selected warpSPH bounce is an explicit comparison control. Cohesion/acos and
shift-history source quirks are retained; this is not a precision remedy.
"""
import copy
import math
from dataclasses import replace
import torch
from warpSPHCore import buildVerletList, SupportScheme
from ...configurations.compute import neighborSearchOptions
from . import dyngpusphOperators as ops
from .workingState import workingState
from .surfaceFlags import diffusionTaper
from .wallClosure import bounceVelocities
from .referenceMotion import referenceMotionBudget


def validate(problem):
    from ...enumTypes import PressureSmoothingScheme
    p=problem.schemeConfig.acParams
    if (problem.state.positions.shape[1]!=2 or bool(problem.config.domain.periodic.any())
            or getattr(problem.schemeConfig,'boundaryProvider',None) is not None):
        raise ValueError('dyngpusph-static-2d requires nonperiodic sampled 2D walls')
    if p.pressureSmoothing is not PressureSmoothingScheme.renormalizedBiLaplacian or p.rkStages!=3:
        raise ValueError('dyngpusph-static-2d currently qualifies AC-2L / SSPRK3 only')
    if (p.cavitationProjection!='off' or p.isolatedZeroPressure or p.bdfShiftCorrection
            or p.wallBounceTarget!='position'):
        raise ValueError('Reference path requires no pressure projections, external source shift history, position-only bounce')
    if bool((problem.state.velocities[problem.state.kinds==1]!=0).any()):
        raise ValueError('Reference path currently supports stationary walls only')
    for body in problem.schemeConfig.rigidBodies:
        if float(body.angularVelocity) != 0.0 or bool((body.linearVelocity != 0).any()):
            raise ValueError('Reference path freezes static wall geometry; moving rigid bodies are unsupported')
    if not bool((problem.state.supports == problem.state.supports[0]).all()):
        raise ValueError('Reference path requires fixed uniform support')
    if not bool((problem.state.densities == float(problem.schemeConfig.fluid.restDensity)).all()):
        raise ValueError('Reference path requires constant physical reference density')


def residual_metrics(position_residual,pressure_residual,velocities,fluid_index,n,dt,rho0,c0,trim=False):
    """Native 1/N norms, always also report full untrimmed diagnostics."""
    vmax=velocities.index_select(0,fluid_index).double().square().sum(-1).max().sqrt().clamp(1e-5,.1*c0)
    v2=position_residual.index_select(0,fluid_index).double().square().sum(-1)
    p2=pressure_residual.index_select(0,fluid_index).double().square()
    if trim:
        count=min(n,int(.95*n+.5))
        v2=torch.sort(v2).values[:count];p2=torch.sort(p2).values[:count]
    else:count=n
    return torch.stack((torch.log10(v2.sum().sqrt()/(count*vmax)),
        torch.log10(p2.sum().sqrt()/count*dt/(rho0*.01*c0*c0))))


def iteration(problem,x,v,p,prior_lambda,prior_normal,prior_edges,frozen_switch,*,warmup,initialize_switch,
              cell_positions,adjacency,track_motion=False):
    """Pure tensor map: every evolving surface/sensor field is an input/output.

    Graph validation therefore qualifies the same auxiliary state consumed by
    the next iteration and remap; there are no Python stage-counter side effects.
    """
    from .dualTime import bdfSource,positionCoordinates
    dt=problem.dtau*problem.schemeConfig.acParams.dtOverDtau
    dtau=problem.dtau/(2 if warmup else 1)
    beta=.75*(problem.state.supports/2)/dtau
    prior_shift=getattr(problem.schemeConfig,'_dyngpusphShiftBDF',torch.zeros_like(v))
    dx_bdf,dv_bdf=bdfSource(problem,x,v)
    dx_bdf=dx_bdf-prior_shift
    slopes=[[],[],[]];tab=problem.tableau
    stages=3
    lam,norm,edges=prior_lambda,prior_normal,prior_edges
    switch=frozen_switch
    pressure_grad=pressure_lap=None
    frozen_diffusion=None
    # Destination-row alpha_PI uses the source's literal C++ (3/2)==1.
    factors=[1/(1+float(tab.b[s])*dtau/dt) for s in range(stages)]
    motion=x.new_zeros(()) if track_motion else None
    for s in range(stages):
        xs,vs,ps=x,v,p
        for l in range(s):
            a=float(tab.a[s,l])*factors[s]
            if a:
                xs=xs+dtau*a*slopes[0][l]
                vs=vs+dtau*a*slopes[1][l]
                ps=ps+dtau*a*slopes[2][l]
        coords=positionCoordinates(problem,xs)
        if track_motion:
            motion=torch.maximum(motion,(coords-cell_positions).norm(dim=-1).max())
        view=workingState(problem.state,coords,vs,ps)
        # Native closes walls before refreshing stage-zero lambda/raw flags.
        closed,wall_normal,average,divF,lapV=ops.closures(view,problem.config,adjacency,
            lam,edges,problem.bodyForce,cell_positions=cell_positions)
        view.pressures=closed
        if s==0:
            lam,norm,pressure_grad,pressure_lap=ops.moments(view,problem.config,adjacency,cell_positions=cell_positions)
            norm=torch.where((view.kinds==1)[:,None],wall_normal,norm)
            edges=ops.raw_surface(view,problem.config,adjacency,lam,norm)
        terms=ops.residual_terms(view,problem.config,adjacency,wall_normal,average,divF,lapV,
            pressure_grad,pressure_lap,edges,
            frozen_switch=None if initialize_switch and s==0 else switch,cell_positions=cell_positions)
        edges=terms[9]
        if initialize_switch and s==0:switch=terms[8]
        if s==0:
            frozen_diffusion=(.1*view.supports/2*beta)*terms[7]*diffusionTaper(lam)
        rP=-beta.square()*view.densities*terms[0]+frozen_diffusion
        rV=terms[1]+problem.nu[:,None]*terms[2]+problem.bodyForce+(
            beta*view.supports/2)[:,None]*terms[3]+ops.momentum_jst(view,terms,lapV,beta)
        vposition=bounceVelocities(problem.bounce,coords,vs)
        kx=torch.where(problem.fluidRows,vposition-dx_bdf,0.)
        kv=torch.where(problem.fluidRows,rV-dv_bdf,0.)
        kp=torch.where(problem.fluid,rP,0.)
        slopes[0].append(kx);slopes[1].append(kv);slopes[2].append(kp)
    xN,vN,pN=x,v,p
    for s in range(stages):
        b=float(tab.b[s])*factors[s]
        xN=xN+dtau*b*slopes[0][s]
        vN=vN+dtau*b*slopes[1][s]
        pN=pN+dtau*b*slopes[2][s]
    # Native writes the last-stage closure into accepted P_np1, without a wall
    # pressure increment. Preserve it for the external Taylor remap.
    pN=torch.where(problem.state.kinds==1,closed,pN)
    metric=residual_metrics(slopes[0][-1],slopes[2][-1],vN,problem.fluidIndex,problem.nFluid,
        dt,float(problem.schemeConfig.fluid.restDensity),float(problem.schemeConfig.fluid.fixedSoundSpeed))
    out=(xN,vN,pN,lam,norm,edges,switch,metric,slopes[0][-1],slopes[2][-1])
    if track_motion:
        motion=torch.maximum(motion,(positionCoordinates(problem,xN)-cell_positions).norm(dim=-1).max())
        return (*out,motion)
    return out


def solve(problem,x0,v0,p0,verbose=False):
    from .dualTime import DualTimeResult,positionCoordinates
    from ...utils.cudaGraph import GraphedTensorFunction
    validate(problem)
    scheme=problem.schemeConfig;ac=scheme.acParams
    # Rebuild once per physical step. The actual CSR cutoff includes the
    # configured Verlet margin; certify every RK geometry against that list.
    search=copy.copy(problem.state);search.supports=2*search.supports
    adjacency=buildVerletList(search,problem.config.domain,verletScale=problem.config.verletScale,
        supportMode=SupportScheme.Gather,verbose=False,**neighborSearchOptions(problem.config))
    cells=x0.clone()
    support=float(problem.state.supports.min())
    bounce_cutoff=None
    if problem.bounce is not None:
        old=problem.adjacency
        drift=(cells-old.queryPositions).norm(dim=-1) if old.queryPositions is not None else torch.zeros_like(p0)
        fluid_drift=float(drift[problem.state.kinds==0].max())
        wall_drift=float(drift[problem.state.kinds==1].max()) if bool((problem.state.kinds==1).any()) else 0.
        bounce_cutoff=float(old.querySupports.min())-fluid_drift-wall_drift
    budget=referenceMotionBudget(support,float(adjacency.querySupports.min()),
        bounceCutoff=bounce_cutoff,dx=float(problem.config.dx))
    x,v,p=x0,v0,p0
    if ac.incrementalPositions:
        low=problem.positionLow if problem.positionLow is not None else torch.zeros_like(x0)
        previous=problem.previousPositionStep
        if previous is None:previous=problem.history[0]-problem.history[1]
        problem=replace(problem,positionOrigin=x0,positionLow=low,previousPositionStep=previous)
        x=torch.zeros_like(x0)
    previous=getattr(scheme,'_dyngpusphSurface',None)
    if previous is None:
        # cudaMemset(isurf,1) gives a tiny positive double, effectively zero;
        # wall lambda is set to one by the first moment pass.
        lam=torch.zeros_like(p);norm=torch.zeros_like(v);edges=torch.zeros_like(p)
    else:lam,norm,edges=previous
    switch=torch.zeros_like(p)
    graphs=[];graph=None;history=[];trimmed=False;reason='iteration-cap';maximum_motion=0.
    def normal(*state):
        return iteration(problem,*state,warmup=False,initialize_switch=False,cell_positions=cells,adjacency=adjacency,track_motion=True)
    for m in range(ac.maxPseudoIterations):
        inputs=(x,v,p,lam,norm,edges,switch)
        if m<3:
            out=iteration(problem,*inputs,warmup=m<2,initialize_switch=m==0,cell_positions=cells,adjacency=adjacency,track_motion=True)
        else:
            if graph is None and getattr(scheme,'cudaGraph',False) and x.is_cuda and getattr(scheme,'_acsphGraphDisabled',None) is None:
                graph=GraphedTensorFunction(normal,'dynGPUSPH reference pseudo-iteration',
                    validate=not getattr(scheme,'_dyngpusphGraphValidated',False));graphs.append(graph)
            out=graph(*inputs) if graph is not None else normal(*inputs)
            if graph is not None:
                if graph.disabled is not None:scheme._acsphGraphDisabled=graph.disabled
                else:scheme._dyngpusphGraphValidated=True
        x,v,p,lam,norm,edges,switch,metric,last_position,last_pressure,motion=out
        # Share the existing convergence read; motion tracking stays graphable.
        values=torch.cat((metric,motion.reshape(1).to(metric.dtype))).detach().cpu().tolist()
        full=values[:2];maximum_motion=max(maximum_motion,values[2])
        scheme._dyngpusphMotionDiagnostics=dict(maximumMotion=maximum_motion,
            limit=budget.limit,pairLimit=budget.pair,coneLimit=budget.cone,bounceLimit=budget.bounce,
            iteration=m+1)
        if not math.isfinite(values[2]) or maximum_motion>=budget.limit:
            raise RuntimeError(f'Reference solve exceeded certified candidate motion: '
                f'{maximum_motion:.9g} m >= {budget.limit:.9g} m at iteration {m+1}')
        if any(math.isnan(value) or value == float('inf') for value in full):
            raise FloatingPointError('Nonfinite dynGPUSPH reference residual')
        checked=full
        if trimmed:
            checked=residual_metrics(last_position,last_pressure,v,problem.fluidIndex,problem.nFluid,
                problem.dtau*ac.dtOverDtau,float(scheme.fluid.restDensity),float(scheme.fluid.fixedSoundSpeed),trim=True).cpu().tolist()
        history.append(dict(iteration=m+1,epsilonV=full[0],epsilonP=full[1],
            checkedV=checked[0],checkedP=checked[1],trimmed=trimmed))
        if m+1>=ac.minPseudoIterations and checked[0]<ac.epsilonV and checked[1]<ac.epsilonV+1:
            reason='trimmed-residuals' if trimmed else 'full-residuals';break
        # Native plateau mechanism, retaining full diagnostics even after
        # selecting a trimmed norm. Fluid selection uses stable physical IDs;
        # the source's sorted/unsorted bc indexing accident is not reproduced.
        if m>=5:
            recent=history[-5:]
            vr=[r['checkedV'] for r in recent];pr=[r['checkedP'] for r in recent]
            if all(math.isfinite(a) for a in vr+pr):
                if max(vr)-min(vr)<.001*sum(map(abs,vr))/5:trimmed=True
                if max(pr)-min(pr)<.0005*sum(map(abs,pr))/5 and checked[0]<ac.epsilonV:
                    reason='pressure-plateau';break
    scheme._dyngpusphSurface=(lam.detach(),norm.detach(),edges.detach())
    scheme._dyngpusphCellPositions=cells
    scheme._dyngpusphSolveDiagnostics=dict(iterations=m+1,epsilonV=full[0],epsilonP=full[1],
        checkedV=checked[0],checkedP=checked[1],exitReason=reason,trimmed=trimmed,
        graphCaptured=any(g.graph is not None for g in graphs),graphReplays=sum(g.replays for g in graphs),
        graphDisabled=getattr(scheme,'_acsphGraphDisabled',None),history=history)
    coordinates=positionCoordinates(problem,x)
    if problem.positionOrigin is not None:
        low=(problem.positionOrigin-coordinates)+(problem.positionLow+x)
        result=DualTimeResult(coordinates,v,p,m+1,full[0],x,low)
    else:result=DualTimeResult(coordinates,v,p,m+1,full[0])
    result.referenceFields=dict(velocities=v,pressures=p,lambdas=lam,normals=norm,edges=edges)
    result.epsilonP=full[1];result.exitReason=reason
    return result


def external_shift(state,config,scheme,dt):
    from .dyngpusphShifting import shift_and_remap
    lam,norm,edges=scheme._dyngpusphSurface
    # These are physical wall velocities and the accepted last-RK wall pressure.
    shift,recorded,p,v,active,*_=shift_and_remap(state,config,dt=dt,lambdas=lam,edges=edges,
        normals=norm,pair_multiplicity=scheme._dyngpusphShiftPairMultiplicity,
        cell_positions=scheme._dyngpusphCellPositions)
    previous=getattr(scheme,'_dyngpusphShift',torch.zeros_like(shift))
    old=getattr(scheme,'_dyngpusphShiftBDF',torch.zeros_like(shift))
    alpha=scheme._dyngpusphBDFAlpha
    scheme._dyngpusphShiftBDF=torch.where((active>0)[:,None],alpha*(recorded-previous),old)
    scheme._dyngpusphShift=recorded
    state.pressures=p;state.velocities=v
    return shift
