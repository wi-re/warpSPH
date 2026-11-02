"""GPU continuation/renewal and actual native pair-physics qualification.

Small repeated numerical fixtures deliberately omit video and viewer registration.
Run sequentially in the normal GPU-enabled Warp environment.
"""
from copy import copy
from dataclasses import replace

import h5py
import pytest
import torch

from warpSPH.evaluation import (MortonStepper,adopt_spatial_layout,UIDPreconditioner,
                               adopt_spatial_continuation)
from warpSPH.evaluation.solverInputs import integrated_description
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.runtime import Continuation,solve
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.schemes.families.ndf import NDF
from warpSPHIntegrators.schemes.families.adaptive_ndf import AdaptiveNDF
from warpSPHIntegrators.schemes.families.variable_adams import VariableAdams
from warpSPHIntegrators.schemes.families.adaptive_adams import AdaptiveAdams
from test_spatialContinuation import fixture,assert_physics,input_order_step

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA in the normal Warp environment')


@pytest.mark.parametrize('fluid',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('method',['rungeKutta4','dormandPrince','ab3','bdf2',VariableAdams(3),AdaptiveAdams(4),NDF(2),AdaptiveNDF(3)])
def test_gpu_method_memory_cold_restore_matches_uninterrupted_uid_physics(fluid,method,tmp_path):
    from warpSPH.io.export import exportSimulationSystem
    from warpSPH.io.continuation import readContinuation
    from warpSPH.io.particleLayout import restoreCheckpoint
    ctx,state=fixture(fluid,fsal=True,device='cuda')
    scheme=get_scheme(method) if isinstance(method,str) else method
    state.state.UIDcounter=int(state.state.UIDs.max())+1
    template=state.initializeNewState();control=state.initializeNewState()
    control.retainStageDerivatives=True
    if isinstance(scheme,(NDF,AdaptiveNDF)):control.ndf_compatible=True
    stepper=MortonStepper(scheme,max_coverage_retries=16)
    history=control_history=None
    if isinstance(scheme,AdaptiveNDF):
        from warpSPHIntegrators.state.history import StepHistory
        control_history=StepHistory(scheme.metadata.history_length)
    kwargs=dict(config=ctx.config,schemeConfig=ctx.schemeConfig)
    if scheme.implicit:kwargs['solver']=JFNKSolver(tol=1e-6,gmres_maxiter=96)
    for i in range(6):
        # This helper uses an independent unsorted resource manager and the
        # native driver, with the same full-attempt envelope renewal policy.
        baseline=input_order_step(scheme,control,control_history,dict(kwargs,rhs=ctx.stepFunction))
        result=stepper(state,1e-4,history=history,**kwargs)
        assert result.step_status.successful and baseline.step_status.successful
        assert_physics(result.state,baseline.state)
        control,control_history=baseline.state,baseline.history
        if isinstance(scheme,(NDF,AdaptiveNDF)):control.ndf_compatible=True
        accepted=adopt_spatial_layout(result)
        state,history=accepted.state,accepted.history
        state=copy(state);state.spatialContext=replace(state.spatialContext,adjacency=None)
    exportSimulationSystem(str(tmp_path),'gpu',fluid,state,continuation=Continuation(state,history),
        integrator=scheme,config=ctx.config,schemeConfig=ctx.schemeConfig,last_dt=1e-4,next_dt=1e-4)
    with h5py.File(tmp_path/'trajectory/gpu.h5') as f:
        restoreCheckpoint(f,template,ctx.config,ctx.scheme)
        memory,_,_=readContinuation(f['integration'],template,integrator=scheme,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert_physics(stepper(state,1e-4,history=history,**kwargs).state,
                   stepper(memory.state,1e-4,history=memory.history,**kwargs).state)


@pytest.mark.parametrize('name',['Backward Euler (implicit)','Gauss-Legendre 2'])
def test_gpu_warm_and_uid_preconditioned_jfnk(name):
    from warpSPHIntegrators.state.fields import BlockState
    ctx,state=fixture(fsal=True,device='cuda');scheme=get_scheme(name)
    stepper=MortonStepper(scheme,max_coverage_retries=16)
    kwargs=dict(config=ctx.config,schemeConfig=ctx.schemeConfig)
    first=stepper(state,1e-4,solver=JFNKSolver(tol=1e-6),**kwargs)
    accepted=adopt_spatial_layout(first).state;calls=[]
    def action(vector,system,metadata):
        substates=system.states if isinstance(system,BlockState) else (system,)
        assert all(torch.equal(s.state.UIDs,state.state.UIDs) for s in substates)
        calls.append(True)
        return vector
    solver=JFNKSolver(tol=1e-6,preconditioner=UIDPreconditioner(state.state.UIDs,action))
    warm=first.stages if name.startswith('Gauss') else first.stages[-1]
    result=stepper(accepted,1e-4,solver=solver,warmStart=warm,**kwargs)
    fresh=stepper(accepted,1e-4,solver=JFNKSolver(tol=1e-6),**kwargs)
    assert calls and result.step_status.successful
    assert_physics(result.state,fresh.state)


def test_gpu_runtime_fsal_has_accepted_uid_memory():
    ctx,state=fixture(fsal=True,device='cuda');records=[]
    result=solve(MortonStepper(get_scheme('dormandPrince')).as_scheme(),state,4e-4,ctx.stepFunction,
        dt=1e-4,adaptive=False,reuse=True,continuation=adopt_spatial_continuation,
        callback=records.append,allow_stagnation=True,config=ctx.config,schemeConfig=ctx.schemeConfig)
    assert result.successful and result.accepted_steps==4
    assert result.state.spatialContext.layout.identity
    assert all(torch.equal(stage.aux[1].UIDs,record.result.state.state.UIDs) for record in records for stage in record.result.stages)


@pytest.mark.parametrize('fluid',['CompSPH','CRKSPH'])
def test_gpu_fd_coverage_retry_keeps_order_and_failure_preserves_input(fluid,monkeypatch):
    from warpSPH.evaluation.coverage import CoverageRetryExhausted
    import warpSPH.evaluation.sortedStep as module
    ctx,state=fixture(fluid,fsal=True,device='cuda')
    ctx.config.verletScale=1.
    before=state.state.supports.clone();orders=[]
    sorter=module.morton_order
    def sort(*args,**kwargs):orders.append(True);return sorter(*args,**kwargs)
    monkeypatch.setattr(module,'morton_order',sort)
    method=get_scheme('Backward Euler (implicit)')
    kwargs=dict(config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-10))
    with pytest.raises(CoverageRetryExhausted):
        MortonStepper(method,max_coverage_retries=0)(state,1e-4,**kwargs)
    assert state.spatialContext is None and state.adjacency is None and state.pairWork is None
    torch.testing.assert_close(state.state.supports,before,rtol=0,atol=0)
    orders.clear()
    result=MortonStepper(method,max_coverage_retries=16)(state,1e-4,**kwargs)
    assert result.step_status.successful and len(orders)==1
    assert result.state.spatialContext.coverage_renewals>=1
