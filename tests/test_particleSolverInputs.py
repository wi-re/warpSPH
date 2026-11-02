"""Declared flat axes, physical warm seeds and per-linearization UID inputs."""
from dataclasses import replace
from copy import copy

import h5py
import pytest
import torch

from warpSPH.evaluation import MortonStepper,UIDPreconditioner,adopt_spatial_layout
from warpSPH.evaluation.continuation import history_view
from warpSPH.evaluation.solverInputs import flat_view,integrated_description,vector_view
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.state.fields import BlockState,flatten_integrated
from warpSPHIntegrators.schemes.families.ndf import NDF
from warpSPHIntegrators.schemes.families.adaptive_ndf import AdaptiveNDF
from warpSPHIntegrators.schemes.families.variable_adams import VariableAdams
from warpSPHIntegrators.schemes.families.adaptive_adams import AdaptiveAdams
from test_spatialContinuation import fixture,assert_physics


def test_packed_history_and_coupled_vectors_follow_declared_field_axes():
    ctx,state=fixture()
    description=integrated_description(state)
    ids=state.state.UIDs
    p=torch.randperm(len(ids),generator=torch.Generator().manual_seed(41))
    size=flatten_integrated(state).numel()
    vector=torch.arange(3*size,dtype=ctx.dtype).reshape(3,size)
    expected=[];offset=0
    for name,shape,dtype in description:
        width=torch.tensor(shape).prod().item()
        expected.append(vector[:,offset:offset+width].reshape(3,*shape)[:,p].reshape(3,width));offset+=width
    torch.testing.assert_close(flat_view(vector,description,ids,ids[p]),torch.cat(expected,dim=-1),rtol=0,atol=0)
    block=BlockState((state,state.initializeNewState()))
    both=torch.cat([vector[0],vector[1]])
    torch.testing.assert_close(vector_view(both,block,ids,ids[p]),torch.cat([torch.cat(expected,dim=-1)[0],torch.cat(expected,dim=-1)[1]]),rtol=0,atol=0)


@pytest.mark.parametrize('name',['Backward Euler (implicit)','Gauss-Legendre 2'])
def test_warm_stages_and_uid_preconditioner_keep_the_same_implicit_equations(name):
    ctx,state=fixture(fsal=True)
    method=get_scheme(name)
    first=MortonStepper(method)(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-6))
    accepted=adopt_spatial_layout(first)
    calls=[]
    def action(vector,system,context):
        substates=system.states if isinstance(system,BlockState) else (system,)
        assert all(torch.equal(s.state.UIDs,state.state.UIDs) for s in substates)
        calls.append(context['compute_evaluation_context'])
        weight=torch.linspace(.95,1.05,len(state.state.UIDs),dtype=vector.dtype)
        diagonal=torch.cat([weight.repeat_interleave(int(torch.tensor(shape[1:]).prod()) if len(shape)>1 else 1)
                            for s in substates for _,shape,_ in integrated_description(s)])
        return vector*diagonal
    solver=JFNKSolver(tol=1e-6,gmres_maxiter=96,preconditioner=UIDPreconditioner(state.state.UIDs,action))
    warm=first.stages if name.startswith('Gauss') else first.stages[-1]
    result=MortonStepper(method)(accepted.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,
                                 solver=solver,warmStart=warm)
    fresh=MortonStepper(method)(accepted.state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig,solver=JFNKSolver(tol=1e-6))
    assert result.step_status.successful and calls and solver.preconditioner.action is action
    assert_physics(result.state,fresh.state)


@pytest.mark.parametrize('method',[VariableAdams(3),VariableAdams(5,'ab'),AdaptiveAdams(4),NDF(2),AdaptiveNDF(3)])
def test_method_owned_memory_survives_resorts_and_portable_checkpoint(method,tmp_path):
    from warpSPH.io.export import exportSimulationSystem
    from warpSPH.io.continuation import readContinuation
    from warpSPH.io.particleLayout import restoreCheckpoint
    from warpSPHIntegrators.runtime import Continuation
    ctx,state=fixture(fsal=True);state.state.UIDcounter=int(state.state.UIDs.max())+1
    template=state.initializeNewState();stepper=MortonStepper(method,max_coverage_retries=16)
    history=None
    kwargs=dict(config=ctx.config,schemeConfig=ctx.schemeConfig)
    if method.implicit:kwargs['solver']=JFNKSolver(tol=1e-6)
    for i in range(6):
        result=stepper(state,1e-4,history=history,**kwargs)
        assert result.step_status.successful
        if result.history.context.get('current') is not None:
            torch.testing.assert_close(result.history.context['current'],flatten_integrated(result.state),rtol=0,atol=0)
        old_ids=result.state.state.UIDs.clone()
        accepted=adopt_spatial_layout(result)
        assert torch.equal(result.state.state.UIDs,old_ids)
        state,history=accepted.state,accepted.history
        # Force the boundary sort path without changing physical equations.
        state=copy(state);state.spatialContext=replace(state.spatialContext,adjacency=None)
    assert history.context['method'] in ('ndf.v1','variable_adams.v1')
    if isinstance(method,(NDF,AdaptiveNDF)):assert history.context['phase']=='warm'
    exportSimulationSystem(str(tmp_path),'memory','CompSPH',state,
        continuation=Continuation(state,history),integrator=method,config=ctx.config,schemeConfig=ctx.schemeConfig,
        last_dt=1e-4,next_dt=1e-4)
    with h5py.File(tmp_path/'trajectory/memory.h5') as f:
        restoreCheckpoint(f,template,ctx.config,ctx.scheme)
        memory,_,_=readContinuation(f['integration'],template,integrator=method,config=ctx.config,schemeConfig=ctx.schemeConfig)
    a=stepper(state,1e-4,history=history,**kwargs)
    b=stepper(memory.state,1e-4,history=memory.history,**kwargs)
    assert_physics(a.state,b.state)


def test_unknown_method_context_axis_is_rejected_before_use():
    ctx,state=fixture(fsal=True)
    result=MortonStepper(VariableAdams(3))(state,1e-4,config=ctx.config,schemeConfig=ctx.schemeConfig)
    result.history.context['undeclared_rows']=torch.ones(32)
    with pytest.raises(ValueError,match='undeclared'):
        history_view(result.history,state.state.UIDs)


@pytest.mark.parametrize('method',[VariableAdams(3),AdaptiveAdams(4),NDF(2),AdaptiveNDF(3)])
def test_generic_runtime_accepts_non_tableau_methods_and_maps_only_accepted_memory(method):
    from warpSPHIntegrators.runtime import solve
    from warpSPH.evaluation import adopt_spatial_continuation
    ctx,state=fixture(fsal=True)
    kwargs=dict(config=ctx.config,schemeConfig=ctx.schemeConfig)
    if method.implicit:kwargs['solver']=JFNKSolver(tol=1e-6)
    result=solve(MortonStepper(method,max_coverage_retries=16).as_scheme(),state,6e-4,ctx.stepFunction,
        dt=1e-4,adaptive=False,reuse=False,continuation=adopt_spatial_continuation,
        allow_stagnation=True,**kwargs)
    assert result.successful and result.accepted_steps==6
    assert torch.equal(result.history.context['particle_uids'],result.state.state.UIDs)
    if result.history.context.get('current') is not None:
        torch.testing.assert_close(result.history.context['current'],flatten_integrated(result.state),rtol=0,atol=0)
