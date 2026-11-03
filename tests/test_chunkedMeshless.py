"""Native meshless stages, closure, sources and accepted restart contracts.

Tiny repeated controls intentionally omit video; these are numerical A/B tests.
"""
from dataclasses import replace
import importlib
import os
from pathlib import Path

import pytest
import torch
from warpSPH.runner import RunConfiguration, RunAllocationError
from warpSPH.cases.compressible.drivenTurbulence import drivenTurbulenceCase as base, buildSystem
from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
from warpSPHCore.radiusSearch import RowChunkPolicy, buildChunkedVerletList, ChunkedPairField


def configuration(scheme, closure='none', steps=4, dim=2, resume=None):
    return RunConfiguration('drivenTurbulence', simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),
        dim=dim, nx=6, nSteps=steps, maxDt=.0002, plot=False, video=False,
        store=False, show=False, quiet=True, progress=False, storeMode='trajectory',
        **(dict(resumeFrom=str(resume)) if resume else {})),
        params=dict(forcingInterval=.000025),
        solver=dict(scheme=scheme, temporalScheme='ssprk', closure=closure))


def chunkedCase(accepted=None, rows=3, communication=None, perturb=False):
    def build(ctx):
        def factory(state,cfg):
            return buildChunkedVerletList(state,cfg.domain,policy=RowChunkPolicy(edgeBudget=4096,
                maxRows=rows,bytesPerEdge=4+state.positions.element_size()),
                supportMode=cfg.supportMode,verletScale=cfg.verletScale, communication=communication)
        system = buildSystem(ctx,neighborhoodFactory=factory)
        if perturb:
            from warpSPH.modules.meshless.geometry import buildGeometry, stashGeometry
            from warpSPH.modules.meshless.primitives import refreshPrimitives
            state=system.state
            state.positions = state.positions + .01*torch.sin(torch.arange(state.positions.numel(),
                device=state.positions.device).reshape_as(state.positions)*1.7)
            geometry=buildGeometry(state.positions,state.supports,state.lam,ctx.config,
                ctx.schemeConfig,system.domain,owner=state)
            refreshPrimitives(state,geometry,ctx.schemeConfig.gamma,ctx.schemeConfig.nngb,ctx.config.dim,updateSupports=False)
            state.lam=geometry.lam
            system.adjacency=geometry.adjacency
            stashGeometry(state.positions,state.supports,system.domain,ctx.config.kernel,ctx.schemeConfig,geometry,owner=state)
            ctx.config.dt=base.timestep(ctx,system)
        return system
    return replace(base,buildSystem=build,**(dict(acceptedStep=accepted) if accepted else {}))


@pytest.mark.parametrize('scheme',['MFM','MFV'])
@pytest.mark.parametrize('closure',['none','project'])
@pytest.mark.parametrize('dim',[2,3])
def test_native_conserved_stages_match_flat(scheme,closure,dim):
    cfg=configuration(scheme,closure,dim=dim)
    flat=cfg.run()
    chunks=configuration(scheme,closure,dim=dim).run(chunkedCase())
    assert not flat.diverged and not chunks.diverged
    assert float(flat.state.t)==float(chunks.state.t)
    for name,value in vars(flat.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(chunks.state.state,name),value,rtol=3e-5,atol=2e-6,msg=name)
    assert abs(chunks.trajectory[-1]['relativeEnergyBudgetResidual'])<5e-6


@pytest.mark.parametrize('lanes', [None, 1])
def test_chunked_meshless_flux_and_limiter_honor_lane_selection(monkeypatch, lanes):
    if not torch.cuda.is_available() or not os.environ.get('WARPSPH_TEST_DEVICE', 'cpu').startswith('cuda'):
        pytest.skip('CUDA required')
    from warpSPHCore import setNeighborLanes
    launcher = importlib.import_module('warpSPHCore.autograd.launcher')
    laneModule = importlib.import_module('warpSPHCore.autograd.lanes')
    flux = importlib.import_module('warpSPH.modules.meshless.flux')._FLUX_SPEC
    limiter = importlib.import_module('warpSPH.modules.reconstruction.meshless')._LIMITER_SPEC
    monkeypatch.setattr(laneModule, '_envLanes', None)
    launch = launcher._launchMaybeTiled
    seen = set()

    def audit(kernel, n, inputs, device, selected, **kwargs):
        for spec in (flux, limiter):
            if kernel in (spec.kernel, spec.tiledKernel):
                expected = spec.tiledKernel if lanes is None else spec.kernel
                assert kernel is expected
                assert (selected or 1) == (32 if lanes is None else 1)
                seen.add(spec.kernel.key)
        return launch(kernel, n, inputs, device, selected, **kwargs)

    monkeypatch.setattr(launcher, '_launchMaybeTiled', audit)
    monkeypatch.setattr(importlib.import_module('warpSPHCore.autograd.prepared'), '_launchMaybeTiled', audit)
    setNeighborLanes(lanes)
    try:
        result = configuration('MFV', steps=2).run(chunkedCase())
    finally:
        setNeighborLanes(None)
    assert not result.diverged
    assert seen == {flux.kernel.key, limiter.kernel.key}
    assert abs(result.trajectory[-1]['relativeEnergyBudgetResidual']) < 5e-6


@pytest.mark.parametrize('scheme',['MFM','MFV'])
@pytest.mark.parametrize('closure',['none','project'])
def test_checkpoint_restarts_native_warm_start_and_budget(tmp_path,scheme,closure):
    full=configuration(scheme,closure).run(chunkedCase())
    first=configuration(scheme,closure,steps=2).run(chunkedCase())
    path=writeChunkedCheckpoint(tmp_path/'state.h5',first.ctx,first.state)
    resumed=configuration(scheme,closure,steps=2,resume=path).run(chunkedCase(rows=1))
    assert not resumed.diverged and resumed.ctx.scratch['completedSteps']==4
    assert float(full.state.t)==float(resumed.state.t)
    for name,value in vars(full.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(resumed.state.state,name),value,rtol=0,atol=0,msg=name)


@pytest.mark.parametrize('scheme',['MFM','MFV'])
@pytest.mark.parametrize('site',['neighbors','weights'])
def test_bulk_failure_preserves_dumpable_conserved_state(tmp_path,monkeypatch,scheme,site):
    held={}
    def accepted(ctx,system,completed):
        held['ctx']=ctx
        if completed==2:
            held['state']={k:v.clone() for k,v in vars(system.state).items() if isinstance(v,torch.Tensor)}
    case=chunkedCase(accepted,perturb=True)
    full=configuration(scheme,'project').run(case)
    def fail():
        if held['ctx'].scratch.get('completedSteps')==2:
            raise torch.OutOfMemoryError('injected meshless bulk allocation')
    with monkeypatch.context() as patch:
        if site=='weights':
            empty=ChunkedPairField.empty
            @classmethod
            def allocation(cls,*args,**kwargs):
                fail()
                return empty(*args,**kwargs)
            patch.setattr(ChunkedPairField,'empty',allocation)
        else:
            module=importlib.import_module('warpSPHCore.radiusSearch.chunked')
            allocate=module._allocateChunks
            def allocation(*args,**kwargs):
                fail()
                return allocate(*args,**kwargs)
            patch.setattr(module,'_allocateChunks',allocation)
        cfg=configuration(scheme,'project')
        cfg.simulation['exportRoot']=str(tmp_path)
        with pytest.raises(RunAllocationError) as caught:
            cfg.run(case)
    error=caught.value
    assert error.recoveryError is None,error.recoveryError
    assert Path(error.checkpointPath).is_file() and error.result.nSteps==2
    for name,value in held['state'].items():
        torch.testing.assert_close(getattr(error.result.state.state,name),value,rtol=0,atol=0,msg=name)
    resumed=configuration(scheme,'project',steps=2,resume=error.checkpointPath).run(chunkedCase(perturb=True))
    for name,value in vars(full.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(resumed.state.state,name),value,rtol=0,atol=0,msg=name)


@pytest.mark.parametrize('scheme,temporal',[('MFM','ssprk'),('MFV','hancock')])
def test_native_temporal_defaults_are_preserved(scheme,temporal):
    from warpSPH.cases import importAll
    from warpSPH.runner.case import getCase
    importAll()
    baseCase=getCase('sod2d')
    def build(ctx):
        system=baseCase.buildSystem(ctx)
        def factory(particles,cfg):
            return buildChunkedVerletList(particles,cfg.domain,policy=RowChunkPolicy(edgeBudget=4096,
                maxRows=3,bytesPerEdge=4+particles.positions.element_size()),
                supportMode=cfg.supportMode,verletScale=cfg.verletScale)
        system.state._meshlessNeighborhoodFactory=factory
        system.state._meshlessGeometry=None
        return system
    def run(case):
        return RunConfiguration('sod2d',simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),
            nx=8,nSteps=2,maxDt=.0001,plot=False,video=False,store=False,quiet=True,show=False,progress=False),
            solver=dict(scheme=scheme)).run(case)
    control=run(baseCase)
    owned=run(replace(baseCase,buildSystem=build))
    assert not owned.diverged and owned.ctx.integrator.temporal==temporal
    for name,value in vars(control.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(owned.state.state,name),value,rtol=3e-5,atol=2e-6,msg=name)


@pytest.mark.parametrize('context',['material','kind','reservoir'])
def test_chunked_meshless_rejects_unqualified_context(context):
    from warpSPH.modules.meshless.chunked import checkMeshlessChunks
    held={}
    def build(ctx):
        system=chunkedCase().buildSystem(ctx)
        held.update(system=system,ctx=ctx)
        return system
    configuration('MFM',steps=1).run(replace(base,buildSystem=build))
    system,ctx=held['system'],held['ctx']
    if context=='material':system.state.materials[0]=1
    elif context=='kind':system.state.kinds[0]=1
    else:system.inflowReservoir=object()
    with pytest.raises(NotImplementedError,match='unqualified|without boundary'):
        checkMeshlessChunks(system,ctx.config,ctx.schemeConfig)
