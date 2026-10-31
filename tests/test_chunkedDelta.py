"""Native periodic delta/Sun2017 chunk contracts; tiny A/B runs omit video."""
from dataclasses import replace
from functools import wraps
import os
import pytest
import torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.periodic.tgvWeaklyCompressible import tgvWeaklyCompressibleCase as base, buildSystem
from warpSPHCore.radiusSearch import buildChunkedVerletList,RowChunkPolicy
from warpSPHCore import SupportScheme


def configuration(scheme,steps=4,resume=None):
    return RunConfiguration('tgv-wc',simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),
        nx=8,nSteps=steps,maxDt=.0001,plot=False,video=False,store=False,show=False,quiet=True,
        progress=False,execution='eager',storeMode='trajectory',integrationScheme='rungeKutta4',
        **(dict(resumeFrom=str(resume)) if resume else {})),
        params=dict(shuffleIters=3,shuffleSeed=413,jitter=.05,initialPressure=True,machTarget=.1,referenceVelocity=1.),
        solver=dict(scheme=scheme))


def chunkedCase(*,communication=None,rows=3,accepted=None):
    def build(ctx):
        return buildSystem(ctx,neighborhoodFactory=lambda state,cfg:buildChunkedVerletList(state,cfg.domain,
            policy=RowChunkPolicy(edgeBudget=4096,maxRows=rows,bytesPerEdge=8),
            supportMode=SupportScheme.SuperSymmetric,verletScale=cfg.verletScale,
            storeI=False,communication=communication))
    return replace(base,buildSystem=build,**(dict(acceptedStep=accepted) if accepted else {}))


@pytest.mark.parametrize('scheme',['deltaSPH','sun2017DeltaSPH'])
@pytest.mark.parametrize('rows',[1,3])
def test_native_initialization_stages_and_shifting_match_flat(scheme,rows):
    records=[]
    def case(chunked):
        original=chunkedCase(rows=rows) if chunked else base
        def build(ctx):
            system=original.buildSystem(ctx)
            native=ctx.stepFunction
            @wraps(native)
            def observed(system,*args,**kwargs):
                result=native(system,*args,**kwargs)
                diffusion=getattr(system.deltaStages,'diffusion',None)
                records.append((chunked,kwargs.get('stageIndex'),float(system.t),
                    result[0].dvdt.clone(),result[0].drhodt.clone(),diffusion))
                return result
            ctx.bundle=replace(ctx.bundle,stepFunction=observed)
            return system
        return replace(original,buildSystem=build)
    flat=configuration(scheme).run(case(False))
    chunks=configuration(scheme).run(case(True))
    assert not flat.diverged and not chunks.diverged
    assert float(flat.state.t)==float(chunks.state.t)
    for name,value in vars(flat.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(chunks.state.state,name),value,rtol=0,atol=0,msg=name)
    a=[r for r in records if not r[0]];b=[r for r in records if r[0]]
    assert len(a)==len(b)==16
    for ra,rb in zip(a,b):
        assert ra[1:3]==rb[1:3]
        torch.testing.assert_close(ra[3],rb[3],rtol=0,atol=0)
        torch.testing.assert_close(ra[4],rb[4],rtol=0,atol=0)
    if scheme=='sun2017DeltaSPH':
        for stage in range(0,16,4):
            assert all(b[stage+i][5] is b[stage][5] for i in range(4))
            if stage:assert b[stage][5] is not b[stage-4][5]
    assert chunks.state.state.pressures.min()<0


@pytest.mark.parametrize('scheme',['deltaSPH','sun2017DeltaSPH'])
def test_native_checkpoint_restores_density_and_fresh_next_step_diffusion(tmp_path,scheme):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    full=configuration(scheme).run(chunkedCase())
    first=configuration(scheme,steps=2).run(chunkedCase())
    path=writeChunkedCheckpoint(tmp_path/'state.h5',first.ctx,first.state)
    resumed=configuration(scheme,steps=2,resume=path).run(chunkedCase(rows=1))
    assert not resumed.diverged and resumed.ctx.scratch['completedSteps']==4
    assert float(full.state.t)==float(resumed.state.t)
    for name,value in vars(full.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(resumed.state.state,name),value,rtol=0,atol=0,msg=name)


@pytest.mark.parametrize('scheme',['deltaSPH','sun2017DeltaSPH'])
@pytest.mark.parametrize('site',['rhs','shift'])
def test_allocation_failure_in_rhs_or_finalize_preserves_accepted_state(tmp_path,monkeypatch,scheme,site):
    import importlib
    from pathlib import Path
    from warpSPH.runner import RunAllocationError
    dispatch=importlib.import_module('warpSPHCore.autograd.row_dispatch')
    held={}
    def accepted(ctx,system,completed):
        held['ctx']=ctx
        if completed==2:
            held['state']={k:v.clone() for k,v in vars(system.state).items() if isinstance(v,torch.Tensor)}
            held['time']=float(system.t)
    case=chunkedCase(accepted=accepted)
    full=configuration(scheme).run(case)
    with monkeypatch.context() as patch:
        launch=dispatch._launchLocal
        def fail(spec,*args,**kwargs):
            match='computeDeltaShift' if site=='shift' else 'computeDensityDiffusionDeltaSPH'
            if held['ctx'].scratch.get('completedSteps')==2 and match in spec.kernel.key:
                raise torch.OutOfMemoryError('injected delta bulk '+site)
            return launch(spec,*args,**kwargs)
        patch.setattr(dispatch,'_launchLocal',fail)
        cfg=configuration(scheme);cfg.simulation['exportRoot']=str(tmp_path)
        with pytest.raises(RunAllocationError) as caught:cfg.run(case)
    error=caught.value
    assert error.recoveryError is None,error.recoveryError
    assert Path(error.checkpointPath).is_file() and error.result.nSteps==2
    assert float(error.result.state.t)==held['time']
    for name,value in held['state'].items():
        torch.testing.assert_close(getattr(error.result.state.state,name),value,rtol=0,atol=0,msg=name)
    resumed=configuration(scheme,steps=2,resume=error.checkpointPath).run(chunkedCase(rows=1))
    for name,value in vars(full.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(resumed.state.state,name),value,rtol=0,atol=0,msg=name)


def nativeThreeDimensional(scheme,communication=None):
    from warpSPH.configurations import SimulationConfig
    from warpSPH.configurations.weaklyCompressible import WeaklyCompressibleSPHConfig,Sun2017DeltaSPHConfig
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleSystem,WeaklyCompressibleState
    from warpSPH.sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
    from warpSPH.schemes.weaklyCompressible.deltaSPH import deltaSPH_step
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPHCore import get_torch_precision
    from warpSPHIntegrators.schemes import get_scheme,IntegrationSchemeType
    from warpSPH.io.hdf5 import dumpState,loadState
    dtype=get_torch_precision();device=torch.device(os.environ.get('WARPSPH_TEST_DEVICE','cpu'))
    config=SimulationConfig(device=device,dtype=dtype,dim=3,
        domain=buildDomainDescription(l=1.,dim=3,periodic=True,device=device,dtype=dtype))
    sc=Sun2017DeltaSPHConfig() if scheme=='sun2017DeltaSPH' else WeaklyCompressibleSPHConfig()
    sc.surfaceDetectionConfig.active=False;sc.gravityConfig.active=False;sc.fluid.fixedSoundSpeed=10.
    config.dt=1e-5
    integrator=get_scheme(IntegrationSchemeType.rungeKutta4)
    results=[]
    for chunked in (False,True):
        factory=(lambda state,cfg:buildChunkedVerletList(state,cfg.domain,
            policy=RowChunkPolicy(edgeBudget=4096,maxRows=2,bytesPerEdge=8),
            supportMode=SupportScheme.SuperSymmetric,verletScale=cfg.verletScale,communication=communication)) if chunked else None
        system=setupBasicWeaklyCompressibleInitialState(4,config,sc,WeaklyCompressibleState,WeaklyCompressibleSystem,
            neighborhoodFactory=factory)
        s=system.state
        s.velocities=.05*torch.sin(6.28*s.positions)
        s.densities=1+.002*torch.cos(6.28*s.positions[:,0])
        for _ in range(2):
            system=integrator.function(state=system,dt=config.dt,f=deltaSPH_step,config=config,schemeConfig=sc).state
        results.append(system)
    for name,value in vars(results[0].state).items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(getattr(results[1].state,name),value,rtol=0,atol=0,msg=name)
    return results


@pytest.mark.parametrize('scheme',['deltaSPH','sun2017DeltaSPH'])
def test_three_dimensional_native_steps_and_density_roundtrip(tmp_path,scheme):
    import h5py
    from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState
    from warpSPH.io.hdf5 import dumpState,loadState
    results=nativeThreeDimensional(scheme)
    device=results[1].state.positions.device
    with h5py.File(tmp_path/'native.h5','w') as file:dumpState(results[1].state,file.create_group('state'))
    with h5py.File(tmp_path/'native.h5') as file:restored=loadState(file['state'],device,WeaklyCompressibleState)
    torch.testing.assert_close(restored.densities,results[1].state.densities,rtol=0,atol=0)


def test_single_row_pressure_mask_keeps_field_identity():
    from warpSPH.modules.pressure.wp_surfaceAware import computePressureSurfaceAwareWarp
    from warpSPHCore import OperationProperties,buildVerletList
    captured={}
    def build(ctx):
        system=base.buildSystem(ctx);captured.update(system=system,ctx=ctx)
        return system
    configuration('deltaSPH',steps=1).run(replace(base,buildSystem=build))
    state=captured['system'].state;ctx=captured['ctx']
    pressure=-torch.ones_like(state.densities)
    mask=(torch.arange(len(pressure),device=pressure.device)%2).to(torch.int32)
    flat=buildVerletList(state,ctx.config.domain,supportMode=SupportScheme.SuperSymmetric,queryOrder='input')
    chunks=buildChunkedVerletList(state,ctx.config.domain,policy=RowChunkPolicy(edgeBudget=4096,maxRows=1,bytesPerEdge=8),
        supportMode=SupportScheme.SuperSymmetric)
    kw=dict(queryParticles=state,operationProperties=OperationProperties(kernel=ctx.config.kernel,supportMode=SupportScheme.SuperSymmetric),
        domain=ctx.config.domain,queryPressures=pressure,querySurfaceMask=mask)
    torch.testing.assert_close(computePressureSurfaceAwareWarp(**kw,adjacency=chunks),
        computePressureSurfaceAwareWarp(**kw,adjacency=flat),rtol=0,atol=0)


@pytest.mark.parametrize('operator',['pressure','viscosity'])
def test_flat_reverse_derivatives_and_chunked_ad_rejection(operator):
    import torch.autograd.forward_ad as fwAD
    from warpSPHCore import ParticleState,DomainDescription,OperationProperties,KernelFunctions,buildVerletList,get_torch_precision
    from warpSPH.modules.pressure.wp_surfaceAware import computePressureSurfaceAwareWarp
    from warpSPH.modules.deltaSPH.wp_viscosityDelta import computeVelocityDiffusionDeltaSPH
    device=torch.device(os.environ.get('WARPSPH_TEST_DEVICE','cpu'));dtype=get_torch_precision()
    t=torch.arange(17,device=device,dtype=dtype)
    x=torch.stack((.3*torch.sin(t),.3*torch.cos(t*1.7)),1)
    state=ParticleState(positions=x,supports=x.new_full((17,),.6),masses=x.new_ones(17),densities=x.new_ones(17),
        kinds=torch.zeros(17,device=device,dtype=torch.int32))
    domain=DomainDescription(x.new_full((2,),-.5),x.new_full((2,),.5),torch.ones(2,device=device,dtype=torch.bool),2)
    flat=buildVerletList(state,domain,supportMode=SupportScheme.SuperSymmetric,queryOrder='input')
    chunks=buildChunkedVerletList(state,domain,policy=RowChunkPolicy(edgeBudget=1024,maxRows=1,bytesPerEdge=8),supportMode=SupportScheme.SuperSymmetric)
    props=OperationProperties(kernel=KernelFunctions.Wendland2,supportMode=SupportScheme.SuperSymmetric)
    if operator=='pressure':
        field=-1+.1*torch.sin(t)
        def evaluate(value,adj):
            return computePressureSurfaceAwareWarp(state,props,domain,queryPressures=value,
                querySurfaceMask=(torch.arange(17,device=device)%2).to(torch.int32),adjacency=adj)
    else:
        field=.1*x.cos()
        def evaluate(value,adj):
            return computeVelocityDiffusionDeltaSPH(state,props,domain,queryVelocities=value,referenceVelocities=value,
                inviscid=False,nu=.1,c_s=10.,alpha=.01,approachOnly=False,adjacency=adj)
    tangent=.07*torch.sin(torch.arange(field.numel(),device=device,dtype=dtype).reshape_as(field)*1.3)
    primal=field.clone().requires_grad_();value=evaluate(primal,flat)
    cotangent=torch.cos(torch.arange(value.numel(),device=device,dtype=dtype).reshape_as(value)*.7)
    reverse,=torch.autograd.grad((value*cotangent).sum(),primal)
    # These retained flat operators have reverse AD but no registered JVP.
    with fwAD.dual_level(),pytest.raises(NotImplementedError,match='JVPSpec'):
        evaluate(fwAD.make_dual(field,tangent),flat)
    tol=3e-3 if dtype==torch.float32 else 1e-7
    eps=.01 if dtype==torch.float32 else 1e-5
    fd=(evaluate(field+eps*tangent,flat)-evaluate(field-eps*tangent,flat))/(2*eps)
    torch.testing.assert_close((reverse*tangent).sum(),(fd*cotangent).sum(),rtol=tol,atol=tol)
    with pytest.raises(NotImplementedError,match='AD'):
        evaluate(primal,chunks)


def test_default_initial_relaxation_is_preserved():
    if os.environ.get('WARPSPH_TEST_DEVICE','cpu')!='cpu':
        pytest.skip('Default 128-iteration initialization is covered on CPU; CUDA uses bounded parity controls.')
    results=[]
    for case in (base,chunkedCase()):
        cfg=configuration('sun2017DeltaSPH',steps=1)
        cfg.params.pop('shuffleIters');cfg.params.pop('jitter')
        assert cfg.resolve(case).params['shuffleIters']==128
        results.append(cfg.run(case))
    assert not any(r.diverged for r in results)
    for name,value in vars(results[0].state.state).items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(getattr(results[1].state.state,name),value,rtol=0,atol=0,msg=name)


def test_delta_checkpoint_rejects_invalid_clock_before_publication(tmp_path):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    result=configuration('deltaSPH',steps=1).run(chunkedCase())
    result.ctx.config.dt=0.
    path=tmp_path/'invalid.h5'
    with pytest.raises(ValueError,match='positive next timestep'):
        writeChunkedCheckpoint(path,result.ctx,result.state)
    assert not path.exists()


def test_native_delta_restart_requires_explicit_chunk_target(tmp_path):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    first=configuration('deltaSPH',steps=1).run(chunkedCase())
    path=writeChunkedCheckpoint(tmp_path/'state.h5',first.ctx,first.state)
    with pytest.raises(NotImplementedError,match='explicit chunked target'):
        configuration('deltaSPH',steps=1,resume=path).run(base)
