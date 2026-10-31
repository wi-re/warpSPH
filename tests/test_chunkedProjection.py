"""Native periodic projections on owned rows; tiny A/B controls omit video."""
from dataclasses import replace
from functools import wraps
import os
import pytest
import torch
from warpSPH.runner import RunConfiguration
from warpSPH.cases.periodic.tgv import tgvCase as base,buildSystem
from warpSPHCore.radiusSearch import buildChunkedVerletList,RowChunkPolicy
from warpSPHCore import SupportScheme,get_torch_precision

SCHEMES=('iisph','dfsph','dfsphPredictor','dfsphShift')
_NATIVE_IISPH_FP64_STOP = pytest.mark.xfail(
    condition=os.environ.get('WARPSPH_TEST_DEVICE','cpu').startswith('cuda') and get_torch_precision()==torch.float64,
    reason='Pre-port 7fa1ba7 native IISPH relaxed nx8 stress also stops at dt limit 1.29715e-11 < minDt; active periodic gate remains separate.',
    raises=ValueError, strict=True)
DEFAULT_SCHEMES=(pytest.param('iisph',marks=_NATIVE_IISPH_FP64_STOP),*SCHEMES[1:])


def configuration(scheme,steps=4,resume=None):
    return RunConfiguration('tgv',simulation=dict(device=os.environ.get('WARPSPH_TEST_DEVICE','cpu'),
        nx=8,nSteps=steps,maxDt=.0001,plot=False,video=False,store=False,show=False,quiet=True,
        progress=False,execution='eager',storeMode='trajectory',
        **(dict(resumeFrom=str(resume)) if resume else {})),
        params=dict(relaxSteps=3,shuffleSeed=413,jitter=.01),solver=dict(scheme=scheme))


def chunkedCase(*,communication=None,rows=3,accepted=None):
    def build(ctx):
        return buildSystem(ctx,neighborhoodFactory=lambda state,cfg:buildChunkedVerletList(state,cfg.domain,
            policy=RowChunkPolicy(edgeBudget=4096,maxRows=rows,bytesPerEdge=8),
            supportMode=SupportScheme.SuperSymmetric,verletScale=cfg.verletScale,
            communication=communication))
    return replace(base,buildSystem=build,**(dict(acceptedStep=accepted) if accepted else {}))


@pytest.mark.parametrize('scheme',DEFAULT_SCHEMES)
@pytest.mark.parametrize('rows',[1,3])
def test_native_pressure_iterations_and_fields_match_flat(scheme,rows):
    records=[]
    def case(chunked):
        original=chunkedCase(rows=rows) if chunked else base
        def build(ctx):
            system=original.buildSystem(ctx);native=ctx.stepFunction
            @wraps(native)
            def observed(system,*args,**kwargs):
                result=native(system,*args,**kwargs);update=result[0]
                metrics=[]
                for name in ('iisphResult','divergenceResult','densityResult'):
                    pressure=getattr(update,name,None)
                    if pressure is not None:
                        metrics.append((name,pressure.iterations,pressure.error,pressure.termination,pressure.pressure.clone(),pressure.source.clone()))
                records.append((chunked,metrics,getattr(update,'orderAudit',None)))
                return result
            ctx.bundle=replace(ctx.bundle,stepFunction=observed)
            return system
        return replace(original,buildSystem=build)
    flat=configuration(scheme).run(case(False));chunks=configuration(scheme).run(case(True))
    assert not flat.diverged and not chunks.diverged
    assert float(flat.state.t)==float(chunks.state.t)
    for name,value in vars(flat.state.state).items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(getattr(chunks.state.state,name),value,rtol=0,atol=0,msg=name)
    a=[r for r in records if not r[0]];b=[r for r in records if r[0]]
    assert len(a)==len(b)==4
    for ra,rb in zip(a,b):
        assert ra[2]==rb[2]
        for pa,pb in zip(ra[1],rb[1]):
            assert pa[:4]==pb[:4]
            for x,y in zip(pa[4:],pb[4:]):torch.testing.assert_close(x,y,rtol=0,atol=0)


@pytest.mark.parametrize('scheme',DEFAULT_SCHEMES)
def test_native_checkpoint_preserves_warm_starts(tmp_path,scheme):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    full=configuration(scheme).run(chunkedCase())
    first=configuration(scheme,steps=2).run(chunkedCase())
    path=writeChunkedCheckpoint(tmp_path/'state.h5',first.ctx,first.state)
    resumedCase=chunkedCase(rows=1)
    build=resumedCase.buildSystem
    def checkedBuild(ctx):
        system=build(ctx);native=ctx.stepFunction
        @wraps(native)
        def checked(system,*args,**kwargs):
            from warpSPH.configurations.compute import neighborSearchOptions
            assert system.adjacency.storeI==neighborSearchOptions(ctx.config)['storeI']
            return native(system,*args,**kwargs)
        ctx.bundle=replace(ctx.bundle,stepFunction=checked)
        return system
    resumed=configuration(scheme,steps=2,resume=path).run(replace(resumedCase,buildSystem=checkedBuild))
    assert not resumed.diverged and resumed.ctx.scratch['completedSteps']==4
    assert float(full.state.t)==float(resumed.state.t)
    for name,value in vars(full.state.state).items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(getattr(resumed.state.state,name),value,rtol=0,atol=0,msg=name)


@pytest.mark.parametrize('scheme',DEFAULT_SCHEMES)
def test_pressure_iteration_allocation_failure_keeps_accepted_fields_and_histories(tmp_path,monkeypatch,scheme):
    allocationRecovery(tmp_path,monkeypatch,scheme,'pressure')


def test_post_drift_shift_allocation_failure_keeps_accepted_fields_and_histories(tmp_path,monkeypatch):
    allocationRecovery(tmp_path,monkeypatch,'dfsphShift','shift')


def allocationRecovery(tmp_path,monkeypatch,scheme,site):
    import importlib
    from pathlib import Path
    from warpSPH.runner import RunAllocationError
    dispatch=importlib.import_module('warpSPHCore.autograd.row_dispatch');held={};counts={}
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
            if held.get('ctx') is not None and held['ctx'].scratch.get('completedSteps')==2:
                target='computeIISPHDensityRate' if site=='pressure' else 'computeIISPHPressureDiagonal'
                if target in spec.kernel.key:
                    counts[target]=counts.get(target,0)+1
                    # IISPH's source calls C twice, DF's divergence source
                    # once. The next C consumes iterative F; the shift's
                    # second diagonal is post-drift.
                    if counts[target]==((3 if scheme=='iisph' else 2) if site=='pressure' else 2):
                        raise torch.OutOfMemoryError('injected projection bulk '+site)
            return launch(spec,*args,**kwargs)
        patch.setattr(dispatch,'_launchLocal',fail)
        cfg=configuration(scheme);cfg.simulation['exportRoot']=str(tmp_path)
        with pytest.raises(RunAllocationError) as caught:cfg.run(case)
    error=caught.value
    assert error.recoveryError is None,error.recoveryError
    assert Path(error.checkpointPath).is_file() and error.result.nSteps==2
    assert float(error.result.state.t)==held['time']
    for name,value in held['state'].items():torch.testing.assert_close(getattr(error.result.state.state,name),value,rtol=0,atol=0,msg=name)
    resumed=configuration(scheme,steps=2,resume=error.checkpointPath).run(chunkedCase(rows=1))
    for name,value in vars(full.state.state).items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(getattr(resumed.state.state,name),value,rtol=0,atol=0,msg=name)


@_NATIVE_IISPH_FP64_STOP
def test_projection_native_file_requires_explicit_chunk_target(tmp_path):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    result=configuration('iisph',steps=1).run(chunkedCase())
    path=writeChunkedCheckpoint(tmp_path/'state.h5',result.ctx,result.state)
    with pytest.raises(NotImplementedError,match='explicit chunked target'):
        configuration('iisph',steps=1,resume=path).run(base)


def nativeThreeDimensional(scheme,communication=None):
    from warpSPH.configurations import SimulationConfig
    from warpSPH.schemes import buildScheme
    from warpSPH.sample.weaklyCompressible import setupBasicWeaklyCompressibleInitialState
    from warpSPH.modules.iisph import initializeIISPHBoundaries
    from warpSPH.utils.domain import buildDomainDescription
    from warpSPHCore import get_torch_precision
    from warpSPHIntegrators.schemes import get_scheme
    dtype=get_torch_precision();device=torch.device(os.environ.get('WARPSPH_TEST_DEVICE','cpu'))
    cfg=SimulationConfig(device=device,dtype=dtype,dim=3,
        domain=buildDomainDescription(l=1.,dim=3,periodic=True,device=device,dtype=dtype))
    bundle=buildScheme(scheme);sc=bundle.SimulationConfig()
    sc.surfaceDetectionConfig.active=False;sc.gravityConfig.active=False;sc.shiftProperties.active=False
    sc.diffusionParams.viscidNu=.01;sc.diffusionParams.inviscid=False
    # Retain native nonnegative projection defaults in this independent 3D gate.
    cfg.dt=1e-4;cfg.calibrateNormalization=False
    integrator=get_scheme(bundle.integrationMethod() if callable(bundle.integrationMethod) else bundle.integrationMethod)
    results=[]
    for chunked in (False,True):
        factory=(lambda state,config:buildChunkedVerletList(state,config.domain,
            policy=RowChunkPolicy(edgeBudget=4096,maxRows=2,bytesPerEdge=8),
            supportMode=SupportScheme.SuperSymmetric,verletScale=config.verletScale,communication=communication)) if chunked else None
        system=setupBasicWeaklyCompressibleInitialState(4,cfg,sc,bundle.SimulationState,bundle.SimulationSystem,
            neighborhoodFactory=factory)
        s=system.state;s.velocities=.01*torch.sin(6.28*s.positions)
        s.pressures=.01*torch.cos(6.28*s.positions[:,0]).abs()
        if scheme!='iisph':
            s.densityPressureImpulse=s.pressures*(cfg.dt*cfg.dt)
            s.divergencePressureImpulse=s.pressures*cfg.dt
        (system.boundarySupport,system.analyticBoundaryProvider,system.contactBoundaryProvider)=initializeIISPHBoundaries(s,sc)
        for _ in range(2):system=integrator.function(state=system,dt=cfg.dt,f=bundle.stepFunction,config=cfg,schemeConfig=sc).state
        results.append(system)
    for name,value in vars(results[0].state).items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(getattr(results[1].state,name),value,rtol=0,atol=0,msg=name)
    return results


@pytest.mark.parametrize('scheme',SCHEMES)
def test_three_dimensional_native_maps_match_flat(scheme):
    nativeThreeDimensional(scheme)


@pytest.mark.parametrize('operator',['rate','force','diagonal','gradient','wall'])
def test_flat_pressure_operator_reverse_derivative_and_chunk_ad_guard(operator):
    from warpSPHCore import ParticleState,DomainDescription,OperationProperties,KernelFunctions,buildVerletList,get_torch_precision
    from warpSPH.modules.iisph.wp_densityRate import computeIISPHDensityRateWarp
    from warpSPH.modules.iisph.wp_pressureAcceleration import computeIISPHPressureAccelerationWarp
    from warpSPH.modules.iisph.wp_pressureDiagonal import computeIISPHPressureDiagonalWarp
    from warpSPH.modules.iisph.wp_gradientSum import computeIISPHGradientSumWarp
    from warpSPH.modules.iisph.wp_wallGradient import computeIISPHWallGradientWarp
    dtype=get_torch_precision();device=torch.device(os.environ.get('WARPSPH_TEST_DEVICE','cpu'))
    t=torch.arange(17,device=device,dtype=dtype)
    x=torch.stack((.3*torch.sin(t),.3*torch.cos(t*1.7)),1)
    state=ParticleState(positions=x,supports=x.new_full((17,),.6),masses=x.new_ones(17),densities=x.new_ones(17),
        kinds=(torch.arange(17,device=device)%3==0).to(torch.int32))
    domain=DomainDescription(x.new_full((2,),-.5),x.new_full((2,),.5),torch.ones(2,device=device,dtype=torch.bool),2)
    flat=buildVerletList(state,domain,supportMode=SupportScheme.SuperSymmetric,queryOrder='input')
    chunks=buildChunkedVerletList(state,domain,policy=RowChunkPolicy(edgeBudget=1024,maxRows=1,bytesPerEdge=8),supportMode=SupportScheme.SuperSymmetric)
    props=OperationProperties(kernel=KernelFunctions.Wendland2,supportMode=SupportScheme.SuperSymmetric)
    ratio=1+.1*t.cos();wall=.01*x;gradient=.02*x.cos();field=.1*x.sin();pressure=.1*t.sin()
    def evaluate(volumes,adj):
        kw=dict(queryParticles=state,operationProperties=props,domain=domain,adjacency=adj,restVolumes=volumes)
        if operator=='rate':return computeIISPHDensityRateWarp(**kw,field=field,wallGradient=wall)
        if operator=='force':return computeIISPHPressureAccelerationWarp(**kw,densityRatio=ratio,wallGradient=wall,gradientSum=gradient,contactPressure=pressure,normalizedPressure=pressure)
        if operator=='diagonal':return computeIISPHPressureDiagonalWarp(**kw,densityRatio=ratio,gradientSum=gradient)
        if operator=='gradient':return computeIISPHGradientSumWarp(**kw,wallGradient=wall)
        return computeIISPHWallGradientWarp(**kw)
    volumes=.01+.001*t.sin();tangent=.001*(1+t.cos())
    primal=volumes.clone().requires_grad_();value=evaluate(primal,flat)
    weight=torch.cos(torch.arange(value.numel(),device=device,dtype=dtype).reshape_as(value)*.7)
    reverse,=torch.autograd.grad((value*weight).sum(),primal)
    eps=.01 if dtype==torch.float32 else 1e-5;tol=3e-3 if dtype==torch.float32 else 1e-7
    fd=(evaluate(volumes+eps*tangent,flat)-evaluate(volumes-eps*tangent,flat))/(2*eps)
    torch.testing.assert_close((reverse*tangent).sum(),(fd*weight).sum(),rtol=tol,atol=tol)
    torch.testing.assert_close(evaluate(volumes,chunks),evaluate(volumes,flat),rtol=0,atol=0)
    with pytest.raises(NotImplementedError,match='AD'):evaluate(primal,chunks)


def test_projection_restart_preserves_recovering_timestep_clock(tmp_path):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    def accepted(ctx,system,completed):
        if completed==2:ctx.config.dt=.00002
    case=chunkedCase(accepted=accepted)
    full=configuration('dfsph').run(case)
    first=configuration('dfsph',steps=2).run(case)
    path=writeChunkedCheckpoint(tmp_path/'state.h5',first.ctx,first.state)
    resumed=configuration('dfsph',steps=2,resume=path).run(case)
    assert float(resumed.state.t)==float(full.state.t)
    assert float(resumed.ctx.config.dt)==float(full.ctx.config.dt)>.00002
    assert float(resumed.ctx.scratch['projectionNominalDt'])==float(full.ctx.scratch['projectionNominalDt'])
    for name,value in vars(full.state.state).items():
        if isinstance(value,torch.Tensor):torch.testing.assert_close(getattr(resumed.state.state,name),value,rtol=0,atol=0,msg=name)


@_NATIVE_IISPH_FP64_STOP
def test_projection_checkpoint_rejects_bad_clock_and_incompatible_solver(tmp_path):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    result=configuration('iisph',steps=1).run(chunkedCase())
    path=writeChunkedCheckpoint(tmp_path/'state.h5',result.ctx,result.state)
    cfg=configuration('iisph',steps=1,resume=path)
    cfg.solver['iisph']={'maxErrorPercent':.02}
    with pytest.raises(ValueError,match='configuration is incompatible'):cfg.run(chunkedCase())
    result.ctx.config.dt=0
    with pytest.raises(ValueError,match='positive next timestep'):
        writeChunkedCheckpoint(tmp_path/'invalid.h5',result.ctx,result.state)
    assert not (tmp_path/'invalid.h5').exists()


def activeConfiguration(scheme,steps=4,resume=None):
    # The existing periodic mean-free control has a compatible signed source.
    # Keep it separate from the native absolute-source/default-relaxation stress.
    cfg=configuration(scheme,steps,resume)
    cfg.simulation.update(nx=12,dt=.001,maxDt=.001)
    cfg.params.update(relaxSteps=0,jitter=0.)
    cfg.solver['iisph']={'densitySourcePolicy':'periodicMeanFree'}
    if scheme=='iisph':cfg.solver['iisph']['relaxationFactor']=.25
    return cfg


@pytest.mark.parametrize('scheme',SCHEMES)
def test_active_periodic_pressure_and_warm_start_restart_are_exact(tmp_path,scheme):
    from warpSPH.io.chunkCheckpoint import writeChunkedCheckpoint
    full=activeConfiguration(scheme).run(chunkedCase(rows=8))
    flat=activeConfiguration(scheme).run(base)
    first=activeConfiguration(scheme,steps=2).run(chunkedCase(rows=8))
    path=writeChunkedCheckpoint(tmp_path/'active.h5',first.ctx,first.state)
    resumed=activeConfiguration(scheme,steps=2,resume=path).run(chunkedCase(rows=5))
    assert not any(r.diverged for r in (full,flat,first,resumed))
    assert bool((full.state.state.pressures!=0).any())
    update=full.ctx.scratch['lastStageUpdate']
    result=getattr(update,'iisphResult',None) or update.densityResult
    assert result.termination=='converged'
    if scheme!='iisph':
        assert bool((first.state.state.densityPressureImpulse!=0).any())
        assert bool((first.state.state.divergencePressureImpulse!=0).any())
    for actual in (flat,resumed):
        assert float(actual.state.t)==float(full.state.t)
        for name,value in vars(full.state.state).items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(getattr(actual.state.state,name),value,rtol=0,atol=0,msg=name)


@pytest.mark.parametrize('unqualified',['freeSurface','externalShift','bandGeometry'])
def test_chunked_projection_scope_rejects_unqualified_paths(unqualified):
    cfg=activeConfiguration('dfsphShift',steps=1)
    if unqualified=='freeSurface':cfg.params['freeSurface']=True
    elif unqualified=='externalShift':cfg.params['shifting']=True
    else:cfg.solver['dfsph']={'pressureBoundary':'band2018'}
    with pytest.raises(NotImplementedError,match='unqualified|native pressure geometry'):
        cfg.run(chunkedCase())
