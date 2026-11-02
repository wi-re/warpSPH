"""Sequential short GPU fixtures; no video/viewer for these repeated A/Bs."""
from dataclasses import replace
import time

import pytest
import torch

from warpSPH.evaluation import MortonStepper,adopt_spatial_layout,canonical_view
from warpSPH.runner import run
from test_nativeParticleOrder import fixture,HISTORY_ROWS
from test_spatialEvaluationResources import setup

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='requires the normal CUDA Warp environment')


def test_native_sampled_wall_ghosts_keep_graphs_and_physical_relations():
    from test_cudaGraph import _marrone
    a=_marrone(16,8,cudaGraph=False,particleOrder='input',progress=True)
    b=_marrone(16,8,cudaGraph=True,particleOrder='morton',progress=True,pipelineOutputs=False)
    def order(state):
        parent=state.UIDs[state.ghostIndices.clamp_min(0).long()]
        key=torch.where(state.kinds==2,-parent.long()-1,state.UIDs.long())
        return torch.argsort(key)
    x,y=a.state.state,b.state.state
    p,q=order(x),order(y)
    for name in ('positions','velocities','densities','pressures','supports','ghostOffsets'):
        atol=2e-4
        if name=='pressures':
            # The acoustic EOS amplifies a few density ulps by c0^2.
            fluid=b.ctx.schemeConfig.fluid
            atol=max(atol,8*torch.finfo(x.densities.dtype).eps*float(fluid.fixedSoundSpeed)**2*float(fluid.restDensity))
        torch.testing.assert_close(getattr(x,name)[p],getattr(y,name)[q],rtol=3e-4,atol=atol,msg=name)
    assert b.ctx.scratch['stepGraph'].disabled is None
    assert b.ctx.scratch['stepGraph'].replays>=4
    valid=y.ghostIndices>=0
    assert torch.equal(y.ghostIndices[y.ghostIndices[valid]],torch.arange(len(y.UIDs),device='cuda')[valid].to(y.ghostIndices.dtype))
    print(f'mDBC wall: native {a.wallTime:.3f}s; Morton graph {b.wallTime:.3f}s, '
          f'{b.series("stepTime_ms").mean():.3f}ms/step',flush=True)


@pytest.mark.parametrize('scheme',['artificialCompressible','iisph','dfsph'])
def test_native_gpu_solver_histories_and_acsph_iteration_graph(scheme,tmp_path):
    from test_nativeParticleOrder import close
    case,opts=fixture(scheme,device='cuda',cudaGraph=True)
    baseline=run(case,**opts)
    ordered=run(case,**dict(opts,particleOrder='morton',store=True,storeInterval=1,exportRoot=str(tmp_path)))
    close(ordered.state,baseline.state)
    if scheme=='artificialCompressible':
        assert getattr(ordered.ctx.schemeConfig,'_acsphGraphDisabled',None) is None
        assert ordered.ctx.schemeConfig._acsphGraphValidated
        assert ordered.ctx.scratch['lastStageUpdate'].pseudoIterations<ordered.ctx.schemeConfig.acParams.maxPseudoIterations
    checkpoint=__import__('pathlib').Path(ordered.exportPath,'trajectory','state_0001.h5')
    resumed=run(case,**dict(opts,nSteps=1),particleOrder='morton',resumeFrom=str(checkpoint),resumeStepOffset=2)
    close(resumed.state,ordered.state)
    print(f'{scheme}: GPU baseline {baseline.wallTime:.3f}s, Morton {ordered.wallTime:.3f}s, '
          f'{ordered.series("stepTime_ms").mean():.3f}ms/step (checkpoint I/O excluded)',flush=True)


@pytest.mark.parametrize('scheme',['CompSPH','CRKSPH'])
def test_compressible_runner_preserves_graph_route(scheme):
    from test_runnerParticleOrder import fixture_case,options,by_uid
    opts=options()
    opts.update(device='cuda',nSteps=12,scheme=scheme,particleOrder='morton',cudaGraph=True,pipelineOutputs=False)
    result=run(fixture_case(),**opts)
    sg=result.ctx.scratch['stepGraph']
    assert sg.disabled is None
    assert sg.captures>=1 and sg.replays-sg.fallbacks>=8
    assert result.ctx.scratch['nativeLayout'].generation==1
    print(f'{scheme} Morton graph: {result.wallTime:.3f}s, {result.series("stepTime_ms").mean():.3f}ms/step; '
          f'captures={sg.captures}, replays={sg.replays}',flush=True)


@pytest.mark.parametrize('pipeline',[False,True])
def test_native_graph_replays_before_and_after_layout_change(pipeline,monkeypatch):
    from warpSPH.evaluation.nativeLayout import NativeMortonLayout
    observed={}
    original=NativeMortonLayout.flags
    def flags(self,system,config):
        value=original(self,system,config)
        observed.setdefault(id(self),[]).append(isinstance(value,bool))
        return value
    monkeypatch.setattr(NativeMortonLayout,'flags',flags)
    case,opts=fixture('deltaSPH',device='cuda',nSteps=16)
    opts['params']['soundSpeed']=10.
    def shift(ctx,system,i):
        if i==7:system.state.positions.add_(.71)
    case=replace(case,postStep=shift)
    eager=run(case,**opts,particleOrder='morton',cudaGraph=False)
    graph=run(case,**opts,particleOrder='morton',cudaGraph=True,pipelineOutputs=pipeline)
    sg=graph.ctx.scratch['stepGraph']
    assert sg.disabled is None
    assert sg.captures>=2 and sg.replays-sg.fallbacks>=8
    assert graph.ctx.scratch['nativeLayout'].generation==2
    reads=observed[id(graph.ctx.scratch['nativeLayout'])][-16:]
    assert sum(reads)>=12 and not reads[7]  # hook invalidated the captured geometry flag
    for name in ('positions','velocities','densities','pressures','UIDs'):
        torch.testing.assert_close(getattr(graph.state.state,name),getattr(eager.state.state,name),rtol=0,atol=0)
    print(f'native graph pipeline={pipeline}: eager {eager.wallTime:.3f}s, graph {graph.wallTime:.3f}s; '
          f'{graph.series("stepTime_ms").mean():.3f} ms/step; captures={sg.captures}, replays={sg.replays}, fallbacks={sg.fallbacks}',flush=True)


@pytest.mark.parametrize('scheme',['CompSPH','CRKSPH'])
@pytest.mark.parametrize('dim',[2,3])
def test_gpu_evolving_adaptive_support_compatible_energy_and_rebuilds(scheme,dim):
    from warpSPH.evaluation.layout import permute_state
    # A Cartesian acoustic perturbation embedded in 2D/3D, using the low-level
    # sampler. The public linearWave case intentionally admits 1D only.
    from warpSPH.cases.compressible.linearWave import linearWaveCase
    from warpSPH.runner import CaseSpec
    from warpSPH.runner.runner import prepareContext
    spec=CaseSpec.fromDict(dict(linearWaveCase.defaults,scheme=scheme,device='cuda',dim=dim,
        nx=8 if dim==3 else 12,n_h=2.,dt=1e-4,integrationScheme='rungeKutta4',quiet=True,
        params=dict(linearWaveCase.params,nIters=0)))
    ctx=prepareContext(linearWaveCase,spec)
    initial=linearWaveCase.buildSystem(ctx)
    linearWaveCase.initialConditions(ctx,initial)
    velocity=torch.zeros_like(initial.state.positions)
    velocity[:,0]=initial.state.velocities[:,0]
    initial.state.velocities=velocity
    ctx.schemeConfig.compatibleEnergy=True
    order=torch.randperm(len(initial.state.UIDs),device='cuda')
    initial.state=permute_state(initial.state,order)
    baseline=initial.initializeNewState();current=initial.initializeNewState()
    adapter=MortonStepper(ctx.integrator,max_coverage_retries=8)
    kwargs=dict(config=ctx.config,schemeConfig=ctx.schemeConfig)
    start=time.perf_counter()
    for i in range(4):
        baseline=ctx.integrator(baseline,1e-4,ctx.stepFunction,**kwargs).state
        result=adapter(current,1e-4,**kwargs)
        assert result.step_status.successful
        current=adopt_spatial_layout(result).state
        a=canonical_view(baseline,include_adjacency=False)[0].state
        b=canonical_view(current,include_adjacency=False)[0].state
        for name in ('positions','velocities','internalEnergies','supports','densities'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=2e-4,atol=3e-6)
        assert bool((b.internalEnergies>0).all())
        if i==1:
            baseline.state.positions.add_(.21);current.state.positions.add_(.21)
        print(f'{scheme}/{dim}D adaptive-H+compatible: step={i+1}/4',flush=True)
    torch.cuda.synchronize()
    wall=time.perf_counter()-start
    print(f'{scheme}/{dim}D paired steps: {wall:.3f}s, {1000*wall/8:.3f}ms/step including UID checks',flush=True)
