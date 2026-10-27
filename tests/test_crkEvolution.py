"""Source-matched CRK volume algebra and opt-in configuration contracts."""
from types import SimpleNamespace

import pytest
import torch
from warpSPHCore import ParticleState, DomainDescription, KernelFunctions, get_torch_precision
from warpSPH.configurations import CRKSPHConfig, MaterialConfig
from warpSPH.configurations.crkSPH import crkSPHConfigToDict, dictToCRKSPHConfig
from warpSPH.modules.crk import crkFactors
from warpSPH.modules.crk.evolution import boundedVolume, updateContinuityState
from warpSPH.systems.compSPH import CompSPHState, CompSPHSystem
from warpSPH.systems.compressibleMonaghan import CompressibleSystemUpdate
from warpSPHIntegrators.state.specs import ComponentUpdateSpec


def test_physical_volumes_preserve_sharp_unequal_spacing_density_without_energy_calibration(monkeypatch):
    import warp as wp
    monkeypatch.setattr(wp.config,'kernel_cache_dir','/tmp/warpsph-density-test-kernels')
    dtype=get_torch_precision()
    x = torch.cat([(torch.arange(128,dtype=dtype)+.5)/256, .5+(torch.arange(32,dtype=dtype)+.5)/64])
    ids = (x >= .5).to(torch.int32)
    target = torch.where(ids == 0, torch.ones_like(x), torch.full_like(x,.25))
    volumes = torch.where(ids == 0, torch.full_like(x,1/256.), torch.full_like(x,1/64.))
    s = ParticleState(x[:, None], 4*volumes, volumes*target, torch.zeros_like(ids), target)
    s.materials = ids
    domain = DomainDescription(torch.tensor([0.],dtype=dtype), torch.tensor([1.],dtype=dtype), torch.ones(1, dtype=torch.bool), 1)
    v, rho, _ = crkFactors(s, domain, KernelFunctions.B7, densityPolicy='sameMaterial', volumes=volumes)
    assert v is volumes
    torch.testing.assert_close(rho, target, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_spheral_volume_bounds_and_reverse_derivative(dim):
    v = torch.tensor([1., 1., 1.], requires_grad=True)
    m, rho = torch.ones(3), torch.full((3,), 2.)
    rates = torch.tensor([100., -100., .1])
    support = torch.full((3,), 3.5)
    out = boundedVolume(v, m, rho, rates, 1., support, dim, 3.5)
    assert out[0] == .25
    assert float(out[1].detach()) == pytest.approx((2., torch.pi, 4*torch.pi/3)[dim-1])
    assert float(out[2].detach()) == pytest.approx(.975)
    grad, = torch.autograd.grad(out.sum(), v)
    torch.testing.assert_close(grad, torch.tensor([0., 0., 1.]))


def test_volume_policy_uses_updated_density_not_stage_density():
    one = torch.ones(2)
    z = torch.zeros(2, dtype=torch.int32)
    s = CompSPHState(positions=torch.zeros(2, 1), velocities=torch.zeros(2, 1),
        supports=one*10, masses=one, densities=one*2, kinds=z, materials=z, UIDs=z, UIDcounter=2,
        internalEnergies=one, crkMassDensities=one*2, crkVolumes=one)
    u = CompressibleSystemUpdate(dxdt=s.positions, dvdt=s.positions, dudt=one*0,
        drhodt=one*3, dEdt=one*0)
    updateContinuityState(CompSPHSystem(state=s), u, ComponentUpdateSpec(.2), SimpleNamespace(kernel=KernelFunctions.B7))
    torch.testing.assert_close(s.densities, one*2.6)
    torch.testing.assert_close(s.crkVolumes, one*(1-.2*3/2.6**2))


def test_paper_stage_advances_only_density_without_evolving_volume():
    one=torch.ones(2);ids=torch.zeros(2,dtype=torch.int32)
    s=CompSPHState(positions=torch.zeros(2,1),velocities=torch.zeros(2,1),supports=one,masses=one,
        densities=2*one,kinds=ids,materials=ids,UIDs=ids,UIDcounter=2,internalEnergies=one,
        crkMassDensities=2*one,crkVolumes=None)
    u=CompressibleSystemUpdate(dxdt=s.positions,dvdt=s.positions,dudt=0*one,drhodt=3*one,dEdt=0*one)
    updateContinuityState(CompSPHSystem(state=s),u,ComponentUpdateSpec(.2),SimpleNamespace(kernel=KernelFunctions.B8),SimpleNamespace(crkEvolution='paper'))
    torch.testing.assert_close(s.densities,2.6*one)
    assert s.crkVolumes is None


def test_paper_mode_configuration_and_independent_midpoint_controls():
    c=CRKSPHConfig(crkEvolution='paper',crkDensityPolicy='massReplacement',materials=[MaterialConfig()],
        crkEnergyPartition='entropy',crkViscosityGradient='cached')
    c.validate();b=dictToCRKSPHConfig(crkSPHConfigToDict(c));assert b.crkEvolution=='paper'
    c.crkDensityPolicy='legacy'
    with pytest.raises(ValueError,match='massReplacement'):c.validate()
    c.crkEvolution='summation';c.validate()


def test_paper_density_uses_midpoint_stage_and_full_step_start_independently(monkeypatch):
    """Trace rho'=lambda*rho through the real Butcher state machinery."""
    import numpy as np
    from warpSPHIntegrators.drivers.runge_kutta import RungeKuttaB
    from warpSPH.modules.crk import evolution
    class Toy(CompSPHSystem):
        def initializeNewState(self,*args,**kwargs):
            return Toy(state=self.state.initializeNewState(),pairWork=self.pairWork,t=self.t)
        def finalize(self,*args,**kwargs):return self
    resets=[]
    def reset(system,*args):
        resets.append(float(system.t));system.state.crkMassDensities=system.state.densities.clone()
        system.state.crkVolumes=None
    monkeypatch.setattr(evolution,'prepareStep',reset)
    ids=torch.zeros(2,dtype=torch.int32);one=torch.ones(2,dtype=torch.float64);rho0=one*2
    state=CompSPHState(positions=torch.zeros(2,1,dtype=torch.float64),velocities=torch.zeros(2,1,dtype=torch.float64),
        densities=rho0.clone(),masses=one,supports=one,kinds=ids,materials=ids,UIDs=torch.arange(2),UIDcounter=2,internalEnergies=one)
    c=SimpleNamespace(crkEvolution='paper',crkViscosityGradient='current',crkEnergyPredictor='raw',crkEnergyPartition='native')
    seen=[];rate=.7;dt=.2
    def rhs(system,dt,**kwargs):
        s=system.state;seen.append(s.densities.clone())
        assert s.crkVolumes is None
        return CompressibleSystemUpdate(dxdt=s.velocities,dvdt=torch.zeros_like(s.velocities),dudt=0*one,drhodt=rate*s.densities,dEdt=0*one),None,s
    tableau=SimpleNamespace(a=np.array([[0.,0.],[.5,0.]]),b=np.array([0.,1.]),c=np.array([0.,.5]))
    result=RungeKuttaB(Toy(state=state),dt,rhs,tableau,schemeConfig=c,config=SimpleNamespace(kernel=KernelFunctions.B8))
    assert resets==[0.] and len(seen)==2
    torch.testing.assert_close(seen[1],rho0*(1+.5*dt*rate))
    torch.testing.assert_close(result.state.state.densities,rho0*(1+dt*rate+.5*(dt*rate)**2))
    torch.testing.assert_close(seen[0],rho0) # midpoint construction must not corrupt the full-step start


@pytest.mark.parametrize('mode',['paper','spheral'])
def test_density_reset_stage_modes_do_not_enter_graph_replay(mode):
    from warpSPH.modules.compSPH.stepMode import _blocker
    ctx=SimpleNamespace(schemeConfig=SimpleNamespace(crkEvolution=mode))
    assert 'density reset' in _blocker(ctx,None,None)


def test_paper_reset_refreshes_eos_without_adjusting_thermal_energy(monkeypatch):
    from warpSPH.modules.crk import evolution
    from warpSPH.modules import eos
    one=torch.ones(2,dtype=torch.float64);u=one*3;rho=one*2
    s=SimpleNamespace(densities=rho,crkMassDensities=rho.clone(),crkVolumes=one,masses=one,internalEnergies=u)
    seen=[]
    monkeypatch.setattr(evolution,'buildVerletList',lambda *a,**k:None)
    monkeypatch.setattr(evolution,'neighborSearchOptions',lambda c:{})
    def factors(state,*a,**k):
        assert k['volumes'] is None and k['densityPolicy']=='massReplacement'
        return one,one*4,None
    monkeypatch.setattr(evolution,'crkFactors',factors)
    def refresh(state,c):
        seen.append(state.densities.clone());state.pressures=(c.gamma-1)*state.densities*state.internalEnergies
    monkeypatch.setattr(eos,'refreshEOS',refresh)
    system=SimpleNamespace(state=s,adjacency=None)
    evolution.prepareStep(system,SimpleNamespace(domain=None,kernel=None,verletScale=1.4),SimpleNamespace(crkEvolution='paper',gamma=1.4))
    assert s.crkVolumes is None and s.internalEnergies is u and len(seen)==1
    torch.testing.assert_close(s.densities,4*one)
    torch.testing.assert_close(s.pressures,4.8*one)


def test_evolution_config_roundtrip_and_required_density_policy():
    c = CRKSPHConfig(crkEvolution='spheral', crkDensityPolicy='sameMaterial', materials=[MaterialConfig()])
    c.validate()
    assert dictToCRKSPHConfig(crkSPHConfigToDict(c)).crkEvolution == 'spheral'
    with pytest.raises(ValueError, match='midpoint'):
        c.validate(SimpleNamespace(integrationScheme=SimpleNamespace(name='rungeKutta4')))
    with pytest.raises(ValueError, match='B-spline'):
        c.validate(SimpleNamespace(integrationScheme=SimpleNamespace(name='rungeKutta2'), kernel=KernelFunctions.Wendland2))
    c.crkDensityPolicy = 'legacy'
    with pytest.raises(ValueError, match='sameMaterial'):
        c.validate()
    c.crkEvolution = 'invalid'
    with pytest.raises(ValueError, match='crkEvolution'):
        c.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA solver roundtrip')
def test_midpoint_evolution_trajectory_retains_density_and_volume(tmp_path, monkeypatch):
    from dataclasses import replace
    from warpSPH.runner import run
    from warpSPH.cases.compressible.twoGasSod import twoGasSodCase
    from warpSPH.io.importIO import loadTrajectory, loadTrajectoryFrame
    from warpSPH.modules.crk import evolution
    sums = []
    original = evolution.crkFactors
    def record(*args, **kwargs):
        sums.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(evolution, 'crkFactors', record)
    def initial(ctx):
        system = twoGasSodCase.buildSystem(ctx)
        # The runner must also initialize fields for existing/custom samplers.
        system.state.crkMassDensities = system.state.crkVolumes = None
        return system
    case = replace(twoGasSodCase, buildSystem=initial)
    result = run(case, nx=40, nSteps=3, tLimit=.02, execution='eager',
        params={'pressureRight': 1., 'equalMass': False},
        schemeOverrides={'crkDensityPolicy': 'sameMaterial', 'crkEvolution': 'spheral'},
        plot=False, quiet=True, progress=False, store=True, storeMode='trajectory', storeInterval=1,
        exportRoot=str(tmp_path))
    f, meta = loadTrajectory(result.exportPath, 'cuda', extraFields=twoGasSodCase.extraFields)
    try:
        restored, _ = loadTrajectoryFrame(f, meta, -1, schemeConfig=result.ctx.schemeConfig)
        for name in ('crkMassDensities', 'crkVolumes', 'densities', 'internalEnergies'):
            torch.testing.assert_close(getattr(restored.state, name), getattr(result.state.state, name))
        assert result.state.state.crkVolumes.min() > 0
        assert len(sums) == result.nSteps  # one density sum per step, not per RHS
    finally:
        f.close()
