"""Closeout invariants for support closure and adaptive dissipative entropy SPH."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from test_pesphOperators import setup, DOUBLE, TOL
from test_pesphHamiltonian import independent_gradient
from warpSPH.enumTypes import PESPHVariant, AdaptiveSupportScheme, ViscositySwitch
from warpSPH.modules.pesph.stage import refreshPESPHState
from warpSPH.modules.pesph.entropyHeating import entropyEnergyGradient
from warpSPH.schemes.compressible.pesph import pesphStep
from warpSPH.systems.pesph import PESPHSystem
from warpSPH.configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy


def adaptive(state,cfg,sc,iterations=64):
    sc=replace(sc,adaptiveSupportScheme=AdaptiveSupportScheme.NumberDensity,
       adaptiveSupportCorrections=True,adaptiveSupportIterations=iterations,
       adaptiveSupportThreshold=1e-12 if DOUBLE else 1e-5)
    return PESPHSystem(state=state,domain=cfg.domain),sc


@pytest.mark.parametrize('calibrated',[False,True])
def test_support_constraint_rejects_an_exhausted_newton_solve(calibrated):
    state,cfg,sc=setup(dim=1,nx=40);cfg.calibrateNormalization=calibrated
    state.supports=state.supports*.63
    system,sc=adaptive(state,cfg,sc,iterations=1)
    # CPU assertions leave the test process usable; invalid CUDA assertions do not.
    for name,value in vars(state).items():
        if isinstance(value,torch.Tensor):setattr(state,name,value.cpu())
    cfg.domain.min=cfg.domain.min.cpu();cfg.domain.max=cfg.domain.max.cpu();cfg.domain.periodic=cfg.domain.periodic.cpu()
    with pytest.raises(RuntimeError,match='support constraint did not converge'):
        refreshPESPHState(system,cfg,sc)


@pytest.mark.parametrize('calibrated',[False,True])
def test_converged_support_guard_accepts_constant_kernel_calibration(calibrated):
    state,cfg,sc=setup(dim=2,nx=8);cfg.calibrateNormalization=calibrated
    system,sc=adaptive(state,cfg,sc)
    p=refreshPESPHState(system,cfg,sc)
    assert (p.pressures>0).all()
    h=state.supports.clone();refreshPESPHState(system,cfg,sc,solveSupport=False)
    assert torch.equal(h,state.supports)


@pytest.mark.parametrize('variant',list(PESPHVariant))
def test_hydrostatic_sampling_preserves_pesph_configured_stage_budget(variant):
    from warpSPH.cases.compressible.hydrostatic import hydrostaticCase
    from warpSPH.runner import RunConfiguration
    ctx=RunConfiguration('hydrostatic',simulation=dict(device='cpu',nx=16),
      solver=dict(scheme='PESPH',variant=variant.name,dissipation=False,adaptiveSupportIterations=37)).prepare(hydrostaticCase)
    system=hydrostaticCase.buildSystem(ctx)
    assert ctx.schemeConfig.adaptiveSupportIterations==37
    refreshPESPHState(system,ctx.config,ctx.schemeConfig)
    assert (system.state.pressures>0).all()


@pytest.mark.skipif(not DOUBLE,reason='independent adaptive total-power oracle needs float64')
@pytest.mark.parametrize('switch',[ViscositySwitch.NoneSwitch,ViscositySwitch.CullenDehnen2010,
    ViscositySwitch.Rosswog2020,ViscositySwitch.Sphenix2022,ViscositySwitch.Wadsley2017])
@pytest.mark.parametrize('policy',[VelocityPairPolicy.Raw,VelocityPairPolicy.MonotoneLimited])
def test_adaptive_nonlocal_viscous_power_and_entropy(switch,policy):
    state,cfg,sc=setup(dim=2,nx=8);system,sc=adaptive(state,cfg,sc)
    sc.dissipation=True;sc.entropyHeating='NonlocalPower';sc.diffusionParams.thermalConductivity=0.
    sc.diffusionParams.velocityPairPolicy=policy.value;sc.viscositySwitchParams.scheme=switch
    p=refreshPESPHState(system,cfg,sc)
    U,GX=independent_gradient(system,cfg,sc)
    GA=entropyEnergyGradient(state,p,cfg,sc.gamma,system.adjacency)
    update,*_=pesphStep(system,1e-4,cfg,sc)
    terms=torch.stack(((GX*update.dxdt).sum(),torch.dot(GA,update.dAdt),
       (state.masses[:,None]*state.velocities*update.dvdt).sum()))
    assert terms.abs().sum()>1e-5,'inactive source is not a dissipative-power test'
    assert abs(terms.sum())<2e-10*terms.abs().sum()
    assert update.dAdt.min()>=-1e-12
    entropyRate=torch.sum(state.masses*update.dAdt/((sc.gamma-1)*state.entropies))
    assert entropyRate>=0
