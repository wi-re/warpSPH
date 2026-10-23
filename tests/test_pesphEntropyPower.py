"""Independent discrete-energy checks for opt-in pressure-entropy source/IC policies."""
from dataclasses import replace
import pytest
import torch
from test_pesphOperators import setup,cubic,adjacency,DOUBLE,TOL
from warpSPH.configurations.pesph import pesphConfigToDict,dictToPESPHConfig
from warpSPH.enumTypes import PESPHVariant,ViscositySwitch
from warpSPH.configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy
from warpSPH.modules.pesph.operators import computePressureState
from warpSPH.modules.pesph.entropyHeating import entropyEnergyGradient,entropyHeatingRate
from warpSPH.modules.pesph.entropyInitialization import matchEntropyThermalEnergy


@pytest.mark.parametrize('dim,nx',[(1,20),(2,6),(3,4)])
def test_entropy_energy_derivative_matches_independent_dense_autograd(dim,nx):
    state,cfg,scheme=setup(dim,nx);adj=adjacency(state,cfg)
    pressure=computePressureState(state,cfg,scheme,adj)
    A=state.entropies.detach().requires_grad_();g=scheme.gamma
    W,_=cubic(state,cfg);w=state.masses*A.pow(1/g);y=W@w
    U=(w*y.pow(g-1)/(g-1)).sum();expected=torch.autograd.grad(U,A)[0]
    actual=entropyEnergyGradient(state,pressure,cfg,g,adj)
    torch.testing.assert_close(actual,expected,rtol=4*TOL,atol=4*TOL)
    assert (actual>0).all()
    scheme.entropyHeating='NonlocalPower';scheme.diffusionParams.thermalConductivity=0.
    for signed in (False,True):
        q=torch.linspace(.1,1.,len(A),device=A.device,dtype=A.dtype)
        if signed:q=q-q.mean()
        rate=entropyHeatingRate(state,pressure,q,cfg,scheme,adj)
        desired=torch.dot(state.masses,q)
        assert abs(torch.dot(expected,rate)-desired)<8*TOL*desired.abs().clamp_min(1)
        if not signed:assert (rate>=0).all()


def test_single_particle_heating_discriminates_against_local_conversion():
    state,cfg,scheme=setup();adj=adjacency(state,cfg);p=computePressureState(state,cfg,scheme,adj)
    G=entropyEnergyGradient(state,p,cfg,scheme.gamma,adj)
    q=torch.zeros_like(state.masses);q[13]=1
    local=(scheme.gamma-1)*p.thermodynamicDensity.pow(1-scheme.gamma)*q
    desired=state.masses[13]
    assert abs(torch.dot(G,local)/desired-1)>1e-3
    scheme.entropyHeating='NonlocalPower';scheme.diffusionParams.thermalConductivity=0.
    rate=entropyHeatingRate(state,p,q,cfg,scheme,adj)
    torch.testing.assert_close(torch.dot(G,rate),desired,rtol=TOL,atol=TOL)


def test_viscous_rhs_closes_discrete_total_power_without_energy_rescaling():
    from warpSPH.schemes.compressible.pesph import pesphStep
    from warpSPH.systems.pesph import PESPHSystem
    state,cfg,scheme=setup(dim=2,nx=6)
    scheme.dissipation=True;scheme.entropyHeating='NonlocalPower'
    scheme.diffusionParams.thermalConductivity=0.
    scheme.viscositySwitchParams.scheme=ViscositySwitch.NoneSwitch
    scheme.diffusionParams.velocityPairPolicy=VelocityPairPolicy.Raw.value
    A=state.entropies.detach().requires_grad_();x=state.positions.detach().requires_grad_();g=scheme.gamma
    candidate=replace(state,positions=x)
    W,_=cubic(candidate,cfg);w=state.masses*A.pow(1/g);y=W@w
    U=(w*y.pow(g-1)/(g-1)).sum();GX,GA=torch.autograd.grad(U,(x,A))
    update,_,_=pesphStep(PESPHSystem(state=state,domain=cfg.domain),1e-4,cfg,scheme)
    residual=(GX*update.dxdt).sum()+torch.dot(GA,update.dAdt)+(state.masses[:,None]*state.velocities*update.dvdt).sum()
    assert abs(residual)<10*TOL
    assert (update.dAdt>=-TOL).all()


@pytest.mark.parametrize('gamma',[1.4,5/3])
def test_thermal_energy_initialization_preserves_cold_blast_targets_without_global_scaling(gamma):
    state,cfg,scheme=setup(dim=2,nx=10);scheme.gamma=gamma
    target=1e-8+10*torch.exp(-state.positions.square().sum(-1)/.02)
    original_mass=state.masses.clone();adj=adjacency(state,cfg)
    matchEntropyThermalEnergy(state,cfg,scheme,target,adj)
    p=computePressureState(state,cfg,scheme,adj)
    torch.testing.assert_close(p.internalEnergies,target,rtol=5e-10 if DOUBLE else 2e-5,atol=0)
    assert torch.equal(state.masses,original_mass)
    assert (state.entropies>0).all()


def test_explicit_policies_roundtrip_and_keep_the_default_guard():
    from warpSPH.configurations import PESPHConfig
    cfg=PESPHConfig(variant=PESPHVariant.PressureEntropy,entropyHeating='NonlocalPower',entropyInitialization='ThermalEnergy')
    cfg.diffusionParams.thermalConductivity=0.;cfg.validate()
    assert pesphConfigToDict(dictToPESPHConfig(pesphConfigToDict(cfg)))==pesphConfigToDict(cfg)
    cfg.diffusionParams.thermalConductivity=.5
    with pytest.raises(ValueError,match='conductivity'):cfg.validate()
    with pytest.raises(ValueError,match='dissipation'):PESPHConfig(variant=PESPHVariant.PressureEntropy).validate()


def test_typed_runner_resolves_both_explicit_entropy_policies():
    from warpSPH.cases import importAll
    from warpSPH.runner import RunConfiguration
    importAll()
    ctx=RunConfiguration('sod',simulation={'device':'cpu'},solver={'scheme':'PESPH',
        'variant':'PressureEntropy','entropyHeating':'NonlocalPower',
        'entropyInitialization':'ThermalEnergy','diffusionParams':{'thermalConductivity':0.}}).prepare()
    assert ctx.schemeConfig.entropyHeating=='NonlocalPower'
    assert ctx.schemeConfig.entropyInitialization=='ThermalEnergy'


def test_nonlocal_power_checkpoint_restores_active_state_and_source_policy(tmp_path):
    from pathlib import Path
    import copy
    from test_pesphIntegration import runOptions,assertSameState
    from warpSPH.cases.compressible.sod import sodCase
    from warpSPH.runner import run
    from warpSPH.io import importSimulationSystem
    from warpSPH.schemes.compressible.pesph import pesphStep
    options=runOptions(tmp_path,PESPHVariant.PressureEntropy)
    options['schemeOverrides'].update(dissipation=True,entropyHeating='NonlocalPower',
        entropyInitialization='ThermalEnergy',diffusionParams={'thermalConductivity':0.,'velocityPairPolicy':'Raw'})
    result=run(sodCase,**options,store=True,storeMode='states',storeInterval=1,exportInterval=1e-4)
    paths=sorted((Path(result.exportPath)/'trajectory').glob('*.h5'))
    restored,_,_,_=importSimulationSystem(str(paths[-1]),result.ctx.config.device,result.ctx.config.dtype)
    from warpSPH.io.importIO import importConfigs
    loadedConfig,loadedScheme=importConfigs(str(Path(result.exportPath)/'config.json'),dictToPESPHConfig)
    assert loadedScheme.entropyHeating=='NonlocalPower'
    assert loadedScheme.entropyInitialization=='ThermalEnergy'
    assert loadedScheme.diffusionParams.thermalConductivity==0.
    assertSameState(result.state.state,restored.state,exact=True)
    rebuilt=copy.deepcopy(result.state);rebuilt.adjacency=None
    for _ in range(2):
        rebuilt=result.ctx.integrator.function(state=rebuilt,dt=1e-4,f=pesphStep,
            config=result.ctx.config,schemeConfig=result.ctx.schemeConfig).state
        restored=result.ctx.integrator.function(state=restored,dt=1e-4,f=pesphStep,
            config=loadedConfig,schemeConfig=loadedScheme).state
    assertSameState(rebuilt.state,restored.state)
