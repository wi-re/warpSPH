"""Independent pair-flux checks for conductivity strength and the Hopkins F18 option."""
from dataclasses import replace
import math
import pytest
import torch
from test_pesphOperators import setup, adjacency, DOUBLE, TOL
from warpSPHCore import OperationProperties, SupportScheme
from warpSPH.configurations.moduleConfigurations.diffusionParameters import ViscosityTerms
from warpSPH.modules.dissipation.wp_conductivity import computeConductivityWarp


def conduction(state,cfg,params,adj):
    return computeConductivityWarp(state,OperationProperties(kernel=cfg.kernel,
        supportMode=SupportScheme.KernelMeanSymmetric),cfg.domain,params,adjacency=adj)


def radial_gradient(state,cfg):
    delta=state.positions[:,None]-state.positions[None,:]
    L=cfg.domain.max-cfg.domain.min
    delta=torch.where(cfg.domain.periodic,delta-L*torch.round(delta/L),delta)
    r=delta.norm(dim=-1);dim=delta.shape[-1];h=state.supports[:,None];q=r/h
    C={1:8/3,2:80/(7*math.pi),3:16/math.pi}[dim]
    dk=-3*(1-q).clamp_min(0).square()+12*(.5-q).clamp_min(0).square()
    F=C*dk/h.pow(dim+1)
    return delta,r,.5*(F+F.T)


@pytest.mark.parametrize('dim,nx',[(1,20),(2,6),(3,4)])
def test_strength_scales_rates_and_zero_removes_the_source(dim,nx):
    state,cfg,scheme=setup(dim,nx);adj=adjacency(state,cfg);p=scheme.diffusionParams
    p.thermalConductivityTerm=ViscosityTerms.Price2008.value
    state.alphas.fill_(1.)
    state.pressures=torch.linspace(.7,2.1,len(state.masses),device=state.masses.device,dtype=state.masses.dtype)
    p.thermalConductivity=1.;full=conduction(state,cfg,p,adj)
    assert full.abs().max()>1e-4
    for strength in (0.,.25,.5,2.):
        p.thermalConductivity=strength
        torch.testing.assert_close(conduction(state,cfg,p,adj),strength*full,rtol=TOL,atol=TOL)
    assert abs(torch.dot(state.masses,full))<TOL*torch.dot(state.masses,full.abs()).clamp_min(1)
    # A physical diffusion source transports heat from hotter to colder particles.
    assert torch.dot(state.masses,full/state.internalEnergies)>0


@pytest.mark.parametrize('dim,nx',[(1,20),(2,6),(3,4)])
def test_hopkins_conductivity_matches_f18_for_unequal_mass_supports_and_alphas(dim,nx):
    state,cfg,scheme=setup(dim,nx);adj=adjacency(state,cfg);p=scheme.diffusionParams
    state.pressures=torch.linspace(.7,2.1,len(state.masses),device=state.masses.device,dtype=state.masses.dtype)
    state.alphas=torch.linspace(.1,1.3,len(state.masses),device=state.masses.device,dtype=state.masses.dtype)
    p.thermalConductivityTerm=ViscosityTerms.Hopkins2015.value;p.thermalConductivity=.25
    p.Cu_l=1.3
    delta,r,F=radial_gradient(state,cfg)
    u=state.internalEnergies;m=state.masses;rho=state.densities;P=state.pressures;a=state.alphas;c=state.soundspeeds
    mu=((state.velocities[:,None]-state.velocities[None,:])*delta).sum(-1)/r.clamp_min(torch.finfo(r.dtype).eps)
    signal=(c[:,None]+c[None,:]-3*mu).clamp_min(0)
    expected=(.25*m[None,:]*.5*(a[:,None]+a[None,:])*1.3*signal*(u[:,None]-u[None,:])
        *abs(P[:,None]-P[None,:])/(P[:,None]+P[None,:])*F/(.5*(rho[:,None]+rho[None,:]))).sum(1)
    actual=conduction(state,cfg,p,adj)
    torch.testing.assert_close(actual,expected,rtol=4*TOL,atol=4*TOL)
    assert abs(torch.dot(m,actual))<4*TOL*torch.dot(m,actual.abs()).clamp_min(1)
    assert torch.dot(m,actual/u)>0
    # Exact pressure balance switches off this source, despite a temperature jump.
    state.pressures.fill_(1.)
    assert torch.equal(conduction(state,cfg,p,adj),torch.zeros_like(u))


def test_hopkins_signal_cutoff_for_fast_expansion():
    state,cfg,scheme=setup(dim=1,nx=4,periodic=False)
    state.velocities=100*state.positions
    state.pressures=torch.linspace(1.,2.,len(state.masses),device=state.masses.device,dtype=state.masses.dtype)
    p=scheme.diffusionParams;p.thermalConductivityTerm=ViscosityTerms.Hopkins2015.value;p.thermalConductivity=.25
    assert torch.equal(conduction(state,cfg,p,adjacency(state,cfg)),torch.zeros_like(state.masses))


@pytest.mark.parametrize('scheme',['Monaghan','PESPH'])
def test_typed_runner_exposes_the_conductivity_selector(scheme):
    from warpSPH.cases import importAll
    from warpSPH.runner import RunConfiguration
    importAll()
    config=RunConfiguration('hydrostatic',simulation={'device':'cpu'},solver={'scheme':scheme,
        'diffusionParams':{'thermalConductivityTerm':'Hopkins2015','thermalConductivity':.25}})
    ctx=config.prepare()
    assert ctx.schemeConfig.diffusionParams.thermalConductivityTerm==ViscosityTerms.Hopkins2015.value
    assert ctx.schemeConfig.diffusionParams.thermalConductivity==.25


def test_thermal_only_hopkins_is_not_advertised_as_momentum_viscosity():
    from warpSPH.runner.configuration import _validateSolver
    with pytest.raises(ValueError):
        _validateSolver('PESPH',{'diffusionParams':{'viscosityTerm':'Hopkins2015'}})
