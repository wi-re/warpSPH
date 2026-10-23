"""Opt-in normal-jump minmod composition: bounded viscosity and conjugate work.

This is an experimental reconstruction policy, separate from native Frontiere
Limited. Check its pair contract independently of simulation pictures.
"""
import torch
import pytest
import warp as wp
from warp.types import vector, matrix

from warpSPHCore import scalar_t, KernelFunctions
from warpSPHCore.util import castTorchToWarpAsBuiltins
from warpSPH.configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy
from warpSPH.enumTypes import PESPHVariant, ViscositySwitch
from warpSPH.modules.reconstruction import reconstructPairVelocity
from warpSPH.modules.dissipation.rates import computeDissipationRates
from warpSPH.modules.pesph import computePressureForce
from warpSPH.modules.pesph.stage import refreshPESPHState
from warpSPH.systems.pesph import PESPHSystem
from test_pesphOperators import setup, TOL

vec2 = vector(length=2, dtype=scalar_t)
mat2 = matrix(shape=(2,2), dtype=scalar_t)


@pytest.mark.parametrize('scheme',['Monaghan','PESPH'])
def test_existing_gradient_correction_option_survives_the_typed_runner(scheme):
    from warpSPH.runner import RunConfiguration
    from warpSPH.cases.compressible.sod import sodCase
    config=RunConfiguration('sod',solver={'scheme':scheme,'diffusionParams':{
        'velocityPairPolicy':'MonotoneLimited','correctReconstructionGradient':True}})
    recovered=RunConfiguration.fromDict(config.toDict()).prepare(sodCase)
    assert recovered.schemeConfig.diffusionParams.correctReconstructionGradient
    assert recovered.schemeConfig.diffusionParams.velocityPairPolicy==VelocityPairPolicy.MonotoneLimited.value


@wp.kernel
def _velocities(x: wp.array(dtype=vec2), vi: wp.array(dtype=vec2), vj: wp.array(dtype=vec2),
                gi: wp.array(dtype=mat2), gj: wp.array(dtype=mat2), policy: wp.int32,
                out: wp.array(dtype=vec2)):
    i = wp.tid()
    out[i] = reconstructPairVelocity(policy, vi[i], vj[i], gi[i], gj[i], x[i],
        scalar_t(1), scalar_t(1), wp.static(KernelFunctions.B7.value), 2,
        scalar_t(.01), scalar_t(.01), scalar_t(0), scalar_t(0), scalar_t(2))


def evaluate(x, vi, vj, gi, gj, policy):
    out = wp.zeros(len(x), dtype=vec2, device=str(x.device))
    wp.launch(_velocities, dim=len(x), device=str(x.device), inputs=[
        *(castTorchToWarpAsBuiltins(v.contiguous()) for v in (x,vi,vj,gi,gj)), policy.value, out])
    return wp.to_torch(out)


def test_admissible_pair_jump_is_symmetric_bounded_and_preserves_smooth_reconstruction():
    # Flat sampled pair with nonzero neighbouring slopes; reconstructed sign
    # reversal; reconstruction of the wrong slope; linear compression; shock.
    dtype = torch.float64 if scalar_t is wp.float64 else torch.float32
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normal = torch.tensor([[1.,0.],[.6,.8]], dtype=dtype, device=dev)
    D = torch.tensor([0.,1.,-1.,-.5,-1.], dtype=dtype, device=dev)
    slope = torch.tensor([4.,4.,-4.,-1.,0.], dtype=dtype, device=dev)
    n = normal[:,None,:].expand(2,5,2).reshape(-1,2)
    x = .5*n
    vi = D.repeat(2)[:,None]*n
    vj = torch.zeros_like(vi)
    gi = slope.repeat(2)[:,None,None]*n[:,:,None]*n[:,None,:]
    # Tangential velocity is irrelevant to the normal bound and must survive.
    vi += torch.tensor([.13,-.21],device=dev,dtype=dtype)
    vj += torch.tensor([.13,-.21],device=dev,dtype=dtype)
    native = evaluate(x,vi,vj,gi,gi,VelocityPairPolicy.Limited)
    bounded = evaluate(x,vi,vj,gi,gi,VelocityPairPolicy.MonotoneLimited)
    raw = ((vi-vj)*n).sum(-1)
    nativeNormal = (native*n).sum(-1)
    actual = (bounded*n).sum(-1)
    expected = torch.maximum(torch.minimum(nativeNormal,raw.clamp_min(0)),raw.clamp_max(0))
    assert nativeNormal[0] < -.1, 'fixture must expose the native flat-pair defect'
    assert torch.allclose(actual,expected,atol=10*TOL,rtol=10*TOL)
    assert actual[0].abs() < 10*TOL
    assert actual[3].abs() < 10*TOL, 'linear compression should still reconstruct away'
    assert torch.allclose(actual[[4,9]],raw[[4,9]],atol=10*TOL,rtol=0), 'shock keeps raw AV'
    reverse = evaluate(-x,vj,vi,gi,gi,VelocityPairPolicy.MonotoneLimited)
    assert torch.allclose(reverse,-bounded,atol=10*TOL,rtol=10*TOL)
    # Projection changes only the normal component.
    tangent = torch.stack([-n[:,1],n[:,0]],dim=-1)
    assert torch.allclose((bounded*tangent).sum(-1),(native*tangent).sum(-1),atol=10*TOL,rtol=10*TOL)


def test_admissible_viscosity_heats_locally_and_keeps_pair_energy_momentum_balance():
    state,cfg,scheme = setup(dim=2,nx=8,variant=PESPHVariant.PressureEnergy)
    scheme.dissipation = True
    scheme.viscositySwitchParams.scheme = ViscositySwitch.NoneSwitch
    scheme.diffusionParams.velocityPairPolicy = VelocityPairPolicy.MonotoneLimited.value
    state.alphas.fill_(1)
    state.alpha0s.fill_(1)
    system = PESPHSystem(state=state,domain=cfg.domain)
    pressure = refreshPESPHState(system,cfg,scheme)
    acc,_ = computePressureForce(state,pressure,cfg,system.adjacency)
    rates = computeDissipationRates(state,1e-4,cfg,scheme,system.adjacency,acc)
    kinetic = (state.masses[:,None]*state.velocities*rates.acceleration).sum()
    heat = (state.masses*rates.viscousHeating).sum()
    assert heat > 1e-6
    assert rates.viscousHeating.min() >= -TOL
    assert abs(kinetic+heat) < TOL*(abs(kinetic)+abs(heat))
    momentum = (state.masses[:,None]*rates.acceleration).sum(0)
    assert momentum.norm() < TOL*(state.masses[:,None]*rates.acceleration).abs().sum()
