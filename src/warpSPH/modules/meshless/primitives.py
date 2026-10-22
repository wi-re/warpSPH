"""Conserved-to-primitive conversion and diagnostic refresh without Torch op chains.

This is forward-only, like MeshlessWarp. The separate pressure and diagnostic
mass floors retain the reference's behavior even below the mass floor.
"""
import math
import torch
import warp as wp
from warpSPHCore.autograd.stream import torchWarpStream
from warpSPHCore import scalar_t
from warpSPHCore.mfm.warpBackend import _forwardOnly
from ..eos.conserved import idealGasPrimitives

wp.set_module_options({'enable_backward': False, 'fuse_fp': False})
BALL = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}


@wp.kernel
def _diagnostics(volume: wp.array(dtype=scalar_t), rho: wp.array(dtype=scalar_t), pressure: wp.array(dtype=scalar_t),
                 gamma: scalar_t, nngb: scalar_t, ball: scalar_t, exponent: scalar_t,
                 support: wp.array(dtype=scalar_t), sound: wp.array(dtype=scalar_t), entropy: wp.array(dtype=scalar_t)):
    i = wp.tid()
    v = nngb * volume[i] / ball
    h = wp.pow(v, exponent)
    if exponent == scalar_t(.5):
        h = wp.sqrt(v)
    if exponent == scalar_t(1.):
        h = v
    support[i] = h
    density = wp.max(rho[i], scalar_t(1.e-30))
    sound[i] = wp.sqrt(gamma * pressure[i] / density)
    entropy[i] = pressure[i] / wp.pow(density, gamma)


def primitives(Q, volume, gamma, *, returnEnergy=False):
    _forwardOnly(Q, volume)
    return idealGasPrimitives(Q, volume, gamma, returnEnergy=returnEnergy)


def refreshPrimitives(state, geom, gamma, nngb, dim):
    rho, vel, pressure, energy = primitives(state.Q, geom.volume, gamma, returnEnergy=True)
    support, sound, entropy = (torch.empty_like(rho) for _ in range(3))
    w = lambda t: wp.from_torch(t.detach(), dtype=scalar_t)
    with torchWarpStream(rho.device):
        wp.launch(_diagnostics, dim=len(rho), device=wp.device_from_torch(rho.device),
            inputs=[w(geom.volume), w(rho), w(pressure), scalar_t(gamma), scalar_t(nngb), scalar_t(BALL[dim]),
                    scalar_t(1. / dim), w(support), w(sound), w(entropy)])
    state.supports, state.soundspeeds, state.entropies = support, sound, entropy
    state.densities, state.velocities, state.pressures = rho, vel, pressure
    state.masses, state.totalEnergies = state.Q[:, 0], state.Q[:, -1]
    state.internalEnergies = energy
    return rho, vel, pressure, energy
