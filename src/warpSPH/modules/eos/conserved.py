"""Ideal-gas primitives from Q=(m,mv,E) and effective volume.

Native forward conversion is fused. Reverse/JVP inputs use Torch arithmetic.
Pressure uses raw specific energy; diagnostic energy retains its mass floor.
This helper does not select material laws or alter meshless EOS support.
"""
import torch
import torch.autograd.forward_ad as fwAD
import warp as wp
from warpSPHCore.autograd.stream import torchWarpStream
from warpSPHCore.type_config import scalar_t, get_torch_precision

wp.set_module_options({'enable_backward': False, 'fuse_fp': False})


@wp.kernel
def _primitives(Q: wp.array2d(dtype=scalar_t), volume: wp.array(dtype=scalar_t), gamma: scalar_t,
                rho: wp.array(dtype=scalar_t), vel: wp.array2d(dtype=scalar_t), pressure: wp.array(dtype=scalar_t),
                energy: wp.array(dtype=scalar_t)):
    i = wp.tid()
    m = Q[i, 0]
    density = m / volume[i]
    speed2 = scalar_t(0.)
    for c in range(vel.shape[1]):
        v = Q[i, c + 1] / m
        vel[i, c] = v
        speed2 += v * v
    kinetic = scalar_t(.5) * speed2
    rho[i] = density
    pressure[i] = wp.max((gamma - scalar_t(1.)) * density * (Q[i, Q.shape[1] - 1] / m - kinetic), scalar_t(1.e-30))
    energy[i] = Q[i, Q.shape[1] - 1] / wp.max(m, scalar_t(1.e-30)) - kinetic


def idealGasPrimitives(Q, volume, gamma, *, returnEnergy=False):
    if (isinstance(gamma, torch.Tensor) or Q.dtype != get_torch_precision() or volume.dtype != Q.dtype
            or (torch.is_grad_enabled() and (Q.requires_grad or volume.requires_grad))
            or fwAD.unpack_dual(Q).tangent is not None or fwAD.unpack_dual(volume).tangent is not None):
        mass = Q[:, 0]
        rho = mass / volume
        vel = Q[:, 1:-1] / mass[:, None]
        kinetic = .5 * (vel * vel).sum(-1)
        rawEnergy = Q[:, -1] / mass - kinetic
        pressure = ((gamma - 1.) * rho * rawEnergy).clamp_min(1.e-30)
        energy = Q[:, -1] / mass.clamp_min(1.e-30) - kinetic
        return (rho, vel, pressure, energy) if returnEnergy else (rho, vel, pressure)
    n, dim = Q.shape[0], Q.shape[1] - 2
    rho, pressure, energy = (torch.empty(n, dtype=Q.dtype, device=Q.device) for _ in range(3))
    vel = torch.empty((n, dim), dtype=Q.dtype, device=Q.device)
    w = lambda t: wp.from_torch(t.detach(), dtype=scalar_t)
    with torchWarpStream(Q.device):
        wp.launch(_primitives, dim=n, device=wp.device_from_torch(Q.device),
            inputs=[w(Q), w(volume), scalar_t(gamma), w(rho), w(vel), w(pressure), w(energy)])
    return (rho, vel, pressure, energy) if returnEnergy else (rho, vel, pressure)

