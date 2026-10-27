"""Pointwise delta RHS assembly and final fluid masking.

Boundary update hooks run between these two operations. Captured forward kernels retain the Torch operation order. Eager execution
and reverse/JVP inputs use Torch, avoiding repeated native view conversion.
"""
import torch
import warp as wp
from warpSPHCore.autograd.stream import torchWarpStream
from warpSPHCore.autograd.wrapper import _isInertTensor
from warpSPHCore.type_config import scalar_t, get_torch_precision

wp.set_module_options({'enable_backward': False, 'fuse_fp': False})


@wp.kernel
def _assemble(velocity: wp.array2d(dtype=scalar_t), pressure: wp.array2d(dtype=scalar_t),
              forcing: wp.array2d(dtype=scalar_t), gravity: wp.array2d(dtype=scalar_t),
              dissipation: wp.array2d(dtype=scalar_t), contact: wp.array2d(dtype=scalar_t),
              continuity: wp.array(dtype=scalar_t), diffusion: wp.array(dtype=scalar_t),
              dx: wp.array2d(dtype=scalar_t), dv: wp.array2d(dtype=scalar_t),
              drho: wp.array(dtype=scalar_t), passive: wp.array(dtype=wp.bool)):
    i = wp.tid()
    for c in range(velocity.shape[1]):
        a = pressure[i, c] + forcing[i, c]
        a = a + gravity[i, c]
        a = a + dissipation[i, c]
        a = a + contact[i, c]
        dv[i, c] = a
        dx[i, c] = velocity[i, c]
    drho[i] = continuity[i] + diffusion[i]
    passive[i] = False


@wp.kernel
def _fluidMask(kinds: wp.array(dtype=wp.int32), keep: wp.array2d(dtype=scalar_t), useKeep: wp.bool,
               dx: wp.array2d(dtype=scalar_t), dv: wp.array2d(dtype=scalar_t),
               rho: wp.array(dtype=scalar_t), kin: wp.array(dtype=scalar_t),
               outX: wp.array2d(dtype=scalar_t), outV: wp.array2d(dtype=scalar_t),
               outRho: wp.array(dtype=scalar_t), outKin: wp.array(dtype=scalar_t)):
    i = wp.tid()
    for c in range(dx.shape[1]):
        x = scalar_t(0.)
        v = scalar_t(0.)
        if kinds[i] == 0:
            x = dx[i, c]
            v = dv[i, c]
            if useKeep:
                v = v * keep[i, 0]
        outX[i, c] = x
        outV[i, c] = v
    r = scalar_t(0.)
    k = scalar_t(0.)
    if kinds[i] == 0:
        r = rho[i]
        k = kin[i]
    outRho[i] = r
    outKin[i] = k


def _native(*tensors):
    # Capture pays marshalling once; eager pointwise Torch is cheaper on the host.
    if not tensors or not tensors[0].is_cuda or not torch.cuda.is_current_stream_capturing():
        return False
    return all(t is None or (t.is_cuda and t.dtype == get_torch_precision() and _isInertTensor(t)) for t in tensors)


def _view(t):
    return wp.from_torch(t.detach()) if t is not None else None


def assembleUpdate(state, pressure, forcing, gravity, dissipation, contact, continuity, diffusion):
    from ...systems.weaklyCompressible import WeaklyCompressibleSystemUpdate
    velocity = state.velocities
    if not _native(velocity, pressure, forcing, gravity, dissipation, contact, continuity, diffusion):
        return WeaklyCompressibleSystemUpdate(dxdt=velocity.clone(),
            dvdt=pressure + forcing + gravity + dissipation + contact,
            drhodt=continuity + diffusion, passive=torch.zeros_like(continuity, dtype=torch.bool), drhodt_kin=continuity)
    dx, dv = torch.empty_like(velocity), torch.empty_like(velocity)
    drho = torch.empty_like(continuity)
    passive = torch.empty_like(continuity, dtype=torch.bool)
    with torchWarpStream(velocity.device):
        wp.launch(_assemble, dim=len(velocity), device=wp.device_from_torch(velocity.device),
            inputs=[_view(t) for t in (velocity, pressure, forcing, gravity, dissipation, contact,
                continuity, diffusion, dx, dv, drho, passive)])
    return WeaklyCompressibleSystemUpdate(dxdt=dx, dvdt=dv, drhodt=drho, passive=passive, drhodt_kin=continuity)


def maskFluidUpdate(update, kinds, keep=None):
    tensors = (update.dxdt, update.dvdt, update.drhodt, update.drhodt_kin)
    if kinds.dtype != torch.int32 or not _native(*tensors, keep):
        if keep is not None:
            update.dvdt = update.dvdt * keep
        nonFluid = (kinds != 0).unsqueeze(-1)
        update.dxdt = torch.where(nonFluid, torch.zeros_like(update.dxdt), update.dxdt)
        update.dvdt = torch.where(nonFluid, torch.zeros_like(update.dvdt), update.dvdt)
        update.drhodt = torch.where(nonFluid.squeeze(-1), torch.zeros_like(update.drhodt), update.drhodt)
        update.drhodt_kin = torch.where(nonFluid.squeeze(-1), torch.zeros_like(update.drhodt_kin), update.drhodt_kin)
        return update
    outputs = tuple(torch.empty_like(t) for t in tensors)
    with torchWarpStream(update.dxdt.device):
        wp.launch(_fluidMask, dim=len(kinds), device=wp.device_from_torch(kinds.device),
            inputs=[_view(kinds), _view(keep), keep is not None, *map(_view, tensors), *map(_view, outputs)])
    update.dxdt, update.dvdt, update.drhodt, update.drhodt_kin = outputs
    return update
