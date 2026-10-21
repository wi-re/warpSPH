"""Material-indexed ideal gases: persistent device table and a fused pointwise closure.

Torch reference evaluation preserves reverse/forward AD when inputs carry a
gradient. Material parameters are constants. Negative energy is not hidden by
an absolute value on the explicit-material path: sound speed becomes NaN.
"""
import math
from contextlib import nullcontext
import torch
import warp as wp
from warpSPHCore import scalar_t
from warpSPHCore.type_config import get_torch_precision

from ...configurations.moduleConfigurations.materials import normalizeMaterials
from .idealGas import idealGasEOS

__all__ = ['EOSParameters', 'MaterialTable', 'materialRuntime', 'evaluateEOS',
           'validateMaterialState', 'validateMaterialOptions', 'refreshEOS']


@wp.struct
class EOSParameters:
    eosType: wp.int32
    gamma: scalar_t


@wp.struct
class MaterialTable:
    entries: wp.array(dtype=EOSParameters)
    count: wp.int32


_uniformTables = {}


def uniformMaterialTable(device):
    """Count-zero sentinel with a real array for Warp's struct adjoint walker.

    A null array field is valid for forward-only use, but Warp 1.17's Tape
    visits array fields even when the count-zero branch never reads them.
    """
    key = str(device)
    if key not in _uniformTables:
        table = MaterialTable()
        table.entries = wp.zeros(1, dtype=EOSParameters, device=key)
        table.count = 0
        _uniformTables[key] = table
    return _uniformTables[key]


class _MaterialRuntime:
    def __init__(self, entries, device, dtype):
        self.key = (tuple(entries), torch.device(device), dtype)
        records = []
        for material in entries:
            record = EOSParameters()
            record.eosType, record.gamma = 0, material.eos.gamma
            if not math.isfinite(float(record.gamma)) or float(record.gamma) <= 1:
                raise ValueError('materials.eos.gamma: must remain finite and greater than 1 in the selected precision')
            records.append(record)
        self.table = MaterialTable()
        stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if torch.device(device).type == 'cuda' else None
        with (wp.ScopedStream(stream, sync_enter=False, sync_exit=False) if stream is not None else nullcontext()):
            self.table.entries = wp.array(records, dtype=EOSParameters, device=str(device))
        self.table.count = len(records)
        self.gammas = torch.tensor([m.eos.gamma for m in entries], device=device, dtype=dtype)


def materialRuntime(config, rho):
    """Return the cached owner; configuration/device/precision changes rebuild it.

    Create before graph capture. Mutating the configuration requires recapturing
    the graph, just as changes to any other captured solver parameter do.
    """
    entries = getattr(config, 'materials', None)
    if entries is None:
        return None
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError('materials: expected a nonempty material list or None')
    if not all(hasattr(m, 'eos') for m in entries):
        entries = config.materials = normalizeMaterials(entries)
    key = (tuple(entries), rho.device, rho.dtype)
    runtime = getattr(config, '_materialRuntime', None)
    if runtime is None or runtime.key != key:
        runtime = config._materialRuntime = _MaterialRuntime(entries, rho.device, rho.dtype)
    return runtime


def validateMaterialOptions(config, scheme=None):
    if getattr(config, 'materials', None) is None:
        return
    config.materials = normalizeMaterials(config.materials)
    name = scheme or config.schemeName
    if name not in ('CRKSPH', 'CompSPH', 'Monaghan', 'Compressible SPH', 'Comp SPH'):
        raise ValueError(f'materials: {name} does not support material EOS tables')
    from ...enumTypes import ViscositySwitch
    from ...configurations.moduleConfigurations.diffusionParameters import ViscosityTerms
    if config.viscositySwitchParams.scheme not in (ViscositySwitch.NoneSwitch, ViscositySwitch.CullenDehnen2010):
        raise ValueError('materials: only NoneSwitch and CullenDehnen2010 are currently qualified')
    if config.diffusionParams.viscosityTerm == ViscosityTerms.RiemannDissipation.value:
        raise ValueError('materials: Riemann artificial viscosity requires a two-material Riemann solver')


def validateMaterialState(state, config):
    """Setup/load check, deliberately outside the captured timestep."""
    if getattr(config, 'materials', None) is None:
        return
    validateMaterialOptions(config)
    ids = state.materials
    if ids.shape != state.densities.shape or ids.device != state.densities.device:
        raise ValueError('materials: IDs must match density shape and device')
    if ids.dtype != torch.int32:
        raise ValueError('materials: particle IDs must have dtype int32')
    if bool((state.kinds != 0).any()):
        raise ValueError('materials: solid boundary rows are not supported with an explicit EOS table')
    if ids.numel() and (bool((ids < 0).any()) or bool((ids >= len(config.materials)).any())):
        raise ValueError('materials: particle ID is outside the configured table')
    materialRuntime(config, state.densities)


@wp.kernel
def _evaluate(rho: wp.array(dtype=scalar_t), ids: wp.array(dtype=wp.int32),
              values: wp.array(dtype=scalar_t), table: MaterialTable, source: wp.int32,
              entropy: wp.array(dtype=scalar_t), energy: wp.array(dtype=scalar_t),
              pressure: wp.array(dtype=scalar_t), sound: wp.array(dtype=scalar_t)):
    i = wp.tid()
    m = ids[i]
    a = scalar_t(wp.nan)
    u = scalar_t(wp.nan)
    p = scalar_t(wp.nan)
    c = scalar_t(wp.nan)
    if m >= 0 and m < table.count and rho[i] > scalar_t(0.0):
        g = table.entries[m].gamma
        r = rho[i]
        v = values[i]
        if source == 0:
            u = v
            p = (g - scalar_t(1.0)) * r * u
            a = u * (g - scalar_t(1.0)) * wp.pow(r, scalar_t(1.0) - g)
            c = wp.sqrt(u * g * (g - scalar_t(1.0)))
        elif source == 1:
            p = v
            u = p / r / (g - scalar_t(1.0))
            a = p / wp.pow(r, g)
            c = wp.sqrt(g * p / r)
        elif source == 2:
            a = v
            u = a * wp.pow(r, g - scalar_t(1.0)) / (g - scalar_t(1.0))
            p = a * wp.pow(r, g)
            c = wp.sqrt(g * wp.pow(r, g - scalar_t(1.0)) * a)
        else:
            c = wp.where(v >= scalar_t(0.0), v, scalar_t(wp.nan))
            p = c * c * r / g
            u = p / r / (g - scalar_t(1.0))
            a = p / wp.pow(r, g)
    entropy[i] = a
    energy[i] = u
    pressure[i] = p
    sound[i] = c


def evaluateEOS(rho, materials, config, *, u=None, P=None, A=None, c=None):
    """Return (entropic function, specific energy, pressure, sound speed).

    Exactly one source is required. With no material list the old scalar
    helper is retained verbatim, including its sound-speed convention.
    Invalid device IDs yield NaNs rather than an out-of-bounds table read.
    """
    sources = [u, P, A, c]
    if sum(v is not None for v in sources) != 1:
        raise ValueError('EOS: supply exactly one of u, P, A or c')
    source = next(i for i, v in enumerate(sources) if v is not None)
    value = sources[source]
    if rho.dtype not in (torch.float32, torch.float64):
        raise ValueError('EOS: density must use float32 or float64')
    if rho.ndim != 1 or value.shape != rho.shape or value.device != rho.device or value.dtype != rho.dtype:
        raise ValueError('EOS: source must match the one-dimensional density shape, device and dtype')
    runtime = materialRuntime(config, rho)
    if runtime is None:
        if c is not None:
            P = c.square() * rho / config.gamma
        return idealGasEOS(A=A, u=u, P=P, rho=rho, gamma=config.gamma)
    if materials.shape != rho.shape or materials.dtype != torch.int32 or materials.device != rho.device:
        raise ValueError('materials: expected int32 IDs matching density shape and device')
    # dual tensors (JVP) can have requires_grad=False; inspect their tangent too.
    dual = torch.autograd.forward_ad.unpack_dual(rho).tangent is not None or torch.autograd.forward_ad.unpack_dual(value).tangent is not None
    if rho.requires_grad or value.requires_grad or dual or rho.dtype != get_torch_precision():
        valid = (materials >= 0) & (materials < runtime.table.count) & (rho > 0)
        if source == 3:
            valid = valid & (c >= 0)
        gamma = runtime.gammas[materials.long().clamp(0, runtime.table.count - 1)]
        if source == 3:
            P = c.square() * rho / gamma
            A, u, P, _ = idealGasEOS(A=None, u=None, P=P, rho=rho, gamma=gamma)
        else:
            A, u, P, _ = idealGasEOS(A=A, u=u, P=P, rho=rho, gamma=gamma)
            c = torch.sqrt(gamma * P / rho)
        return tuple(torch.where(valid, x, torch.full_like(x, float('nan'))) for x in (A, u, P, c))
    outputs = tuple(torch.empty_like(rho) for _ in range(4))
    if rho.numel():
        inputs = [
            wp.from_torch(rho.contiguous()), wp.from_torch(materials.contiguous()),
            wp.from_torch(value.contiguous()), runtime.table, source,
            *(wp.from_torch(t) for t in outputs)]
        stream = wp.stream_from_torch(torch.cuda.current_stream(rho.device)) if rho.is_cuda else None
        wp.launch(_evaluate, dim=rho.numel(), device=str(rho.device), inputs=inputs, stream=stream)
    return outputs


def refreshEOS(state, config):
    state.entropies, _, state.pressures, state.soundspeeds = evaluateEOS(
        state.densities, state.materials, config, u=state.internalEnergies)
