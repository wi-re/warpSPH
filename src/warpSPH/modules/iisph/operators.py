"""Stateless IISPH geometry, continuity and pressure-force reductions."""

from typing import NamedTuple
import torch
import warp as wp
from warpSPHCore.type_config import scalar_t
from .wp_operators import kernels


class IISPHKernelData(NamedTuple):
    indices: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor
    gradient: torch.Tensor
    weights: torch.Tensor


def launchIISPHKernel(name, state, arguments, count=None):
    """Launch a dimension-specialized kernel with explicit tensor arguments."""
    compiled = kernels(state.positions.shape[1])
    values = [wp.from_torch(v.contiguous(), dtype=compiled['V'] if v.ndim == 2 else None)
              if isinstance(v, torch.Tensor) else v for v in arguments]
    wp.launch(compiled[name], dim=len(state.positions) if count is None else count,
              inputs=values, device=str(state.positions.device))


def pairArguments(pairs):
    return [pairs.indices, pairs.offsets, pairs.counts, pairs.gradient]


def geometryParameters(state, config):
    compiled = kernels(state.positions.shape[1])
    norm = 1.0
    if config.calibrateNormalization:
        from warpSPHCore.util.latticeDensity import latticeDensityFactor
        norm = latticeDensityFactor(config.kernel, float(config.n_h), state.positions.shape[1])
    return [compiled['V'](*(config.domain.max-config.domain.min).tolist()),
            compiled['I'](*[int(v) for v in config.domain.periodic.tolist()]),
            config.kernel.value, scalar_t(norm)]


def computeIISPHKernelGeometry(state, config, adjacency, reference=None):
    """Evaluate kernels on Core's live pairs; retain only tensor data."""
    reference = state if reference is None else reference
    i, j = (v.to(torch.int64).contiguous() for v in adjacency.livePairs())
    gradient = torch.empty((len(j), state.positions.shape[1]),
                           dtype=state.positions.dtype, device=state.positions.device)
    weights = torch.empty_like(gradient[:, 0]).contiguous()
    launchIISPHKernel('geometry', state, [state.positions, reference.positions, state.supports,
        i, j, *geometryParameters(state, config), gradient, weights], len(j))
    return IISPHKernelData(j, adjacency.edgeOffsets.to(torch.int64).contiguous(),
                          adjacency.numNeighbors.to(torch.int64).contiguous(), gradient, weights)


def computeIISPHDensityRate(state, field, coefficients):
    """C(field), using rest volumes; positive means compression."""
    out = torch.empty_like(coefficients.restVolumes)
    launchIISPHKernel('rate', state, pairArguments(coefficients.pairs) +
        [coefficients.restVolumes, state.kinds.to(torch.int32), coefficients.wallGradient, field, out])
    return out


def computeIISPHContactPressure(state, normalizedPressure, restVolumes, contactWeights):
    """Interpolate q=p/rho0 at the wall contact points."""
    if contactWeights is None:
        return torch.zeros_like(normalizedPressure)
    pairs, denominator = contactWeights
    out = torch.empty_like(normalizedPressure)
    launchIISPHKernel('contact_value', state, [pairs.indices, pairs.offsets, pairs.counts,
        pairs.weights, restVolumes, state.kinds.to(torch.int32), denominator, normalizedPressure, out])
    return out


def computeIISPHPressureAcceleration(state, normalizedPressure, coefficients):
    """Linear F(q), excluding the prescribed gravity extension."""
    contact = computeIISPHContactPressure(state, normalizedPressure,
        coefficients.restVolumes, coefficients.contactWeights)
    out = torch.empty_like(state.positions)
    launchIISPHKernel('force', state, pairArguments(coefficients.pairs) +
        [coefficients.restVolumes, state.kinds.to(torch.int32), coefficients.densityRatio,
         coefficients.gradientSum, coefficients.wallGradient, contact, normalizedPressure, out])
    return out


def computeIISPHPressureOperator(state, normalizedPressure, coefficients):
    """Apply C F without assembling a pressure matrix."""
    return computeIISPHDensityRate(state,
        computeIISPHPressureAcceleration(state, normalizedPressure, coefficients), coefficients)
