"""Invert the pressure-entropy closure for prescribed internal energies.

Hopkins (2013), Eqs. 19-22: w=m*A**(1/gamma), y=W*w, and
u=(w/m)*y**(gamma-1)/(gamma-1). Only the constrained A values may change.
This local driven-boundary extension is not a new hydrodynamic force law.
"""
import torch
from .operators import computePressureSums
from .thermodynamics import requirePositive


def matchBoundaryInternalEnergy(state, config, scheme, target, mask, adjacency, *, maxIterations=128):
    if not bool(mask.any()):
        return 0
    requirePositive(target[mask], 'prescribed boundary internal energy')
    gamma = scheme.gamma
    original = state.entropies
    tolerance = max(1e-11, 8*torch.finfo(target.dtype).eps)
    # Log-space damping keeps weights positive. Evaluate the actual stored-A
    # roundtrip and EOS, not a more accurate but different algebraic u formula.
    # Form the relative correction in FP64: subtracting large absolute logs in
    # FP32 can stall before the closure's own representable residual floor.
    damping = 1/gamma
    from .thermodynamics import closePressure
    candidate = original
    for iteration in range(maxIterations):
        weights = state.masses*candidate.pow(1/gamma)
        sums = computePressureSums(state, config, weights, adjacency)
        current = closePressure(state, sums, weights, scheme.variant, gamma, adaptive=False).internalEnergies
        residual = ((current-target).abs()/target)[mask].max()
        if bool(residual <= tolerance):
            state.entropies = torch.where(mask, candidate, original)
            return iteration
        correction = (target.double()/current.double()).pow(damping)
        candidate = ((weights.double()*correction/state.masses.double()).pow(gamma)).to(original.dtype)
        candidate = torch.where(mask, candidate, original)
    raise RuntimeError('PESPH prescribed internal-energy boundary did not converge')
