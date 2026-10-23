"""Experimental energy-preserving entropy initialization for a shared ideal gas.

Invert u_i=A_i**(1/gamma)*y_i(A)**(gamma-1)/(gamma-1) for every prescribed u_i.
A log-space Jacobi map is a contraction with factor gamma-1 for 1<gamma<2:
log w_new=log[m*(gamma-1)*u]-(gamma-1)*log(W*w).
The fixed point preserves the prescribed specific thermal energies without a
global normalization or temperature floor. Smoothed pressure is an outcome,
so this is separate from the historical pressure-volume initialization policy.
"""
import torch
from .operators import computePressureSums
from .thermodynamics import requirePositive


def matchEntropyThermalEnergy(state,config,scheme,target,adjacency):
    gamma=scheme.gamma
    requirePositive(target,'target initial thermal energy')
    tolerance=max(1e-10,32*torch.finfo(target.dtype).eps)
    weights=state.masses*state.entropies.pow(1/gamma)
    logTarget=torch.log(state.masses*(gamma-1)*target)
    for _ in range(128):
        y=computePressureSums(state,config,weights,adjacency).weightedSum
        requirePositive(y,'initial weighted pressure sum')
        current=weights/state.masses*y.pow(gamma-1)/(gamma-1)
        if ((current-target).abs()/target).max()<tolerance:
            state.entropies=(weights/state.masses).pow(gamma)
            return
        weights=torch.exp(logTarget-(gamma-1)*torch.log(y))
    raise RuntimeError('PESPH ThermalEnergy initialization did not converge')
