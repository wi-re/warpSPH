"""Experimental continuous heat allocation through the discrete entropy energy.

For entropy weights w=m*A**(1/gamma), U(A)=sum w*y**(gamma-1)/(gamma-1),
y_i=sum_j W_ij(h_i)*w_j. At fixed positions/number-density supports,
G_j=dU/dA_j includes the transpose gather sum. The NonlocalPower policy
assigns dA_j=m_j*q_j/G_j, matching the prescribed total continuous power.
This is a local experimental source allocation, not SWIFT's local conversion
or the finite-event injection iteration of Borrow et al. (2021), section 4.
It does not prescribe each particle's du/dt: neighbouring energies respond.
"""
import torch
from warpSPHCore import OperationProperties, SupportScheme, WarpOperation, warpOperation
from warpSPHCore.profiling import record_function
from .thermodynamics import requirePositive


def entropyEnergyGradient(state, pressure, config, gamma, adjacency=None):
    """Exact d(sum m*u)/dA, with an O(N) buffer and the core scatter operator."""
    with record_function('warpSPH[PESPH entropy energy derivative]'):
        y=pressure.sums.weightedSum
        coefficient=pressure.weights*y.pow(gamma-2)
        transpose=warpOperation(state,OperationProperties(kernel=config.kernel,
            operation=WarpOperation.Interpolate,supportMode=SupportScheme.Scatter,
            n_h=getattr(config,'n_h',None),
            calibrateNormalization=getattr(config,'calibrateNormalization',False)),
            domain=config.domain,adjacency=adjacency,queryValues=coefficient,
            queryVolumes=torch.ones_like(coefficient))
        derivative=(state.masses/gamma)*state.entropies.pow(1/gamma-1)*(y.pow(gamma-1)/(gamma-1)+transpose)
        requirePositive(derivative,'discrete thermal energy derivative')
        return derivative


def entropyHeatingRate(state, pressure, heating, config, scheme, adjacency=None):
    if scheme.entropyHeating=='NonlocalPower':
        return state.masses*heating/entropyEnergyGradient(state,pressure,config,scheme.gamma,adjacency)
    return (scheme.gamma-1)*pressure.thermodynamicDensity.pow(1-scheme.gamma)*heating
