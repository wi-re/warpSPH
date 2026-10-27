"""The MFV (meshless finite-volume) step: the MFM fluxes plus the GIZMO mass-flux limiter, which redistributes mass across faces (dm != 0).

`mfvStep` is a spatial flux RHS in mode 'MFV'. The integrators package owns
Hancock staging; the historical experimental primitive kick is excluded.
"""

from .step import runMeshlessStep

__all__ = ['mfvStep']




def mfvStep(system, dt, config, schemeConfig, verbose = False, **kwargs):
    return runMeshlessStep(system, dt, config, schemeConfig, mode='MFV', **kwargs)
