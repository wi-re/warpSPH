"""The MFM (meshless finite-mass) step: the conserved face fluxes of Hopkins (2015); the mass of each particle is conserved (dm = 0).

`mfmStep` is `runMeshlessStep` in mode 'MFM'. Its 'hancock' update is the plain Euler update of the conserved vector. ``temporalScheme='auto'``
resolves to the package-owned 'ssprk' driver (the D4 measurement). This function
returns spatial rates only and refuses a driver without its stage lifecycle.
"""

from .step import runMeshlessStep

__all__ = ['mfmStep']


def mfmStep(system, dt, config, schemeConfig, verbose = False, **kwargs):
    return runMeshlessStep(system, dt, config, schemeConfig, mode='MFM', **kwargs)
