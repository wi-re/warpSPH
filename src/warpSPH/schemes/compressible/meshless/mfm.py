"""The MFM (meshless finite-mass) step: the conserved face fluxes of Hopkins (2015); the mass of each particle is conserved (dm = 0).

`mfmStep` is `runMeshlessStep` in mode 'MFM'. Its 'hancock' update is the plain Euler update of the conserved vector. ``temporalScheme='auto'``
resolves to 'ssprk' (the D4 measurement): see `temporal.py` for the three schemes and the exact-delta contract.
"""

from .step import runMeshlessStep
from .temporal import conservedHancockUpdate

__all__ = ['mfmStep']


def mfmStep(system, dt, config, schemeConfig, verbose = False):
    return runMeshlessStep(system, dt, config, schemeConfig, mode='MFM', hancockUpdate=conservedHancockUpdate)
