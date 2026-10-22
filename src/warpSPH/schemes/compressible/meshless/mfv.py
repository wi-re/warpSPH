"""The MFV (meshless finite-volume) step: the MFM fluxes plus the GIZMO mass-flux limiter, which redistributes mass across faces (dm != 0).

`mfvStep` is `runMeshlessStep` in mode 'MFV'. ``temporalScheme='auto'`` resolves to 'hancock' (the D4 measurement). The 'hancock' update has a second,
EXPERIMENTAL form selected by ``mfvUpdate='primitive'`` (`gizmoPrimitiveUpdate`).
"""

from ....modules.meshless.updates import gizmoPrimitiveUpdate

from .step import runMeshlessStep
from .temporal import conservedHancockUpdate

__all__ = ['mfvStep', 'gizmoPrimitiveUpdate']




def mfvStep(system, dt, config, schemeConfig, verbose = False, **kwargs):
    update = gizmoPrimitiveUpdate if schemeConfig.mfvUpdate == 'primitive' else conservedHancockUpdate
    return runMeshlessStep(system, dt, config, schemeConfig, mode='MFV', hancockUpdate=update, **kwargs)
