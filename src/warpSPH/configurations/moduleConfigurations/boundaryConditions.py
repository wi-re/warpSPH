"""`BoundaryCondition` and its supporting enums (`BCType`, `BoundaryConditionType`,
`VectorProjectionType`): a declarative per-boundary spec of an SDF plus
per-quantity Dirichlet/update/forcing callables, applied by
`modules/boundaryConditions/bcs.py` and populated by case builders (e.g.
`cases/wallBounded/lidDrivenCavity.py`, `cases/periodic/kolmogorov.py`, `caseUtils/weaklyCompressible.py`).
Callables are pickled via `dill` (falling back to stdlib `pickle` on decode for
configs written before the `dill` switch) so a boundary condition survives a
TOML/HDF5 round-trip through `boundaryConditionToDict`/`dictToBoundaryCondition`.
"""

__all__ = [
    'BCType', 'BoundaryConditionType', 'VectorProjectionType', 'BoundaryCondition',
    'boundaryConditionToDict', 'dictToBoundaryCondition',
]

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Tuple, Dict, Any
from warpSPH.geometry.sdf import *
import torch


class BCType(Enum):
    constant = 1
    noSlip = 2
    freeSlip = 3
    extended = 4
    zeros = 5


class BoundaryConditionType(Enum):
    constant = "constant"
    dynamic = "dynamic"
    reflective = "reflective"
    extending = "extending"
    forcing = "forcing"
    dynamic_forcing = "dynamic_forcing"

class VectorProjectionType(Enum):
    none = "none"
    normal = "normal"
    tangential = "tangential"

@dataclass
class BoundaryCondition:
    type: BoundaryConditionType
    sdf: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] # SDF function defining the boundary returns distance and normal

    # The functions take as input:
    # - The current particle state (as an Any type, which can be a custom dataclass or a dictionary containing all relevant particle properties)
    # - The simulation configuration (as an Any type, which can be a custom dataclass or a dictionary containing all relevant simulation parameters)
    # - The scheme configuration (as an Any type, which can be a custom dataclass or a dictionary containing all relevant scheme parameters)
    # - The particle positions (torch.Tensor of shape [N, dim])
    # - The distance to the boundary (torch.Tensor of shape [N])
    # - The normal to the boundary (torch.Tensor of shape [N, dim])
    # - The current time (torch.Tensor scalar)
    # - The current time step (torch.Tensor scalar)
    # The function returns
    # - The new value of the quantity being updated
    # This function needs to be defined for each quantity that is updated by the boundary condition (e.g. density, velocity, energy, etc.)
    dirichletFunctions: Dict[str, Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None
    # Update functions work the same as dirichlet functions but instead target the update of the quantity instead of its value. This is useful for dynamic boundary conditions where the update of the quantity depends on its current value and the state of the system.
    updateFunctions: Dict[str, Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None
    # Forcing functions work the same but return the forcing term. This gives a more convenient wrapper for some types of boundary conditions (e.g. dynamic_forcing) where the boundary condition is more naturally expressed as a forcing term rather than a direct update to the quantity.
    forcingFunctions: List[Callable[[Any, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None

    # Vector projection type for handling vector quantities at the boundary
    vectorProjectionType: VectorProjectionType = VectorProjectionType.none


    
import os, pickle
import dill
import codecs


def _encode_callable(fn: Callable) -> str:
    # dill can serialize local lambdas/closures used in case builders.
    return codecs.encode(dill.dumps(fn), 'base64').decode()


def _decode_callable(encoded_fn: str) -> Callable:
    raw = codecs.decode(encoded_fn.encode(), 'base64')
    try:
        return dill.loads(raw)
    except Exception:
        # Backward compatibility for configs written with pickle.
        return pickle.loads(raw)

def boundaryConditionToDict(bc: BoundaryCondition) -> Dict[str, Any]:
    return {
        'type': bc.type.name if isinstance(bc.type, BoundaryConditionType) else bc.type,
        'sdf': _encode_callable(bc.sdf),
        'dirichletFunctions': {varName: _encode_callable(fn) for varName, fn in bc.dirichletFunctions.items()} if bc.dirichletFunctions is not None else None,
        'updateFunctions': {varName: _encode_callable(fn) for varName, fn in bc.updateFunctions.items()} if bc.updateFunctions is not None else None,
        'forcingFunctions': [_encode_callable(fn) for fn in bc.forcingFunctions] if bc.forcingFunctions is not None else None,
        'vectorProjectionType': bc.vectorProjectionType.name if isinstance(bc.vectorProjectionType, VectorProjectionType) else bc.vectorProjectionType
    }

def dictToBoundaryCondition(bcDict: Dict[str, Any]) -> BoundaryCondition:
    return BoundaryCondition(
        type=BoundaryConditionType[bcDict['type']] if 'type' in bcDict else None,
        sdf=_decode_callable(bcDict['sdf']) if 'sdf' in bcDict else None,
        dirichletFunctions={varName: _decode_callable(fn) for varName, fn in bcDict['dirichletFunctions'].items()} if 'dirichletFunctions' in bcDict and bcDict['dirichletFunctions'] is not None else None,
        updateFunctions={varName: _decode_callable(fn) for varName, fn in bcDict['updateFunctions'].items()} if 'updateFunctions' in bcDict and bcDict['updateFunctions'] is not None else None,
        forcingFunctions=[_decode_callable(fn) for fn in bcDict['forcingFunctions']] if 'forcingFunctions' in bcDict and bcDict['forcingFunctions'] is not None else None,
        vectorProjectionType=VectorProjectionType[bcDict['vectorProjectionType']] if 'vectorProjectionType' in bcDict else VectorProjectionType.none
    )