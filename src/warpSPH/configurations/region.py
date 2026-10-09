"""`RegionType` and `ParticleRegion`: the declarative fluid/boundary/inlet/outlet
region descriptor built by `regions/region.py`'s `buildRegion` and stored on
`WeaklyCompressibleSPHConfig.regions`/`IncompressibleSPHConfig.regions`; read
throughout `regions/`, `rigidBody/`, `modules/mdbc`, `modules/noise`, and
`initializers/weaklyCompressible.py`. Also re-exports `BCType` (defined in
`moduleConfigurations.boundaryConditions`) since several `cases/*.py` modules
import it from here rather than its defining module. Note: `parseInitialConditions`
(below) builds its output dict but has no `return` statement, so it always
yields `None`; its sibling helpers `parseParticleSet`/`unparseParticleSet` are
defined but never called anywhere -- `ParticleRegion.toDict`/`fromDict` always
pass `particles=None` with those calls commented out.
"""

__all__ = ['RegionType', 'ParticleRegion', 'BCType']

from dataclasses import dataclass
from enum import Enum
from .moduleConfigurations.boundaryConditions import BoundaryConditionType, BoundaryCondition, BCType
from ..geometry import ParticleSet
import torch

from typing import Callable, List, Tuple, Dict, Any
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


class RegionType(Enum):
    Fluid = 1
    Boundary = 2
    Inlet = 4
    Outlet = 5

def parseParticleSet(particleSet: ParticleSet) -> dict:
    return {
        'positions': particleSet.positions.detach().cpu().numpy().tolist(),
        'supports': particleSet.supports.detach().cpu().numpy().tolist(),
        'masses': particleSet.masses.detach().cpu().numpy().tolist(),
        'densities': particleSet.densities.detach().cpu().numpy().tolist()
    }

def unparseParticleSet(particleSetDict: dict) -> ParticleSet:
    return ParticleSet(
        positions=torch.tensor(particleSetDict['positions']),
        supports=torch.tensor(particleSetDict['supports']),
        masses=torch.tensor(particleSetDict['masses']),
        densities=torch.tensor(particleSetDict['densities'])
    )

def parseInitialConditions(initialConditions: dict) -> dict:
    outDict = {}
    for key, value in initialConditions.items():
        if isinstance(value, torch.Tensor):
            outDict[key] = value.detach().cpu().numpy().tolist()
        elif isinstance(value, Callable):
            outDict[key] = _encode_callable(value)
        else:
            outDict[key] = value

def unparseInitialConditions(initialConditionsDict: dict) -> dict:
    outDict = {}
    for key, value in initialConditionsDict.items():
        if isinstance(value, (list, tuple)):
            outDict[key] = torch.tensor(value)
        elif isinstance(value, dict):
            outDict[key] = unparseInitialConditions(value)
        elif isinstance(value, str):
            outDict[key] = _decode_callable(value)
        else:
            outDict[key] = value
    return outDict

@dataclass
class ParticleRegion:
    sdf: callable
    type: RegionType
    particles: ParticleSet
    contour: list = None

    initialConditions: dict = None
    kind: BCType = BCType.constant
    # Analytic boundary (a `warpSPHBoundaries` Body): the region keeps its `sdf` (so `filterRegion` clips the fluid
    # against it as against any boundary) but is not sampled into boundary particles; `particles` is empty and
    # `rigidBody.buildAnalyticRigidBody` makes its RigidBody. None: a particle boundary (unchanged).
    representation: object = None

    def toDict(self):
        return {
            'sdf': _encode_callable(self.sdf),
            'type': self.type.name,
            'contour': None,
            'initialConditions': parseInitialConditions(self.initialConditions),
            'kind': self.kind.name,
            'particles': None, #parseParticleSet(self.particles)
        }

    @staticmethod
    def fromDict(regionDict: dict) -> 'ParticleRegion':
        return ParticleRegion(
            sdf=_decode_callable(regionDict['sdf']),
            type=RegionType[regionDict['type']] if isinstance(regionDict['type'], str) else regionDict['type'],
            contour=None,
            initialConditions=unparseInitialConditions(regionDict.get('initialConditions', {})),
            kind=BCType[regionDict.get('kind', BCType.constant.name)] if isinstance(regionDict.get('kind', BCType.constant.name), str) else regionDict.get('kind', BCType.constant),
            particles=None, #unparseParticleSet(regionDict['particles'])
        )