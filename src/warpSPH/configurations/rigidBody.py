"""`RigidBody`, the per-body state for the mDBC rigid-body coupling: center of
mass, orientation, velocities, mass/inertia, and boundary/ghost particle arrays,
read and mutated throughout `rigidBody/` (`build.py`, `update.py`, `integrate.py`,
`transformation.py`, `ghostParticles.py`) and stored on
`WeaklyCompressibleSPHConfig.rigidBodies`/`IncompressibleSPHConfig.rigidBodies`.
`toDict`/`fromDict` only round-trip the body-level scalars (COM, orientation,
velocities, mass, inertia, `sdf`, `bodyID`, `kind`) -- the per-particle
boundary/ghost array fields are commented out and so are dropped on
serialize/deserialize; that dead code is left in place here.
"""

__all__ = ['RigidBody']

from .moduleConfigurations.boundaryConditions import BoundaryCondition, BoundaryConditionType, boundaryConditionToDict, dictToBoundaryCondition, BCType
import numpy as np
import torch
from dataclasses import dataclass, field

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




@dataclass(slots = True)
class RigidBody:
    centerOfMass: torch.Tensor
    orientation: torch.Tensor
    angularVelocity: torch.Tensor
    linearVelocity: torch.Tensor
    mass: torch.Tensor
    inertia: torch.Tensor
    
    # All in world coordinates
    particlePositions: torch.Tensor
    particleVelocities: torch.Tensor
    particleMasses: torch.Tensor
    particleUIDs: torch.Tensor
    particleIndices: torch.Tensor
    particleBoundaryDistances: torch.Tensor
    particleBoundaryNormals: torch.Tensor
    
    ghostParticlePositions: torch.Tensor
    ghostParticleIndices: torch.Tensor  
    ghostParticleUIDs: torch.Tensor
    ghostParticleBoundaryDistances: torch.Tensor
    ghostParticleBoundaryNormals: torch.Tensor
    
    
    sdf: callable
    bodyID: int = 0  
    kind: BCType = BCType.constant
    # Analytic representation of the body (a `warpSPHBoundaries` Body, or anything with the same
    # `center` / `angle` / `massProperties(rho)` surface): None for a body made of boundary particles (the
    # mDBC path, unchanged); set for a body whose boundary integrals come from a boundary provider
    # (`rigidBody.build.buildAnalyticRigidBody`: the particle / ghost arrays are then empty).
    representation: object = None

    def toDict(self) -> dict:
        # These fields are tensors as constructed, but a caller driving a body
        # by hand assigns plain Python numbers -- `body.angularVelocity = 1.0`
        # is what the moving-obstacle example does -- so serialising has to
        # accept both rather than assuming a tensor. `bodyID` is annotated
        # `int` but is populated from a tensor, so it arrives as a numpy int32,
        # which `json.dump` also refuses.
        def value(field):
            if isinstance(field, torch.Tensor):
                return field.detach().cpu().numpy().tolist()
            if isinstance(field, np.generic):
                return field.item()
            return field

        return {
            'centerOfMass': value(self.centerOfMass),
            'orientation': value(self.orientation),
            'angularVelocity': value(self.angularVelocity),
            'linearVelocity': value(self.linearVelocity),
            'mass': value(self.mass),
            'inertia': value(self.inertia),
            # 'particlePositions': self.particlePositions.detach().cpu().numpy().tolist(),
            # 'particleVelocities': self.particleVelocities.detach().cpu().numpy().tolist(),
            # 'particleMasses': self.particleMasses.detach().cpu().numpy().tolist(),
            # 'particleUIDs': self.particleUIDs.detach().cpu().numpy().tolist(),
            # 'particleIndices': self.particleIndices.detach().cpu().numpy().tolist(),
            # 'particleBoundaryDistances': self.particleBoundaryDistances.detach().cpu().numpy().tolist(),
            # 'particleBoundaryNormals': self.particleBoundaryNormals.detach().cpu().numpy().tolist(),
            # 'ghostParticlePositions': self.ghostParticlePositions.detach().cpu().numpy().tolist(),
            # 'ghostParticleIndices': self.ghostParticleIndices.detach().cpu().numpy().tolist(),
            # 'ghostParticleUIDs': self.ghostParticleUIDs.detach().cpu().numpy().tolist(),
            # 'ghostParticleBoundaryDistances': self.ghostParticleBoundaryDistances.detach().cpu().numpy().tolist(),
            # 'ghostParticleBoundaryNormals': self.ghostParticleBoundaryNormals.detach().cpu().numpy().tolist(),
            'sdf': _encode_callable(self.sdf),
            'bodyID': value(self.bodyID),
            'kind': self.kind.name
        }

    @staticmethod
    def fromDict(bodyDict: dict) -> 'RigidBody':
        return RigidBody(
            centerOfMass=torch.tensor(bodyDict['centerOfMass']),
            orientation=torch.tensor(bodyDict['orientation']),
            angularVelocity=torch.tensor(bodyDict['angularVelocity']),
            linearVelocity=torch.tensor(bodyDict['linearVelocity']),
            mass=torch.tensor(bodyDict['mass']),
            inertia=torch.tensor(bodyDict['inertia']),
            # particlePositions=torch.tensor(bodyDict['particlePositions']),
            # particleVelocities=torch.tensor(bodyDict['particleVelocities']),
            # particleMasses=torch.tensor(bodyDict['particleMasses']),
            # particleUIDs=torch.tensor(bodyDict['particleUIDs']),
            # particleIndices=torch.tensor(bodyDict['particleIndices']),
            # particleBoundaryDistances=torch.tensor(bodyDict['particleBoundaryDistances']),
            # particleBoundaryNormals=torch.tensor(bodyDict['particleBoundaryNormals']),
            # ghostParticlePositions=torch.tensor(bodyDict['ghostParticlePositions']),
            # ghostParticleIndices=torch.tensor(bodyDict['ghostParticleIndices']),
            # ghostParticleUIDs=torch.tensor(bodyDict['ghostParticleUIDs']),
            # ghostParticleBoundaryDistances=torch.tensor(bodyDict['ghostParticleBoundaryDistances']),
            ghostParticleBoundaryNormals=torch.tensor(bodyDict['ghostParticleBoundaryNormals']),
            sdf=_decode_callable(bodyDict['sdf']),
            bodyID=bodyDict.get('bodyID', 0),
            kind=BCType[bodyDict.get('kind', BCType.constant.name)] if isinstance(bodyDict.get('kind', BCType.constant.name), str) else bodyDict.get('kind', BCType.constant)
        )