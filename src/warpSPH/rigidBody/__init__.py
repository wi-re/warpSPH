"""Rigid-body kinematics for mDBC obstacles/boundaries: assembling a
`configurations.rigidBody.RigidBody` from the particles tagged as one boundary
material (`build.buildRigidBody`), adding that boundary's mirrored ghost-particle
layer (`ghostParticles.addBoundaryGhostParticles`), advancing its prescribed pose
each step (`integrate.integrateRigidBody`) and re-projecting its particle/ghost
positions and velocities through the resulting transform
(`transformation.getTransformationMatrix`, `update.updateBodyParticlesWCSPH`).
`systems/weaklyCompressible.py` and `systems/incompressible.py` call the
integrate/update pair every step to move rigid obstacles such as
`cases/movingObstacle.py`'s spinning hexagon.
"""

from .build import buildRigidBody, buildAnalyticRigidBody
from .ghostParticles import addBoundaryGhostParticles
from .integrate import integrateRigidBody
from .transformation import getTransformationMatrix
from .update import updateBodyParticlesWCSPH

__all__ = [
    'buildRigidBody',
    'buildAnalyticRigidBody',
    'addBoundaryGhostParticles',
    'integrateRigidBody',
    'getTransformationMatrix',
    'updateBodyParticlesWCSPH'
]