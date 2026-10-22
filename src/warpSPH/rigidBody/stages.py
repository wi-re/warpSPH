"""Advance prescribed/free boundary bodies after the fluid step."""
import torch
from .integrate import integrateRigidBody
from .update import updateBodyParticlesWCSPH
from ..modules.gravity import computeGravity


def advanceBodies(self, dt, config, schemeConfig):
    for rigidBody in schemeConfig.rigidBodies:
        if (getattr(schemeConfig, 'analyticSpinAxisymmetric', False) and not getattr(rigidBody, 'dynamic', False)
                and getattr(rigidBody, 'representation', None) is not None):
            continue                                          # a prescribed spin of an axisymmetric analytic body: the pose never matters (the oracle resets `angle = 0` every step), so it is not advanced: every step sees the same state, as a graph replay needs
        dudt, dwdt = 0, 0
        if getattr(rigidBody, 'dynamic', False):
            # a free analytic body: the fluid's load (booked by the scheme's right-hand side) and gravity drive it (acceleration of the centre of mass and of the rotation)
            if rigidBody.load is None:
                raise RuntimeError('a dynamic rigid body needs the loads of the scheme\'s right-hand side (analytic boundaries only)')
            g = computeGravity(self.state, config, schemeConfig, self.adjacency)
            g = (g.reshape(-1, g.shape[-1])[0] if g.dim() > 1 else g).to(rigidBody.centerOfMass.dtype)
            load = rigidBody.load.sum(0).to(rigidBody.centerOfMass.dtype)
            dudt = load[:2] / rigidBody.mass + g
            dwdt = load[2] / rigidBody.inertia
        rigidBody = integrateRigidBody(rigidBody, dudt, dwdt, dt)
        if getattr(rigidBody, 'representation', None) is None:                 # an analytic body has no particles to move (and the update reads host values, which a graph capture forbids)
            self.state = updateBodyParticlesWCSPH(self.state, rigidBody)



def advanceUnforcedBodies(system, dt, schemeConfig):
    """Legacy projection coupling: advance poses with zero external force.

    This deliberately does not consume recorded fluid loads. Replacing it
    with `advanceBodies` would enable a different rigid/fluid coupling model.
    """
    for body in schemeConfig.rigidBodies:
        body = integrateRigidBody(body, 0, 0, dt)
        system.state = updateBodyParticlesWCSPH(system.state, body)
