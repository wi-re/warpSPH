"""Explicit-Euler pose integrator for a `RigidBody`: advances orientation and
center of mass by its current angular/linear velocity, and would advance those
velocities by `dudt`/`dwdt` if a caller ever passed a nonzero value. Both call
sites (`systems/weaklyCompressible.py`, `systems/incompressible.py`) pass
`dudt=dwdt=0` every step, so in practice a rigid body's velocity is whatever a
case set once (e.g. `cases/wallBounded/movingObstacle.py`'s `angularVelocity`), not
something the fluid drives back.
"""

__all__ = ['integrateRigidBody']


def integrateRigidBody(rigidBody, dudt, dwdt, dt):
    rigidBody.angularVelocity += dwdt * dt
    rigidBody.linearVelocity += dudt * dt
    
    rigidBody.orientation += rigidBody.angularVelocity * dt
    rigidBody.centerOfMass += rigidBody.linearVelocity * dt

    return rigidBody
