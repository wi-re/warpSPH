"""Assembles a `RigidBody` snapshot from a particle state's fluid/ghost
particles for one boundary material ID: center of mass, moment of inertia
about it, and per-particle positions/boundary distances/normals stored
relative to that center of mass. Called once per rigid body during
initialization (`initializers/weaklyCompressible.py`) to seed each body before
its pose is integrated; `angularVelocity`/`linearVelocity` always start at
zero here regardless of any velocity the particles were sampled with -- a case
sets the prescribed rate afterward (e.g. `cases/movingObstacle.py`).
"""

import torch
from ..configurations.weaklyCompressible import RegionType, ParticleRegion, RigidBody

__all__ = ['buildRigidBody', 'buildAnalyticRigidBody']


def buildRigidBody(particleState, regions, bodyId):
    particleIndices = torch.logical_and(particleState.kinds == 1, particleState.materials == bodyId)
    ghostIndices = torch.logical_and(particleState.kinds == 2, particleState.materials == bodyId)

    boundaryRegions = [region for region in regions if region.type == RegionType.Boundary]
    currentRegion = boundaryRegions[bodyId] if len(boundaryRegions) > bodyId else None

    # print(torch.sum(particleIndices), torch.sum(ghostIndices))
    if(torch.sum(particleIndices) == 0):
        print('No particles in body', bodyId)
        return None

    masses = particleState.masses[particleIndices]
    positions = particleState.positions[particleIndices]
    device = particleState.positions.device
    dtype = particleState.positions.dtype

    ghostPositions = particleState.positions[ghostIndices]

    mass = torch.sum(masses)
    centerOfMass = torch.sum(masses.view(-1,1) * positions, dim = 0) / mass
    angularVelocity = torch.tensor(0.0, device = device, dtype = dtype)
    linearVelocity = torch.tensor([0.0, 0.0], device = device, dtype = dtype)
    inertia = torch.sum(masses * torch.linalg.norm(positions - centerOfMass, dim = 1)**2)
    orientation = torch.tensor(0.0, device = device, dtype = dtype)

    return RigidBody(
        centerOfMass=centerOfMass,
        orientation=orientation,
        angularVelocity=angularVelocity,
        linearVelocity=linearVelocity,
        mass=mass,
        inertia=inertia,

        particlePositions=positions - centerOfMass,
        ghostParticlePositions=ghostPositions - centerOfMass,
        particleVelocities=particleState.velocities[particleIndices],


        particleMasses = masses,
        particleUIDs = particleState.UIDs[particleIndices],
        ghostParticleUIDs = particleState.UIDs[ghostIndices],
        particleIndices = particleIndices,
        ghostParticleIndices=ghostIndices,

        particleBoundaryDistances=torch.linalg.norm(particleState.ghostOffsets[particleIndices], dim = -1),
        ghostParticleBoundaryDistances=torch.linalg.norm(particleState.ghostOffsets[ghostIndices], dim = -1),
        particleBoundaryNormals=particleState.ghostOffsets[particleIndices],
        ghostParticleBoundaryNormals=particleState.ghostOffsets[ghostIndices],

        bodyID=bodyId,
        sdf = currentRegion.sdf,
        kind= currentRegion.kind
    )


def buildAnalyticRigidBody(region, bodyId, particleState, restDensity = 1.0, bodyDensity = None):
    """The `RigidBody` of a boundary region whose boundary is analytic (`region.representation`, a
    `warpSPHBoundaries` Body): no boundary particles, so every per-particle array is empty and the
    index masks select nothing -- `integrateRigidBody` and `updateBodyParticlesWCSPH` run on it unchanged
    (an empty gather / scatter), and `modules/mdbc/velocity.py`'s `_ghostBodyVelocity` finds no owned rows.

    Mass, centre of mass and inertia come from the representation (`massProperties`, uniform density
    `bodyDensity`, default the fluid's rest density) where the solid is bounded; an unbounded solid (a
    tank wall, a hole in an infinite solid) has infinite mass and inertia. The pose is the body's own
    (`center`, `angle`, rates). The body-frame origin must be the centre of mass for a finite-mass body
    (`RigidBody.centerOfMass` is the pose reference): `massProperties` reports the offset, and a body
    whose origin is elsewhere is refused rather than silently rotated about the wrong point.
    """
    rep = region.representation
    device = particleState.positions.device
    dtype = particleState.positions.dtype
    n = particleState.positions.shape[0]
    t = lambda v: torch.as_tensor(v, device = device, dtype = dtype)
    empty = torch.zeros((0, 2), device = device, dtype = dtype)
    noRows = torch.zeros(n, dtype = torch.bool, device = device)

    center = t(rep.center).clone()
    try:
        mp = rep.massProperties(restDensity if bodyDensity is None else bodyDensity)
    except ValueError:
        mass, inertia = float('inf'), float('inf')                     # an unbounded solid: a wall that does not move under load
    else:
        offset = (mp['com'][0] ** 2 + mp['com'][1] ** 2) ** 0.5
        if offset > 1e-9 * max(1.0, float(mp['area']) ** 0.5):
            raise ValueError(f"analytic body {bodyId}: its origin is {offset:.3g} away from its centre of mass; "
                             "place the body-frame origin at the centre of mass (massProperties(rho)['com'])")
        mass, inertia = mp['mass'], mp['inertia']

    return RigidBody(
        centerOfMass = center,
        orientation = t(float(rep.angle)),
        angularVelocity = t(float(rep.angularVelocity)),
        linearVelocity = t(rep.linearVelocity).clone(),
        mass = t(mass),
        inertia = t(inertia),

        particlePositions = empty.clone(),
        ghostParticlePositions = empty.clone(),
        particleVelocities = empty.clone(),

        particleMasses = torch.zeros(0, device = device, dtype = dtype),
        particleUIDs = torch.zeros(0, device = device, dtype = torch.int64),
        ghostParticleUIDs = torch.zeros(0, device = device, dtype = torch.int64),
        particleIndices = noRows,
        ghostParticleIndices = noRows.clone(),

        particleBoundaryDistances = torch.zeros(0, device = device, dtype = dtype),
        ghostParticleBoundaryDistances = torch.zeros(0, device = device, dtype = dtype),
        particleBoundaryNormals = empty.clone(),
        ghostParticleBoundaryNormals = empty.clone(),

        bodyID = bodyId,
        sdf = region.sdf,
        kind = region.kind,
        representation = rep,
    )
