"""Geometry-only analytic wall contribution to velocity diffusion."""

import torch
from warpSPHCore import KernelFunctions
from ...boundary.provider import bindBodies
from ...configurations.region import BCType
from .wallTerms import viscousPrefactor


def mirrorViscousAcceleration(velocity, wallVelocity, normal, laplacian, *,
                             viscosity, wallVolume, density, freeSlip, approachOnly=False):
    """The shared local mirror closure for a solid Laplacian integral."""
    relative = velocity - wallVelocity
    if freeSlip:
        speed = (relative * normal).sum(-1, keepdim=True)
        if approachOnly:
            speed = speed.clamp_max(0)
        relative = speed * normal
    return (-2 * viscosity * wallVolume * laplacian / density)[:, None] * relative


def computeWallViscosity(state, config, schemeConfig, wall=None, *, perBody=False, approachOnly=None, support=None):
    """Integrate the mirrored wall velocity without evaluating wall pressure.

    Free slip preserves tangential relative velocity and reverses its normal
    component. No slip reverses both. The existing analytic wall Laplacian
    integrates this local mirror closure, with the shared pair coefficient.
    """
    from warpSPHBoundaries.scene.viscosity import lap_factor
    provider = schemeConfig.boundaryProvider
    bodies = getattr(provider, 'rigidBodies', schemeConfig.rigidBodies)
    velocity = state.velocities.double()
    if wall is None:
        bindBodies(provider, bodies)
        if support is None:
            support = float(state.supports.max())
        integrals = provider.aggregate(state, support, config.kernel, laplacian=True)
        kinematics = provider.scene.kinematics(state.positions.double())
        packing = float(getattr(schemeConfig, 'analyticWallMass', 1.0))
    else:
        support, integrals, kinematics, packing = wall.support, wall.agg, wall.kin, wall.wm
    if approachOnly is None:
        approachOnly = schemeConfig.diffusionParams.approachOnly
    factor = viscousPrefactor(schemeConfig, config, support)
    family = {KernelFunctions.Wendland2: 'w2', KernelFunctions.Wendland4: 'w4'}[config.kernel]
    laplacian = lap_factor(support, family) * integrals.out['lap']
    acceleration = torch.zeros_like(velocity)
    parts = []
    for bi, body in enumerate(bodies):
        if body.kind == BCType.extended:
            raise NotImplementedError('Extended boundary velocity has no analytic viscosity closure')
        normal = torch.zeros_like(velocity)
        if body.kind == BCType.freeSlip:
            _, normal, covered = provider.signed_distance(state.positions, body=bi)
            if bool((~covered & (laplacian[bi] != 0)).any()):
                raise ValueError('Free-slip viscosity requires a covered analytic surface normal')
        wallVelocity = 0 if body.kind == BCType.zeros else kinematics.velocity[bi]
        part = mirrorViscousAcceleration(velocity, wallVelocity, normal, laplacian[bi],
            viscosity=factor / 8, wallVolume=packing, density=state.densities.double(),
            freeSlip=body.kind == BCType.freeSlip, approachOnly=approachOnly)
        acceleration += part
        parts.append(part)
    result = torch.stack(parts) if perBody else acceleration
    return torch.where((state.kinds == 0)[:, None], result.to(state.velocities), 0)
