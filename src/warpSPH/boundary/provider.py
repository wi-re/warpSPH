"""The boundary provider: a structural protocol and the adapter to the `warpSPHBoundaries` package.

The protocol is what the scheme consumes (it is defined here, in the scheme's own package; the boundary
package depends on warpSPHCore only and never imports warpSPH):

    bodies                                        the rigid bodies (pose, velocity, acceleration), bound to the integrated state
    aggregate(ps, support, kernel, laplacian, ...)  the wall integrals at the particles `ps`: `.out` (lam, G, Cov, cover, lap, tens),
                                                  `.evaluate(...)` for the hydrostatic term A(a1), `.cone_area(axis, halfAngle)`,
                                                  `.dir_extreme(vecs)` ([B, N]: the largest |cos| between a vector and a direction to a wall point in support: Michel's U_char)
    signed_distance(x, body, supportMax, want_body) (d, n, hit[, body index]) with d > 0 in the fluid

`buildBoundaryProvider` makes one from the analytic regions of a scene; `bindBodies` writes the integrated
`RigidBody` state (device tensors, updated in place by `integrateRigidBody`) into the provider's bodies at
the start of every right-hand-side evaluation, so a captured graph reads the live pose.
"""
from typing import Any, Protocol, Sequence

import torch

from ..configurations.region import RegionType

__all__ = ['BoundaryProvider', 'buildBoundaryProvider', 'bindBodies']


class BoundaryProvider(Protocol):
    bodies: Sequence[Any]

    def aggregate(self, ps: Any, support: float, kernel: Any = ..., laplacian: bool = False, fixedAdjacency: bool = True, lean: bool = False) -> Any: ...

    def signed_distance(self, x: torch.Tensor, body: Any = None, supportMax: float = None, want_body: bool = False) -> Any: ...


def buildBoundaryProvider(regions, device):
    """The provider of the analytic boundary regions (None when there are none): one scene holding the
    representations in region order."""
    reps = [r.representation for r in regions if r.type == RegionType.Boundary and getattr(r, 'representation', None) is not None]
    if not reps:
        return None
    from warpSPHBoundaries.scene import AnalyticBoundary, Scene
    for i, b in enumerate(reps):
        b.bodyId = i
    return AnalyticBoundary(Scene(reps, str(device)))


def bindBodies(provider, rigidBodies):
    """The integrated state of the analytic `RigidBody` objects into the provider's bodies (device tensors; the
    pose enters the kernels as (cos, sin) tensors, never as host floats, so this is capturable). The body
    objects of the provider are the `representation` of the RigidBody."""
    F64 = torch.float64
    for rb in rigidBodies:
        b = getattr(rb, 'representation', None)
        if b is None:
            continue
        b.center = rb.centerOfMass.to(F64)
        angle = torch.as_tensor(rb.orientation).to(F64)
        b.angle, b._cs = angle, (torch.cos(angle), torch.sin(angle))
        b.linearVelocity = torch.as_tensor(rb.linearVelocity).to(F64)
        b.angularVelocity = torch.as_tensor(rb.angularVelocity).to(F64)
        b.linearAcceleration = torch.zeros_like(b.linearVelocity)
        b.angularAcceleration = torch.zeros_like(angle)
