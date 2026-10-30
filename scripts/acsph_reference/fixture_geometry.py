"""Geometry contracts for the exact-particle Lobovsky comparison fixture."""
import torch
from contextlib import contextmanager


@contextmanager
def volume_ghost_initializer(audit):
    """Fixture-only unit correction; restore shared initializer hooks on exit.

    State masses already contain rho. Use (m/rho)^(1/dim) for both the
    grid-snap node placement and SDF mollifier, without changing physical ICs.
    This diagnostic context is scoped to single-threaded case initialization.
    """
    import warpSPH.initializers.weaklyCompressible as initializer
    import warpSPH.rigidBody.ghostParticles as geometry
    original_add = initializer.addBoundaryGhostParticles
    original_place = geometry._gridSnapGhostOffsets
    original_hybrid = geometry._hybridGhostOffsets

    def add(regions, state):
        mode = geometry._ghostPlacementMode()
        if mode not in ('gridsnap', 'hybrid'):
            raise ValueError('Volume-spacing Lobovsky control requires gridsnap/hybrid ghost placement')
        dim = state.positions.shape[-1]
        volume = state.masses / state.densities
        if not bool(torch.isfinite(volume).all() & (volume > 0).all()):
            raise ValueError('Ghost spacing requires finite positive particle volumes')
        spacing = float(volume.mean().pow(1. / dim))
        audit.update(mode='volume', spacing=spacing,
                     placement=mode,
                     legacy_mass_spacing=float(state.masses.mean().pow(1. / dim)))

        def place(bpos, sdfs, ignored_spacing, support, **kwargs):
            return original_place(bpos, sdfs, spacing, support, **kwargs)

        def hybrid(bpos, sdfs, ignored_spacing, support):
            return original_hybrid(bpos, sdfs, spacing, support)

        geometry._gridSnapGhostOffsets = place
        geometry._hybridGhostOffsets = hybrid
        try:
            return original_add(regions, state)
        finally:
            geometry._gridSnapGhostOffsets = original_place
            geometry._hybridGhostOffsets = original_hybrid

    initializer.addBoundaryGhostParticles = add
    try:
        yield
    finally:
        initializer.addBoundaryGhostParticles = original_add
        geometry._gridSnapGhostOffsets = original_place
        geometry._hybridGhostOffsets = original_hybrid


def lobovsky_tank_sdf(x):
    """Tank SDF, positive on the fluid side, as (N values, N×2 normals).

    ParticleRegion/ghost construction expects this tuple. A bare N-vector is
    silently indexed as sdf(x)[0], broadcasting one particle's value to all
    rows in the merged-surface finite-difference normal calculation.
    """
    lo = x.new_tensor([0., 0.]); hi = x.new_tensor([1.61, 1.4])
    q = torch.maximum(lo-x, x-hi)
    values = -(q.clamp_min(0).norm(dim=-1)+q.max(dim=-1).values.clamp_max(0))
    inward = torch.maximum(lo, torch.minimum(x, hi))-x
    length = inward.norm(dim=-1, keepdim=True)
    nearest = torch.cat((x-lo, hi-x), dim=-1).argmin(dim=-1)
    basis = torch.cat((torch.eye(2, device=x.device, dtype=x.dtype),
        -torch.eye(2, device=x.device, dtype=x.dtype)))
    normals = torch.where(length>0, inward/length.clamp_min(1e-30), basis[nearest])
    return values, normals


def audit_flat_wall_normals(state, margin=.05):
    """Reject invalid derived ghost normals before an expensive comparison."""
    ghost = state.kinds == 2
    boundary = state.ghostIndices[ghost].long()
    offsets = state.ghostOffsets[ghost]
    length = offsets.norm(dim=-1)
    unit = offsets/length.clamp_min(1e-30).unsqueeze(-1)
    p = state.positions[boundary]
    vertical = (p[:, 1]>margin)&(p[:, 1]<1.4-margin)
    horizontal = (p[:, 0]>margin)&(p[:, 0]<1.61-margin)
    result = {}
    for name, mask, expected in (
            ('bottom', horizontal&(p[:, 1]<0), [0., 1.]),
            ('top', horizontal&(p[:, 1]>1.4), [0., -1.]),
            ('left', vertical&(p[:, 0]<0), [1., 0.]),
            ('right', vertical&(p[:, 0]>1.61), [-1., 0.])):
        mask = mask&(length>0)
        count = int(mask.sum())
        if not count:
            raise ValueError('No usable derived ghost normals on flat '+name+' wall')
        error = float((unit[mask]-unit.new_tensor(expected)).abs().max())
        result[name] = {'rows': count, 'max_component_error': error}
        if error>1e-3:
            raise ValueError(f'Invalid derived {name} wall normals (max error {error:g}); check the SDF tuple contract')
    return result
