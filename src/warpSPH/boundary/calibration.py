"""Apply the calibrated lattice (`modules/analyticBoundary/latticeCalibration.py`) to a sampled fluid block in an analytic tank.

A summation-density (incompressible) scheme reads the analytic wall as `rho_i = sum_j V_j W_ij + mu int_solid W dA`; a sampled regular lattice is an exact rest state of that sum only if the
particle mass, the wall mass and the distance of the first row from the wall plane are the lattice's own (`latticeCalibration`). `calibrateAnalyticLattice` sets the first two and moves the walls
the fluid touches (the tank box is rebuilt with per-wall offsets, the boundary provider with it) so the first row sits at the calibrated distance. The delta+ scheme integrates its density
and does not need it (OPEN_PROBLEMS 31).
"""
import numpy as np
import torch

from ..modules.analyticBoundary.latticeCalibration import latticeCalibration

__all__ = ['calibrateAnalyticLattice']


def _spacing(values):
    u = np.unique(np.round(values, 9))
    return float(np.median(np.diff(u))) if len(u) > 1 else None


def calibrateAnalyticLattice(ctx, system, interior, touch=1.5, verbose=False):
    """Calibrate the sampled fluid in the analytic tank `interior` (box domain of the tank, before any wall offset); returns the calibration dict plus `offsets`.

    particle mass = `restDensity * V'`, `schemeConfig.analyticWallMass = mu`, and every wall of the box within `touch` spacings of the first fluid row is moved outward by `d_wall - distance`
    (the others stay at the box). Only the tank (the first analytic body) is calibrated: the calibration is derived for a flat wall.
    """
    from ..caseUtils.weaklyCompressible import analyticTankBody
    from . import buildBoundaryProvider
    sc = ctx.schemeConfig
    provider = sc.boundaryProvider
    if provider is None:
        return None
    # the tank is the first analytic body (region order); other bodies (an obstacle) keep their own surface, a sloped or curved wall is not a calibrated lattice (open: body-fitted packing)
    st = system.state
    fluid = st.kinds == 0
    pos = st.positions[fluid].double().cpu().numpy()
    dx, dy = _spacing(pos[:, 0]), _spacing(pos[:, 1])
    h = float(st.supports[fluid].max())
    cal = latticeCalibration(dx, dy, h, ctx.config.kernel)
    rho0 = sc.fluid.restDensity
    st.masses = torch.where(fluid, torch.full_like(st.masses, rho0 * cal['V']), st.masses)
    sc.analyticWallMass = cal['mu']

    lo, hi = np.array([float(interior.min[0]), float(interior.min[1])]), np.array([float(interior.max[0]), float(interior.max[1])])
    dist = np.array([pos[:, 0].min() - lo[0], pos[:, 1].min() - lo[1], hi[0] - pos[:, 0].max(), hi[1] - pos[:, 1].max()])      # lo x, lo y, hi x, hi y
    wanted = np.array([cal['dwallX'], cal['dwallY'], cal['dwallX'], cal['dwallY']])
    spacing = np.array([dx, dy, dx, dy])
    offsets = np.where(dist < touch * spacing, wanted - dist, 0.0)
    loOff, hiOff = offsets[:2], offsets[2:]

    body = analyticTankBody(interior, offset=(loOff, hiOff))
    old = provider.rigidBodies[0]
    tankRegion = next(r for r in ctx.config.regions if getattr(r, 'representation', None) is old.representation)
    body.bodyId = old.representation.bodyId
    tankRegion.representation = body
    old.representation = body
    newProvider = buildBoundaryProvider(ctx.config.regions, st.positions.device)
    newProvider.rigidBodies = provider.rigidBodies
    sc.boundaryProvider = newProvider
    cal['offsets'] = offsets
    cal['spacing'] = (dx, dy)
    if verbose:
        print(f"[latticeCalibration] spacing ({dx:.5f}, {dy:.5f}) h {h:.5f}: particle mass x{cal['V'] / (dx * dy):.5f}, wall mass {cal['mu']:.5f}, "
              f"first-row distance wanted ({cal['dwallX'] / dx:.4f} dx, {cal['dwallY'] / dy:.4f} dy), wall offsets lo {loOff / np.array([dx, dy])} hi {hiOff / np.array([dx, dy])} (in spacings)")
    return cal
