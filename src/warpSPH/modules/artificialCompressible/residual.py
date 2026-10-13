"""The ACSPH spatial residual: the right-hand sides of Eqs. (23) and (25)."""

import torch
from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         SupportScheme, WarpOperation, warpOperation)

from ..deltaSPH import computeVelocityDiffusion
from ..pressure import computePressureForceSurfaceAware
from .wallClosure import unilateralWallDivergence, unilateralWallForce


def spatialResidual(view, config, schemeConfig, adjacency, k1, k2, diffusion,
                     bodyForce, nu, unilateralWall=False, taper=None):
    """`(r_p, r_v)`: the right-hand sides of Eqs. (23) and (25) at `view`.

    Eq. (23):  Dp/Dtau = -k1 rho sum_j (v_j - v_i).gradW V_j + k2 D^p
    Eq. (25):  Dv/Dtau + Dv/Dt = -(1/rho) sum_j (p_i+p_j) gradW V_j
                                 + nu K sum_j (v_ij.x_ij)/|x_ij|^2 gradW V_j + f

    `diffusion` is `D^p`, passed in rather than computed here: it is frozen
    across the RK stages (Antuono/Jameson) and only re-evaluated once per
    dual-time iteration. `view.pressures` is expected to already carry the
    extrapolated wall values (`wallPressures`).
    """
    rho = view.densities

    divV = warpOperation(
        view,
        OperationProperties(kernel=config.kernel, operation=WarpOperation.Divergence,
                            supportMode=SupportScheme.SuperSymmetric,
                            operationMode=OperationDirection.AllToAll,
                            gradientMode=GradientScheme.Difference),
        queryValues=view.velocities, domain=config.domain, adjacency=adjacency)
    if unilateralWall:
        divV = unilateralWallDivergence(view, config, adjacency, divV)
    if taper is not None:
        diffusion = diffusion * taper
    rP = -k1 * rho * divV + k2 * diffusion

    # `computePressureSurfaceAwareWarp` returns `-sum_j V_j p_ij gradW`, which
    # is `rho * dv/dt`; Eq. (25) carries the explicit `1/rho_i`. delta-SPH
    # omits that division because it runs at `restDensity = 1`, where it is a
    # no-op; ACSPH does not assume that.
    if unilateralWall:
        rV = unilateralWallForce(view, config, schemeConfig, adjacency) / rho.unsqueeze(-1)
    else:
        rV = computePressureForceSurfaceAware(view, config, schemeConfig, adjacency) / rho.unsqueeze(-1)

    rV = rV + viscosity(view, config, adjacency, nu)

    return rP, rV + bodyForce


class ViscosityShim:
    """Adapter presenting `computeVelocityDiffusion`'s expected
    `schemeConfig.diffusionParams` / `.fluid` surface over ACSPH's config.
    ACSPH has no `diffusionParams` block -- its viscosity is one number derived
    from Eq. (25) -- so rather than bolt a delta-SPH block onto its config just
    to satisfy an accessor, the accessor is satisfied here."""

    class _Params:
        def __init__(self, nu):
            self.inviscid = False
            self.inviscidAlpha = 0.0
            self.viscidNu = nu

    class _Fluid:
        fixedSoundSpeed = 0.0

    def __init__(self, nu):
        self.diffusionParams = self._Params(nu)
        self.fluid = self._Fluid()


def viscosity(view, config, adjacency, nu):
    """Eq. (25)'s viscous term,
    `nu K sum_j (v_ij . x_ij)/|x_ij|^2 gradW_ij V_j` with `K = 2(dim+2)`.

    Two adaptations of `computeVelocityDiffusion`, both exact rather than
    approximate:

    - it divides by `mean(rho_i, rho_j)` where Eq. (25) does not. Density is
      invariant here, so passing `nu * rho0` compensates exactly.
    - `approachOnly=False` lifts the artificial-viscosity clamp, which is what
      makes this the Monaghan-Gingold velocity Laplacian rather than a
      one-sided half of it (see `wp_viscosityDelta.py`'s docstring).

    `nu` is per-particle where the kernel takes a scalar; the kernel's `nu`
    enters linearly, so scaling the output by `nu_i / nu_mean` afterwards is
    exact for the `nu_i` factor. (`nu_j` does not appear -- Eq. 25 has a single
    `nu`.) The rescale is skipped entirely when `nu` is uniform, which it is
    unless it was derived from a varying `h`."""
    rho0, nuScalar, nuVaries = viscosityScalars(view.densities, nu)
    out = computeVelocityDiffusion(view, config, ViscosityShim(nuScalar * rho0),
                                   adjacency, approachOnly=False)
    if nuVaries:
        out = out * (nu / nuScalar).unsqueeze(-1)
    return out


VISCOSITY_SCALARS: dict = {}


def viscosityScalars(densities, nu):
    """`(rho0, mean nu, nu non-uniform)` -- three host reads that used to run
    on every RK stage of every pseudo-iteration (~1200 syncs per real step at
    200 pseudo-iterations). `densities` and `nu` are the same tensor objects
    for a whole real step (`workingState` only swaps x/v/p), so the values are
    cached against those objects -- held, so an id can never be recycled."""
    c = VISCOSITY_SCALARS
    if c.get('densities') is densities and c.get('nu') is nu:
        return c['values']
    rho0 = float(densities[0]) if densities.numel() else 1.0
    if isinstance(nu, torch.Tensor):
        values = (rho0, float(nu.mean()), float(nu.std()) > 0.0)
    else:
        values = (rho0, float(nu), False)
    c.update(densities=densities, nu=nu, values=values)
    return values
