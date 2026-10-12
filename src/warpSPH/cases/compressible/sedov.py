"""Sedov-Taylor blast wave, compressible.

The script/notebook forms of this case live in `examples/compressible/06-sedov/`
(`sedov_1d.py`/`.ipynb`, `sedov_2d.py`/`.ipynb`, `sedov_3d.py`/`.ipynb`). All
three dimensionalities are the *same* case -- identical sampler, identical
scheme, identical stopping rule -- so this is one `Case` run as ``--dim 1``,
``--dim 2`` or ``--dim 3``; only `nx` (and, for the notebooks, `PRESET`)
differs.

The run ends when the shock reaches `goalRadius`, which is a time derived from
the analytic self-similar solution rather than a number chosen by hand.
"""

from __future__ import annotations

from typing import Dict, List

from ...caseUtils import SedovSolution, beta, buildSedov, radius
from ...configurations.meshless import MeshlessConfig
from ...runner import Case, RunContext, caseMain, registerCase
from .common import (COMPRESSIBLE_DEFAULTS, COMPRESSIBLE_PARAMS,
                           compressibleDiagnostics, configureCompressible,
                           paramExtraData)
from ..common.plotting import ProfileAxis, profilePlot

__all__ = ['sedovCase', 'sedovSolution', 'goalTime', 'drawSedov']


def sedovSolution(ctx: RunContext) -> SedovSolution:
    return SedovSolution(
        nDim=ctx.spec.dim,
        gamma=ctx.param('gamma'),
        rho0=ctx.param('rho0'),
        E0=ctx.param('E0'),
        h0=2 / ctx.spec.nx,
    )


def goalTime(ctx: RunContext, solution: SedovSolution) -> float:
    """The time at which the shock front reaches ``goalRadius``."""
    nu1 = 1.0 / (solution.nu + 2.0)
    return (ctx.param('goalRadius')
            * (solution.alpha * ctx.param('rho0') / ctx.param('E0')) ** nu1) ** (1.0 / (2.0 * nu1))


def buildSystem(ctx: RunContext):
    solution = sedovSolution(ctx)
    ctx.scratch['solution'] = solution
    ctx.spec = ctx.spec.merged(tLimit=goalTime(ctx, solution))
    kwargs = dict(
        config=ctx.config,
        nx=ctx.spec.nx, dim=ctx.spec.dim, domainExtent=ctx.spec.L,
        periodicDomain=ctx.spec.periodic,
        rho0=ctx.param('rho0'), E0=ctx.param('E0'),
        initialization=ctx.param('initialization'),
        gamma=ctx.param('gamma'), kernel=ctx.config.kernel,
        targetNeighbors=ctx.config.targetNeighbors,
        dtype=ctx.config.dtype, device=ctx.config.device)
    if isinstance(ctx.schemeConfig, MeshlessConfig):
        from ...caseUtils.meshless import toMeshlessSystem
        from ...systems.compressibleMonaghan import CompressibleState, CompressibleSystem
        comp = buildSedov(CompressibleSystem, CompressibleState, **kwargs)
        return toMeshlessSystem(comp, ctx.config, ctx.schemeConfig)
    return buildSedov(ctx.SimulationSystem, ctx.SimulationState, **kwargs)


# -- plotting ----------------------------------------------------------------
# One set of radial profile panels for every dimension: `profilePlot` scatters
# each particle against its own distance from the origin at `dim>1` (unsigned,
# vector quantities read as their magnitude) and against raw signed `x` at
# `dim==1` (matching the two reference notebooks this case replaces, which
# plotted the same two shock-radius estimates the two ways below). Two
# reference targets are kept at every dimension: `r2`/`vs`/`rho2` come from
# `SedovSolution`'s full self-similar solve, `rt` from the closed-form
# `beta`-fit estimate the notebooks also overlaid -- the gap between the two
# is itself a diagnostic worth seeing.

def _mirror(ctx: RunContext, values: List[float]) -> List[float]:
    """Add the negative of each value too, but only for the signed `dim==1` x-axis."""
    return list(values) + [-v for v in values] if ctx.spec.dim == 1 else list(values)


def _shock(ctx: RunContext, state):
    """`(shockSpeed, r2, v2, rho2, P2)` and the front radius the beta-fit predicts."""
    solution = ctx.scratch['solution']
    t = max(float(state.t), 1e-12)
    return solution.shockState(t), radius(beta(ctx.spec.dim), ctx.param('E0'), t,
                                          ctx.param('rho0'), ctx.spec.dim)


def _fronts(ctx: RunContext, state):
    (_, r2, _, _, _), rt = _shock(ctx, state)
    return _mirror(ctx, [r2, rt])


def _shockDensity(ctx: RunContext, state):
    (_, _, _, rho2, _), _ = _shock(ctx, state)
    return [ctx.param('rho0'), rho2]


def _shockVelocity(ctx: RunContext, state):
    (vs, _, v2, _, _), _ = _shock(ctx, state)
    return _mirror(ctx, [vs, v2])


setupPlot, updatePlot, drawSedov = profilePlot(
    [
        ProfileAxis('internalEnergies', 'Internal energy', yscale='log', vlines=_fronts),
        ProfileAxis('densities', 'Density', yscale='log', vlines=_fronts,
                    hlines=_shockDensity),
        ProfileAxis('velocities', 'Velocity', component=0, vlines=_fronts,
                    hlines=_shockVelocity),
        ProfileAxis('supports', 'Support', vlines=_fronts),
    ],
    shape=(2, 2), figsize=(9, 6), xlim=(0, 1),
)


def diagnostics(ctx: RunContext, state) -> Dict[str, float]:
    return compressibleDiagnostics(ctx, state)


sedovCase = registerCase(Case(
    name='sedov',
    scheme='CRKSPH',
    description='Sedov-Taylor blast wave (1D, 2D or 3D), compressible SPH.',
    buildSystem=buildSystem,
    configureScheme=configureCompressible,
    diagnostics=diagnostics,
    setupPlot=setupPlot,
    updatePlot=updatePlot,
    extraData=paramExtraData,
    defaults=dict(
        COMPRESSIBLE_DEFAULTS,
        caseName='06-sedovTaylorBlastwave',
        # `nx` is per-dimension particle count along one axis of the domain,
        # so the total particle count grows as `nx**dim`: 1D uses the
        # notebook's nx=800, 2D and 3D each dial it back (see sedov_2d.py/
        # sedov_3d.py) to keep the 2D/3D examples in a comparable budget.
        dim=1,
        nx=800,
        L=2.0,
        # Replaced in buildSystem by the analytic time to reach goalRadius.
        tLimit=1.0,
        plotInterval=25,
        storeInterval=500,
    ),
    params=dict(
        COMPRESSIBLE_PARAMS,
        E0=1.0,
        goalRadius=0.8,
        # 'hat' deposits E0 on the single particle nearest the origin, same as
        # 'singular', then smooths that spike with one SPH interpolation pass
        # over the finalized adaptive supports -- spreading it over one
        # smoothing scale instead of leaving a single-particle delta, which is
        # what 'singular' does and what makes it numerically harsh at low
        # resolution. 'quadrant' instead spreads E0 evenly over the 2^dim
        # innermost particles, with no smoothing.
        initialization='hat',
    ),
))


if __name__ == '__main__':
    caseMain(sedovCase)
