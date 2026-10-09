"""What every compressible example case does the same way.

The fifteen `examples/compressible/*.ipynb` notebooks share a setup block
verbatim: CRKSPH, the B7 kernel, `gamma`/`rho0` onto the scheme config, the
viscosity switch and the Owen adaptive-support scheme. They also share their
diagnostics -- kinetic, thermal and total energy, with total energy the
conserved quantity the runs are judged on.

That block lives here once, so each case module is only its own geometry,
initial condition and plot.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..configurations.moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig
from ..enumTypes import AdaptiveSupportScheme, ViscositySwitch
from ..modules.timestep.compressible import computeTimestep
from ..runner import RunContext, resolveEnum

__all__ = ['COMPRESSIBLE_DEFAULTS', 'COMPRESSIBLE_PARAMS', 'configureCompressible',
           'compressibleDiagnostics', 'ALPHA_ACTIVE_THRESHOLD', 'avReportDiagnostics', 'compressibleTimestep', 'paramExtraData']


#: `CaseSpec` fields shared by the compressible examples. A case merges these
#: into its own `defaults` and overrides what it needs (`nx`, `dim`, `tLimit`).
COMPRESSIBLE_DEFAULTS = dict(
    kernel='B7',
    integrationScheme='rungeKutta2',
    supportMode='KernelMeanSymmetric',
    gradientMode='Difference',
    laplacianMode='Brookshaw',
    samplingScheme='regular',
    periodic=True,
    n_h=4.0,
    # Left unset: every compressible sampler ends by calling `computeTimestep`,
    # so the CFL-derived value is what the run actually starts from.
    dt=None,
    adaptiveDt=True,
    cflFactor=0.3,
    minDt=1e-8,
    plotInterval=25,
    storeInterval=500,
)

#: Case parameters shared by the compressible examples.
COMPRESSIBLE_PARAMS = dict(
    gamma=5 / 3,
    rho0=1.0,
    # None: the scheme's own default (Monaghan: Rosswog 2020 + limited reconstruction since 2026-10-08; CompSPH /
    # CRKSPH: none). A named switch gets a plain `ViscositySwitchConfig`, as it always did.
    viscositySwitch=None,
    adaptiveSupportScheme='Owen',
    adaptiveSupportCorrections=False,
    markerSize=2,
)


def applyViscositySwitchParam(ctx: RunContext) -> None:
    """The case parameter `viscositySwitch`: None keeps the scheme's own default switch config (Monaghan: Rosswog
    2020 at alpha in [0, 1] since 2026-10-08; CompSPH / CRKSPH: none); a named switch gets a fresh
    `ViscositySwitchConfig` (its own default alpha range), exactly as before the Monaghan default changed."""
    if ctx.param('viscositySwitch') is not None:
        ctx.schemeConfig.viscositySwitchParams = ViscositySwitchConfig()
        ctx.schemeConfig.viscositySwitchParams.scheme = resolveEnum(
            ViscositySwitch, ctx.param('viscositySwitch'))


def configureCompressible(ctx: RunContext) -> None:
    """Stamp the shared scheme settings onto `ctx.schemeConfig`."""
    schemeConfig = ctx.schemeConfig
    schemeConfig.gamma = ctx.param('gamma')
    schemeConfig.rho0 = ctx.param('rho0')
    applyViscositySwitchParam(ctx)
    schemeConfig.adaptiveSupportScheme = resolveEnum(
        AdaptiveSupportScheme, ctx.param('adaptiveSupportScheme'))
    schemeConfig.adaptiveSupportCorrections = ctx.param('adaptiveSupportCorrections')
    # solid-wall knobs, for the cases that have walls (cases/compressibleWalls.py WALL_PARAMS)
    params = ctx.spec.params
    if 'wallSlip' in params:
        schemeConfig.wallSlip = params['wallSlip']
    if 'wallRiemannState' in params:
        schemeConfig.wallRiemannState = params['wallRiemannState']
    if 'wallLatticeSupport' in params:
        schemeConfig.wallLatticeSupport = {'auto': None, 'on': True, 'off': False}[params['wallLatticeSupport']]


#: A particle counts as "active" for the AV report when its switched alpha
#: exceeds this (AV_PLAN Part 0, Group B: `fraction(alpha > 0.1)`).
ALPHA_ACTIVE_THRESHOLD = 0.1


def compressibleDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """Energies, entropy, angular momentum and the switch's alpha statistics.

    Total energy is the conserved one; every sum is over the fluid rows only
    (solid wall rows, `kinds != 0`, are excluded). The AV_PLAN Part 0 additions (all read
    from the state, so every compressible case and scheme picks them up):

    * `entropy` -- `sum m_i s_i` with `s = P / rho^gamma`, `P = (gamma-1) rho u`.
      Monotone non-decreasing in an adiabatic flow with only dissipative AV, so
      its growth is the integrated dissipation.
    * `angularMomentum` -- `sum m_i (x_i x v_i)` about the coordinate origin:
      the z component in 2D, the norm in 3D, 0 in 1D. Only meaningful where the
      run does not wrap positions across a periodic boundary (Gresho, Yee, Noh
      about their centre); elsewhere read it as a drift indicator only.
    * `alphaMean/alphaMax/alphaActiveFraction` -- of `state.alphas`, the value
      the pair operator actually uses; `NaN` when the scheme has none.
    """
    particles = state.state
    # solid wall rows (`kinds != 0`, modules/compressibleWall) hold no energy of the
    # gas: their state is re-derived from the fluid every evaluation
    kinds = getattr(particles, 'kinds', None)
    fluid = (kinds == 0) if kinds is not None else torch.ones_like(particles.masses, dtype=torch.bool)
    mass = torch.where(fluid, particles.masses, torch.zeros_like(particles.masses))
    kinetic = 0.5 * (torch.linalg.norm(particles.velocities, dim=-1) ** 2 * mass).sum()
    thermal = (particles.internalEnergies * mass).sum()

    gamma = float(ctx.schemeConfig.gamma)
    rho = particles.densities
    entropy = (mass * (gamma - 1.0) * particles.internalEnergies * rho ** (1.0 - gamma)).sum()

    pos, vel = particles.positions, particles.velocities
    dim = pos.shape[-1]
    if dim == 2:
        angular = (mass * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])).sum()
    elif dim == 3:
        angular = torch.linalg.norm((mass[:, None] * torch.linalg.cross(pos, vel)).sum(dim=0))
    else:
        angular = torch.zeros((), dtype=mass.dtype, device=mass.device)

    out = {
        'kineticEnergy': kinetic,
        'thermalEnergy': thermal,
        'totalEnergy': kinetic + thermal,
        'entropy': entropy,
        'angularMomentum': angular,
    }
    alphas = particles.alphas
    if alphas is not None:
        out['alphaMean'] = alphas.mean()
        out['alphaMax'] = alphas.max()
        out['alphaActiveFraction'] = (alphas > ALPHA_ACTIVE_THRESHOLD).to(alphas.dtype).mean()
    else:
        nan = torch.full((), float('nan'), dtype=mass.dtype, device=mass.device)
        out.update(alphaMean=nan, alphaMax=nan, alphaActiveFraction=nan)
    return {k: v.detach().cpu().item() for k, v in out.items()}


def avReportDiagnostics(ctx: RunContext, state) -> Dict[str, float]:
    """The extra per-step numbers `scripts/av_report.py` records on top of the
    case's own diagnostics: AV dissipation power split linear/quadratic
    (`modules/dissipation/avPower.py`) and the neighbour-count statistics of the
    adjacency the last RHS evaluation left on the system (Group F). Kept out of
    `compressibleDiagnostics` because the power split costs three extra
    neighbour loops."""
    from ..modules.dissipation.avPower import computeAVPowerSplit, computeChenNixonRatio
    out = computeAVPowerSplit(state, ctx.config, ctx.schemeConfig)
    out.update(computeChenNixonRatio(state, ctx.config, ctx.schemeConfig))
    n = getattr(state.adjacency, 'numNeighbors', None)
    if n is not None:
        n = n.detach().double()
        out.update(neighboursMean=n.mean().item(), neighboursMin=n.min().item(),
                   neighboursMax=n.max().item())
    return out


def compressibleTimestep(ctx: RunContext, state) -> "float | torch.Tensor":
    """The acoustic-CFL `dt`, recomputed from the current state.

    Attach this as a case's `timestep` hook to get the notebooks'
    ``while t < tLimit`` behaviour: dt tracks the sound speed as the shock
    develops instead of staying at whatever the initial state implied.
    """
    return computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)


def paramExtraData(ctx: RunContext, state) -> Dict[str, Any]:
    """Record the case's own parameters on every exported frame.

    Lists and dicts are dropped: the HDF5 attribute writer only takes scalars
    and strings, and a nested region description is not what a frame needs.
    """
    return {k: v for k, v in ctx.spec.params.items()
            if not isinstance(v, (list, dict))}
