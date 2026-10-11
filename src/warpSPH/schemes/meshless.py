"""Full MFM / MFV step (COMPRESSIBLE_SPH_PLAN E1 + D4): wraps the core `warpSPHCore.mfm` backend.

Unlike the other compressible steps, this one owns the *entire* time step internally and hands
the integrator the exact per-step deltas ``(x^{n+1} - x^n) / dt`` -- the same exact-delta
contract as `schemes/artificialCompressible.py`. The spatial part is the meshless geometry (a
warm-started graph-Laplacian closure solve from the previous step's ``lam``), the primitives
(rho, v, P) derived from the conserved ``Q`` and the geometric volume, and the re-derived
supports (``h = (N_ngb V / C_nu)^(1/nu)``, lagged one step behind as in the core driver). The
temporal part is the D4-selected scheme (``temporalScheme``):

- ``'hancock'`` -- the core driver's single stage: the face-flux divergence rate at ``dt``
  (the MUSCL-Hancock internal prediction), an Euler update of the conserved ``Q``, and a
  midpoint drift of the positions. This is the G1 port-fidelity setting.
- ``'ssprk'``  -- two-stage SSPRK: the rate at ``dt = 0`` (no internal prediction) at the start,
  a stage state at the midpoint drift (a fresh geometry warm-started from the first), the rate
  there, and ``Q^{n+1} = 1/2 (Q* + Q + dt f2)``; the final drift is time-centred.
- ``'kdk'``    -- kick half, drift the full ``dt`` with the half-kicked velocity, kick half with
  the rate at the drifted state.

``'auto'`` resolves per mode (the D4 measurement): MFM -> ``'ssprk'``, MFV -> ``'hancock'``.
Because the step performs the temporal integration itself, the integrator must be
single-evaluation (``forwardEuler`` / ``explicitEuler``); any other integrator would run the
step once per stage and blend -- a plausible-looking but wrong update -- so the entry rejects it
(``validateIntegrationScheme``). The geometry is built by the backend from the frontend's own
positions / supports, so no frontend Verlet list is needed (the returned adjacency is ``None``).
"""

import math
import torch

from warpSPHIntegrators.integration import IntegrationSchemeType

from ..configurations import SimulationConfig
from ..configurations.meshless import MeshlessConfig
from ..systems.meshless import MeshlessSystem, MeshlessSystemUpdate
from warpSPHCore.mfm import MeshlessWarp, primitives

__all__ = ['meshlessStep', 'supports_from_volume', 'resolve_temporal_scheme',
           'validateIntegrationScheme']

#: unit-ball volumes per dimension: the h = (N_ngb V / C_nu)^(1/nu) normalisation
BALL = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}

#: the D4-measured per-mode default temporal scheme (what 'auto' resolves to)
_AUTO_TEMPORAL_SCHEME = {'MFM': 'ssprk', 'MFV': 'hancock'}
_KNOWN_TEMPORAL_SCHEMES = ('hancock', 'ssprk', 'kdk')

#: the integrators that evaluate the step exactly once per time step -- the only ones that can
#: drive a step which owns its own full time step. A multi-stage integrator would run the step
#: once per stage and blend the results, which is plausible-looking but wrong. Mirrors
#: `artificialCompressible._EXACT_DELTA_INTEGRATORS`.
_EXACT_DELTA_INTEGRATORS = {IntegrationSchemeType.forwardEuler, IntegrationSchemeType.explicitEuler}


def resolve_temporal_scheme(schemeConfig: MeshlessConfig) -> str:
    """Resolve ``temporalScheme`` to a concrete scheme ('auto' -> the D4 per-mode default)."""
    ts = schemeConfig.temporalScheme
    if ts == 'auto':
        return _AUTO_TEMPORAL_SCHEME[schemeConfig.mode]
    if ts in _KNOWN_TEMPORAL_SCHEMES:
        return ts
    raise ValueError(
        f'temporalScheme {ts!r} is not recognised; expected "auto" or one of '
        f'{list(_KNOWN_TEMPORAL_SCHEMES)}.')


def validateIntegrationScheme(config: SimulationConfig) -> None:
    """Refuse any integrator that would run this step more than once per time step.

    `meshlessStep` performs the entire temporal integration internally and returns the exact
    per-step delta, so it must be evaluated exactly once per step. Mirrors
    `artificialCompressible.validateIntegrationScheme`.
    """
    scheme = getattr(config, 'integrationScheme', None)
    if scheme in _EXACT_DELTA_INTEGRATORS:
        return
    raise ValueError(
        f'the MFM / MFV step owns its full time step (D4) and must run under a '
        f'single-evaluation integrator (forwardEuler / explicitEuler); got {scheme!r}. Set '
        f'config.integrationScheme to IntegrationSchemeType.forwardEuler.')


def wrap(pos, domain):
    """Wrap positions into the domain (periodic axes only)."""
    L = domain.max - domain.min
    per = domain.periodic
    return torch.where(per, domain.min + torch.remainder(pos - domain.min, L), pos)


def supports_from_volume(V, nngb, dim):
    """The per-particle support from its effective volume: ``h = (N_ngb V / C_nu)^(1/nu)``."""
    return (nngb * V / BALL[dim]) ** (1.0 / dim)


def meshlessStep(
    system: MeshlessSystem,
    dt: float,
    config: SimulationConfig,
    schemeConfig: MeshlessConfig,
    verbose = False,
):
    # the step owns the full time step, so the integrator must evaluate it exactly once
    validateIntegrationScheme(config)

    currentSystem = system
    currentState = currentSystem.state
    dim = config.dim
    gamma = float(schemeConfig.gamma)
    scheme = resolve_temporal_scheme(schemeConfig)
    domain = system.domain

    def build(pos, h, lam0):
        return MeshlessWarp.build(
            pos, h, domain, config.kernel,
            cond_max=schemeConfig.condMax, closure=schemeConfig.closure,
            closure_power=schemeConfig.closurePower, rtol=None, cg_tol=schemeConfig.cgTol,
            cg_iters=schemeConfig.cgIters, lam0=lam0, guards=True,
            face_cond_max=schemeConfig.faceCondMax, centred_weights=True, area_cap=False,
            closure_solver=schemeConfig.closureSolver,
            mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse)

    def rates_at(g, Q):
        rho_g, vel_g, P_g = primitives(Q, g.volume, gamma)
        return g.rates(
            rho_g, vel_g, P_g, gamma, dt=0.0, mode=schemeConfig.mode, order=schemeConfig.order,
            beta_min=schemeConfig.betaMin, beta_max=schemeConfig.betaMax,
            cond_crit=schemeConfig.condCrit, psi1=schemeConfig.psi1, psi2=schemeConfig.psi2,
            timeCentredFrame=schemeConfig.timeCentredFrame, retry=schemeConfig.retry,
            massFluxLimit=schemeConfig.massFluxLimit)

    # 1. the start-of-step geometry: build (or rebuild) the graph, warm-starting the closure
    #    solve from the previous step's lam
    geom = build(currentState.positions, currentState.supports, currentState.lam)

    # 2. the primitives from the conserved Q and the geometric volume
    rho, vel, P = primitives(currentState.Q, geom.volume, gamma)

    # 3. re-derive the supports from the volumes (lagged one step, as in the core driver) and
    #    refresh the start-of-step diagnostic primitives
    currentState.supports = supports_from_volume(geom.volume, schemeConfig.nngb, dim)
    masses = currentState.Q[:, 0]
    e_tot = currentState.Q[:, -1] / masses.clamp_min(1e-30)
    u = e_tot - 0.5 * torch.einsum('ij,ij->i', vel, vel)
    currentState.densities = rho
    currentState.velocities = vel
    currentState.pressures = P
    currentState.masses = masses
    currentState.totalEnergies = currentState.Q[:, -1]
    currentState.internalEnergies = u
    currentState.soundspeeds = torch.sqrt(gamma * P / rho.clamp_min(1e-30))
    currentState.entropies = P / rho.clamp_min(1e-30) ** gamma

    Q = currentState.Q
    pos = currentState.positions

    if scheme == 'hancock':
        # the core driver's single stage: the rate at dt (the MUSCL-Hancock internal
        # prediction), an Euler update of the conserved Q, and a midpoint drift of the
        # positions. dQdt is exactly the rate, so the integrator's Q += dt*dQdt is that Euler
        # update, and dxdt is the time-centred velocity, so pos += dt*dxdt is the midpoint
        # drift.
        dQdt = geom.rates(
            rho, vel, P, gamma, dt=dt, mode=schemeConfig.mode, order=schemeConfig.order,
            beta_min=schemeConfig.betaMin, beta_max=schemeConfig.betaMax,
            cond_crit=schemeConfig.condCrit, psi1=schemeConfig.psi1, psi2=schemeConfig.psi2,
            timeCentredFrame=schemeConfig.timeCentredFrame, retry=schemeConfig.retry,
            massFluxLimit=schemeConfig.massFluxLimit)
        Qn = Q + dt * dQdt
        vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
        dxdt = 0.5 * (vel + vel_n)
        finalLam = geom.lam
    elif scheme == 'ssprk':
        # two-stage SSPRK: f1 at the start (dt = 0, no internal prediction), a stage state at
        # the midpoint drift (a fresh geometry warm-started from the first), f2 there, and
        # Q^{n+1} = 1/2 (Q* + Q + dt f2); the final drift is time-centred.
        f1 = rates_at(geom, Q)
        Qs = Q + dt * f1
        vels = Qs[:, 1:-1] / Qs[:, 0:1]
        poss = wrap(pos + 0.5 * dt * (vel + vels), domain)
        g2 = build(poss, currentState.supports, geom.lam)
        f2 = rates_at(g2, Qs)
        Qn = 0.5 * Qs + 0.5 * (Q + dt * f2)
        vel_n = Qn[:, 1:-1] / Qn[:, 0:1]
        dQdt = (Qn - Q) / dt
        dxdt = 0.5 * (vel + vel_n)
        finalLam = g2.lam
    else:  # 'kdk'
        # kick half, drift the full dt with the half-kicked velocity, kick half with the rate
        # at the drifted state.
        f1 = rates_at(geom, Q)
        Qh = Q + 0.5 * dt * f1
        velh = Qh[:, 1:-1] / Qh[:, 0:1]
        poss = wrap(pos + dt * velh, domain)
        g2 = build(poss, currentState.supports, geom.lam)
        f2 = rates_at(g2, Qh)
        Qn = Qh + 0.5 * dt * f2
        dQdt = (Qn - Q) / dt
        dxdt = velh
        finalLam = g2.lam

    # 4. hand the integrator the exact per-step deltas: it applies pos += dt*dxdt (wrapping) and
    #    Q += dt*dQdt. The closure warm start for the next step is the last geometry's lam.
    currentState.lam = finalLam
    update = MeshlessSystemUpdate(dxdt=dxdt, dQdt=dQdt)
    adjacency = None
    return update, adjacency, currentState
