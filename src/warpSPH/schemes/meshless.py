"""Thin MFM / MFV step (COMPRESSIBLE_SPH_PLAN E1): wraps the core `warpSPHCore.mfm` backend.

Builds the meshless geometry (warm-starting the closure solve from the previous step's
``lam``), derives the primitives (rho, v, P) from the conserved ``Q`` and the geometric
volume, re-derives the supports from the volumes (``h = (N_ngb V / C_nu)^(1/nu)``), evaluates
the face-flux divergence rates (the MUSCL-Hancock / Riemann stage of Hopkins 2015), and hands
the integrator the conserved rates ``dQ/dt`` (plus ``dx/dt = v``). The temporal integration is
the integrator's job (the default is decided by the D4 measurement); this step is the spatial
RHS. The geometry is built by the backend from the frontend's own positions / supports, so no
frontend Verlet list is needed (the returned adjacency is ``None``).
"""

import math
import torch

from ..configurations import SimulationConfig
from ..configurations.meshless import MeshlessConfig
from ..systems.meshless import MeshlessSystem, MeshlessSystemUpdate
from warpSPHCore.mfm import MeshlessWarp, primitives

__all__ = ['meshlessStep', 'supports_from_volume']

#: unit-ball volumes per dimension: the h = (N_ngb V / C_nu)^(1/nu) normalisation
BALL = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}


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
    currentSystem = system
    currentState = currentSystem.state
    dim = config.dim
    gamma = float(schemeConfig.gamma)

    # 1. the meshless geometry: build (or rebuild) the graph, warm-starting the closure solve
    geom = MeshlessWarp.build(
        currentState.positions, currentState.supports, system.domain, config.kernel,
        cond_max=schemeConfig.condMax, closure=schemeConfig.closure,
        closure_power=schemeConfig.closurePower, rtol=None, cg_tol=schemeConfig.cgTol,
        cg_iters=schemeConfig.cgIters, lam0=currentState.lam, guards=True,
        face_cond_max=schemeConfig.faceCondMax, centred_weights=True, area_cap=False,
        closure_solver=schemeConfig.closureSolver,
        mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse)
    currentState.lam = geom.lam

    # 2. the primitives from the conserved Q and the geometric volume
    rho, vel, P = primitives(currentState.Q, geom.volume, gamma)

    # 3. re-derive the supports from the volumes (lagged one step, as in the core driver)
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

    # 4. the face-flux divergence rates (the spatial RHS, in conserved form)
    rates = geom.rates(
        rho, vel, P, gamma, dt=dt, mode=schemeConfig.mode, order=schemeConfig.order,
        beta_min=schemeConfig.betaMin, beta_max=schemeConfig.betaMax,
        cond_crit=schemeConfig.condCrit, psi1=schemeConfig.psi1, psi2=schemeConfig.psi2,
        timeCentredFrame=schemeConfig.timeCentredFrame, retry=schemeConfig.retry,
        massFluxLimit=schemeConfig.massFluxLimit)

    update = MeshlessSystemUpdate(dxdt=vel, dQdt=rates)
    adjacency = None
    return update, adjacency, currentState
