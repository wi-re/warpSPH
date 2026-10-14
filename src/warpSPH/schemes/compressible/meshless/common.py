"""What the MFM and MFV steps share: building the geometry, the start-of-step primitives and diagnostics, the supports, and the geometry
the timestep hands over.

The meshless geometry (`warpSPHCore.mfm.MeshlessWarp`) is the radius search, the moment matrices and their inversions, the effective face areas and
(optionally, `closure='project'`) the graph-Laplacian closure solve. The step builds it at the start-of-step positions and supports, derives the
primitives (rho, v, P) from the conserved ``Q`` and the geometric volume, re-derives the supports from the volumes, and then performs one of the
temporal schemes of `temporal.py`.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from warpSPHCore.mfm import MeshlessWarp, primitives

__all__ = ['BALL', 'supports_from_volume', 'wrap', 'stashGeometry', 'takeGeometry', 'buildGeometry', 'ratesAtRest',
           'startOfStepGeometry', 'refreshPrimitives', 'StepContext']

#: unit-ball volumes per dimension: the h = (N_ngb V / C_nu)^(1/nu) normalisation
BALL = {1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}


def supports_from_volume(V, nngb, dim):
    """The per-particle support from its effective volume: ``h = (N_ngb V / C_nu)^(1/nu)``."""
    return (nngb * V / BALL[dim]) ** (1.0 / dim)


def wrap(pos, domain):
    """Wrap positions into the domain (periodic axes only)."""
    L = domain.max - domain.min
    per = domain.periodic
    return torch.where(per, domain.min + torch.remainder(pos - domain.min, L), pos)


# ------------------------------------------------------------------------------------------------------ the geometry hand-over

#: The timestep's geometry (`modules/timestep/meshless.computeTimestep`, built with closure 'none' at the start-of-step positions / supports),
#: handed to the step that follows so the step's first build does not repeat the radius search, moments and inversions: the step only adds the
#: closure solve (`MeshlessWarp.close`). ONE slot, consumed by the next step; a mismatch (other positions, supports, domain, kernel or condition
#: limits) falls back to a fresh build. (Process-global by construction: the timestep and the step are separate calls of the runner loop that share no
#: object the integrator does not copy.)
_pendingGeometry = None


def stashGeometry(positions, supports, domain, kernel, schemeConfig, geometry):
    """Offer `geometry` (built at `positions` / `supports`) to the next meshless step."""
    global _pendingGeometry
    _pendingGeometry = (positions, positions._version, supports, supports._version, domain, kernel,
                        schemeConfig.condMax, schemeConfig.faceCondMax, geometry)


def takeGeometry(positions, supports, domain, kernel, schemeConfig):
    """The stashed geometry if it was built for exactly these inputs (consumed either way), else None."""
    global _pendingGeometry
    pending, _pendingGeometry = _pendingGeometry, None
    if pending is None:
        return None
    pos, posVersion, sup, supVersion, dom, ker, condMax, faceCondMax, geometry = pending
    if dom is not domain or ker != kernel or condMax != schemeConfig.condMax or \
            faceCondMax != schemeConfig.faceCondMax:
        return None
    if pos is positions and sup is supports and pos._version == posVersion and sup._version == supVersion:
        return geometry
    if pos.shape != positions.shape or sup.shape != supports.shape:
        return None
    # another tensor object (e.g. the integrator's copy of the state): compare the values
    if bool(torch.equal(pos, positions)) and bool(torch.equal(sup, supports)):
        return geometry
    return None


# --------------------------------------------------------------------------------------------------------- the step's context

@dataclass
class StepContext:
    """Everything a temporal scheme reads: the start-of-step state and primitives, the start-of-step geometry, and the two ways to evaluate more."""
    dt: Any
    gamma: float
    config: Any
    schemeConfig: Any
    domain: Any
    pos: torch.Tensor           # start-of-step positions
    supports: torch.Tensor      # the re-derived supports (what the next step uses)
    Q: torch.Tensor             # conserved (m, m v, m e_tot)
    rho: torch.Tensor           # primitives at the start of the step
    vel: torch.Tensor
    P: torch.Tensor
    u: torch.Tensor             # specific internal energy
    geom: Any                   # the start-of-step geometry

    def build(self, positions, lam0):
        """A fresh geometry at `positions` (this step's supports), warm-started from `lam0`."""
        return buildGeometry(positions, self.supports, lam0, self.config, self.schemeConfig, self.domain)

    def ratesAtRest(self, geometry, Q):
        """The face-flux rate of the conserved vector `Q` on `geometry` with no internal prediction (dt = 0)."""
        return ratesAtRest(geometry, Q, self.gamma, self.schemeConfig)

    def rates(self, massFluxLimit, dt=None):
        """The start-of-step rate with the internal (MUSCL-Hancock) prediction over `dt` (default: the step's)."""
        return self.geom.rates(
            self.rho, self.vel, self.P, self.gamma, dt=self.dt if dt is None else dt, mode=self.schemeConfig.mode,
            order=self.schemeConfig.order, beta_min=self.schemeConfig.betaMin, beta_max=self.schemeConfig.betaMax,
            cond_crit=self.schemeConfig.condCrit, psi1=self.schemeConfig.psi1, psi2=self.schemeConfig.psi2,
            timeCentredFrame=self.schemeConfig.timeCentredFrame, retry=self.schemeConfig.retry,
            massFluxLimit=massFluxLimit, faceGuard=self.schemeConfig.faceGuard)


# ------------------------------------------------------------------------------------------------------------- geometry / rates

def buildGeometry(positions, supports, lam0, config, schemeConfig, domain):
    return MeshlessWarp.build(
        positions, supports, domain, config.kernel,
        cond_max=schemeConfig.condMax, closure=schemeConfig.closure,
        closure_power=schemeConfig.closurePower, rtol=None, cg_tol=schemeConfig.cgTol,
        cg_iters=schemeConfig.cgIters, lam0=lam0, guards=True,
        face_cond_max=schemeConfig.faceCondMax, centred_weights=True, area_cap=False,
        closure_solver=schemeConfig.closureSolver,
        mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse,
        closure_cap=schemeConfig.closureCap)


def ratesAtRest(geometry, Q, gamma, schemeConfig):
    rho, vel, P = primitives(Q, geometry.volume, gamma)
    return geometry.rates(
        rho, vel, P, gamma, dt=0.0, mode=schemeConfig.mode, order=schemeConfig.order,
        beta_min=schemeConfig.betaMin, beta_max=schemeConfig.betaMax,
        cond_crit=schemeConfig.condCrit, psi1=schemeConfig.psi1, psi2=schemeConfig.psi2,
        timeCentredFrame=schemeConfig.timeCentredFrame, retry=schemeConfig.retry,
        massFluxLimit=schemeConfig.massFluxLimit, faceGuard=schemeConfig.faceGuard)


def startOfStepGeometry(state, domain, config, schemeConfig):
    """The start-of-step geometry: the timestep's one at the same positions / supports when it is there (only the closure solve is added),
    else a fresh build, warm-starting the closure solve from the previous step's lam."""
    geom = takeGeometry(state.positions, state.supports, domain, config.kernel, schemeConfig)
    if geom is None:
        return buildGeometry(state.positions, state.supports, state.lam, config, schemeConfig, domain)
    if schemeConfig.closure == 'project':
        geom.close(closure_power=schemeConfig.closurePower, cg_tol=schemeConfig.cgTol,
                   cg_iters=schemeConfig.cgIters, lam0=state.lam,
                   closure_solver=schemeConfig.closureSolver,
                   mg_min_particles=schemeConfig.mgMinParticles, mg_max_coarse=schemeConfig.mgMaxCoarse,
                   closure_cap=schemeConfig.closureCap)
    return geom


def refreshPrimitives(state, geom, gamma, nngb, dim):
    """The primitives from the conserved Q and the geometric volume; the supports re-derived from the volumes (lagged one step, as in the core
    driver) and the start-of-step diagnostic fields refreshed on `state`. Returns ``(rho, vel, P, u)``."""
    rho, vel, P = primitives(state.Q, geom.volume, gamma)
    state.supports = supports_from_volume(geom.volume, nngb, dim)
    masses = state.Q[:, 0]
    e_tot = state.Q[:, -1] / masses.clamp_min(1e-30)
    u = e_tot - 0.5 * torch.einsum('ij,ij->i', vel, vel)
    state.densities = rho
    state.velocities = vel
    state.pressures = P
    state.masses = masses
    state.totalEnergies = state.Q[:, -1]
    state.internalEnergies = u
    state.soundspeeds = torch.sqrt(gamma * P / rho.clamp_min(1e-30))
    state.entropies = P / rho.clamp_min(1e-30) ** gamma
    return rho, vel, P, u
