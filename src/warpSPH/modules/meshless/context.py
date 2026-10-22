"""What the MFM and MFV steps share: building the geometry, the start-of-step primitives and diagnostics, the supports, and the geometry
the timestep hands over.

The meshless geometry (`warpSPHCore.mfm.MeshlessWarp`) is the radius search, the moment matrices and their inversions, the effective face areas and
(optionally, `closure='project'`) the graph-Laplacian closure solve. The step builds it at the start-of-step positions and supports, derives the
primitives (rho, v, P) from the conserved ``Q`` and the geometric volume, re-derives the supports from the volumes, and then performs one of the
temporal schemes of `temporal.py`.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .primitives import BALL, primitives, refreshPrimitives
from .operators import computeRates
from .geometry import buildGeometry, startOfStepGeometry, stashGeometry, takeGeometry

__all__ = ['BALL', 'supports_from_volume', 'wrap', 'stashGeometry', 'takeGeometry', 'buildGeometry', 'ratesAtRest',
           'startOfStepGeometry', 'refreshPrimitives', 'StepContext']

#: unit-ball volumes per dimension: the h = (N_ngb V / C_nu)^(1/nu) normalisation


def supports_from_volume(V, nngb, dim):
    """The per-particle support from its effective volume: ``h = (N_ngb V / C_nu)^(1/nu)``."""
    return (nngb * V / BALL[dim]) ** (1.0 / dim)


def wrap(pos, domain):
    """Wrap positions into the domain (periodic axes only)."""
    L = domain.max - domain.min
    per = domain.periodic
    return torch.where(per, domain.min + torch.remainder(pos - domain.min, L), pos)


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
        return computeRates(self.geom,
            self.rho, self.vel, self.P, self.gamma, dt=self.dt if dt is None else dt, mode=self.schemeConfig.mode,
            order=self.schemeConfig.order, beta_min=self.schemeConfig.betaMin, beta_max=self.schemeConfig.betaMax,
            cond_crit=self.schemeConfig.condCrit, psi1=self.schemeConfig.psi1, psi2=self.schemeConfig.psi2,
            timeCentredFrame=self.schemeConfig.timeCentredFrame, retry=self.schemeConfig.retry,
            massFluxLimit=massFluxLimit, faceGuard=self.schemeConfig.faceGuard)


# ------------------------------------------------------------------------------------------------------------- geometry / rates

def ratesAtRest(geometry, Q, gamma, schemeConfig):
    rho, vel, P = primitives(Q, geometry.volume, gamma)
    return computeRates(geometry,
        rho, vel, P, gamma, dt=0.0, mode=schemeConfig.mode, order=schemeConfig.order,
        beta_min=schemeConfig.betaMin, beta_max=schemeConfig.betaMax,
        cond_crit=schemeConfig.condCrit, psi1=schemeConfig.psi1, psi2=schemeConfig.psi2,
        timeCentredFrame=schemeConfig.timeCentredFrame, retry=schemeConfig.retry,
        massFluxLimit=schemeConfig.massFluxLimit, faceGuard=schemeConfig.faceGuard)

