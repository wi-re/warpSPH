"""Shared helpers for the scripts/gradcheck_*.py family.

Not a standalone entrypoint (leading underscore) -- imported by the
per-module gradcheck scripts. See docs/historic_plans/CLEANUP_PLAN.md's Phase 4.1 for the
methodology this mirrors (warpSPHCore's own scripts/_gradcheck_common.py,
scripts/gradcheck_*_native.py): one standalone script per kernel-bearing
module, each calling torch.autograd.gradcheck directly against the real
entry point, run as a subprocess because warpSPHCore_PRECISION is baked
into every @wp.kernel/@wp.func at first warpSPHCore import and cannot
change mid-process.

These fixtures are *vendored* from warpSPHCore's scripts/_gradcheck_common.py
rather than imported -- that module lives under warpSPHCore's scripts/, not
its installed package, so it is not importable from here. Keep the vendored
copies behaviourally identical; do not add repo-specific logic to them. Add
repo-local helpers (this repo's per-scheme state objects, minimal
SimulationConfig/SchemeConfig construction) as separate functions below
instead, for modules that need more than a bare ParticleState.

Every gradcheck_*.py script must set warpSPHCore_PRECISION=float64 via
os.environ.setdefault *before* importing this module (which imports
warpSPHCore) -- gradcheck's numerical Jacobian needs the precision headroom,
and the setting can't change after the first warpSPHCore import in a
process. This repo additionally has its own precision-locking entry point,
warpSPHBootstrap.bootstrap(); a gradcheck script should set the env var
directly instead, matching how warpSPHCore's own scripts bypass any
higher-level config helper for the same reason.

All cases here are deliberately 1D/small (single particle / a short line),
so failures are easy to reason about by hand and self- vs. non-self-particle
gradient contributions are easy to separate out.
"""

from __future__ import annotations

import torch

from warpSPHCore import ParticleState, radiusSearchCompactHashMap, buildCompactHashMap
from warpSPHCore.enumTypes import KernelFunctions, SupportScheme
from warpSPHCore.dataTypes import DomainDescription

DEVICE = torch.device("cpu")
DTYPE = torch.float64
KERNEL = KernelFunctions.Wendland2


# --------------------------------------------------------------------------
# Domain / particle-case construction (vendored from warpSPHCore's
# scripts/_gradcheck_common.py -- see this module's docstring for why).
# --------------------------------------------------------------------------

def make_domain(dim: int = 1, margin: float = 10.0) -> DomainDescription:
    return DomainDescription(
        min=torch.tensor([-margin] * dim, dtype=DTYPE, device=DEVICE),
        max=torch.tensor([margin] * dim, dtype=DTYPE, device=DEVICE),
        periodic=torch.tensor([False] * dim, device=DEVICE),
        dim=dim,
    )


def single_particle_case(h: float = 1.0):
    """One particle at the origin. Isolates the self-interaction term:
    no neighbors, so any nonzero d(output)/d(position) gradient here would
    be a bug -- a symmetric kernel's gradient at r=0 is exactly zero."""
    positions = torch.tensor([[0.0]], dtype=DTYPE, device=DEVICE, requires_grad=True)
    supports = torch.tensor([h], dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.tensor([1.0], dtype=DTYPE, device=DEVICE, requires_grad=True)
    return positions, supports, masses


def line_case(n: int, xmin: float = -1.0, xmax: float = 1.0, h: float | None = None):
    """n particles evenly spaced on [xmin, xmax]. Small and regular enough
    to inspect self vs. non-self gradient terms entry-by-entry."""
    x = torch.linspace(xmin, xmax, n, dtype=DTYPE, device=DEVICE).unsqueeze(-1)
    positions = x.detach().clone().requires_grad_(True)  # fresh leaf tensor
    if h is None:
        spacing = (xmax - xmin) / max(n - 1, 1)
        h = max(2.5 * spacing, 1e-3)  # a few particle spacings so neighborhoods overlap
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return positions, supports, masses


def grid_case_2d(n_per_side: int = 3, spacing: float = 0.4, h: float | None = None):
    """n_per_side x n_per_side particles on a regular 2D grid centered at the
    origin. For modules that need a genuine 2D domain rather than the
    degenerate 1D case line_case gives -- kept small since gradcheck's
    numerical Jacobian cost grows with total element count across all
    differentiable inputs."""
    coords = torch.linspace(-(n_per_side - 1) / 2 * spacing, (n_per_side - 1) / 2 * spacing, n_per_side, dtype=DTYPE, device=DEVICE)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    x = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    positions = x.detach().clone().requires_grad_(True)  # fresh leaf tensor
    n = positions.shape[0]
    if h is None:
        h = max(2.5 * spacing, 1e-3)
    supports = torch.full((n,), h, dtype=DTYPE, device=DEVICE, requires_grad=True)
    masses = torch.full((n,), 1.0, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return positions, supports, masses


def build_adjacency(positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, domain: DomainDescription, mode=SupportScheme.Gather):
    """Adjacency is treated as non-differentiable and frozen: built once
    from detached positions and reused across every forward call in a
    gradcheck, rather than rebuilt per-call. This matches the standard SPH
    modeling assumption (no contribution from the neighbor search itself to
    the gradient) and keeps the function gradcheck evaluates numerically
    smooth -- rebuilding the neighbor list under finite-difference
    perturbation risks discontinuities right at the support-radius boundary."""
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    adjacency = radiusSearchCompactHashMap(p, domain, mode=mode)
    return adjacency, kinds


def build_grid_adjacency(positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, domain: DomainDescription, mode=SupportScheme.Gather):
    """Same non-differentiable/frozen contract as build_adjacency, but returns a genuine
    CompactHashMap (grid traversal, useAdjacency=False) instead of the CSR AdjacencyList
    radiusSearchCompactHashMap returns by default despite its name. Use this helper
    instead of build_adjacency when a script specifically needs to exercise the grid/
    compact-hash-map traversal branch (e.g. as a dual-path regression guard)."""
    kinds = torch.zeros(positions.shape[0], dtype=torch.int32, device=DEVICE)
    grid = buildCompactHashMap(
        positions.detach(), positions.detach(),
        supports.detach(), supports.detach(),
        periodicity=domain.periodic,
        domainDescription=domain,
        mode=mode,
    )
    return grid, kinds


def compute_densities(positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, kinds: torch.Tensor, domain: DomainDescription, adjacency, mode=SupportScheme.Gather):
    """Realistic per-particle densities via the (separately, upstream-verified)
    Density op, detached and re-leafed as an independent gradcheck input --
    this module's own backward is not what's under test here, only realistic
    magnitudes for whatever consumes densities downstream."""
    from warpSPHCore import OperationProperties, warpOperation
    from warpSPHCore.enumTypes import OperationDirection, WarpOperation

    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    rho = warpOperation(
        p,
        OperationProperties(kernel=KERNEL, operation=WarpOperation.Density, supportMode=mode, operationMode=OperationDirection.AllToAll),
        domain,
        adjacency=adjacency,
    )
    return rho.detach().clone().requires_grad_(True)


# --------------------------------------------------------------------------
# Repo-local helpers: this repo's per-scheme state objects. Tier 0
# (wp_surfaceAware.py, sdf.py/domainSDF.py) needed nothing beyond a bare
# ParticleState; Tier 1's compSPH/CRK/dissipation kernels read `.velocities`,
# `.internalEnergies`, `.pressures`, `.soundspeeds`, `.alphas` off their
# `queryParticles` via `hasattr(...)` duck-typing, which a bare ParticleState
# doesn't carry. Real call sites pass a full CompSPHState/CompressibleState
# (see systems/compSPH.py, systems/compressibleMonaghan.py); both declare the
# same required fields; CompressibleState is used here since it's already
# what scripts/troubleshoot_balanceTerm_segfault.py proved works standalone.
# --------------------------------------------------------------------------

def make_compressible_state(positions, supports, masses, densities, velocities, internalEnergies, *, pressures=None, soundspeeds=None, alphas=None, kinds=None):
    """Builds a real CompressibleState from independent differentiable
    leaves, rather than deriving pressures/soundspeeds/densities through the
    actual physics pipeline (EOS, adjacency, ...) -- consistent with this
    file's operator-level gradcheck philosophy: check each module's own
    backward against its direct inputs, not the full simulation's
    self-consistency. `pressures`/`soundspeeds`/`alphas` left None reproduce
    the "not provided" branch each module's own `hasattr(...)` check takes
    (individual_cs/viscositySwitch/explicitPressure all False)."""
    from warpSPH.systems.compressibleMonaghan import CompressibleState

    n = positions.shape[0]
    if kinds is None:
        kinds = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    state = CompressibleState(
        positions=positions,
        velocities=velocities,
        supports=supports,
        masses=masses,
        densities=densities,
        kinds=kinds,
        materials=torch.zeros(n, dtype=torch.int32, device=DEVICE),
        UIDs=torch.arange(n, dtype=torch.int64, device=DEVICE),
        UIDcounter=n,
        internalEnergies=internalEnergies,
    )
    state.pressures = pressures
    state.soundspeeds = soundspeeds
    state.alphas = alphas
    return state


def compute_crk_state(positions: torch.Tensor, supports: torch.Tensor, masses: torch.Tensor, kinds: torch.Tensor, domain: DomainDescription, adjacency, kernel=KERNEL):
    """CRK correction terms (A, B, gradA, gradB) plus the apparent-volume and
    CRK-consistency density, via warpSPHCore's own computeCRKFactors (already
    gradchecked upstream by warpSPHCore's own gradcheck_crk_native.py). Built
    once from detached positions and returned frozen -- same
    non-differentiable/frozen contract as build_adjacency: this op's own
    backward is not what a caller of this helper is testing."""
    from warpSPHCore import computeCRKFactors

    p = ParticleState(positions=positions.detach(), supports=supports.detach(), masses=masses.detach(), densities=None, kinds=kinds)
    apparentVolume, densities, crkState = computeCRKFactors(p, domain, kernel, adjacency=adjacency)
    return apparentVolume.detach().clone(), densities.detach().clone(), crkState
