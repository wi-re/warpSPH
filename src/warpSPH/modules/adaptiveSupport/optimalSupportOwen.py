"""Owen (1998)-style adaptive smoothing length: table-lookup mapping the local kernel-sum "psi" statistic to a target neighbor count.

Each iteration rebuilds the Verlet neighbor list, evaluates the psi_0/psi_0_H
statistics (``computePsi0Warp``), looks up the corresponding neighbor count
``n_h`` from a per-``(kernel, dim)`` lookup table (``owenLUT.computeOwen``,
cached in module-level ``PsiLUTs``), and blends toward the new support with a
ratio-dependent relaxation factor ``a`` in ``computeNewSupport`` (``a =
0.4*(1+s**-3)`` growing, ``0.4*(1+s**2)`` shrinking) rather than jumping
straight to the target. Supports are clamped to ``[hMin*0.25, hMax*4.0]``
each iteration; iteration stops early once the relative support change drops
below ``compConfig.adaptiveSupportThreshold``. Note: this file's own
``n_h_to_nH``/``nH_to_n_h`` are local re-implementations that shadow the
otherwise-canonical versions in ``warpSPH.utils.support`` -- not re-exported
here to avoid that ambiguity spreading further.
"""

from ...systems import CompressibleState
from ...configurations import SimulationConfig, CompressibleSPHConfig
import numpy as np
import torch
from types import SimpleNamespace
from warpSPHCore import *
from typing import Optional, Union

from .wp_psi0 import computePsi0Warp
from .owenLUT import computeOwen, interpolateLUT

from torch.profiler import profile, ProfilerActivity

from warpSPHCore.profiling import record_function
__all__ = ['evaluateOptimalSupportOwen']

#: Owen's psi lookup table, cached per ``(kernel, dim)``.
#:
#: This used to be a single unkeyed global, built on the first call and reused
#: for every call after it -- but `computeOwen` slices the table by dimension
#: (`psi[:, dim-1]`) and generates it from one specific kernel, so a process
#: that touched two dimensions got the *first* one's table for both, and its
#: supports came out silently wrong. Nothing caught it because every case in
#: the repo ran in 1D or 2D but never both in one process, until `sod3d`
#: arrived and the test suite started building a 1D and a 2D Sod back to back.
#: Keyed like this, a single-dimension process behaves exactly as before.
PsiLUTs = {}

# assumes convention from CRKSPH with $\eta$ = 1
def n_h_to_nH(n_h, dim):
    spacing = 1 / n_h
    v = spacing**dim
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    return vH / v

def nH_to_n_h(nH, dim):
    vH = 2.0 if dim == 1 else (np.pi if dim == 2 else (4/3) * np.pi)
    v = vH / nH
    return (1 / v)**(1/dim)


def computeNewSupport(target, n_h, h):        
    n_h_Target = target
    s = n_h_Target / n_h
    a = torch.where(s >= 1.0, 0.4 * (1 + s**-3), 0.4 * (1 + s**2))
    h_i_new = (1 - a + a * s) * h
    # h_i_new = s * h
    return h_i_new

def _listFits(adjacency, particles) -> bool:
    """A list the solver may run on without validating it first: built for as many particles as there are now.
    (`buildVerletList` rebuilds on a changed particle count; a solve that skips the per-iteration check must not
    run a kernel on a list sized for other particles -- walls appended after the setup's list, for instance.)"""
    if adjacency is None or getattr(adjacency, 'queryPositions', None) is None or getattr(adjacency, 'referencePositions', None) is None:
        return False
    n = particles.positions.shape[0]
    return adjacency.queryPositions.shape[0] == n and adjacency.referencePositions.shape[0] == n and adjacency.numNeighbors.shape[0] == n


def evaluateOptimalSupportOwen(
        particles: CompressibleState,
        config: SimulationConfig,
        compConfig: CompressibleSPHConfig,
        kernel_: Optional[KernelFunctions] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None,
        supportScheme: Optional[SupportScheme] = None,
        verbose = False,
        validateOnce: bool = False,
        rebuildSlack: float = 0.0,
        computeDensity: bool = True,
        grid = None,
        gridRadius: Optional[float] = None,
        info: Optional[dict] = None,
        ):
    """`validateOnce=False` (default): the neighbour list is validated, and rebuilt if needed, at the start of
    every iteration. `validateOnce=True`: the iterations run on the incoming list without a check (the particles do
    not move, only `h` changes) and the list is validated ONCE afterwards against the largest support any iteration
    used (and the particle motion); if that finds the list too short the list is rebuilt and the iterations are
    run again from the incoming supports. The returned list then covers every iterate and the final supports, so
    the caller needs no check of its own. `rebuildSlack` rebuilds that final check early (see `buildVerletList`).
    `computeDensity=False` skips the closing summation density and returns the particles' own densities instead.
    `grid` (a `CompactHashMap` built for supports up to `gridRadius`): the iterations and the closing density run on
    the grid and no neighbour list is looked at, validated or built here (`adjacency` is handed back untouched). Every
    iterate is limited to `gridRadius` so the grid stays valid; `info['outgrown']` receives a device flag that is true
    when that limit was binding (the caller then makes a grid for a larger radius).
    Also: the early stop on the support change is evaluated on the device (a finished iteration freezes `h`)
    instead of a host read per iteration."""
    with record_function("[warpSPH] - evaluateOptimalSupport - Owen"):
        kernel = kernel_ if kernel_ is not None else config.kernel
        # Note `kernel`, not `kernel_`: the table is built from whichever kernel
        # this call resolved to, so a caller that leaves `kernel_` at None gets
        # a table for `config.kernel` rather than one built from None.
        table = getattr(compConfig, 'owenTable', 'lattice')
        key = (kernel, config.domain.dim, table)
        PsiLUT_fn = PsiLUTs.get(key)
        if PsiLUT_fn is None:
            PsiLUT_fn = PsiLUTs[key] = computeOwen(
                kernel, dim = config.domain.dim, nMin = 2.0, nMax = 6.0, nLUT = 2**12, table = table)
        # particles, domain, kernel, targetNeighbors, PsiLUT_fn, nIter = 16, neighborhood = None, verbose = False,eps = 1e-3, neighborhoodAlgorithm = 'compact'):
        hs = [particles.supports]
        # print(particles)
        psis = []

        supportMode = config.supportMode if supportScheme is not None else SupportScheme.SuperSymmetric
        targetNeighbors = config.targetNeighbors
        dim = config.domain.dim
        
        nhTarget = nH_to_n_h(targetNeighbors, dim = dim)
        verletScale = config.verletScale
        # verletScale = 1.0 if config is None else (config['neighborhood']['verletScale'] if 'verletScale' in config else 1.0)
        hMin = particles.supports.min()
        hMax = particles.supports.max()

        nIter = compConfig.adaptiveSupportIterations
        adaptiveHThreshold = compConfig.adaptiveSupportThreshold
        # verbose = False
        
        suppliedSupports = particles.supports
        peakSupports = particles.supports
        converged = None                       # device flag: an iteration changed h by less than the threshold
        attempt = 0
        i = 0
        onGrid = grid is not None
        deviceStop = validateOnce or onGrid
        outgrown = None
        while i < nIter:
            if verbose:
                print('----------------------------------')
                print(f'Iteration {i}, target: {nhTarget}')
            if onGrid:
                adjacency_ = grid
            elif not validateOnce or not _listFits(adjacency, particles):
                adjacency = buildVerletList(particles, domain = config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=adjacency, verbose=False)
            if not onGrid:
                adjacency_ = adjacency

            
            # psi_0, psi_0_H = computePsi_0(particles, kernel_, neighbors.get('noghost'), supportScheme, config)
            psi_0, psi_0_H = computePsi0Warp(
                particles,
                operationProperties=OperationProperties(
                    kernel = config.kernel,
                    supportMode = SupportScheme.Gather
                ),
                domain = config.domain,
                adjacency = adjacency_
            )
            if verbose:
                print(f'Psi: {psi_0_H.min()} | {psi_0_H.max()} | {psi_0_H.mean()}')
                
            n_h_i = PsiLUT_fn.fromPsiH(psi_0_H).to(dtype = particles.supports.dtype, device = particles.supports.device)
            if verbose:
                print(f'n_h: {n_h_i.min()} | {n_h_i.max()} | {n_h_i.mean()}')
            
            h = computeNewSupport(nhTarget, n_h_i, particles.supports)
            psis.append(psi_0_H)
            h = h.clamp(min = hMin * 0.25, max = hMax * 4.0)
            if onGrid:
                binding = h.max() > gridRadius
                outgrown = binding if outgrown is None else (outgrown | binding)
                h = h.clamp(max = gridRadius)
            hMin = h.min()
            hMax = h.max()
            
            h_ratio = h / particles.supports
            if deviceStop:
                # a converged iteration freezes h (what the early `break` did), without reading the flag on the host
                if converged is not None:
                    h = torch.where(converged, particles.supports, h)
                    h_ratio = torch.where(converged, torch.ones_like(h_ratio), h_ratio)
                peakSupports = torch.maximum(peakSupports, h)
            
            if verbose: 
                print(f'Support: {h.min()} | {h.max()} | {h.mean()}')
                print(f'Ratio: {(h_ratio).min()} | {(h_ratio).max()} | {(h_ratio).mean()}')
                
            particles.supports = h

            hs.append(h)

            if deviceStop:
                done = (h_ratio - 1).abs().max() < adaptiveHThreshold
                converged = done if converged is None else (converged | done)
            elif (h_ratio - 1).abs().max() < adaptiveHThreshold:
                if verbose: 
                    print('Stopping Early')
                # print('Stopping Early')
                break
            i += 1
            if validateOnce and not onGrid and i == nIter and attempt == 0:
                # one validation for the whole solve: the particles with the largest support any iteration used
                checked = buildVerletList(SimpleNamespace(positions=particles.positions, supports=peakSupports), domain = config.domain, verletScale = verletScale,
                                          supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=adjacency,
                                          verbose=False, rebuildSlack=rebuildSlack)
                if checked is not adjacency:
                    # the list was too short for some iterate: use the new one and solve again from the supplied supports
                    adjacency = checked
                    attempt = 1
                    i = 0
                    particles.supports = suppliedSupports
                    peakSupports = suppliedSupports
                    converged = None
                    hs = [suppliedSupports]
                    psis = []

        if info is not None and outgrown is not None:
            info['outgrown'] = outgrown
        if not computeDensity:
            # the caller recomputes its density (in its own sense) right after: hand back the one the particles have
            return particles.densities, hs[-1], adjacency, psis, hs
        densities = warpOperation(
            particles,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Density,
                supportMode = supportScheme,
            ),
            domain = config.domain,
            adjacency = grid if onGrid else adjacency
        )
        
        return densities, hs[-1], adjacency, psis, hs