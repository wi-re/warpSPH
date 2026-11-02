"""Monaghan-style adaptive smoothing length: Newton iteration on the implicit constraint h = h(rho(h)).

Each iteration rebuilds the Verlet neighbor list, re-estimates density at the
current support, and takes a Newton step ``h -= F(h)/dF_dh`` where ``F(h) =
h - computeH(rho, m, targetNeighbors, dim)`` and ``dF_dh`` comes from
``computeOmegaWarp`` (the grad-h correction term). Steps are clamped to
``[hMin*0.25, hMax*4.0]`` each iteration and non-finite steps are held at the
previous value; iteration stops early once the relative change in ``h`` drops
below ``compParams.adaptiveSupportThreshold``, or after
``adaptiveSupportIterations``.

``numberDensity=True`` runs the same iteration with every particle mass set to 1, so
``rho`` is the number density ``n_i = sum_j W_ij`` and the constraint is ``h = eta n^(-1/d)``
(Hopkins 2015, App. F2): the one PESPH's grad-h terms (Frontiere 2017 Eqs. G.7-G.9)
differentiate. For equal masses it is the mass-density solve exactly; for unequal masses it
does not depend on the masses at all. It returns ``rho_i = m_i n_i`` as the density, and the
list of per-iteration number densities in place of the density history.
"""

from warpSPH.configurations.compute import neighborSearchOptions

from ...configurations.compressibleConfig import CompressibleSPHConfig

from .wp_omega import computeOmegaWarp
from ...systems.baseState import *
from warpSPHCore import *
from ...configurations import SimulationConfig
from ...utils.support import volumeToSupport, nH_to_n_h
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
from types import SimpleNamespace
from .listValidation import listFits
from typing import Optional
import torch

__all__ = ['evaluateOptimalSupportMonaghan']

def computeH(rho, m, targetNeighbors, dim):
    safe_rho = torch.clamp(torch.nan_to_num(rho, nan=1.0, posinf=1.0, neginf=1.0), min=1e-12)
    V = m / safe_rho
    return volumeToSupport(V, targetNeighbors, dim)
    return targetNeighbors / 2 * V

def F(h, rho, m, targetNeighbors, dim):
    return h - computeH(rho, m, targetNeighbors, dim)
from warpSPHCore import *

def evaluateOptimalSupportMonaghan(
        particleState: BaseState,
        config: SimulationConfig,
        compParams: CompressibleSPHConfig,
        supportScheme: SupportScheme = SupportScheme.Scatter,
        adjacency: Optional[AdjacencyList] = None,
        numberDensity: bool = False,
        validateOnce: bool = False,
        rebuildSlack: float = 0.0,
        computeDensity: bool = True,
        listBuilder = None,
):
    """Validate all Newton iterates once when requested, retaining eager fallback.

    A speculative solve may use an existing list while supports change. Its
    largest support (including the input and output) is checked afterwards. If
    the list was too short, discard the solve and restart from the input with
    per-iteration validation. No state from an invalid list is accepted.
    computeDensity=False skips only the closing sum, not the Newton densities.
    """
    buildList = buildVerletList if listBuilder is None else listBuilder
    if numberDensity:
        realMasses = particleState.masses
        particleState.masses = torch.ones_like(realMasses)
        try:
            n, h, adjacency, ns, supports = evaluateOptimalSupportMonaghan(
                particleState, config, compParams, supportScheme, adjacency,
                validateOnce=validateOnce, rebuildSlack=rebuildSlack, computeDensity=computeDensity, listBuilder=listBuilder)
        finally:
            particleState.masses = realMasses
        return (realMasses * n if computeDensity else n), h, adjacency, ns, supports
    with record_function("[warpSPH] - evaluateOptimalSupport - Monaghan"):
        verletScale = config.verletScale
        nIter = compParams.adaptiveSupportIterations
        hThreshold = compParams.adaptiveSupportThreshold
        iterState = particleState
        originalDensities = iterState.densities
        suppliedSupports = iterState.supports
        speculative = validateOnce
        if speculative and not listFits(adjacency, iterState):
            adjacency = buildList(
                iterState, domain=config.domain, verletScale=verletScale,
                supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=adjacency,
                verbose=False, **neighborSearchOptions(config))

        while True:
            iterState.supports = suppliedSupports
            iterState.densities = originalDensities
            hMin, hMax = suppliedSupports.min(), suppliedSupports.max()
            peakSupports = suppliedSupports
            rhos, supports = [originalDensities], [suppliedSupports]
            retryEager = False
            for i in range(nIter):
                with record_function(f"[evalOS] Iteration {i}"):
                    if not speculative:
                        with record_function("[evalOS] buildVerletList"):
                            adjacency = buildList(
                                iterState,
                                domain=config.domain,
                                verletScale=verletScale,
                                supportMode=SupportScheme.SuperSymmetric,
                                priorNeighborhood=adjacency,
                                verbose=False,
                                **neighborSearchOptions(config))

                    iterState.densities = warpOperation(
                        iterState,
                        OperationProperties(
                            kernel = config.kernel,
                            operation = WarpOperation.Density,
                            supportMode = supportScheme,
                        ),
                        domain = config.domain,
                        adjacency=adjacency
                    )
                    h_prev = iterState.supports

                    safe_rho = torch.clamp(torch.nan_to_num(iterState.densities, nan=1.0, posinf=1.0, neginf=1.0), min=1e-12)
                    F_ = F(h_prev, safe_rho, iterState.masses, targetNeighbors = config.targetNeighbors, dim = config.dim)
                    dFdh_ = computeOmegaWarp(iterState,
                            OperationProperties(
                                kernel = config.kernel,
                                supportMode = supportScheme,
                            ),
                            domain = config.domain,
                            adjacency=adjacency)
                    dFdh_safe = torch.nan_to_num(dFdh_, nan=0.0, posinf=0.0, neginf=0.0)
                    step = F_ / (dFdh_safe + 1e-6)
                    step = torch.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
                    h_new = h_prev - step
                    h_new = torch.where(torch.isfinite(h_new), h_new, h_prev)

                    h_new = h_new.clamp(min = hMin * 0.25, max = hMax * 4.0)
                    hMin = h_new.min()
                    hMax = h_new.max()

                    # Relative *change* must vanish for unchanged positive supports.
                    # Adding an absolute length to h_prev biases this check and
                    # prevents convergence at tight tolerances/small length scales.
                    relativeChange = (h_new - h_prev).abs() / h_prev.abs().clamp_min(torch.finfo(h_prev.dtype).tiny)
                    iterState.supports = h_new
                    if speculative:
                        peakSupports = torch.maximum(peakSupports, h_new)

                    rhos.append(iterState.densities)
                    supports.append(iterState.supports)

                    if speculative:
                        # A bad speculative density must not generate an enormous
                        # fictitious support and then an enormous list rebuild.
                        # Share the existing convergence read with this early
                        # escape; the eager retry starts from the supplied h.
                        done, outside = torch.stack((relativeChange.max() < hThreshold,
                            (h_new > adjacency.querySupports).any())).tolist()
                        if outside:
                            retryEager = True
                            break
                        if done:
                            break
                    elif relativeChange.max() < hThreshold:
                        break

            if retryEager:
                speculative = False
                continue
            checked = buildList(
                SimpleNamespace(positions=iterState.positions,
                                supports=peakSupports if speculative else iterState.supports),
                domain=config.domain, verletScale=verletScale,
                supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=adjacency,
                verbose=False, rebuildSlack=rebuildSlack, **neighborSearchOptions(config))
            if speculative and checked is not adjacency:
                adjacency = checked
                speculative = False
                continue
            adjacency = checked
            break

        if computeDensity:
            densities = warpOperation(
                iterState,
                OperationProperties(kernel=config.kernel, operation=WarpOperation.Density,
                                    supportMode=supportScheme),
                domain=config.domain, adjacency=adjacency)
        else:
            densities = originalDensities
        particleState.densities = originalDensities
        return densities, iterState.supports, adjacency, rhos, supports
