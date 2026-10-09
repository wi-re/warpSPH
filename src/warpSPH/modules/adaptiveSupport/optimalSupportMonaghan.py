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

from ...configurations.compressibleConfig import CompressibleSPHConfig

from .wp_omega import computeOmegaWarp
from ...systems.baseState import *
from warpSPHCore import *
from ...configurations import SimulationConfig
from ...utils.support import volumeToSupport, nH_to_n_h
from torch.profiler import profile, ProfilerActivity
from warpSPHCore.profiling import record_function
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
):
    if numberDensity:
        realMasses = particleState.masses
        particleState.masses = torch.ones_like(realMasses)
        try:
            n, h, adjacency, ns, supports = evaluateOptimalSupportMonaghan(
                particleState, config, compParams, supportScheme, adjacency)
        finally:
            particleState.masses = realMasses
        return realMasses * n, h, adjacency, ns, supports
    with record_function("[warpSPH] - evaluateOptimalSupport - Monaghan"):
        rhos = [particleState.densities]
        supports = [particleState.supports]

        # verletScale = 2**(1/particleState.positions.shape[1])
        verletScale = config.verletScale
        # nIter = 16
        nIter = compParams.adaptiveSupportIterations
        # hThreshold = 1e-3
        hThreshold = compParams.adaptiveSupportThreshold

        hMin = particleState.supports.min()
        hMax = particleState.supports.max()

        iterState = particleState#.initializeNewState()
        originalDensities = iterState.densities.clone()
        # adjacency = None

        for i in range(nIter):
            with record_function(f"[evalOS] Iteration {i}"):
                with record_function("[evalOS] buildVerletList"):
                    adjacency = buildVerletList(iterState, domain = config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=adjacency, verbose=False)

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

                h_diff = h_new - h_prev
                h_ratio = h_new / (h_prev + 1e-6)
                iterState.supports = h_new

                rhos.append(iterState.densities)
                supports.append(iterState.supports)
                        
                # print(f'Iteration: {i} | h_ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
                # print(f'Densities: {iterState.densities.min()} | {iterState.densities.max()} | {iterState.densities.mean()}')
                # print(f'Supports: {h_new.min()} | {h_new.max()} | {h_new.mean()}')
                # print(f'Iteration: {i} | Support: {h_new.min()} | {h_new.max()} | {h_new.mean()} | Ratio: {h_ratio.min()} | {h_ratio.max()} | {h_ratio.mean()}')
                if (h_ratio - 1).abs().max() < hThreshold:
                    # print('Stopping Early')
                    break

        adjacency = buildVerletList(iterState, domain = config.domain, verletScale = verletScale, supportMode = SupportScheme.SuperSymmetric, priorNeighborhood=adjacency, verbose=False)
        densities = warpOperation(
            iterState,
            OperationProperties(
                kernel = config.kernel,
                operation = WarpOperation.Density,
                supportMode = supportScheme,
            ),
            domain = config.domain,
            adjacency=adjacency
        )
        particleState.densities = originalDensities
        return densities, iterState.supports, adjacency, rhos, supports