"""Shared building blocks for the viscosity switches: CRKSPH gradient-correction matrix, shear/rotation tensor, and a plain divergence estimator.

``computeM`` is the CRKSPH gradient-renormalization matrix (its caller inverts
it with ``torch.linalg.pinv`` to get ``M_inv``); ``computeShearTensor``
returns the velocity-gradient divergence, trace-free symmetric (shear) and
antisymmetric (rotation) parts, optionally corrected by that matrix.
``computeDivergence`` picks between a naive SPH divergence and the
Cullen-corrected one via ``schemeConfig.viscositySwitchParams.divergenceScheme``.
"""

from warpSPHCore import *
from ...systems.compressibleMonaghan import CompressibleState
from ...configurations import *
from typing import Union, Tuple, Dict, Optional
# from ...enumTypes import SupportScheme
import torch
from .wp_computeM import computeMWarp

__all__ = ['computeM', 'computeShearTensor', 'computeDivergence']

# Correction Term from CRKSPH
def computeM(
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None
    ):
    return -computeMWarp(
        particleState,
        operationProperties=OperationProperties(
            kernel = simulationConfig.kernel,
            supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode,
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
    )
    # kernelValues = neighborhood[1]
    # # CHECK THIS, does it really use the term without densities? # requires custom op
    # return -SPHOperation(particles, kernelValues.x_ij * particles.densities[neighborhood[0].col].view(-1,1), kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Naive)
    
    # # i, j        = neighborhood.row, neighborhood.col
    # # h_i         = particles_a.supports[i]
    # # m_j         = particles_b.masses[j]
    # # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    
    # # gradW_i = kernel.jacobian(x_ij, h_i)   

    # # dyadicProduct = m_j.view(-1,1,1) * torch.einsum('ij,ik->ijk', x_ij, gradW_i)
    
    # # return -scatter_sum(dyadicProduct, i, dim = 0, dim_size = particles_a.positions.shape[0])

def computeShearTensor(
        correctionMatrix: torch.Tensor,
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode
    Vs = warpOperation(
        particleState,
        OperationProperties(
            kernel = simulationConfig.kernel,
            operation = WarpOperation.Gradient,
            supportMode = supportMode,
            gradientMode = GradientScheme.Difference
        ),
        domain = simulationConfig.domain,
        adjacency = adjacency,
        queryValues=particleState.velocities
    )

    # Vs = SPHOperation(particles, particles.velocities, kernel, neighborhood[0], neighborhood[1], operation = Operation.Gradient, supportScheme=supportScheme, gradientMode=GradientMode.Difference)
    
    # i, j        = neighborhood.row, neighborhood.col
    # v_i, v_j    = particles_a.velocities[i], particles_b.velocities[j]
    # m_j         = particles_b.masses[j]
    # rho_j       = particles_b.densities[j]
    # h_i         = particles_a.supports[i]
    # x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)

    # gradW_i = kernel.jacobian(x_ij, h_i)
    # v_ij = v_i - v_j
    
    # dyadicProduct = -(m_j / rho_j).view(-1,1,1) * torch.einsum('ij,ik->ijk', v_ij, gradW_i) # don't forget the minus sign! (B2) computes D_i in the Cullen paper but that is not the quantity we are interested in. Instead we care about v\otimes\nabla W_ij which requires a minus sign.
    
    # Vs = scatter_sum(dyadicProduct, i, dim = 0, dim_size = particles_a.positions.shape[0])
    if schemeConfig.viscositySwitchParams.correctVelocityGradient:
        if correctionMatrix is not None:
            # Vs[c, g] = dv_c/dx_g and, for v = A x, Vs = A M: the correction acts on the gradient index, Vs M^-1
            # (M^-1 Vs = M^-1 A M kept the trace but not the shear / rotation, OPEN_PROBLEMS §22)
            Vs = torch.einsum('ijk, ikl -> ijl', Vs, correctionMatrix)
    trace = torch.einsum('...ii', Vs)
    
    traces = torch.eye(Vs.shape[1], device=Vs.device) * trace.view(-1, 1, 1) / Vs.shape[1]
    
    Shear = (Vs + Vs.transpose(1,2))/2 - traces
    Rotation = (Vs - Vs.transpose(1,2)) / 2
    
    return trace, Shear, Rotation

# from ..momentum.consistent

def computeDivergence(
        particleState: CompressibleState,
        simulationConfig: SimulationConfig,
        schemeConfig: CompressibleSPHConfig,
        supportScheme: Optional[SupportScheme] = None,
        adjacency: Optional[Union[AdjacencyList, CompactHashMap]] = None):
    supportMode = supportScheme if supportScheme is not None else simulationConfig.supportMode
    viscousDivergenceScheme = schemeConfig.viscositySwitchParams.divergenceScheme
    
    if viscousDivergenceScheme == 'naive':
        return warpOperation(
            particleState,
            operationProperties=OperationProperties(
                kernel = simulationConfig.kernel,
                operation = WarpOperation.Divergence,
                supportMode = supportMode,
                gradientMode = GradientScheme.Difference
            ),
            domain = simulationConfig.domain,
            adjacency = adjacency,
            queryValues=particleState.velocities
        )
        # return sph_op(psphState, psphState, domain, wrappedKernel, actualNeighbors, 'superSymmetric', 'divergence', 'difference', quantity=(psphState.velocities, psphState.velocities))
    elif viscousDivergenceScheme == 'cullen':
        if schemeConfig.viscositySwitchParams.correctVelocityGradient:
            M = computeM(particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
            M_inv = torch.linalg.pinv(M)        
        else:
            M = None
            M_inv = None
        div, S, Rot = computeShearTensor(M_inv, particleState, simulationConfig, schemeConfig, supportScheme, adjacency)
        return div
    