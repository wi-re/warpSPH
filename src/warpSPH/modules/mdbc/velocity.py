"""mDBC boundary-particle velocity conditions.

Computes ghost/boundary-particle velocities by dispatching, per boundary
material, to one of several `BCType` policies read from region config: zero,
constant (unchanged), no-slip (mirrored fluid velocity), free-slip
(tangential-only reflection), or "extended" (MLS extrapolation via
`interpolateLiuLiu`). Each non-extended policy interpolates fluid velocities
to ghost points via a Shepard-normalized SPH gather. No-ops (returns
`currentState.velocities` unchanged) when there are no boundary particles.

Two measured deviations from the published mDBC forms, neither fixed -- see
`DFSPH_IMPROVEMENT_PLAN.md` Part 9's addendum and
`scripts/probe_boundaryVelocityModes.py --mode verify`, which grades both
conditions against the wall normal:

- **Both slip conditions project the normal component out rather than
  reflecting it.** Decomposed against the wall normal, `noSlip` measures
  (normal, tangential) = (0, -1) and `freeSlip` (0, +1), where the published
  forms are (-1, -1) and (-1, +1). The tangential half is exactly right in
  both; the boundary particle simply never opposes an approaching fluid
  particle's normal velocity, so a wall contributes half the compression
  signal it should to the SPH divergence. Running the reflecting form instead
  was measured and is *worse* on the bounded DFSPH case, with or without
  `mdbcNoPenetrationShift` -- so this is recorded, not "fixed" blind.
- **`noSlip`'s `2 * u_wall` term is dead.** It reads `currentState.velocities`
  at the *ghost* rows, and nothing writes a body velocity there except
  `rigidBody/update.py`'s `BCType.constant` branch, so on a moving no-slip wall
  it degenerates to a stationary one. `lidDrivenCavity` does not show it
  because `enforceDirichlet` runs after this function and re-imposes the lid
  velocity on the boundary rows.
"""

import warp as wp
from warp.types import vector, matrix
from typing import Any
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Optional, Union, Tuple
from warpSPHCore import *

__all__ = ['computeBoundaryVelocities']



from warpSPH.configurations.moduleConfigurations.boundaryConditions import BCType
from warpSPH.configurations.region import RegionType
from warpSPH.configurations.simulationConfig import SimulationConfig
from ...enumTypes import *
from ...configurations.moduleConfigurations.gravity import GravityType, gravityConfiguration
from ...configurations.weaklyCompressible import WeaklyCompressibleSPHConfig

from ..liu import interpolateLiuLiu
from ._util import stateHasBoundaryParticles


def noSlip(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    """
    Computes the no-slip boundary condition for ghost particles based on the velocities of fluid particles.
    """
    # Interpolate fluid velocities to ghost particles
    qVel = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = currentState.velocities,
    )
    
    # Compute Shepard values for normalization
    shepValue = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = torch.ones_like(currentState.densities),
    )

    # Normalize interpolated velocities
    qVel = qVel / (shepValue.view(-1,1) + 1e-7)

    # The no-slip condition. Two caveats, both measured -- see this module's
    # docstring: `bodyVelocity` is read at the *ghost* rows, where nothing
    # writes a moving body's velocity, and the `projected_vels` step below
    # drops the normal component instead of reflecting it, so what this
    # actually returns is `-tangential(u_f)`.
    bodyVelocity = currentState.velocities
    bIndices = currentState.ghostIndices[currentState.kinds == 2]
    u_g = 2 * bodyVelocity - qVel

    r_ib = torch.linalg.norm(currentState.ghostOffsets, dim=-1)
    n_b = currentState.ghostOffsets / (r_ib.view(-1,1) + 1e-7)
    projected_vels = u_g - torch.einsum('nd, nd -> n', u_g, n_b).view(-1,1) * n_b

    u_g[bIndices,:] = projected_vels[currentState.kinds == 2,:]

    return u_g

def freeSlip(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    """
    Computes the free-slip boundary condition for ghost particles based on the velocities of fluid particles.
    """
    # Interpolate fluid velocities to ghost particles
    qVel = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = currentState.velocities,
    )
    
    # Compute Shepard values for normalization
    shepValue = warpOperation(
        currentState,
        OperationProperties(
            kernel = config.kernel,
            operation = WarpOperation.Interpolate,
            supportMode = SupportScheme.Gather,
            operationMode = OperationDirection.FluidToGhost
        ),
        domain = config.domain,
        adjacency = adjacency,
        queryValues = torch.ones_like(currentState.densities),
    )

    # Normalize interpolated velocities
    qVel = qVel / (shepValue.view(-1,1) + 1e-7)

    # The free-slip condition. NOTE: the published form is
    # `u_g = u_f - 2 * (u_f . n_b) * n_b` (reflect the normal component); what
    # is computed below is `u_f - 1 * (u_f . n_b) * n_b` (project it out), so
    # the boundary particle ends up with no normal velocity rather than the
    # reversed one. Measured and left as is -- see this module's docstring.
    bodyVelocity = currentState.velocities
    bIndices = currentState.ghostIndices[currentState.kinds == 2]
    
    r_ib = torch.linalg.norm(currentState.ghostOffsets, dim=-1)
    n_b = currentState.ghostOffsets / (r_ib.view(-1,1) + 1e-7)

    projected_vels = qVel - torch.einsum('nd, nd -> n', qVel, n_b).view(-1,1) * n_b
    projected_vels[bIndices,:] = projected_vels[currentState.kinds == 2,:]


    return projected_vels

def extendedVelocity(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:

    uvs = [currentState.velocities[:,d] for d in range(currentState.velocities.shape[1])]

    extendedVelocities = [interpolateLiuLiu(
        currentState.positions[currentState.kinds == 2],
        referenceParticles = currentState,
        referenceQuantities = uv,
        config = config,
        neighbor_threshold = 4,
        direction = OperationDirection.FluidToGhost,
        supportScale = 1.0
    ) for uv in uvs] # res[:,0], res[:,1:], neighCounts, A_g, b, wellConditioned


    ghostMask = currentState.kinds == 2
    relPos = currentState.ghostOffsets[ghostMask]

    velocities = [currentState.velocities.new_zeros(currentState.velocities.shape[0], device = currentState.velocities.device, dtype = currentState.velocities.dtype) for uv in extendedVelocities]

    for d in range(currentState.velocities.shape[1]):
        u_interp, u_interp_grad, numNeighbors, A_g, b, wellConditioned = extendedVelocities[d]

        shepardNominator = b[:,0]
        shepardDenominator = A_g[:,0,0]
        shepardDensity = shepardNominator / shepardDenominator
        bIndices = currentState.ghostIndices[ghostMask]

        vel = velocities[d]
        vel[bIndices] = torch.where(numNeighbors > 0, shepardDensity, vel[bIndices])

        # `wellConditioned` (interpolateLiuLiu's neighbour-count-AND-determinant
        # gate, DELTASPH_VALIDATION_PLAN.md Part 3) replaces a bare
        # `numNeighbors > 9`. Also fixed: the fallback branch read
        # `vel[ghostMask]` (always zero -- ghost rows are never written here)
        # instead of `vel[bIndices]`, which discarded the Shepard value just
        # written above for every ill-conditioned/low-neighbour row.
        vel[bIndices] = torch.where(wellConditioned, (u_interp - torch.einsum('nu, nu -> n',(-relPos), u_interp_grad)), vel[bIndices])

    extendedVelocities = torch.stack(velocities, dim = -1)
    return extendedVelocities


def constantVelocity(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    return currentState.velocities

def zeroVelocity(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    return currentState.velocities.new_zeros(currentState.velocities.shape[0], currentState.velocities.shape[1], device = currentState.velocities.device, dtype = currentState.velocities.dtype)


def computeBoundaryVelocities(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    if not stateHasBoundaryParticles(currentState, config):
        return currentState.velocities
    with record_function("[warpSPH] - (mdbc) - computeBoundaryVelocities"):

        materials = currentState.materials[currentState.kinds == 1]
        uniqueMaterials = torch.unique(materials)


        boundaryRegions = [region for region in config.regions if region.type == RegionType.Boundary]
        kinds = [region.kind for region in boundaryRegions]
        ghostMask = currentState.kinds == 2
        boundaryMask = currentState.kinds == 1

        BCTypes = [region.kind for region in boundaryRegions]
        BCTypesT = torch.tensor([bct.value for bct in BCTypes], device = currentState.velocities.device, dtype = torch.int64)

        boundaryMaterials = currentState.materials.clone()
        boundaryMaterials[currentState.kinds != 1] = 0
        BCmask = BCTypesT[boundaryMaterials.long()]
        
        outputVelocities = currentState.velocities.clone()

        if BCType.zeros in BCTypes:
            zeroVelocities = zeroVelocity(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.zeros.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying zero boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, zeroVelocities, outputVelocities)
        if BCType.constant in BCTypes:
            constantVelocities = constantVelocity(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.constant.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying constant boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, constantVelocities, outputVelocities)
        if BCType.extended in BCTypes:
            extendedVelocities = extendedVelocity(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.extended.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying extended boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, extendedVelocities, outputVelocities)        
        if BCType.noSlip in BCTypes:
            noSlipVelocities = noSlip(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.noSlip.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying no-slip boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, noSlipVelocities, outputVelocities)            
        if BCType.freeSlip in BCTypes:
            freeSlipVelocities = freeSlip(currentState, config, schemeConfig, adjacency)
            mask = BCmask.view(-1,1) == BCType.freeSlip.value
            mask = mask & boundaryMask.view(-1,1)
            # print("Applying free-slip boundary condition to", torch.sum(mask).item(), "particles.")
            outputVelocities = torch.where(mask, freeSlipVelocities, outputVelocities)

        return outputVelocities
            


        # qVel = warpOperation(
        #     currentState,
        #     OperationProperties(
        #         kernel = config.kernel,
        #         operation = WarpOperation.Interpolate,
        #         supportMode = SupportScheme.Gather,
        #         operationMode = OperationDirection.FluidToGhost
        #     ),
        #     domain = config.domain,
        #     adjacency = adjacency,
        #     queryValues = currentState.velocities,
        # )
        # shepValue = warpOperation(
        #     currentState,
        #     OperationProperties(
        #         kernel = config.kernel,
        #         operation = WarpOperation.Interpolate,
        #         supportMode = SupportScheme.Gather,
        #         operationMode = OperationDirection.FluidToGhost
        #     ),
        #     domain = config.domain,
        #     adjacency = adjacency,
        #     queryValues = torch.ones_like(currentState.densities),
        # )

        # qVel = qVel / (shepValue.view(-1,1) + 1e-7)

        # bodyVelocity = currentState.velocities

        # bIndices = currentState.ghostIndices[currentState.kinds == 2]

        # u_g = 2 * bodyVelocity - qVel

        # r_ib = torch.linalg.norm(currentState.ghostOffsets, dim = -1)
        # n_b = currentState.ghostOffsets / (r_ib.view(-1,1) + 1e-7)

        # projected_vels = u_g - torch.einsum('nd, nd -> n', u_g, n_b).view(-1,1) * n_b

        # boundaryVelocities = currentState.velocities.clone()
        # boundaryVelocities[bIndices] = u_g[ghostMask]

        # projectedVelocities = currentState.velocities.clone()
        # projectedVelocities[bIndices] = projected_vels[ghostMask]

        # return boundaryVelocities, projectedVelocities