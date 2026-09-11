"""mDBC boundary-particle velocity conditions.

Computes ghost/boundary-particle velocities by dispatching, per boundary
material, to one of several `BCType` policies read from region config: zero,
constant (unchanged), no-slip (mirrored fluid velocity), free-slip
(tangential-only reflection), or "extended" (MLS extrapolation via
`interpolateLiuLiu`). Each non-extended policy interpolates fluid velocities
to ghost points via a Shepard-normalized SPH gather. No-ops (returns
`currentState.velocities` unchanged) when there are no boundary particles.

Both slip conditions are written on the **relative** velocity
`w = Shepard(u_fluid) - u_body` and add the wall's own velocity back
(`_ghostBodyVelocity`), so the boundary particle's normal component tracks the
wall rather than being pinned to zero:

    noSlip    u_g = u_body - w_t             (normal component = u_body . n)
    freeSlip  u_g = u_body + w_t - w_n       (published form: normal reflected)

That matters for the continuity equation, not just for advection: a wall moving
into the fluid has to enter `div(v)` with its own normal velocity, otherwise the
density change the wall drives is missing while gravity's forcing is not, and
the two diverge. diffSPH's delta-SPH does the same thing by a different route --
it restores `boundaryBodyVelocities` before `computeMomentum`
(`schemes/deltaSPH.py:169`) instead of folding the body velocity into the BC.

Graded against the wall normal (`scripts/probe_boundaryVelocityModes.py --mode
verify`) the decomposition (normal, tangential) is now `freeSlip` **(-1, +1)**,
the published form, and `noSlip` **(0, -1)**.

**`freeSlip` reflects the fluid's normal component as of
`DELTASPH_VALIDATION_PLAN.md` 5.7.** It used to project it out (0, +1), so the
wall never opposed an approaching fluid particle and contributed half the
compression signal it should to the SPH divergence. Measured cost on Marrone
3.1: with a non-reflecting, non-slipping bed the dam-break tongue thinned to
2.8 dx and shed fliers 8 dx ahead of the front, which reached the far wall
early and drove the P1 probe to ~10x the reference; the bed also went to
*negative* pressure under the front (an attracting wall). Switching this case's
wall to free slip restored the tongue to 10-14 dx and halved the flier lead.

**`noSlip` still projects rather than reflects** ((0, -1) against the published
(-1, -1)) -- deliberately, not overlooked. `DFSPH_IMPROVEMENT_PLAN.md` Part 9's
addendum measured the reflecting form as *worse* on the bounded DFSPH case, with
or without `mdbcNoPenetrationShift`. Revisit it with that case in hand, not
blind.

For a stationary wall (`u_body = 0`), `noSlip` is bit-for-bit what this module
computed before; `freeSlip` is not, by design.
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


def _shepardFluidVelocity(currentState: Any, config: SimulationConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    """Shepard-normalized fluid velocity gathered onto the ghost nodes.

    Rows other than `kinds == 2` come back zero: the gather runs
    `FluidToGhost`, so only ghost rows accumulate.
    """
    props = OperationProperties(
        kernel = config.kernel,
        operation = WarpOperation.Interpolate,
        supportMode = SupportScheme.Gather,
        operationMode = OperationDirection.FluidToGhost,
    )
    qVel = warpOperation(
        currentState, props, domain = config.domain, adjacency = adjacency,
        queryValues = currentState.velocities,
    )
    shepValue = warpOperation(
        currentState, props, domain = config.domain, adjacency = adjacency,
        queryValues = torch.ones_like(currentState.densities),
    )
    return qVel / (shepValue.view(-1,1) + 1e-7)


def _ghostBodyVelocity(currentState: Any, schemeConfig: Any) -> torch.Tensor:
    """The wall's own rigid-body velocity, broadcast onto its ghost nodes.

    **Only particles owned by a `RigidBody` get a non-zero value; everything
    else is zero.** That restriction is the whole subtlety. The BC arithmetic
    runs on ghost rows, but a body velocity is written on the *boundary* rows
    (`rigidBody/update.py`), so this has to map across via `ghostIndices` --
    and it is tempting to just read `currentState.velocities` there. That is
    wrong: for a wall with no rigid body those rows hold whatever the last
    step's BC left behind, not a prescribed wall motion, so feeding it back in
    as `u_body` makes the condition compound on its own output. Measured:
    `scripts/probe_boundaryVelocityModes.py --mode verify` graded `freeSlip`'s
    normal slope at **-3** instead of -1, because `u_g . n = 2(u_body . n) -
    f . n` with `u_body . n` already equal to `-f . n` from the previous step.

    This term was historically dead (it read the ghost rows, which are always
    zero), which is why the compounding never showed up before it was fixed.
    """
    ghostMask = currentState.kinds == 2
    bodyVelocity = torch.zeros_like(currentState.velocities)
    bodies = getattr(schemeConfig, 'rigidBodies', None) or []
    if bodies:
        owned = torch.zeros(currentState.velocities.shape[0], dtype=torch.bool,
                            device=currentState.velocities.device)
        for body in bodies:
            idx = getattr(body, 'particleIndices', None)
            if idx is not None:
                owned[idx] = True
        bodyVelocity = torch.where(owned.view(-1, 1), currentState.velocities,
                                   bodyVelocity)
    bodyVelocity[ghostMask] = bodyVelocity[currentState.ghostIndices[ghostMask]]
    return bodyVelocity


def _wallNormals(currentState: Any) -> torch.Tensor:
    r_ib = torch.linalg.norm(currentState.ghostOffsets, dim=-1)
    return currentState.ghostOffsets / (r_ib.view(-1,1) + 1e-7)


def noSlip(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    """
    Computes the no-slip boundary condition for ghost particles based on the velocities of fluid particles.
    """
    qVel = _shepardFluidVelocity(currentState, config, adjacency)

    # No-slip on the *relative* velocity: reverse the fluid's tangential slip
    # about the wall, then add the wall's own velocity back, so the normal
    # component is `u_body . n` rather than 0. At `u_body = 0` this is
    # `-tangential(u_f)`, bit-for-bit what this function returned before.
    # The fluid's normal component stays projected out rather than reflected --
    # a separate, measured deviation; see the module docstring.
    bodyVelocity = _ghostBodyVelocity(currentState, schemeConfig)
    bIndices = currentState.ghostIndices[currentState.kinds == 2]
    n_b = _wallNormals(currentState)

    w = qVel - bodyVelocity
    w_t = w - torch.einsum('nd, nd -> n', w, n_b).view(-1,1) * n_b
    u_g = bodyVelocity - w_t

    out = currentState.velocities.clone()
    out[bIndices,:] = u_g[currentState.kinds == 2,:]

    return out

def freeSlip(currentState: Any, config: SimulationConfig, schemeConfig: WeaklyCompressibleSPHConfig, adjacency: Optional[Union[AdjacencyList, CompactHashMap]]) -> torch.Tensor:
    """
    Computes the free-slip boundary condition for ghost particles based on the velocities of fluid particles.
    """
    qVel = _shepardFluidVelocity(currentState, config, adjacency)

    # Free slip, published form, on the *relative* velocity: keep the fluid's
    # tangential share and **reflect** its normal component, then add the
    # wall's own velocity back.
    #
    #     u_g = u_body + w_t - w_n ,   w = Shepard(u_f) - u_body
    #
    # Decomposed against the wall normal that is (normal, tangential) =
    # (-1, +1), which is what `scripts/probe_boundaryVelocityModes.py --mode
    # verify` grades against. This used to *project* the normal component out
    # (`u_g = u_body + w_t`, giving (0, +1)), so the wall never opposed an
    # approaching fluid particle and contributed half the compression signal it
    # should to the SPH divergence. Measured consequence on Marrone 3.1
    # (`DELTASPH_VALIDATION_PLAN.md` 5.7): under a dragged bed the dam-break
    # tongue thinned to 2.8 dx and shed fliers 8 dx ahead of the front, which
    # slammed the far wall early and drove the P1 probe to 10x the reference.
    bodyVelocity = _ghostBodyVelocity(currentState, schemeConfig)
    bIndices = currentState.ghostIndices[currentState.kinds == 2]
    n_b = _wallNormals(currentState)

    w = qVel - bodyVelocity
    w_n = torch.einsum('nd, nd -> n', w, n_b).view(-1,1) * n_b
    u_g = bodyVelocity + (w - w_n) - w_n          # = u_body + w_t - w_n

    out = currentState.velocities.clone()
    out[bIndices,:] = u_g[currentState.kinds == 2,:]

    return out

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