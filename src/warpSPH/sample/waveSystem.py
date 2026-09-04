"""Assemble a ``WaveSystemv3`` from the grids the case generator produced.

Takes the per-particle ``u``/``v``/``c``/``damping`` grids plus the two id
grids from :func:`warpSPH.caseUtils.waveEquation.gencase.genInitial`, resolves
source ids to amplitudes and obstacle/boundary ids to wave speeds, optionally
smooths the initial conditions, and returns the system together with its
timestep.

This is the last stage of the wave pipeline, not an entry point -- there is no
``Case`` wrapping it, so a caller has to run the earlier stages themselves.
"""

import torch
from ..systems.waveSystem import computeDt, WaveSystemv3, WaveSystemStatev3, sampleInitialWaveState
from ..caseUtils.waveEquation.damping import DampingProfiles, createDampingProfile
from ..caseUtils.waveEquation.sample import smoothValuesWarp
from warpSPHCore import *

def finalizeWaveSystemSetup(
    particleState,
    uGrid, vGrid, cGrid, dampGrid,
    uSourceGrid, cSourceGrid,
    sourceMagnitudes, obstacleSpeeds,
    config, caseConfig,
):


    boundaryIds = torch.unique(cSourceGrid)
    boundaryIds = boundaryIds[boundaryIds != 0]  # Exclude background (0)
    # print("Boundary IDs:", boundaryIds)

    cGrid = torch.full_like(cGrid, caseConfig.defaultSpeed)
    # print(torch.sum(cGrid), '/', cGrid.numel(), ' default: ', caseConfig.defaultSpeed)
    for bId in boundaryIds:
        if bId == -1:
            speed = caseConfig.defaultBoundarySpeed
        else:
            speed = obstacleSpeeds[bId-1]
        # Not `torch.full_like(cGrid, speed)`: `full_like`'s fill value must be
        # a plain Number, so it silently requires `speed` to already be
        # detached. Broadcasting the add keeps the graph when `speed` is a
        # `requires_grad` tensor (see `docs/historic_plans/WAVE_EQUATION_PLAN.md` step 5).
        cGrid = torch.where(cSourceGrid == bId, torch.zeros_like(cGrid) + speed, cGrid)
        # print(f"Set speed {speed} for boundary ID {bId}")
    # print(torch.sum(cGrid))

    uGrid = torch.zeros_like(uGrid)
    sourceIds = torch.unique(uSourceGrid)
    sourceIds = sourceIds[sourceIds != 0]  # Exclude background (0)
    for sId in sourceIds:
        magnitude = sourceMagnitudes[sId-1]  # source IDs are 1-indexed in uSourceGrid
        uGrid = torch.where(uSourceGrid == sId, torch.zeros_like(uGrid) + magnitude, uGrid)
        # print(f"Set magnitude {magnitude} for source ID {sId}")

        
    if caseConfig.smoothICs:
        # if args.verbose:
            # print("Smoothing initial conditions...")
        uGrid = smoothValuesWarp(
            uGrid,
            particleState,
            caseConfig.smoothIterations, None,
            config
        )
    # uGrid[particleState.positions[:,0] > 0] = 0.0

        
    waveState = WaveSystemStatev3(
        positions=particleState.positions,
        supports=particleState.supports,
        masses=particleState.masses,
        densities=particleState.densities,
        
        kinds=torch.zeros(particleState.positions.shape[0], device=config.device, dtype=torch.int32),
        materials=torch.zeros(particleState.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDs=torch.arange(particleState.positions.shape[0], device=config.device, dtype=torch.int32),
        UIDcounter=particleState.positions.shape[0],
        
        u=uGrid,
        v=vGrid,
        c=cGrid,
        damping=dampGrid
    )
    adjacency = buildVerletList(waveState, config.domain, 1.0, SupportScheme.SuperSymmetric, None)


    waveSystem = WaveSystemv3(
        state = waveState,
        adjacency = adjacency,
        domain = config.domain,
        t = torch.tensor(0.0, device=config.device, dtype=config.dtype)
    )

    # print(obstacleSpeeds + [caseConfig.defaultSpeed])
    dt = config.dt if not config.adaptiveDt else computeDt(waveSystem, config, caseConfig, None, obstacleSpeeds, False)
    # config.dt = dt

    return waveSystem, dt


def _wendlandKernelBump(distances: torch.Tensor, radius: float) -> torch.Tensor:
    """A C^2 Wendland (`KernelFunctions.Wendland2`, multi-D branch) radial
    bump, reimplemented directly in torch so it stays autograd-differentiable
    w.r.t. whatever `distances` was computed from (e.g. a source `position`
    tensor). `warpSPHCore`'s own kernel functions
    (`warpSPHCore.kernels.kernelFunctions.wendland2.wendland2_k`, matching
    shape) are `@wp.func`-compiled and not torch-callable outside a full
    neighbour-search operation, so a single-point evaluation goes through
    this instead.
    """
    q = torch.clamp(distances / radius, max=1.0)
    shape = (1.0 - q) ** 4 * (1.0 + 4.0 * q)
    return torch.where(distances < radius, shape, torch.zeros_like(shape))


def sampleSmoothPointSourceWaveSystem(
    nx, config, caseConfig,
    position: torch.Tensor, magnitude: torch.Tensor, radius: float = 0.15,
) -> WaveSystemv3:
    """A plain (no id-grid sources/obstacles) wave system with a single
    kernel-weighted bump stamped onto `u`, centred at `position` and scaled
    by `magnitude`.

    Used for the 1D/3D case variants (`shape_generation.py`'s SDF shapes are
    2D-only) and for checking gradients reach a source's placement: unlike
    the SDF id-grid path, which paints `u` through a `torch.where(sdf < 0,
    ...)` step function (zero gradient w.r.t. position almost everywhere),
    this contributes to `u` directly as a smooth function of `position`, so
    `position` and `magnitude` can be real leaf tensors a caller
    `requires_grad_()`s and get a non-zero gradient back through.
    """
    particleState = sampleInitialWaveState(nx, config, caseConfig)

    if caseConfig.domainDamping:
        particleState.damping = createDampingProfile(
            particleState, config, DampingProfiles.borderDamping_strong)

    distances = torch.linalg.norm(particleState.positions - position, dim=-1)
    particleState.u = magnitude * _wendlandKernelBump(distances, radius)

    adjacency = buildVerletList(particleState, config.domain, 1.0, SupportScheme.SuperSymmetric, None)
    return WaveSystemv3(
        state=particleState,
        adjacency=adjacency,
        domain=config.domain,
        t=torch.tensor(0.0, device=config.device, dtype=config.dtype),
    )