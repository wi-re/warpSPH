"""Adaptive timestep for the meshless (MFM / MFV) system (COMPRESSIBLE_SPH_PLAN E2).

The per-particle signal timestep `2 C L / v_sig` (Hopkins 2015 Eq. 24-25; GIZMO's
`core/timestep.cc`), the global minimum. `L` is GIZMO's particle size `V^(1/d)`
(`cflLength='particle'`, the default) or the core backend's kernel radius `h`
(`cflLength='support'`). `MeshlessWarp.timestep` reads only the
neighbourhood and the primitives -- not the closure potential (`lam`) nor the
moment inverse -- so the geometry is built with `closure="none"`: the CG /
multigrid solve is the step's work, and the dt must not pay for it (the
adjacency and the moments are still built; those are the cheap part).

Kept as a tensor (not `.item()`'d), for the same reason as `compressible.py`:
an adaptive dt is a function of state, so the gradient back to state stays for
anyone differentiating through a rollout.
"""

from warpSPH.configurations.compute import neighborSearchOptions
from typing import Optional, Union

import torch

from ...configurations.meshless import MeshlessConfig
from ...configurations.simulationConfig import SimulationConfig
from ...systems.meshless import MeshlessSystem
from warpSPHCore.mfm import MeshlessWarp
from ..meshless.primitives import primitives, particleGamma
from ..meshless.geometry import stashGeometry, takeGeometry, cachedAdjacency

__all__ = ['computeTimestep']


def computeTimestep(
    system: MeshlessSystem,
    config: SimulationConfig,
    compParams: MeshlessConfig,
    dt: Optional[Union[float, torch.Tensor]] = None,
):
    if not config.adaptiveDt:
        if dt is not None:
            return dt
        raise ValueError("dt must be provided if adaptiveDt is False")

    state = system.state
    gamma = particleGamma(state, compParams)
    # The start-of-step geometry without the closure solve: `timestep` never
    # reads `lam`, so this is adjacency + moments only. Same positions /
    # supports / domain / kernel the step builds from, so the two geometries
    # agree on everything the signal speed depends on.
    reservoir = getattr(system, 'inflowReservoir', None)
    if reservoir is not None:
        g = reservoir.build(state, config, compParams, system.t)
        dts = reservoir.timestep(state, g, compParams)
    else:
        g = takeGeometry(state.positions, state.supports, system.domain, config.kernel, compParams, owner=state)
        if g is None:
            g = MeshlessWarp.build(
                state.positions, state.supports, system.domain, config.kernel,
                adjacency=cachedAdjacency(state.positions, state.supports, system.domain, compParams, owner=state,
                                          storeI=neighborSearchOptions(config)['storeI'], config=config),
                storeI=neighborSearchOptions(config)['storeI'],
                cond_max=compParams.condMax, closure="none", rtol=None, lam0=None,
                guards=True, face_cond_max=compParams.faceCondMax,
                centred_weights=True, area_cap=False)
        # the step that follows starts from these positions / supports: hand it this geometry so it
        # only adds the closure solve (`schemes/meshless.stashGeometry`)
        stashGeometry(state.positions, state.supports, system.domain, config.kernel, compParams, g, owner=state)
        rho, vel, P = primitives(state.Q, g.volume, gamma)
        # Core timestep uses pressure only through sqrt(gamma*P/rho); supplying
        # gamma_i*P_i with scalar gamma=1 gives each material its true signal speed.
        dts = (g.timestep(rho, vel, P*gamma, 1., compParams.cfl) if isinstance(gamma, torch.Tensor)
               else g.timestep(rho, vel, P, gamma, compParams.cfl))          # 2 C h / v_sig, h the kernel radius
    if compParams.cflLength == 'particle':
        # GIZMO's Courant length is the particle size L = V^(1/d) (core/predict.cc Get_Particle_Size,
        # core/timestep.cc: dt = CourantFac L / (0.5 MaxSignalVel)), not the kernel radius: the same
        # formula with L in place of h, i.e. a factor L / h ~ 1/4 at ~50 neighbours
        dts = dts * (g.volume ** (1.0 / state.positions.shape[1]) / state.supports)
    elif compParams.cflLength != 'support':
        raise ValueError("cflLength must be 'particle' or 'support'")
    initial_dt = torch.min(dts)
    # Floor reporting belongs to the runner's existing timestep-stall diagnostics.
    # No device scalar is read by Python on this path.
    initial_dt = torch.clamp(initial_dt, min=config.minDt, max=config.maxDt)
    if dt is not None:
        previous = torch.as_tensor(config.dt, dtype=initial_dt.dtype, device=initial_dt.device)
        initial_dt = torch.where(initial_dt > previous,
            torch.minimum(initial_dt, config.dtGrowthFactor * previous), initial_dt)
    return initial_dt
