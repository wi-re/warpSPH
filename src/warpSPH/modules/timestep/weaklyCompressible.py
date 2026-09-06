"""Adaptive timestep for weakly-compressible / delta-SPH systems.

Computes `dt` from up to three individually-togglable constraints (via
`compParams.dt_*Constraint` flags): a viscous diffusion limit, a CFL acoustic
limit (using the fixed sound speed), and an acceleration limit derived from
the last system update's max acceleration -- following Sun et al.'s delta-SPH
timestep procedure (cited in-code). Clamped to `[minDt, maxDt]` and to at
most `config.dtGrowthFactor` times the previous `dt`. Returns the fixed `dt`
unchanged when `config.adaptiveDt` is disabled.

`setupWeaklyCompressibleTimestep` is a one-time setup helper (not a per-step
timestep call): it back-solves the artificial EOS sound speed `c0` from a
target `dt`/`dx`/target-neighbor-count, stores it on `schemeConfig`, and warns
if the resulting Mach number would exceed 0.1.
"""

from warpSPH.utils.support import volumeToSupport

from ...systems.weaklyCompressible import WeaklyCompressibleState, WeaklyCompressibleSystem, WeaklyCompressibleSystemUpdate
from ...configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from typing import Optional
from warpSPHCore import *
from ...modules.eos import idealGasEOS
import torch
import warp as wp

__all__ = ['computeTimestep', 'setupWeaklyCompressibleTimestep']


def computeTimestep(
    system: WeaklyCompressibleSystem,
    config: SimulationConfig,
    compParams: WeaklyCompressibleSPHConfig,
    dt: Optional[float] = None,
    systemUpdate: Optional[WeaklyCompressibleSystemUpdate] = None,
):
# See Sun et al: The delta SPH-model: Simple procedures for a further improvement of the SPH scheme
    if not config.adaptiveDt:
        return dt
    
    timestepCFL = config.cflFactor
    maxDt = config.maxDt
    minDt = config.minDt
    
    alpha = compParams.diffusionParams.inviscidAlpha
    c_s = compParams.fluid.fixedSoundSpeed
    particleSupport = system.state.supports.min()
    kernelScale = float(sphKernelScale(config.kernel.value, config.dim))

    dtype =  config.dtype
    device = config.device

    # if verbose:
    #     print(f'[SPH] - Adaptive Timestep Update')
    #     print(f'\tCFL: {timestepCFL}, maxDt: {maxDt}, minDt: {minDt}')
    #     print(f'\tDiffusion: {diffusionAlpha}, {diffusionScheme}')
    #     print(f'\tFluid: {fluidCs}, {particleSupport}, {kernelScale}')


    # with record_function("[SPH] - Adaptive Timestep Update"):
    state = system.state
    dim = state.positions.shape[1]

    # nu = alpha * c_s * particleSupport / (2 * (dim +2))
    if compParams.diffusionParams.inviscid:
        nu = alpha * c_s * particleSupport / (2 * (dim +2))
    else:
        nu = compParams.diffusionParams.viscidNu

    # nu = config.get('diffusion', {}).get('nu', nu) if diffusionScheme == 'deltaSPH_viscid' else nu
    dt_v = 0.125 * particleSupport**2 / nu / kernelScale
    dt_v = torch.tensor(dt_v, dtype = dtype, device = device) if not isinstance(dt_v, torch.Tensor) else dt_v

    dt_c = timestepCFL * particleSupport / c_s / kernelScale
    dt_c = torch.tensor(dt_c, dtype = dtype, device = device) if not isinstance(dt_c, torch.Tensor) else dt_c

    # acceleration timestep condition -- Sun et al. 2017 Eq. (5) `dt_a =
    # 0.25 min sqrt(h / ||a||)`. The per-particle acceleration is the update's
    # `dvdt` (`WeaklyCompressibleSystemUpdate`); older callers passed an object
    # carrying `.velocities` (a dv), so accept either.
    dudt = getattr(systemUpdate, 'dvdt', None) if systemUpdate is not None else None
    if dudt is None and systemUpdate is not None:
        dudt = getattr(systemUpdate, 'velocities', None)
    if dudt is not None and dudt.numel():
        finite = dudt[~torch.isnan(dudt).any(dim=-1)] if dudt.ndim > 1 else dudt[~torch.isnan(dudt)]
        if finite.numel():
            max_accel = torch.max(torch.linalg.norm(finite, dim = -1))
            dt_a = 0.25 * torch.sqrt(particleSupport / (max_accel + 1e-7)) / kernelScale
        else:
            dt_a = torch.tensor(maxDt, dtype = dtype, device = device)
    else:
        dt_a = torch.tensor(maxDt, dtype = dtype, device = device)
    dt_a = torch.tensor(dt_a, dtype = dtype, device = device) if not isinstance(dt_a, torch.Tensor) else dt_a
    # if verbose:
        # print(f'\tViscosity: {dt_v}, Acoustic: {dt_c}, Acceleration: {dt_a}')

    # dt = config['timestep']['dt']
    if dt is None:
        dt = torch.tensor(maxDt, dtype = dtype, device = device)
    else:
        dt = torch.tensor(dt, dtype = dtype, device = device) if not isinstance(dt, torch.Tensor) else dt
    new_dt = dt
    if compParams.dt_viscosityConstraint:
        new_dt = dt_v
    if compParams.dt_accelerationConstraint:
        new_dt = torch.min(new_dt, dt_a)
    if compParams.dt_acousticConstraint:
        new_dt = torch.min(new_dt, dt_c)
    new_dt = torch.min(new_dt, torch.tensor(maxDt, dtype = dtype, device = device))
    new_dt = torch.max(new_dt, torch.tensor(minDt, dtype = dtype, device = device))
    if dt is not None and new_dt > dt:
        new_dt = torch.clamp(new_dt, max=config.dtGrowthFactor * dt)
    return new_dt


def setupWeaklyCompressibleTimestep(
        config, schemeConfig, compressibleSystem, targetDt, verbose = True,
        cSound = None, uMaxExpected = None, machTarget = 0.1,
):
    """Fix the artificial sound speed `c0` and the initial `dt`.

    Two ways to pick `c0`:

    * **Sun et al. 2017 Eq. (2)** -- the physical way. Pass `cSound` directly,
      or `uMaxExpected` and `machTarget` (default 0.1) so
      `c0 = uMaxExpected / machTarget` (`>= 10 U_max` at the default). `dt` is
      then the acoustic-CFL step `cflFactor * h / (c0 * kernelScale)` (with `h`
      the support radius). This keeps the run genuinely weakly compressible
      regardless of `dx`.
    * **legacy back-solve** -- neither given: `c0` is inverted out of a fixed
      `targetDt` through the acoustic CFL, `c0 = 0.3 h_vol / (kernelScale
      targetDt)`, and `dt = targetDt`. This makes `c0 ~ 1/dx` and can push the
      Mach number well past 0.1 at fine resolution (see the warning below);
      prefer the Sun Eq. (2) path for any physical-scale case.
    """
    dx = config.dx
    h = float(compressibleSystem.state.supports.min())
    kernelScale = float(sphKernelScale(config.kernel.value, config.dim))

    if cSound is not None or uMaxExpected is not None:
        c0 = float(cSound) if cSound is not None else float(uMaxExpected) / float(machTarget)
        dt = float(config.cflFactor) * h / (c0 * kernelScale)
        schemeConfig.fluid.fixedSoundSpeed = c0
        if verbose:
            print(f'Sun Eq. (2) sound speed: c0 = {c0:.4g}, acoustic dt = {dt:.4g}')
        uMax = torch.max(torch.linalg.norm(compressibleSystem.state.velocities, dim=1))
        if verbose:
            print(f'Max velocity (init): {uMax}, Max Mach: {uMax / c0}')
        config.dt = dt
        return c0, dt

    c0 = 0.3 * volumeToSupport(dx**2, config.targetNeighbors, 2) / float(sphKernelScale(config.kernel.value, 2)) / targetDt
    if verbose:
        print(f'Computed c0: {c0}, target c0: {schemeConfig.fluid.fixedSoundSpeed}, diff: {abs(c0 - schemeConfig.fluid.fixedSoundSpeed)}')

    schemeConfig.fluid.fixedSoundSpeed = c0
    # dt = computeTimestep(
    #     system = compressibleSystem,
    #     config = config,
    #     compParams = schemeConfig,
    #     dt = None,
    #     systemUpdate = None,
    # )
    dt = targetDt
    if verbose:
        print(f'Computed dt: {dt}, target dt: {targetDt}, diff: {abs(dt - targetDt)}')
    uMax = torch.max(torch.linalg.norm(compressibleSystem.state.velocities, dim=1))
    if verbose:
        print(f'Max velocity: {uMax}, CFL: {uMax * dt / dx}, Max Mach: {uMax / c0}')
    if uMax / c0 > 0.1:
        print(f'Warning: Max Mach number ({uMax / c0:.2f}) is greater than 0.1, which may lead to instability in the simulation. Consider changing the relevant parameters.')

    config.dt = dt
    return c0, dt
