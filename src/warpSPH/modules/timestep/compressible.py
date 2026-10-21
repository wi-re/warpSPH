"""Adaptive timestep for the (fully) compressible system.

Computes the CFL-limited timestep from the minimum sound-crossing time over
the current sound speeds (via the ideal-gas EOS) and smoothing lengths,
clamped to `[config.minDt, config.maxDt]` and to at most
`config.dtGrowthFactor` times the previous `dt`. Falls back to the
caller-supplied fixed `dt` (raising if none given) when `config.adaptiveDt`
is disabled. Returns a `torch.Tensor`, not a Python float, when adaptive --
`c_s` depends on state, so keeping it a tensor preserves the gradient back to
state for anyone differentiating through a rollout; float-converting is left
to the caller (see runner.py's `_scalar`).
"""

from warpSPH.utils.support import volumeToSupport

from ...systems.compressibleMonaghan import CompressibleState, CompressibleSystem
from ...configurations.compressibleConfig import CompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from typing import Optional, Union
from warpSPHCore import *
from ...modules.eos import evaluateEOS
import torch
import warp as wp

__all__ = ['computeTimestep']

def computeTimestep(
    system: CompressibleSystem,
    config: SimulationConfig,
    compParams: CompressibleSPHConfig,
    dt: Optional[Union[float, torch.Tensor]] = None,
):
    if config.adaptiveDt:
        # 
        targetCFL = config.cflFactor
        # The initial state contains density and pressure
        # we can get the internal energies for each state via the gas EOS
        u = system.state.internalEnergies
        # c_s = system.soundspeeds.max()            
        _, _, _, c_s = evaluateEOS(
            system.state.densities, system.state.materials, compParams,
            u=system.state.internalEnergies)


        # print(f"Initial internal energy: u (min: {u.min()}, max: {u.max()}, mean: {u.mean()})")
        # print(f"Initial sound speed: c_s (min: {c_s.min()}, max: {c_s.max()}, mean: {c_s.mean()})")

        # u_left = leftState.p / ((gamma - 1) * leftState.rho)
        # u_right = rightState.p / ((gamma - 1) * rightState.rho)
        # # this then gives us the speed of sound for each state, which we can use to determine the initial timestep based on the CFL condition
        # c_s_left =  np.sqrt(u_left * gamma * (gamma - 1))
        # c_s_right =  np.sqrt(u_right * gamma * (gamma - 1))

        # We cab then get the minimum support radius based on the initial sampling and use this to get the initial timestep
        # The sampling is done such that with a sampling ratio of 1 th left and right states each have nx particles
        # The sampling ratio then reduces the number of particles by that factor
        # accordingly, the higher resolution is always the unmultiplied L/2 /nx
        # L = config.domain.max - config.domain.min
        # dx = L / (nx * 2)
        h = volumeToSupport(config.dx, config.targetNeighbors, config.dim)
        h = system.state.supports.min()
        xi = float(sphKernel_xi(config.kernel.value, config.dim))
        # xi = 1

        dt_cfl_left = targetCFL * h / (c_s * xi)
        # dt_cfl_right = targetCFL * h / (c_s + h * xi)
        initial_dt = torch.min(dt_cfl_left)
        if initial_dt < config.minDt:
            print(f"Warning: initial dt {initial_dt.item()} is less than minDt {config.minDt}. Clamping to minDt.")
        initial_dt = torch.clamp(initial_dt, min=config.minDt, max=config.maxDt)
        if dt is not None and initial_dt > config.dt:
            initial_dt = torch.clamp(initial_dt, max=config.dtGrowthFactor * config.dt)
        # Kept as a tensor (not .item()'d) so a genuinely adaptive dt -- itself a
        # function of state via c_s above -- carries a gradient back to state for
        # anyone differentiating through a rollout. Plain-float call sites must
        # convert explicitly at their own reporting/bookkeeping boundary (see
        # runner.py's `_scalar`), not have that decision made for them here.
        return initial_dt
    else:
        if dt is not None:
            return dt
        else:
            raise ValueError("dt must be provided if adaptiveDt is False")