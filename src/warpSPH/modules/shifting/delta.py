"""delta^+ particle shifting (Sun et al. 2017 Eq. (7)): iterates a raw
kernel-gradient shift term (`computeDeltaShiftWarp`, imported from
`warpSPH.sample.wp_deltaShift`) and rescales it to a position delta, where
`Ma = v_max / c0` is a per-call Mach number estimate (density, position and
velocities are restored to their pre-call values afterwards; only the
accumulated position `delta` is returned to the caller). An equivalent
shifting-*velocity* form (Michel 2022 scaling) is present in a comment but not
used. `v_max` falls back to `c_max = 0.1` when the finite velocity magnitudes
are all ~zero, to avoid a zero shift.

**Two scalings live here**, selected by `ShiftProperties.sun2017Eq7Shift`:
`16 h^2` (with `R = 0.2`), which is Eq. (7) literally, and the historical
`2 h^2` (with `R = 0.25`), which is **1/8 of it** -- see the scaling block in
`computeDeltaShift` for the factor-by-factor comparison against the paper, and
`scripts/probe_deltaPlusShiftMagnitude.py` for the measurement. The historical
value is still the default for `ShiftingScheme.deltaSPH`; `--scheme
sun2017DeltaSPH` selects Eq. (7).
"""

# 17. If finalize, compute shifting and update positions and velocities

from warpSPHCore import *
import torch

from warpSPH.sample.wp_deltaShift import computeDeltaShiftWarp

__all__ = ['computeDeltaShift', 'computeDeltaShiftWarp']

    # for i in tqdm(range(shiftIters), leave = False):
def computeDeltaShift(currentState, config, schemeConfig, domain, adjacency, iters = -1):
    original_positions = currentState.positions.clone()
    original_densities = currentState.densities.clone()
    for i in range(schemeConfig.shiftProperties.iterations if iters == -1 else iters):
            
        # adjacency = buildVerletList(
        #     currentState, 
        #     config.domain, verletScale = config.verletScale, supportMode = SupportScheme.SuperSymmetric,
        #     priorNeighborhood = adjacency,
        #     verbose = False)


        # currentState.densities =warpOperation(
        #     currentState,
        #     operationProperties = OperationProperties(
        #         operation=WarpOperation.Density,
        #         kernel = config.kernel, 
        #         supportMode = SupportScheme.Gather
        #     ),
        #     domain = domain,
        #     adjacency = adjacency
        # )
        # display(currentState)

        # Sun et al. 2017 Eq. (7)'s literal constants rather than the
        # historical ones -- see the scaling block below for what differs and
        # why it is opt-in.
        eq7 = getattr(schemeConfig.shiftProperties, 'sun2017Eq7Shift', False)

        c0 = schemeConfig.fluid.fixedSoundSpeed if schemeConfig.fluid.fixedSoundSpeed is not None else 1.0

        velocity_magnitudes = torch.linalg.vector_norm(currentState.velocities, dim=-1)
        finite_velocity_magnitudes = velocity_magnitudes[torch.isfinite(velocity_magnitudes)]
        v_max = (
            torch.max(finite_velocity_magnitudes)
            if finite_velocity_magnitudes.numel() > 0
            else torch.tensor(float('nan'), device=currentState.velocities.device)
        )
        c_max = v_max / c0
        h_min = currentState.supports.min()
        # NaN (no finite velocities) compares False against 1e-6 just like the
        # `if` it replaces, so c_max is left as NaN in that case -- same as before.
        c_max = torch.where(c_max < 1e-6, c_max.new_tensor(0.1), c_max)
        # print(f'Iteration {i}, max velocity: {v_max.item()}, min support: {h_min.item()}, c_max: {c_max.item()}')


        shift = computeDeltaShiftWarp(
            currentState,
            operationProperties = OperationProperties(
                operation=WarpOperation.Density,
                kernel = config.kernel, 
                supportMode = SupportScheme.Gather
            ),
            referenceParticles = currentState,
            domain = domain,
            # supportMode = SupportScheme.Gather,
            # kernel = KernelFunctions.Wendland2,
            # operationMode = OperationDirection.AllToAll,
            adjacency = adjacency,

            CFL = schemeConfig.shiftProperties.CFL, computeMach = schemeConfig.shiftProperties.computeMach, c_max = c_max.cpu().item(),
            rho0 = 1.0, dx = config.dx if not isinstance(config.dx, torch.Tensor) else config.dx.cpu().item(),
            # Sun's R = 0.2 (Eq. 7, Monaghan's tensile-control value) under
            # `sun2017Eq7Shift`; `computeDeltaShiftWarp`'s own historical
            # default is 0.25. `n = 4` agrees either way.
            **({'R': 0.2} if eq7 else {})
        ) #* schemeConfig.fluid.fixedSoundSpeed * config.dt
        # The compute function returns the unscaled term
        # \sum_j 0.5 * m_j / (rho_i + rho_j) * [ 1 + R * (w_ij / W_0)^n ] * gradW_ij
        # The scaling factor is applied here to get the final shift amount
        Ma = c_max
        CFL = schemeConfig.shiftProperties.CFL
        kernelScale = float(sphKernelScale(config.kernel.value, config.dim))
        h = currentState.supports / kernelScale
        dt = config.dt

        # Sun et al. 2017 Eq. (7), verbatim (`literature/sun2017_*.pdf` p. 28):
        #
        #   dr_i = -CFL Ma (2 h_ij)^2 sum_j [1 + R (W_ij/W(dx_i))^n]
        #          grad_i W_ij  2 m_j/(rho_i + rho_j)  phi_ij
        #
        # so the scaling that turns the raw sum above into Eq. (7) is
        # `(2h)^2 = 4 h^2` for the prefactor, times 4 to lift the raw sum's
        # `0.5 m_j/(rho_i+rho_j)` weight to Eq. (7)'s `2 m_j/(rho_i+rho_j)`:
        # 16 h^2 in total.
        #
        # The historical scaling here was `2 h^2` with the raw weight left
        # alone -- **1/8 of Eq. (7)** -- justified by an in-code comment that
        # dropped the `(2h)^2`'s factor of 2 "because we include the 2 from
        # the mean density term in the computation of the shift". That
        # accounting does not hold: the `2` in `2 m_j/(rho_i+rho_j)` is Eq.
        # (7)'s own volume weight, a separate factor from the `(2h)^2`
        # prefactor, and the raw sum carries `0.5 m_j/(rho_i+rho_j)` rather
        # than either. `scripts/probe_deltaPlusShiftMagnitude.py` measures the
        # resulting ratio directly: 0.131x Eq. (7), flat across
        # L/dx = 50/100/200.
        #
        # Landing this as the unconditional default would change the physics of
        # every case using `ShiftingScheme.deltaSPH`, so it is opt-in
        # (`ShiftProperties.sun2017Eq7Shift`, default False) and turned on by
        # `Sun2017DeltaSPHConfig` / `--scheme sun2017DeltaSPH`, the same way
        # `freezeDiffusionAcrossStages` is.
        scalingDeltaPlus = -CFL * Ma * (16.0 if eq7 else 2.0) * h**2

        # If we follow the approach of Michel 2022 we can rewrite the shift as a shifting velocity instead of a shift amount, and then scale it by dt to get the final shift amount
        # The scaling factor here is
        # - Ma * c0 * 2 h
        # Note that the acoustic time step is dt = CFL * h / c0, so the scaling factor is equivalent to the delta^+ scaling factor for a fixed time conservative timestep
        # scalingMichel = - Ma * c0 * 2 * h * dt

        dt_c = schemeConfig.shiftProperties.CFL * h.min().cpu().item() / c0# / kernelScale

        # print('-' * 80)
        # print(f'scalingDeltaPlus: {scalingDeltaPlus.mean().item()}, scalingMichel: {scalingMichel.mean().item()}, \n[CFL: {CFL}, Ma: {Ma.item()}, c0: {c0}, h: {h.mean().item()}, dt: {dt}], ratio: {scalingDeltaPlus.mean().item() / scalingMichel.mean().item()}')
        # print(f'dt_c: {dt_c}, dt: {dt}, ratio: {dt_c / dt}')


        shift = shift * scalingDeltaPlus.unsqueeze(-1)

        # print(f'Iteration {i}, shift magnitude: {shift.norm(dim=1).mean().item()}, max: {shift.norm(dim=1).max().item()} [dx: {config.dx.item()}/dt: {config.dt}/mean support: {currentState.supports.mean().item()}]')

        # print(f'Iteration {i}, shift magnitude: {shift.norm(dim=1).mean().item()}, max: {shift.norm(dim=1).max().item()} [dx: {config.dx.item()}/dt: {config.dt}/mean support: {currentState.supports.mean().item()}]')
        currentState.positions = currentState.positions + shift

    delta = currentState.positions - original_positions
    currentState.positions = original_positions
    currentState.densities = original_densities

    return delta, adjacency