"""Michel et al. 2022 (`literature/michel2022`) interior particle-shifting
law, Eq. (22): the "consistent" PST whose characteristic velocity
(`wp_michelUChar.computeUCharWarp`'s `U_char`) is a *relative* quantity,
giving it Galilean and local rotation invariance without a Mach number --
unlike `modules.shifting.delta`'s Sun-style law, this is usable by ACSPH
(which has no sound speed to form a Mach number from; see `PST_ALE_PLAN.md`
Part 1.2).

`computeMichelShift` mirrors `modules.shifting.delta.computeDeltaShift`'s
contract (restores positions/densities, returns the accumulated position
delta) so it drops into `modules.shifting.wrapper.solveShifting`'s dispatch
almost unchanged in shape, plus two additions: `beta`, the per-particle
Eq. (21/48) coefficient, and `dt`. Its interior value is `(R/dx)**3`
(counterbalancing the lowest-degree truncation term, Eq. 11), but Eq. (48)
requires it to decay *before* the interior law is evaluated near a free
surface (both branches of the norm clamp below depend on it) rather than as
a post-hoc vector projection -- so the caller (`wrapper.py`, which already
runs free-surface detection ahead of the shift dispatch) computes the
surface-adjusted `beta` and passes it in; this function only evaluates
Eq. (22) with whatever `beta` it is given.

**`dt` matters and is not optional.** Eq. (22) is a *velocity*
(ACSPH_PLAN.md Eq. (58): `dx_i/dt = u_i + delta_u_i`) -- unlike
`computeDeltaShift`'s Sun-style law, whose `-CFL*Ma*2*h^2` scaling bakes in
an implicit acoustic timestep (`delta.py`'s own comment: "the acoustic time
step is dt = CFL*h/c0... equivalent to the delta+ scaling factor") and so
already returns a position-shaped displacement. Michel's `U_char` (a genuine
velocity, `max_j|(u_j-u_i).x_hat_ij|`) has no such factor baked in -- an
earlier version of this function added the raw Eq. (22) value straight to
`positions`, which is a units error (confirmed numerically: it produced a
shift ~8x the local particle spacing per call on a representative case,
against Sun's ~0.07x). ACSPH has no acoustic timestep to reuse (no sound
speed), so this multiplies by the real simulation `dt` instead, matching
Eq. (58) literally: over one real step, `delta_u_i * dt` is exactly the
displacement that velocity contributes.
"""

from warpSPHCore import *
import torch

from warpSPH.sample.wp_deltaShift import computeDeltaShiftWarp
from .wp_michelUChar import computeUCharWarp

__all__ = ['computeMichelShift']


def computeMichelShift(currentState, config, schemeConfig, domain, adjacency, beta, dt, iters=-1,
                        returnVelocity=False):
    """`returnVelocity=True` additionally returns Eq. (22)'s own
    `delta_u` (the last iteration's shifting *velocity*, pre-`dt`) as a third
    element -- what Michel's own `delta_u_max` convergence-rate figures (Fig.
    1, PST_ALE_PLAN.md Part 7) are measured on, not the dt-scaled position
    delta this function returns by default for `solveShifting`. Existing
    callers are unaffected: the default is `False` and the return shape is
    unchanged, matching this codebase's `returnNormals`/`returnEigVals`
    convention for optional extra outputs elsewhere."""
    original_positions = currentState.positions.clone()
    original_densities = currentState.densities.clone()

    rho0 = schemeConfig.fluid.restDensity
    dim = currentState.positions.shape[1]
    dx = config.dx if not isinstance(config.dx, torch.Tensor) else config.dx.cpu().item()

    for _ in range(schemeConfig.shiftProperties.iterations if iters == -1 else iters):
        # Eq. (2)/(3): the tensile-corrected, pure-volume-weighted kernel
        # gradient sum. R=0.2, n=4 match Eq. (3) exactly.
        gradCtilde = computeDeltaShiftWarp(
            currentState,
            operationProperties=OperationProperties(
                operation=WarpOperation.Density,
                kernel=config.kernel,
                supportMode=SupportScheme.Gather,
            ),
            referenceParticles=currentState,
            domain=domain,
            adjacency=adjacency,
            CFL=0.0, computeMach=False, c_max=0.0,
            rho0=rho0, dx=dx,
            R=0.2, n=4, volumeWeighted=True,
        )

        # Eq. (20): U_char_i = U_lim_i = max_j |(u_j-u_i).x_hat_ij|
        U_char = computeUCharWarp(
            currentState,
            operationProperties=OperationProperties(
                operation=WarpOperation.Density,
                kernel=config.kernel,
                supportMode=SupportScheme.Gather,
            ),
            referenceParticles=currentState,
            domain=domain,
            adjacency=adjacency,
        )

        R_i = currentState.supports
        achievedDx_i = torch.pow(currentState.masses / rho0, 1.0 / dim)
        capLen = 0.5 * (R_i / achievedDx_i)

        gradNorm = torch.linalg.norm(gradCtilde, dim=-1)
        scaledNorm = beta * R_i * gradNorm
        gradDir = torch.nn.functional.normalize(gradCtilde, dim=-1)

        full = -0.5 * (U_char * beta * R_i).unsqueeze(-1) * gradCtilde
        capped = -0.5 * (U_char * capLen).unsqueeze(-1) * gradDir

        useCapped = (scaledNorm >= capLen).unsqueeze(-1)
        delta_u = torch.where(useCapped, capped, full)

        # Eq. (22)'s delta_u is a velocity; dt converts it to the
        # displacement contributed over one real step (Eq. 58). See this
        # module's docstring.
        currentState.positions = currentState.positions + delta_u * dt

    delta = currentState.positions - original_positions
    currentState.positions = original_positions
    currentState.densities = original_densities

    if returnVelocity:
        return delta, adjacency, delta_u
    return delta, adjacency
