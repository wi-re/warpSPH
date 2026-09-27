"""Adaptive real timestep for artificial-compressibility SPH, De Courcy et al.
2024 Eq. (46) (`ACSPH_PLAN.md` Sec. 4.5).

Three things separate it from `weaklyCompressible.py`'s:

- **The acoustic constraint is gone**, replaced by an advective one,
  `CFL_t h / |v|_max`. ACSPH has no `c0`; nothing in the scheme propagates at a
  sound speed, which is the entire reason it exists.
- **A symmetric growth/shrink clamp**, `[0.8, 1.2] x dt^{n-1}`. Not a stability
  device: BDF2's accuracy degrades when the step ratio swings, so Eq. (46)
  bounds it in both directions. `weaklyCompressible.py` clamps growth only.
- **`CFL_t <= 0.4` is a hard ceiling**, not a guideline. Tables 1-2 measure a
  sharp accuracy cliff above it -- error jumps ~2.4x from 0.4 to 0.6 and ~10x
  by 1.0 -- so exceeding it is a warning, not a silent choice.

    WARNING -- Eq. (46)'s first term is dimensionally a length. The paper writes

        dt^n = max( min( CFL_t h , CFL_t h / |v|_max , 0.125 h^2 / nu ,
                         1.2 dt^{n-1} ) , 0.8 dt^{n-1} )

    (verified against the rendered page 10) and `CFL_t h` is a length, not a
    time. What it *does*, though, is unambiguous from the structure: paired
    with `CFL_t h / |v|_max` it is

        min( CFL_t h , CFL_t h / |v|_max )  ==  CFL_t h / max(1, |v|_max)

    i.e. the advective constraint with its denominator floored at one, which
    stops `dt` running away when the flow is slow. In a code whose velocities
    are O(1) that floor is a reference velocity of 1 left implicit -- which is
    exactly the dimension that is missing.

    Implemented that way, with the floor exposed as `REFERENCE_VELOCITY` so it
    is a stated assumption rather than a hidden 1. It matters: without it
    nothing bounds `dt` in a quiescent case. On `hydrostaticColumn` at rest,
    `|v|_max ~ 0` makes the advective term infinite and `nu = 0` makes the
    viscous one infinite too, so `dt` grew 1.2x every step straight to
    `config.maxDt` -- and the corner error grew with it.

    See ACSPH_PLAN.md Sec. 5.6 -- **ask the authors** whether their `h` there
    carries an implicit reference velocity, and whether the body-force
    constraint every delta-SPH timestep carries is genuinely absent or was
    simply not written down.
"""

from typing import Optional

import torch
from warpSPHCore import sphKernelScale, sphKernel_xi

from ...configurations.artificialCompressible import ArtificialCompressibleSPHConfig
from ...configurations.simulationConfig import SimulationConfig
from ...systems.artificialCompressible import ArtificialCompressibleSystem

__all__ = ['computeTimestep', 'CFL_T_CEILING']

#: Sec. 4.3 / Tables 1-2. Above this the BDF2 error rises sharply; the paper
#: operates at ~0.2.
CFL_T_CEILING = 0.4

#: Eq. (46)'s symmetric step-ratio clamp, `[shrink, grow]`. Present to protect
#: BDF2 accuracy, not stability.
STEP_RATIO_BOUNDS = (0.8, 1.2)

#: The floor on `|v|_max` in the advective constraint -- Eq. (46)'s first term,
#: read as a reference velocity left implicit. See the module docstring. Change
#: it (not `cflT`) if a case's natural velocity scale is not 1.
REFERENCE_VELOCITY = 1.0

_warnedCfl = False


def computeTimestep(
    system: ArtificialCompressibleSystem,
    config: SimulationConfig,
    compParams: ArtificialCompressibleSPHConfig,
    dt: Optional[float] = None,
    systemUpdate=None,
):
    """Eq. (46). Returns `dt` unchanged when `config.adaptiveDt` is off.

    `h` is the paper's smoothing length (`supports / xi`), matching
    `schemes/artificialCompressible.acParameters`; the extra `/ kernelScale`
    both this and `weaklyCompressible.py` apply converts to the kernel's own
    length unit, which is what the published CFL constants are quoted in.
    """
    global _warnedCfl
    if not config.adaptiveDt:
        return dt

    acParams = compParams.acParams
    cflT = acParams.cflT
    if cflT > CFL_T_CEILING and not _warnedCfl:
        _warnedCfl = True
        print(f"[warpSPH] acsph: CFL_t = {cflT} exceeds the measured ceiling "
              f"{CFL_T_CEILING} (De Courcy et al. 2024 Tables 1-2: the BDF2 error "
              f"jumps ~2.4x from 0.4 to 0.6). Proceeding, but the real-time "
              f"accuracy is not the paper's.")

    dtype, device = config.dtype, config.device
    state = system.state
    dim = state.positions.shape[1]
    kernelScale = float(sphKernelScale(config.kernel.value, dim))
    xi = sphKernel_xi(config.kernel.value, dim)
    h = float(state.supports.min()) / xi

    def asTensor(value):
        return value if isinstance(value, torch.Tensor) else \
            torch.tensor(float(value), dtype=dtype, device=device)

    candidates = [asTensor(config.maxDt)]

    # Advective: `CFL_t h / max(REFERENCE_VELOCITY, |v|_max)`, which is
    # Eq. (46)'s first two terms together. Replaces the acoustic constraint --
    # there is no sound speed here to divide by -- and the floor is what keeps
    # `dt` finite when the flow is at rest.
    fluid = state.kinds == 0
    vMax = float(state.velocities[fluid].norm(dim=-1).max()) if bool(fluid.any()) else 0.0
    candidates.append(asTensor(cflT * h / max(REFERENCE_VELOCITY, vMax) / kernelScale))

    if compParams.dt_viscosityConstraint:
        if acParams.referenceSoundSpeedForViscosity is not None:
            nu = acParams.alphaNu * h * acParams.referenceSoundSpeedForViscosity / (2 * (dim + 2))
        else:
            nu = acParams.nu
        candidates.append(asTensor(0.125 * h ** 2 / max(float(nu), 1e-30) / kernelScale))

    # Not in Eq. (46) at all -- every other delta-SPH timestep in this repo and
    # in the literature carries a body-force constraint, and a scheme that runs
    # `dambreak` under gravity needs one. Kept behind the existing
    # `dt_accelerationConstraint` flag so the paper's literal constraint set is
    # still reachable by turning it off.
    if compParams.dt_accelerationConstraint and systemUpdate is not None:
        dvdt = getattr(systemUpdate, 'dvdt', None)
        if dvdt is not None and dvdt.numel():
            finite = dvdt[~torch.isnan(dvdt).any(dim=-1)]
            if finite.numel():
                aMax = float(finite.norm(dim=-1).max())
                candidates.append(asTensor(
                    cflT * (h / (aMax + 1e-12)) ** 0.5 / kernelScale))

    newDt = torch.stack([c.reshape(()) for c in candidates]).min()

    # Eq. (46)'s symmetric clamp, applied before the absolute bounds so a
    # `minDt`/`maxDt` floor cannot be reached by a step ratio BDF2 rejects.
    if dt is not None:
        previous = asTensor(dt)
        shrink, grow = STEP_RATIO_BOUNDS
        newDt = torch.clamp(newDt, min=shrink * previous, max=grow * previous)

    newDt = torch.clamp(newDt, min=asTensor(config.minDt), max=asTensor(config.maxDt))
    return newDt
