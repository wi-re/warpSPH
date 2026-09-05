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

    WARNING -- Eq. (46)'s first term is dimensionally impossible as printed.
    The paper writes

        dt^n = max( min( CFL_t h , CFL_t h / |v|_max , 0.125 h^2 / nu ,
                         1.2 dt^{n-1} ) , 0.8 dt^{n-1} )

    (verified against the rendered page 10) and `CFL_t h` is a *length*, not a
    time. The other two constraints are the standard advective and viscous
    pair; what is missing from the list is the body-force constraint every
    delta-SPH timestep carries, `~ sqrt(h / |a|_max)`, which is exactly where
    `CFL_t h` sits. So the most likely reading is that the first term is that
    constraint with its square root lost in typesetting.

    This module therefore implements the two well-formed constraints plus an
    acceleration constraint `CFL_t sqrt(h / |a|_max)` under the existing
    `dt_accelerationConstraint` flag, and does **not** implement `CFL_t h`.
    `config.maxDt` supplies the absolute cap the literal term would otherwise
    be doing. See ACSPH_PLAN.md Sec. 5.6 -- **ask the authors.**
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

    # Advective: CFL_t h / |v|_max. Replaces the acoustic constraint -- there
    # is no sound speed here to divide by.
    fluid = state.kinds == 0
    vMax = float(state.velocities[fluid].norm(dim=-1).max()) if bool(fluid.any()) else 0.0
    candidates.append(asTensor(cflT * h / (vMax + 1e-12) / kernelScale))

    if compParams.dt_viscosityConstraint:
        if acParams.referenceSoundSpeedForViscosity is not None:
            nu = acParams.alphaNu * h * acParams.referenceSoundSpeedForViscosity / (2 * (dim + 2))
        else:
            nu = acParams.nu
        candidates.append(asTensor(0.125 * h ** 2 / max(float(nu), 1e-30) / kernelScale))

    # See the module docstring: this stands in for Eq. (46)'s dimensionally
    # impossible `CFL_t h` term, on the reading that it is the body-force
    # constraint with a lost square root.
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
