"""Central enum registry for warpSPH: the scheme/kernel/BC/viscosity-switch
selectors used throughout `configurations/`, `modules/`, `schemes/`, and
`runner/` to pick a numerical formulation. Values are the strings/ints
stored in casefile YAML and parsed back into these enums by
`io.parsers`.
"""

from enum import Enum
import torch

__all__ = [
    'EnergyScheme',
    'AdaptiveSupportScheme',
    'ViscositySwitch',
    'CompressibleSPHScheme',
    'WeaklyCompressibleSPHScheme',
    'IncompressibleSPHScheme',
    'ArtificialCompressibleSPHScheme',
    'PressureSmoothingScheme',
    'WaveEquationScheme',
    'EquationOfState',
    'DensityDiffusionScheme',
    'PressureForceScheme',
    'isArtificialCompressibleScheme',
]

# @torch.jit.script
class EnergyScheme(Enum):
    equalWork = 0
    PdV = 1
    diminishing = 2
    monotonic = 3
    hybrid = 4
    CRK = 5

# @torch.jit.script
class AdaptiveSupportScheme(Enum):
    NoScheme = 0
    Monaghan = 1
    Owen = 2

# @torch.jit.script
class ViscositySwitch(Enum):
    Balsara1995 = 0
    Colagrossi2004 = 1
    CullenDehnen2010 = 2
    CullenHopkins = 3
    MorrisMonaghan1997 = 4
    Rosswog2000 = 5
    NoneSwitch = 6


# @torch.jit.script
class CompressibleSPHScheme(Enum):
    Monaghan = 0
    CompSPH = 1
    CRKSPH = 2

# @torch.jit.script
class WeaklyCompressibleSPHScheme(Enum):
    deltaSPH = 0

# @torch.jit.script
class IncompressibleSPHScheme(Enum):
    #: VD+PS DFSPH (Bender & Koschier 2015/2017): a divergence-free Jacobi
    #: pass plus a constant-density Jacobi pass, applied as a momentum-neutral
    #: position shift for most cases or folded into the velocity update
    #: in-step for body-force (gravity) cases (`schemes/divergenceFree.py`'s
    #: `INSTEP_CD`/`_RESTORE_PS_SHIFT` gates). **The recommended default for
    #: general use** -- as of DFSPH_IMPROVEMENT_PLAN.md Part 56/57/58, it does
    #: not diverge on any case in the suite (including every wall-bounded one),
    #: and where it has quality caveats they are modest and non-fatal: a
    #: closed box's density sits ~5-8% high against a strict 5% target
    #: (`randomFlowIncompressible --bounded`/`--obstacle`), and two cases
    #: (`impact`, `columnCollapse`) develop growing particle pairing after an
    #: impact while the walls themselves hold exactly. Reach for `band2018pb`
    #: instead specifically when a closed, wall-dominated case needs tighter
    #: incompressibility than the above and can tolerate rougher particle
    #: distribution -- see that scheme's own docstring below.
    divergenceFree = 0
    #: Reference DFSPH (Bender & Koschier 2015/2017) as implemented in
    #: SPlisHSPlasH's `TimeStepDFSPH.cpp`: the constant-density and
    #: divergence-free corrections are applied to the *velocity* as warm-started
    #: pressure impulses, not as the momentum-neutral position shift the
    #: `divergenceFree` (VD+PS) scheme uses. Exists as the ground-truth
    #: comparison for the hydrostatic-column failure -- see
    #: `DFSPH_IMPROVEMENT_PLAN.md` Part 23 and `schemes/dfsphReference.py`.
    dfsphReference = 1
    #: Plain IISPH (Ihmsen et al. 2014): a single constant-density projection
    #: per step, applied as a velocity impulse -- no divergence-free pass, no
    #: VD+PS position shift. Shares `dfsphReference`'s body with the divergence
    #: solve off (`iisph_step`). Part 33 measured it as the first scheme in the
    #: codebase to hold `hydrostaticColumn`; it also holds `staticBlob` where
    #: the two-solve `dfsphReference` diverges.
    iisph = 2
    #: Direct port of the omniSPH incompressible solver loop
    #: (`~/dev/omniSPH/simulation/{SPH,fluidMechanics}.cpp`): DFSPH-style
    #: divergence + density solves on one neighbourhood, accelerations
    #: accumulated into a single semi-implicit Euler step. No free-surface
    #: gauge, deficiency guard, damped warm start, or masking -- just the
    #: loop as omniSPH runs it. See `schemes/omniIncompressible.py`.
    omniIncompressible = 3
    #: IISPH with Pressure Boundaries (Band et al. 2018, ACM TOG 37(2):14):
    #: the extended PPE where `kind == 1` boundary samples are pressure
    #: unknowns with their own equation + diagonal, iterated in the same
    #: relaxed-Jacobi loop as the fluid -- removes the near-wall rank
    #: deficiency that stalls the `omniIncompressible` / `iisph`
    #: constant-density solve at a wall corner (DFSPH_IMPROVEMENT_PLAN.md
    #: Parts 41-44). See `schemes/band2018pb.py`.
    #: **A real trade-off against `divergenceFree`, not a strict upgrade**
    #: (Part 57/58): tighter density control on closed wall-bounded cases
    #: (~4-5% vs `divergenceFree`'s ~7-8%), at the cost of growing particle
    #: pairing/voids there instead. Its bigger limitation is structural: the
    #: paper has **no free-surface treatment at all** (every scenario in it is
    #: a closed tank), so it inherits real free-surface problems on
    #: `dambreak`-like violent impacts and fails `staticBlob` outright (a
    #: free-space body just is not its regime, Part 51). Use it deliberately
    #: for closed, wall-dominated flows where incompressibility matters more
    #: than particle-distribution smoothness -- not as a general-purpose
    #: replacement for `divergenceFree`.
    band2018pb = 4

# @torch.jit.script
class ArtificialCompressibleSPHScheme(Enum):
    """Artificial-compressibility SPH (De Courcy et al. 2024,
    `literature/decourcy2024_*`; `ACSPH_PLAN.md`).

    A third structurally-distinct incompressible baseline: where DFSPH iterates
    a pressure-Poisson-like Jacobi solve on a velocity constraint, this
    integrates a *differential* pressure equation `Dp/Dtau = -k1 rho div v +
    k2 D^p` to steady state in a pseudo-time loop nested inside each real step,
    with a BDF2 source carrying the real-time derivatives. It is a delta-SPH
    code with the equation of state removed -- which is why so much of it is a
    config flag away here (`ACSPH_PLAN.md` Part 3).

    Its own family rather than a `WeaklyCompressibleSPHScheme` member: the
    pressure is an *integrated* field and the density a constant, which is the
    opposite of every weakly-compressible state in this package.
    """
    artificialCompressible = 0


class PressureSmoothingScheme(Enum):
    """The `D^p` pressure-smoothing operator of De Courcy et al. 2024
    Eqs. (32)-(37) -- ACSPH's analogue of `DensityDiffusionScheme`, which is
    also what implements it (`modules/deltaSPH/densityDiffusion.py`'s
    `computeScalarFieldDiffusion`, with the pressure in place of the density).
    """
    #: AC-2, Eq. (32). Plain Molteni-Colagrossi Laplacian of pressure. A
    #: **negative control**, not a candidate default: it cannot hold a
    #: hydrostatic gradient and diffuses the free surface (paper Sec. 4.1.1).
    #: == `DensityDiffusionScheme.densityOnly`.
    laplacian = 0
    #: AC-2L, Eqs. (33)-(34). Antuono-renormalised bi-Laplacian. **The paper's
    #: working default** and the operator in every head-to-head against
    #: delta-SPH. == `DensityDiffusionScheme.deltaSPH`.
    renormalizedBiLaplacian = 1
    #: AC-4, Eq. (35). Nested (bi-harmonic) Laplacian: a second pass over AC-2's
    #: output, no renormalisation. Inherits AC-2's truncation error, weaker.
    biharmonic = 2
    #: AC-JST, Eqs. (36)-(37). Jameson-Schmidt-Turkel blend of AC-2L and AC-4,
    #: switched on a pressure-oscillation sensor, AC-2L alone near the free
    #: surface. Not pairwise-symmetric, therefore **not locally conservative**
    #: (the paper says so; see ACSPH_PLAN.md Sec. 5.1 on its `eps_4` typo).
    jst = 3


# @torch.jit.script
def isArtificialCompressibleScheme(scheme) -> bool:
    """Is `scheme` any member of `ArtificialCompressibleSPHScheme`?

    The counterpart of `isIncompressibleScheme`, and for the same reason: a
    case that branches on "does this scheme solve for pressure rather than
    evaluate an EOS" must ask about the family, not identity-test one member.
    """
    return isinstance(scheme, ArtificialCompressibleSPHScheme)


# @torch.jit.script
def isIncompressibleScheme(scheme) -> bool:
    """Is `scheme` any member of `IncompressibleSPHScheme`?

    Cases that need "the pressure-projection path" (its own CFL, no acoustic
    term, a solved `pressures` field rather than an EOS one) must test *this*,
    not `scheme is IncompressibleSPHScheme.divergenceFree`. That identity test
    was written when `divergenceFree` was the only incompressible scheme; it
    silently routes `iisph` / `omniIncompressible` / `band2018pb` down the
    weakly-compressible branch, which is wrong in every case that has one.
    """
    return isinstance(scheme, IncompressibleSPHScheme)


class WaveEquationScheme(Enum):
    waveEquation = 0


class EquationOfState(Enum):
    stiffTait = "stiffTait"
    Tait = "Tait"
    isoThermal = "isoThermal"
    Polytropic = "polytropic"
    Murnaghan = "murnaghan"


class DensityDiffusionScheme(Enum):
    deltaSPH = 0
    denormalized = 1
    densityOnly = 2
    deltaOnly = 3
    denormalizedOnly = 4

class PressureForceScheme(Enum):
    conservative = 0
    nonConservative = 1
    Antuono = 2
    i = 3
    j = 4
    symmetric = 5