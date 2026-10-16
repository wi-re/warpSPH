"""Sod shock tube in 2D and 3D -- the 1D setup extruded, sampled at equal mass.

Geometry, per dimension:

* **x** repeats `sod.py`'s arrangement exactly. The domain is periodic on
  ``[-L/2, L/2]``, the left (dense) state occupies the middle half
  ``|x| <= L/4`` and the right state the two outer quarters, so there are two
  interfaces, at ``x = ±L/4``. Nothing is reflected explicitly: the mirror
  symmetry of that arrangement makes ``x = 0`` and ``x = ±L/2`` behave as
  reflecting walls for as long as the waves have not reached them, which is
  what makes a periodic box a usable shock tube. The analytic solution
  therefore describes the window ``x in [0, L/2]``, which is what
  `plotSodProfile` and `sodSolution.solve` are lined up against.
* **y (and z)** are plain periodic directions -- a slab, since the solution is
  uniform along them. Its width is given in *dense-side particle spacings*
  (`transverseSpacings`) rather than as a length, because the constraint it has
  to satisfy is a multiple of the particle spacing: a slab narrower than twice
  the largest support radius lets a particle interact with its own periodic
  image, silently. Fixing the width in spacings makes the transverse particle
  count independent of `nx` (so the count grows as `nx * spacings**(dim-1)`,
  linearly in resolution) and keeps that constraint satisfied at every
  resolution instead of only the one the default was picked at.
  `buildSodND` checks it against the supports it actually produced and raises
  rather than running a quietly wrong simulation.

Equal mass, not equal spacing
-----------------------------

The point of this sampler. Giving both states the same lattice would make the
dense side's particles four times heavier than the light side's (Sod's default
``rho_l/rho_r = 4``), and a 4:1 mass jump across the contact discontinuity is
exactly where SPH's density estimate misbehaves. So the light side is sampled
*coarser* instead, by ``(rho_l/rho_r)**(1/dim)`` in every direction, which
keeps ``mass = cell volume * density`` equal on both sides. That is the same
trade the 1D case already makes -- its ``samplingRatio=4`` default is
``rho_l/rho_r``, which is why 1D masses come out exactly equal -- and the same
one `sampleTriplePointEqualMass` makes in 2D with its ``sqrt(8)``.

"Coarser by ``ratio**(1/dim)``" is not generally an integer number of
particles, though, and both of the light side's counts have to be integers for
its lattice to close on the periodic box. In 3D it cannot be made exact at all:
``4**(1/3)`` is irrational, so no pair of commensurate periodic lattices has
exactly equal masses. What the sampler does instead is pick

1. the transverse count, from the isotropic ideal ``dx * ratio**(1/dim)``, and
2. the x count, to make ``mass = cell volume * density`` match the dense side
   as closely as one integer allows, given (1),

and then retry (1) one count either side, keeping whichever pair matches mass
best among those whose cells stay within `MAX_CELL_ASPECT` of cubic. That last
step is worth its five lines: at ``dim=3, nx=100`` it turns a 1.4% mass error
into 0.29%, and without the aspect bound it would happily take 0.04% at the
price of cells stretched by a third, which the (isotropic) kernel would feel
more than the mass jump it bought off.

Measured over ``nx`` from 25 to 400: exact in 1D and 2D at every resolution
whose counts divide evenly, otherwise within 4.2%; within 1.4% in 3D, with
cells never worse than 1.17:1. Both numbers are reported by
`sodSamplingReport` at build time and asserted in `tests/test_physics.py`.
Compare 75%, which is what equal *spacing* would give.

`buildSod1D` is deliberately left alone rather than being reimplemented on top
of this: it carries an explicit `samplingRatio` knob rather than deriving the
count, and the backprop notebook's recorded numbers are tied to its exact
output. This builder does handle ``dim=1``, where it agrees with `buildSod1D`
on spacings and masses but differs in one respect, on purpose: it lays the
light state out as a cell-centred lattice on the *periodic interval*
``[L/4, 3L/4]``, wrapped back into the box, instead of sampling a block
centred on the origin and pushing its two halves apart. Both give the same
mirror-symmetric arrangement, but the 1D path's ``pos[pos < 0] -= L/4`` /
``pos[pos > 0] += L/4`` leaves a particle sitting at ``x = 0`` -- in the middle
of the *dense* state, carrying the light state's mass -- whenever
``nx // samplingRatio`` is odd (``nx=100`` does it; the ``nx=800`` default does
not). `tests/test_physics.py` checks the agreement and this difference.
"""

from __future__ import annotations

from warpSPH.configurations.compute import neighborSearchOptions

import math
from typing import NamedTuple

import torch

from warpSPHCore import SupportScheme, buildVerletList

from ....configurations.compressibleConfig import CompressibleSPHConfig
from ....enumTypes import AdaptiveSupportScheme
from ....modules import evaluateOptimalSupport, idealGasEOS
from ....modules.timestep.compressible import computeTimestep
from ....utils.support import volumeToSupport
from .sod import sodInitialState

__all__ = ['buildSodND', 'sodSampling', 'sodSamplingReport', 'SodSampling',
           'MAX_CELL_ASPECT']

#: How far from cubic a light-side cell may get while chasing an equal mass.
#: Only binding at the finer resolutions, where the extra transverse count on
#: offer would buy a third decimal of mass match for a third of a cell width.
MAX_CELL_ASPECT = 1.2


class SodSampling(NamedTuple):
    """The two lattices `buildSodND` samples, as counts and spacings.

    `dense`/`light` each hold ``(nx, nTransverse, dx, dTransverse, mass)``.
    Both states span half the periodic domain, so both `nx` counts cover an
    x-extent of ``L/2`` -- contiguous for the dense state, split across the two
    outer quarters for the light one.
    """

    dense: tuple
    light: tuple
    transverseExtent: float
    dim: int

    @property
    def massRatio(self) -> float:
        """Light-side particle mass over dense-side. 1.0 is the goal."""
        return self.light[4] / self.dense[4]

    @property
    def anisotropy(self) -> float:
        """Worst cell aspect ratio over both lattices. 1.0 is a cubic cell."""
        if self.dim == 1:
            return 1.0
        return max(max(dx, dt) / min(dx, dt)
                   for _, _, dx, dt, _ in (self.dense, self.light))


def sodSampling(nx: int, dim: int, L: float, transverseSpacings: int,
                leftState: sodInitialState, rightState: sodInitialState,
                equalMass: bool = True) -> SodSampling:
    """Resolve the two lattices, without sampling anything yet.

    Pure integer/float arithmetic, so a case (or a test) can ask what a given
    `nx` would produce -- particle counts, mass match, cell anisotropy --
    before paying for the sampling itself.
    """
    if dim not in (1, 2, 3):
        raise ValueError(f'dim must be 1, 2 or 3, got {dim}')

    # Both states fill half the (periodic) domain: the dense one the middle
    # block |x| <= L/4, the light one the interval [L/4, 3L/4], which wraps
    # around to become the two outer quarters.
    denseExtent = lightExtent = L / 2

    denseDx = denseExtent / nx
    # Measuring the slab in dense-side spacings makes the dense lattice exactly
    # isotropic and exactly commensurate with the periodic box by construction,
    # with no rounding at all on this side.
    denseNt = int(transverseSpacings) if dim > 1 else 1
    transverse = denseNt * denseDx if dim > 1 else 0.0
    denseDt = denseDx if dim > 1 else 1.0
    denseMass = denseDx * denseDt ** (dim - 1) * leftState.rho

    def lightLattice(nt: int):
        """The best x count for a given transverse count, and what it costs."""
        dt = transverse / nt if dim > 1 else 1.0
        # The one free integer, spent on matching the mass rather than on
        # isotropy -- see this module's docstring.
        n = max(1, round(lightExtent * dt ** (dim - 1) * rightState.rho / denseMass))
        dx = lightExtent / n
        mass = dx * dt ** (dim - 1) * rightState.rho
        aspect = max(dx, dt) / min(dx, dt) if dim > 1 else 1.0
        return (abs(mass / denseMass - 1), aspect), (n, nt, dx, dt, mass)

    if equalMass:
        ratio = leftState.rho / rightState.rho
        isotropic = max(1, round(transverse / (denseDx * ratio ** (1 / dim)))) if dim > 1 else 1
        # Rounding the transverse count to the isotropic ideal and *then*
        # rounding the x count leaves whatever mass error the first rounding
        # forced on the second. Trying the neighbouring transverse counts costs
        # nothing and regularly beats it: 1.4% -> 0.8% at the 3D default,
        # 1.4% -> 0.29% at dim=3, nx=100.
        candidates = sorted({max(1, isotropic + k) for k in (-1, 0, 1)})
        # Mass first, but only among candidates whose cells stay near-cubic:
        # squashing a cell by a third to buy another decimal of mass match is a
        # bad trade, since the kernel is isotropic and the density estimate
        # feels the stretch directly.
        _, (lightNx, lightNt, lightDx, lightDt, lightMass) = min(
            (lightLattice(nt) for nt in candidates),
            key=lambda entry: (entry[0][1] > MAX_CELL_ASPECT,) + entry[0])
    else:
        lightNt, lightDt = denseNt, denseDt
        lightNx = max(1, round(lightExtent / denseDx))
        lightDx = lightExtent / lightNx
        lightMass = lightDx * lightDt ** (dim - 1) * rightState.rho

    return SodSampling(
        dense=(nx, denseNt, denseDx, denseDt, denseMass),
        light=(lightNx, lightNt, lightDx, lightDt, lightMass),
        transverseExtent=transverse, dim=dim,
    )


def sodSamplingReport(sampling: SodSampling) -> str:
    """One-line-per-side summary, printed by `buildSodND` unless quiet."""
    dim = sampling.dim
    lines = []
    for name, (n, nt, dx, dt, mass) in (('dense (left) ', sampling.dense),
                                        ('light (right)', sampling.light)):
        shape = 'x'.join([str(n)] + [str(nt)] * (dim - 1))
        lines.append(f'  {name}: {shape} = {n * nt ** (dim - 1):>8d} particles, '
                     f'dx={dx:.5g}' + (f', dTransverse={dt:.5g}' if dim > 1 else '')
                     + f', mass={mass:.5g}')
    lines.append(f'  mass ratio light/dense: {sampling.massRatio:.6f} '
                 f'({abs(sampling.massRatio - 1):.3%} off equal mass), '
                 f'worst cell aspect: {sampling.anisotropy:.4f}')
    return '\n'.join(lines)


def _lattice(xAxis: torch.Tensor, transverse: float, nt: int, dim: int,
             device, dtype) -> torch.Tensor:
    """Extrude `xAxis` over `dim - 1` cell-centred transverse axes."""
    axes = [xAxis]
    for _ in range(dim - 1):
        dt = transverse / nt
        axes.append(torch.linspace(-transverse / 2 + dt / 2, transverse / 2 - dt / 2, nt,
                                   device=device, dtype=dtype))
    grid = torch.meshgrid(*axes, indexing='ij')
    return torch.stack([g.reshape(-1) for g in grid], dim=-1)


def _cellCentred(low: float, high: float, n: int, device, dtype) -> torch.Tensor:
    """`n` cell centres spanning `[low, high]`."""
    dx = (high - low) / n
    return torch.linspace(low + dx / 2, high - dx / 2, n, device=device, dtype=dtype)


def buildSodND(
    SimulationSystem, SimulationState,
    leftState: sodInitialState,
    rightState: sodInitialState,
    gamma: float,
    config,
    transverseSpacings: int,
    equalMass: bool = True,
    adaptiveSupportScheme: AdaptiveSupportScheme = AdaptiveSupportScheme.Owen,
    verbose: bool = True,
):
    """Sample a 1D/2D/3D Sod shock tube, equal-mass by default.

    `config.domain`'s transverse bounds are overwritten with the lattice-snapped
    extent (and `config.dx`/`config.dt` are set), the same way `buildSod1D`
    writes back the timestep it derives -- the sampling is what decides them.
    """
    dim, device, dtype = config.dim, config.device, config.dtype
    L = float(config.domain.max[0] - config.domain.min[0])

    sampling = sodSampling(config.nx, dim, L, transverseSpacings, leftState, rightState,
                           equalMass=equalMass)
    (denseNx, denseNt, denseDx, denseDt, denseMass) = sampling.dense
    (lightNx, lightNt, lightDx, lightDt, lightMass) = sampling.light
    transverse = sampling.transverseExtent

    if verbose:
        print(f'Sod {dim}D, L={L:g}' +
              (f', transverse slab {transverse:g} = {transverseSpacings} dense spacings'
               if dim > 1 else '') + ':')
        print(sodSamplingReport(sampling))

    # The periodic box has to agree with the lattice exactly in the transverse
    # directions, or the wrap-around spacing is wrong.
    if dim > 1:
        domainMin = config.domain.min.clone()
        domainMax = config.domain.max.clone()
        domainMin[1:] = -transverse / 2
        domainMax[1:] = transverse / 2
        config.domain.min, config.domain.max = domainMin, domainMax

    # The dense state in the middle block, and the light state on the periodic
    # interval [L/4, 3L/4] wrapped back into the box -- which is the two outer
    # quarters, mirror-symmetric about x=0, with no point stranded at x=0 in
    # the middle of the dense state (see the module docstring). Laying it out
    # as one wrapped interval rather than two independent quarter-blocks also
    # halves the granularity of the equal-mass rounding, since the count is
    # then free to be odd.
    lightX = _cellCentred(L / 4, 3 * L / 4, lightNx, device, dtype)
    lightX = torch.where(lightX >= L / 2, lightX - L, lightX)
    blocks = [
        (_lattice(_cellCentred(-L / 4, L / 4, denseNx, device, dtype),
                  transverse, denseNt, dim, device, dtype), 0),
        (_lattice(lightX, transverse, lightNt, dim, device, dtype), 1),
    ]
    positions = torch.cat([block for block, _ in blocks], dim=0)
    materials = torch.cat([torch.full((block.shape[0],), tag, dtype=torch.int32, device=device)
                           for block, tag in blocks], dim=0)

    isDense = materials == 0
    masses = torch.where(isDense, denseMass, lightMass).to(dtype)
    densities = torch.where(isDense, leftState.rho, rightState.rho).to(dtype)
    velocities = torch.zeros_like(positions)
    velocities[:, 0] = torch.where(isDense, float(leftState.v), float(rightState.v)).to(dtype)

    # Support from each side's own cell volume, so the coarser light side gets
    # the wider kernel its spacing calls for.
    supports = torch.where(
        isDense,
        volumeToSupport(denseDx * denseDt ** (dim - 1), config.targetNeighbors, dim),
        volumeToSupport(lightDx * lightDt ** (dim - 1), config.targetNeighbors, dim),
    ).to(dtype)

    UIDs = torch.arange(positions.shape[0], dtype=torch.int32, device=device)
    particleState = SimulationState(
        positions=positions,
        velocities=velocities,
        supports=supports,
        masses=masses,
        densities=densities,

        kinds=torch.zeros_like(materials),
        materials=materials,
        UIDs=UIDs,
        UIDcounter=UIDs.max() + 1,

        internalEnergies=None,
        totalEnergies=None,
        entropies=None,
        pressures=None,
        soundspeeds=None,

        divergence=torch.zeros_like(densities),
        alpha0s=torch.ones_like(densities),
        alphas=torch.ones_like(densities),
    )

    # Relax the supports onto the sampling, then take the EOS off the analytic
    # density -- `buildSod1D`'s `smoothIC=False` path, which is Sod's default.
    # (The SPH-estimated density is deliberately not used here: it is smoothed
    # across the interface, and the pressure it would produce no longer matches
    # the initial condition being posed. That is what `smoothIC` exists to fix
    # in 1D, and it is a 1D-only knob -- this builder always samples a sharp
    # interface.)
    compParams = CompressibleSPHConfig(
        gamma=gamma, adaptiveSupportCorrections=True, adaptiveSupportIterations=16,
        adaptiveSupportThreshold=1e-3, adaptiveSupportScheme=adaptiveSupportScheme)
    _, h_optimal, *_ = evaluateOptimalSupport(particleState, config, compParams,
                                              supportScheme=SupportScheme.Gather)
    particleState.supports = h_optimal

    # A support radius reaching more than half way around the periodic slab
    # lets a particle interact with its own image, and nothing downstream
    # detects that -- it just returns wrong densities. Checked against the
    # relaxed supports, since those are what the run actually uses; the light
    # (coarser) side is what sets it.
    if dim > 1:
        widest = float(h_optimal.max())
        if 2 * widest > transverse:
            raise ValueError(
                f'Transverse slab of {transverse:g} ({transverseSpacings} dense spacings) is '
                f'narrower than twice the largest support radius ({widest:g}), so particles '
                f'would interact with their own periodic images. Raise transverseSpacings to '
                f'at least {math.ceil(2 * widest / denseDx)}.')

    pressures = torch.where(isDense, leftState.p, rightState.p).to(dtype)
    u = pressures / ((gamma - 1) * particleState.densities)
    A_, u_, P_, c_s = idealGasEOS(A=None, u=u, P=None, rho=particleState.densities, gamma=gamma)

    kineticEnergy = torch.linalg.norm(velocities, dim=-1) ** 2 / 2
    particleState.internalEnergies = u_
    particleState.totalEnergies = (u_ + kineticEnergy) * masses
    particleState.pressures = P_
    particleState.soundspeeds = c_s
    particleState.entropies = A_

    adjacency = buildVerletList(
        particleState,
        domain=config.domain,
        verletScale=2 ** (1 / dim),
        supportMode=config.supportMode,
        **neighborSearchOptions(config))
    system = SimulationSystem(state=particleState, adjacency=adjacency, domain=config.domain)

    config.dx = denseDx
    config.dt = computeTimestep(system, config, compParams, dt=None)
    return system
