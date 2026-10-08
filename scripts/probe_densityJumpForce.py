#!/usr/bin/env python3
"""The Cha, Inutsuka & Nayakshin (2010) Fig. 1 test (GODUNOV_SPH_PLAN): the acceleration of particles across a density jump in exact
pressure equilibrium. The exact answer is zero everywhere; the standard SPH momentum equation gives a repulsion at the jump (a loss of
zeroth-order consistency), which damps Kelvin-Helmholtz modes and opens a gap in the contact. The simplified Godunov SPH (Cha & Whitworth) is
identically the standard SPH force at constant pressure; Inutsuka's convolution form (layer 3) must vanish.

The state is the 1D Sod lattice with `rho = 1 | 0.25` (two jumps: the middle and the periodic seam), the particle pressures then set to exactly
`P0` and the velocities to zero. Prints the spurious force per scheme and writes the profile figure.

    python scripts/probe_densityJumpForce.py [--out docs/av/godunov_densityJump] [--ratio 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
try:
    import warpSPHBootstrap  # noqa: F401
except ImportError:
    pass

from warpSPHCore import OperationProperties, SupportScheme                     # noqa: E402
from warpSPH.cases.sod import sodCase                                           # noqa: E402
from warpSPH.runner import run                                                  # noqa: E402

P0 = 1.0
GAMMA = 1.4


def contactState(ratio: float = 4.0, scheme: str = 'Monaghan', uniformSupport: bool = False):
    """`(system, config)` of the 1D lattice with a `ratio : 1` density jump, particle pressures exactly `P0`, at rest."""
    res = run(sodCase, scheme=scheme, nSteps=1, progress=False, quiet=True,
              params=dict(left_rho=1.0, right_rho=1.0 / ratio, left_pressure=P0, right_pressure=P0,
                          left_velocity=0.0, right_velocity=0.0, smoothIC=False))
    system, config = res.state, res.ctx.config
    st = system.state
    st.velocities = torch.zeros_like(st.velocities)
    st.pressures = torch.full_like(st.pressures, P0)
    if uniformSupport:
        # Inutsuka's derivation (identity Eq. 14) is exact for a spatially constant h; with h following the density the two halves h_i / h_j are an approximation
        st.supports = torch.full_like(st.supports, float(st.supports.mean()))
    return system, config


def sphForce(system, config):
    from warpSPH.modules.pressure import computePressureForceSymmetric
    return computePressureForceSymmetric(system.state, config, supportScheme=SupportScheme.KernelMeanSymmetric,
                                         adjacency=system.adjacency)


def gsphForce(system, config, order=1):
    from warpSPH.configurations.gsph import GSPHConfig
    from warpSPH.configurations.moduleConfigurations.diffusionParameters import (
        RiemannSolver, VelocityPairPolicy, resolveReconstructionLimiter)
    from warpSPH.modules.godunov import computeGodunovWarp
    from warpSPH.modules.reconstruction import computeStateGradients, computeVelocityJacobian
    st = system.state
    p = GSPHConfig().diffusionParams
    p.riemannSolver = RiemannSolver.Adaptive.value
    p.velocityPairPolicy = VelocityPairPolicy.Limited.value if order == 2 else VelocityPairPolicy.Raw.value
    p = resolveReconstructionLimiter(p, config.n_h)
    J = G = None
    if order == 2:
        J = computeVelocityJacobian(st, config, SupportScheme.SuperSymmetric, system.adjacency)
        G = computeStateGradients(st, config, SupportScheme.SuperSymmetric, system.adjacency)
    a, _ = computeGodunovWarp(st, OperationProperties(kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric),
                              domain=config.domain, params=p, gamma=GAMMA, order=order, adjacency=system.adjacency,
                              queryVelocityTensor=J, queryStateGradients=G)
    return a


def inutsukaForce(system, config, order=1, cubic=True, widthFactor=1.4142135623730951, hMode='spacing'):
    """Inutsuka (2002): Gaussian density (re-summed on the same particles; the pressures stay exactly `P0`) and the convolution-form force."""
    from warpSPH.configurations.gsph import InutsukaGSPHConfig
    from warpSPH.configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy, resolveReconstructionLimiter
    from warpSPH.modules.godunov import computeGaussianDensityWarp, computeInutsukaWarp
    from warpSPH.modules.reconstruction import computeStateGradients, computeVelocityJacobian
    from warpSPHCore import buildVerletList
    st = system.state
    # the Gaussian's h: eta x the local particle spacing (the scheme's choice); the list must reach 3 sqrt2 h
    # `spacing`: the scheme's Gaussian h = the local particle spacing (eta = 1); `support`: a wider Gaussian, h = support / 3
    hG = (st.masses / st.densities) ** (1.0 / config.dim) if hMode == 'spacing' else st.supports / 3.0
    if float(st.supports.std() / st.supports.mean()) < 1e-6:        # the constant-h case (`--uniformSupport`): constant Gaussian h as well
        hG = torch.full_like(hG, float(hG.mean()))
    reach = float((3.0 * 2.0 ** 0.5 * hG / st.supports).max())
    adj = buildVerletList(st, config.domain, verletScale=max(1.5, 1.05 * reach), supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=None, verbose=False)
    props = OperationProperties(kernel=config.kernel, supportMode=SupportScheme.KernelMeanSymmetric)
    saved = st.densities
    gradRho, rho = computeGaussianDensityWarp(st, props, domain=config.domain, adjacency=adj, widthFactor=widthFactor, gaussianH=hG)
    st.densities = rho
    p = InutsukaGSPHConfig().diffusionParams
    p.velocityPairPolicy = VelocityPairPolicy.Limited.value if order == 2 else VelocityPairPolicy.Raw.value
    p = resolveReconstructionLimiter(p, config.n_h)
    J = G = None
    if order == 2:
        J = computeVelocityJacobian(st, config, SupportScheme.SuperSymmetric, adj)
        G = computeStateGradients(st, config, SupportScheme.SuperSymmetric, adj)
    a, _ = computeInutsukaWarp(st, props, domain=config.domain, params=p, gamma=GAMMA, gradRho=gradRho, order=order, cubic=cubic,
                               adjacency=adj, queryVelocityTensor=J, queryStateGradients=G, gaussianH=hG)
    st.densities = saved
    return a, rho


def jumpMask(system, width=3.0):
    """Particles within `width` smoothing lengths of a density jump (where the sorted densities cross the geometric mean of the two plateau
    values; the lattice is periodic, so the seam counts too)."""
    st = system.state
    x = st.positions[:, 0]
    rho = st.densities
    mid = torch.sqrt(rho.max() * rho.min())
    order = torch.argsort(x)
    xs, rs = x[order], rho[order]
    crossing = ((rs[:-1] - mid) * (rs[1:] - mid)) < 0
    xj = 0.5 * (xs[:-1] + xs[1:])[crossing]
    L = float(x.max() - x.min()) + float(xs[1] - xs[0])
    d = (x[:, None] - xj[None, :]).abs()
    d = torch.minimum(d, L - d).min(dim=1).values
    return d < width * st.supports


def spuriousForce(a, system):
    """`(max |a| at the jumps, max |a| in the bulk)` in units of `P0 / (rho h)`."""
    st = system.state
    scale = P0 / (st.densities * st.supports)
    m = jumpMask(system)
    r = (a[:, 0].abs() / scale)
    return float(r[m].max()), float(r[~m].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--out', default=str(REPO / 'docs' / 'av' / 'godunov_densityJump'))
    ap.add_argument('--ratio', type=float, default=4.0)
    ap.add_argument('--uniformSupport', action='store_true', help='constant smoothing length (the exact case of the derivation)')
    args = ap.parse_args()
    system, config = contactState(args.ratio, uniformSupport=args.uniformSupport)
    st = system.state
    forces = {'standard SPH (symmetric)': sphForce(system, config), 'GSPH simplified, 1st order': gsphForce(system, config, 1),
              'GSPH simplified, 2nd order': gsphForce(system, config, 2)}
    rhoG = None
    for name, kw in (('GSPH Inutsuka, cubic V (h = spacing)', dict(cubic=True)), ('GSPH Inutsuka, linear V (h = spacing)', dict(cubic=False)),
                     ('Inutsuka cubic, density width h', dict(cubic=True, widthFactor=1.0)),
                     ('GSPH Inutsuka, cubic V, h = support/3', dict(cubic=True, hMode='support')),
                     ('GSPH Inutsuka, linear V, h = support/3', dict(cubic=False, hMode='support'))):
        forces[name], rhoG = inutsukaForce(system, config, **kw)
    print(f'density ratio {args.ratio}, N = {st.positions.shape[0]}, P0 = {P0}: max |a| in units of P0/(rho h)')
    print(f'{"scheme":34s} {"at the jumps":>14s} {"in the bulk":>14s}')
    for name, a in forces.items():
        j, b = spuriousForce(a, system)
        print(f'{name:34s} {j:14.4f} {b:14.4f}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = st.positions[:, 0].cpu()
    order = torch.argsort(x)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(x[order], st.densities.cpu()[order], '.', ms=2)
    ax[0].set_ylabel('density')
    for name, a in forces.items():
        ax[1].plot(x[order], a[:, 0].cpu()[order], '.', ms=2, label=name)
    ax[1].set_ylabel('acceleration (exact: 0)')
    ax[1].set_xlabel('x')
    ax[1].legend(markerscale=4, fontsize=7)
    fig.suptitle(f'Pressure equilibrium across a {args.ratio:g}:1 density jump (Cha et al. 2010 Fig. 1)' + (', constant h' if args.uniformSupport else ', adaptive h'))
    fig.tight_layout()
    name = 'densityJumpForce_constantH.png' if args.uniformSupport else 'densityJumpForce.png'
    fig.savefig(out / name, dpi=130)
    print('wrote', out / name)


if __name__ == '__main__':
    main()
