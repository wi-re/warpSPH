#!/usr/bin/env python3
"""Is `modules/deltaSPH/wp_densityDelta.py`'s `psi_ij` the operator the papers
write? (ACSPH_PLAN.md step 3.)

ACSPH reuses this kernel as its pressure-smoothing operator: De Courcy et al.
2024's AC-2 (Eq. 32) is the `densityOnly` branch and AC-2L (Eq. 33) is the
`deltaSPH` branch with the pressure in place of the density. Part 3 of the plan
argues they are the same operator up to a projection that is a no-op for an
isotropic kernel. This probe checks that argument numerically, against a
from-scratch `O(N^2)` torch reference that shares no code with the kernel under
test (its own Wendland C2, its own neighbour loop) -- the point being to catch
exactly the class of defect an internal consistency check cannot see.

Three claims, one command:

  (1) The warp kernel == the torch reference, unprojected form.
      Validates the reference. Anything but ~1e-13 in float64 means one of the
      two is wrong and the rest of the output is meaningless.

  (2) Projected (De Courcy Eq. 33, gradient contracted onto `xhat_ij` first)
      == unprojected (Marrone et al. 2011 Eq. 6), when the kernel gradient is
      not renormalised. `((g_i+g_j).xhat)(xhat.gradW) = W'(r)(g_i+g_j).x_ij/r
      = (g_i+g_j).gradW` for any `gradW || x_ij`, i.e. any isotropic kernel.

  (3) ... and they genuinely differ once `useGradientRenormalization` puts an
      `L_i` in front of `gradW`, which breaks that parallelism. This is the
      caveat Part 3 flags: ACSPH must keep gradient renormalisation off on this
      operator, or implement the projected form explicitly.

Plus the property that makes the operator what it is, on both forms:

  (4) A field linear in space is annihilated pair-by-pair, and a quadratic one
      is annihilated up to discretisation error -- the Antuono correction turns
      the Molteni-Colagrossi Laplacian into a bi-Laplacian. **This is what
      found the sign error fixed on 2026-09-05** (the gradient term entered
      `psi` with the wrong sign, so the two terms added instead of cancelling
      and the operator was twice the *uncorrected* Laplacian). It is pinned as
      a regression test in `tests/test_deltaSPHDiffusion.py`; it is repeated
      here because this script is where the two forms are compared side by side.

    python scripts/probe_deltaSPHPsiProjection.py
    python scripts/probe_deltaSPHPsiProjection.py --n 16 --jitter 0.2
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=20, help='particles per side')
parser.add_argument('--jitter', type=float, default=0.05,
                    help='lattice disorder, in units of dx (0 = perfect lattice, '
                         'which makes several of these cancel for the wrong reason)')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

import math

from warpSPHBootstrap import bootstrap

bootstrap(precision='float64')

import torch

from warpSPHCore import (GradientScheme, OperationDirection, OperationProperties,
                         ParticleState, SupportScheme, WarpOperation,
                         computeRenormalizationMatrices, warpOperation)
from warpSPH.configurations.simulationConfig import SimulationConfig
from warpSPH.enumTypes import DensityDiffusionScheme
from warpSPH.modules.deltaSPH import computeScalarFieldDiffusion
from warpSPH.modules.density.gradRhoL import computeGradRhoL
from warpSPH.utils.domain import buildDomainDescription

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
DTYPE = torch.float64


# --------------------------------------------------------------------------
# The independent reference. Wendland C2 in 2D, written out from Dehnen & Aly
# rather than imported, so a defect in warpSPHCore's kernel cannot hide here.
# --------------------------------------------------------------------------

def wendland2GradTorch(xij, h):
    """`grad_i W(x_i - x_j, h)`, shape-preserving over the pair axes. Equals
    warpSPHCore's `sphGradient_` for `dim=2`: `xhat * dk/dq * C_d / h^(d+1)`
    with `k(q) = (1-q)^4 (1+4q)`, `C_d = 7/pi`, and zero outside `q <= 1`."""
    r = xij.norm(dim=-1)
    q = (r / h).clamp(max=1.0)
    dkdq = -20.0 * q * (1.0 - q) ** 3
    cD = 7.0 / math.pi
    scale = torch.where(r > 0, dkdq * cD / h ** 3 / r.clamp_min(1e-300),
                        torch.zeros_like(r))
    return scale.unsqueeze(-1) * xij


def referenceOperator(positions, volumes, h, field, gradField, projected, L=None):
    """`sum_j psi_ij . gradW_ij V_j`, dense over all pairs.

    `projected=False`  -- Marrone et al. 2011 Eq. (6):
        psi_ij = -(g_i + g_j) - 2 (f_j - f_i) x_ij / |x_ij|^2
    `projected=True`   -- De Courcy et al. 2024 Eq. (33):
        psi.gradW = 2 { (f_i - f_j) - (g_i + g_j).x_ij / 2 } (x_ij.gradW)/|x_ij|^2

    `L` (per-particle, `[N, d, d]`) renormalises `gradW_ij -> L_i gradW_ij`,
    which is the switch claim (3) is about."""
    xij = positions.unsqueeze(1) - positions.unsqueeze(0)          # x_i - x_j
    r2 = (xij ** 2).sum(-1)
    gradW = wendland2GradTorch(xij, h)
    if L is not None:
        gradW = torch.einsum('iab,ijb->ija', L, gradW)
    g = gradField.unsqueeze(1) + gradField.unsqueeze(0)            # g_i + g_j
    df = field.unsqueeze(1) - field.unsqueeze(0)                   # f_i - f_j
    inv = torch.where(r2 > 0, 1.0 / r2.clamp_min(1e-300), torch.zeros_like(r2))

    xdotGrad = (xij * gradW).sum(-1)
    if projected:
        brace = 2.0 * (df - 0.5 * (g * xij).sum(-1))
        contrib = brace * xdotGrad * inv
    else:
        contrib = -(g * gradW).sum(-1) + 2.0 * df * xdotGrad * inv
    return (contrib * volumes.unsqueeze(0)).sum(-1)


# --------------------------------------------------------------------------

def buildCase():
    n, dx = args.n, 1.0 / args.n
    xs = (torch.arange(n, device=DEVICE, dtype=DTYPE) + 0.5) * dx
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')
    pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1).contiguous()
    torch.manual_seed(args.seed)
    pos = pos + args.jitter * dx * torch.randn_like(pos)
    N = pos.shape[0]
    config = SimulationConfig(
        device=DEVICE, dtype=DTYPE, dim=2,
        domain=buildDomainDescription(l=4.0, dim=2, periodic=False,
                                      device=DEVICE, dtype=DTYPE))
    h = config.n_h * dx
    state = ParticleState(
        positions=pos,
        supports=torch.full((N,), h, device=DEVICE, dtype=DTYPE),
        masses=torch.full((N,), dx ** 2, device=DEVICE, dtype=DTYPE),
        densities=torch.ones(N, device=DEVICE, dtype=DTYPE),
        kinds=torch.zeros(N, dtype=torch.int32, device=DEVICE))
    # Only rows a full support radius from every wall are comparable: the
    # reference has no boundary treatment either, but truncation makes the
    # absolute magnitudes meaningless there.
    interior = ((pos > 1.5 * h) & (pos < 1.0 - 1.5 * h)).all(-1)
    return state, config, h, interior


def main():
    state, config, h, interior = buildCase()
    pos = state.positions
    volumes = state.masses / state.densities
    _, _, L = computeRenormalizationMatrices(
        queryParticles=state,
        operationProperties=OperationProperties(
            kernel=config.kernel, operation=WarpOperation.Gradient,
            operationMode=OperationDirection.AllToAll,
            supportMode=SupportScheme.SuperSymmetric),
        domain=config.domain, returnEigVals=True)
    Lmat = L.renormalizationMatrices if hasattr(L, 'renormalizationMatrices') else L

    a = torch.tensor([0.7, -1.3], device=DEVICE, dtype=DTYPE)
    fields = {
        'linear   f = a.x + c': (pos @ a + 2.0, None),
        'quadratic f = |x|^2 ': ((pos ** 2).sum(-1), None),
        'cubic     f = x^3+y^3': ((pos ** 3).sum(-1), None),
    }

    print(f"\n=== n={args.n}^2 = {pos.shape[0]} particles, jitter={args.jitter} dx, "
          f"h={h:.4g}, {int(interior.sum())} interior rows, float64 ===")

    def rms(v):
        return float(v[interior].pow(2).mean().sqrt())

    print("\n(1)+(2)  warp kernel vs torch reference vs projected form"
          "  [gradient renormalisation OFF]")
    print(f"{'field':>22} {'|warp|':>12} {'warp-unproj':>13} {'unproj-proj':>13}")
    for label, (f, _) in fields.items():
        gradFieldL = computeGradRhoL(state, config, None, None, L, field=f)
        warpD = computeScalarFieldDiffusion(
            state, config, None, DensityDiffusionScheme.deltaSPH,
            gradFieldL=gradFieldL, field=f)
        unproj = referenceOperator(pos, volumes, h, f, gradFieldL, projected=False)
        proj = referenceOperator(pos, volumes, h, f, gradFieldL, projected=True)
        print(f"{label:>22} {rms(warpD):12.4e} {rms(warpD - unproj):13.4e} "
              f"{rms(unproj - proj):13.4e}")

    print("\n(3)  the same two forms with `L_i gradW_ij` in place of `gradW_ij`"
          "  [renormalisation ON]")
    print(f"{'field':>22} {'|unproj|':>12} {'unproj-proj':>13} {'relative':>10}")
    for label, (f, _) in fields.items():
        gradFieldL = computeGradRhoL(state, config, None, None, L, field=f)
        unproj = referenceOperator(pos, volumes, h, f, gradFieldL, projected=False, L=Lmat)
        proj = referenceOperator(pos, volumes, h, f, gradFieldL, projected=True, L=Lmat)
        rel = rms(unproj - proj) / max(rms(unproj), 1e-300)
        print(f"{label:>22} {rms(unproj):12.4e} {rms(unproj - proj):13.4e} {rel:10.3f}")

    print("\n(4)  what the corrected operator annihilates (warp kernel, "
          "rms over interior)")
    print(f"{'field':>22} {'deltaSPH (AC-2L)':>18} {'densityOnly (AC-2)':>20}")
    for label, (f, _) in fields.items():
        gradFieldL = computeGradRhoL(state, config, None, None, L, field=f)
        ac2l = computeScalarFieldDiffusion(
            state, config, None, DensityDiffusionScheme.deltaSPH,
            gradFieldL=gradFieldL, field=f)
        ac2 = computeScalarFieldDiffusion(
            state, config, None, DensityDiffusionScheme.densityOnly, field=f)
        print(f"{label:>22} {rms(ac2l):18.4e} {rms(ac2):20.4e}")
    print("\nA linear field must give ~0 for AC-2L and NOT for AC-2 -- that is "
          "\nDe Courcy Sec. 4.1.1's discriminator (AC-2 cannot hold a hydrostatic "
          "\ngradient) reduced to one number.\n")


if __name__ == '__main__':
    main()
