"""Probe (`ACSPH_PLAN.md` Part 8 step 9 / §4.4, `impact`): the instantaneous
kinetic-energy drop on normal impact of two identical 2D rectangular jets,
against Marrone et al. 2015's exact closed-form (`literature/marrone2015`,
their Appendix A, Eq. A.34) -- the analytic reference `ACSPH_PLAN.md` Part 7
names but had not yet used.

Marrone's own case (their §5.2) is two identical rectangular jets, half-width
`H` (transverse), length `L` (from the impact plane to the free end),
impacting head-on at `t=0`. Their incompressible pressure-impulse solution
gives the *exact* instantaneous energy ratio right after impact,

    Ek(0+)/Ek(0-) = (8/pi^2) * Sum_{n=1..inf} 1/(2n-1)^2 * (1 - tanh((2n-1)r)/((2n-1)r))
    r = pi*L/(2H)                                                        (Eq. A.34)

with the two limits `r->0` (thin jet, fully inelastic, ratio->0) and `r->inf`
(filament on a rigid wall, no loss, ratio->1) as sanity checks the series
itself is verified against below.

**Mapping onto `impactCase`'s own parameterisation**: `shapeArgs('box', size,
aspect) = [[size, size*aspect]]`, i.e. half-extents `(H, size*aspect)` -- so
`size=H` directly, and since each jet's near edge sits at the impact plane
once `touching=True` closes the gap, the far (free) edge sits at
`2*(size*aspect)` from the plane, i.e. `L = 2*size*aspect`. That gives the
clean identity `r = pi*L/(2H) = pi*aspectRatio` -- Marrone's own aspect ratio
IS this case's own `aspectRatio` param, times pi, no further conversion
needed.

**Before/after KE**: no gravity, no restoring potential in `impactCase`, so
before contact the two jets simply translate at their prescribed velocity --
`KE(0-)` is exactly the initial diagnostic value, unaffected by anything
upstream of contact. `KE(0+)` is read off the first step where `impact.py`'s
own `gap` diagnostic (nearest-particle separation between the two bodies)
first drops below `contactGapDx * dx`, i.e. the step contact is actually
established -- ACSPH has no acoustic ringing to wait out, so this is a
direct, not a filtered/time-averaged, reading.

Usage:
  python scripts/probe_acsphImpactEnergyLoss.py [--nx 64] [--aspectRatio 0.5]
      [--contactGapDx 1.5]
"""
from __future__ import annotations

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=64)
parser.add_argument('--aspectRatio', type=float, nargs='+', default=[0.25, 0.5, 1.0])
parser.add_argument('--size', type=float, default=0.5)
parser.add_argument('--impactVelocity', type=float, default=1.0)
parser.add_argument('--contactGapDx', type=float, default=1.5,
                     help='gap (in units of dx) below which contact counts as established')
parser.add_argument('--maxSteps', type=int, default=400)
args = parser.parse_args()


def _analyticEnergyRatio(r: float, nTerms: int = 200) -> float:
    """Eq. (A.34), summed to `nTerms` (converges fast -- `1/(2n-1)^2` tail)."""
    total = 0.0
    for n in range(1, nTerms + 1):
        k = 2 * n - 1
        total += (1.0 / k ** 2) * (1.0 - math.tanh(k * r) / (k * r))
    return (8.0 / math.pi ** 2) * total


def _selfCheck():
    """Sanity: the two limits the paper itself states (Eqs. A.35-A.36)."""
    r0 = _analyticEnergyRatio(1e-6)
    rInf = _analyticEnergyRatio(50.0)
    print(f"[self-check] r->0 limit: {r0:.6f} (expect ~0); "
          f"r->inf limit: {rInf:.6f} (expect ~1)")


_selfCheck()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import numpy as np

from warpSPH.cases.impact import impactCase as case
from warpSPH.runner import run


def _configure(aspectRatio):
    base = case.configureScheme

    def wrapped(ctx):
        base(ctx)

    return wrapped


hdr = (f"{'aspectRatio':>11} {'r=pi*aspect':>11} {'analyticRatio':>13} "
       f"{'measuredRatio':>13} {'contactStep':>11} {'KE(0-)':>9} {'KE(0+)':>9}")
print(hdr)
print('-' * len(hdr))

for aspectRatio in args.aspectRatio:
    r = math.pi * aspectRatio
    analyticRatio = _analyticEnergyRatio(r)

    result = run(case, scheme='artificialCompressible', nx=args.nx, nSteps=args.maxSteps,
                params=dict(shape='box', size=args.size, aspectRatio=aspectRatio,
                            arrangement='pair', impactAxis=1, touching=True, gap=1.0,
                            impactVelocity=args.impactVelocity, impactAngle=0.0, spin=0.0,
                            lateralOffset=0.0),
                store=False, plot=False, quiet=True, progress=False)

    tr = [row for row in result.trajectory if 'gap' in row and 'kineticEnergy' in row]
    if not tr:
        print(f"{aspectRatio:11.3f} {r:11.4f} {analyticRatio:13.4f}   (no gap/KE recorded)")
        continue

    ke0 = tr[0]['kineticEnergy']
    dx = result.ctx.config.dx
    contactRow = next((row for row in tr if row['gap'] <= args.contactGapDx * dx), None)
    if contactRow is None:
        print(f"{aspectRatio:11.3f} {r:11.4f} {analyticRatio:13.4f}   "
              f"(never made contact within {args.maxSteps} steps; min gap "
              f"{min(row['gap'] for row in tr):.4g}, dx={dx:.4g})")
        continue

    kePost = contactRow['kineticEnergy']
    measuredRatio = kePost / ke0
    print(f"{aspectRatio:11.3f} {r:11.4f} {analyticRatio:13.4f} {measuredRatio:13.4f} "
          f"{contactRow['step']:11d} {ke0:9.4f} {kePost:9.4f}", flush=True)
