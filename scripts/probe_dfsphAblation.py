"""Ablation to narrow down why PS-restore (DFSPH_FINDINGS.md 1.16) holds
`tgv` and `randomFlowIncompressible --bounded` but NaNs `hydrostaticColumn`.

`hydrostaticColumn` alone has {free surface, body force, quiescent start}.
Add those one at a time to a *periodic, no-wall* base that PS-restore already
holds, and see which one breaks it:

  obstacle_static -- periodic box + a static circular obstacle (solid internal
                     boundary, no free surface, no gravity).
  obstacle_osc    -- `drivenSquare`: the obstacle oscillates back and forth
                     (a *moving* solid boundary), still fluid otherwise.
  gravity_osc     -- periodic random flow + spatially-uniform gravity scaled
                     by sin(2*pi t / period) (a mean-zero *body force*, no
                     free surface, no walls).

Each run twice: baseline (as shipped) vs PS-restore (`_RESTORE_PS_SHIFT` +
`dfsph.INSTEP_CD = False`).

    python scripts/probe_dfsphAblation.py [nsteps]
"""
import dataclasses
import sys

import numpy as np

import warpSPH.systems.incompressible as I
from warpSPH.cases import drivenSquare, randomFlowIncompressible
from warpSPH.configurations.moduleConfigurations.gravity import GravityType
from warpSPH.runner import run
from warpSPH.schemes import dfsph as D

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
GRAV = (9.81, 1.0)  # (magnitude-scale, period) for the oscillating body force


def _reset():
    D.DIVERGENCE_SOLVER = 'omni'; D.XSPH_SCALE = 0.0
    D.SOLVE_ORDER = 'div_then_cd'; D.INSTEP_CD = True; D.GRAVITY_OSC = None
    I._RESTORE_PS_SHIFT = False; I._PS_SHIFT_MODE = 'cd'
    I._PS_POSITION_SHIFT = True; I._PS_VELOCITY_RESAMPLE = True
    I._PS_SHIFT_AS_VELOCITY = False


def _gravityOscCase():
    base = randomFlowIncompressible.randomFlowIncompressibleCase
    orig = base.configureScheme

    def configureScheme(ctx):
        orig(ctx)
        g = ctx.schemeConfig.gravityConfig
        g.active = True
        g.type = GravityType.Directional
        g.magnitude = 1.0            # the sin() amplitude carries the real scale
        g.direction = [0.0, -1.0]

    return dataclasses.replace(base, configureScheme=configureScheme,
                               name='gravityOscProbe')


ARMS = [
    ('obstacle_static', randomFlowIncompressible.randomFlowIncompressibleCase,
     dict(params=dict(obstacle=True, bounded=False)), None),
    ('obstacle_osc', drivenSquare.drivenSquareCase, dict(), None),
    ('gravity_osc', _gravityOscCase(),
     dict(params=dict(bounded=False)), GRAV),
]


def run_arm(case, kw, gravosc, ps):
    _reset()
    if ps:
        I._RESTORE_PS_SHIFT = True
        D.INSTEP_CD = False
    if gravosc is not None:
        D.GRAVITY_OSC = gravosc
    r = run(case, nx=48, scheme='divergenceFree', kernel='Wendland2',
            quiet=True, progress=False, plot=False, store=False,
            integrationScheme='semiImplicitEuler',
            dt=1e-3, minDt=1e-3, maxDt=1e-3, adaptiveDt=False, nSteps=NS, **kw)
    rows = [x for x in r.trajectory if x.get('step', -1) >= 0]
    ke = np.array([x.get('kineticEnergy', float('nan')) for x in rows])
    vm = np.array([x.get('maxVelocity', float('nan')) for x in rows])
    dmax = np.array([x.get('maxDensity', float('nan')) for x in rows])
    dmin = np.array([x.get('minDensity', float('nan')) for x in rows])
    finite = np.isfinite(ke)
    kestr = (f'{ke[0]:.3g}->{ke[finite][-1]:.3g} '
             f'(x{ke[finite][-1] / ke[0]:.3f}, pk x{np.nanmax(ke) / ke[0]:.2f})'
             if finite.any() else 'NaN')
    band = float(np.nanmax(dmax - dmin)) if finite.any() else float('nan')
    return (f'KE {kestr}  |v|pk {np.nanmax(vm):.3g}  dBand {band:.2e}  '
            f'div={r.diverged}  ran {int(finite.sum())}/{len(rows)}')


print(f'dfsph PS-restore ablation  nx=48  {NS} steps  fixed dt=1e-3\n')
for tag, case, kw, gravosc in ARMS:
    for ps, label in ((False, 'baseline  '), (True, 'PS-restore')):
        try:
            out = run_arm(case, dict(kw), gravosc, ps)
        except Exception as e:
            out = f'ERROR {type(e).__name__}: {e}'
        print(f'  {tag:16s} {label}  {out}')
    print()

_reset()
