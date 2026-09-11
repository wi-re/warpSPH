"""Which of the four changed knobs breaks sloshingTank? Equal physical time."""
import sys, os
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run

T = 0.15
legs = [
    ('OLD  nx150 dt2e-4 backsolve RK2',   dict(nx=150, integrationScheme='rungeKutta2',
                                               params=dict(targetDt=2e-4, soundSpeed=None))),
    ('nx200 dt2e-4 backsolve RK2',        dict(nx=200, integrationScheme='rungeKutta2',
                                               params=dict(targetDt=2e-4, soundSpeed=None))),
    ('nx200 dt1e-4 cs20      RK2',        dict(nx=200, integrationScheme='rungeKutta2',
                                               params=dict(targetDt=1e-4, soundSpeed=20.0))),
    ('nx200 dt1e-4 cs20      SIE',        dict(nx=200, integrationScheme='semiImplicitEuler',
                                               params=dict(targetDt=1e-4, soundSpeed=20.0))),
    ('nx200 dt2e-4 backsolve SIE',        dict(nx=200, integrationScheme='semiImplicitEuler',
                                               params=dict(targetDt=2e-4, soundSpeed=None))),
    ('OLD  nx150 dt2e-4 backsolve SIE',   dict(nx=150, integrationScheme='semiImplicitEuler',
                                               params=dict(targetDt=2e-4, soundSpeed=None))),
]
case = getCase('sloshingTank')
for name, kw in legs:
    try:
        r = run(case, scheme='deltaSPH', tLimit=T, quiet=True, store=False,
                progress=False, plot=False, **kw)
        d = r.diagnostics
        rho = d['maxDensity']
        print(f'{name:36s}  diverged={r.diverged!s:5s} t={r.t:.4f} '
              f'rho[{min(d["minDensity"]):.4f},{max(rho):.4g}] '
              f'vmax={max(d["maxVelocity"]):.4g}', flush=True)
    except Exception as e:
        print(f'{name:36s}  ERROR {type(e).__name__}: {e}', flush=True)
