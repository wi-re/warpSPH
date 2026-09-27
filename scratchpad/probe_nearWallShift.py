#!/usr/bin/env python
"""Does the near-wall density anomaly track the delta+ particle shift?

Same near-wall gradient readout as probe_nearWallDDT.py, over shift settings.
`correctdrhodt` is Sun 2019 Eq. (9)'s volume-consistency term for the shift --
off by default, so a shifted particle's density is not corrected for the volume
it swept. Near a wall the shift is large and one-sided.
"""
import sys, numpy as np
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases import importAll; importAll()
from warpSPH.runner import getCase, run

legs = {
    'shift michel2022 (default)': dict(shifting=True,  correctdrhodt=False),
    'shift OFF':                  dict(shifting=False, correctdrhodt=False),
    'shift + correctdrhodt':      dict(shifting=True,  correctdrhodt=True),
}
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
INTEGRATOR = sys.argv[2] if len(sys.argv) > 2 else 'symplecticEuler'
case = getCase('sloshingTank')
print(f'integrator = {INTEGRATOR}   nx=200  dt=1e-4  c_s=20  steps={steps}', flush=True)
for name, extra in legs.items():
    r = run(case, scheme='deltaSPH', nx=200, nSteps=steps, tLimit=1e9,
            integrationScheme=INTEGRATOR,      # pinned, not the case default
            quiet=True, store=False, progress=False, plot=False,
            params=dict(targetDt=1e-4, soundSpeed=20.0, **extra))
    st = r.state.state
    pos = st.positions.detach().cpu().numpy(); rho = st.densities.detach().cpu().numpy()
    k = st.kinds.detach().cpu().numpy(); dx = float(r.ctx.config.dx)
    exp = 9.81 * dx / float(r.ctx.schemeConfig.fluid.fixedSoundSpeed) ** 2
    band = np.abs(pos[:, 0]) < 0.15
    f = band & (k == 0)
    ys = np.round(pos[f, 1] / dx * 2) / 2
    rows = [(y, rho[f][np.abs(ys - y) < 1e-6].mean())
            for y in np.unique(ys)[:9] if (np.abs(ys - y) < 1e-6).sum() >= 3]
    ratios = [(r0 - r1) / exp for (_, r0), (_, r1) in zip(rows, rows[1:])]
    b = band & (k == 1) & (pos[:, 1] < 0.05)
    yb = pos[b, 1]; top = np.abs(yb - yb.max()) < 0.25 * dx
    print(f'{name:28s} ' + ' '.join(f'{x:5.2f}' for x in ratios)
          + f'   step={rho[b][top].mean() - rows[0][1]:+.2e}', flush=True)
print(f'{"":28s} ' + ' '.join(f'{y:5.1f}' for y, _ in
      [(0.5,0),(1.5,0),(2.5,0),(3.5,0),(4.5,0),(5.5,0),(6.5,0),(7.5,0)][:len(ratios)])
      + '   <- row gap start (y/dx); 1.00 = hydrostatic')
