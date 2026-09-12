"""Why does UID 4804's density collapse while it slides along the ceiling?

For a wall whose particles share one constant velocity, the continuity term
factors:

    drho/dt_i = rho_i (v_i - v_wall) . SUM_j (m_j/rho_j) grad W_ij

and for a tangentially symmetric flat wall that gradient sum is purely normal,
so a purely tangential relative velocity must give exactly zero. The measured
collapse (rho 0.995 -> 0.54) therefore has to come from one of:

  (a) a tangential component in SUM grad W  -- the stencil is not symmetric;
  (b) a normal component in the relative velocity;
  (c) the wall particles not sharing one velocity (the tank is rolling, so
      v_j = omega x r_j varies across the stencil).

This extracts all three. SUM_j (m_j/rho_j) grad W_ij is recovered from the real
operator rather than re-derived: with the query particle's velocity zeroed and
every other velocity set to a uniform `e`, the difference-form divergence
reduces to `e . SUM_j (m_j/rho_j) grad W_ij`, so probing e = x_hat and y_hat
gives the two components.
"""
import glob
import os

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import h5py
import numpy as np
import torch

from warpSPH.io.hdf5 import loadState
from warpSPH.systems.weaklyCompressible import WeaklyCompressibleState
from warpSPH.cases.sloshingTank import sloshingTankCase
from warpSPH.runner import run
from warpSPH.modules.momentum import computeMomentum
from warpSPHCore import SupportScheme, buildVerletList

UID = 4804
runDir = sorted(glob.glob('export/16-sloshingTank-wcsph_*'), key=os.path.getmtime)[-1]

r = run(sloshingTankCase, scheme='deltaSPH', nx=225, nSteps=1, tLimit=1e9,
        quiet=True, store=False, progress=False)
cfg, scheme = r.ctx.config, r.ctx.schemeConfig
dev = cfg.domain.min.device
dx = float(cfg.dx)


def gradSum(st, adj, idx):
    """SUM_j (m_j/rho_j) grad W_ij for particle `idx`, per component."""
    out = []
    for e in ([1.0, 0.0], [0.0, 1.0]):
        probe = st.velocities.clone()
        probe[:] = torch.tensor(e, device=dev, dtype=probe.dtype)
        probe[idx] = 0.0
        saved = st.velocities
        st.velocities = probe
        drhodt = computeMomentum(st, cfg, scheme, adj)
        st.velocities = saved
        # drhodt = -rho_i * div, div = e . SUM grad  =>  SUM_e = -drhodt/rho_i
        out.append(-float(drhodt[idx]) / float(st.densities[idx]))
    return np.array(out)


for step in (43000, 43500, 44000, 44500, 45000):
    f = os.path.join(runDir, 'trajectory', f'state_{step:04d}.h5')
    if not os.path.exists(f):
        continue
    h = h5py.File(f, 'r')
    st = loadState(h['state'], dev, WeaklyCompressibleState)
    m = (st.UIDs == UID).nonzero()
    if m.numel() == 0:
        print(f'step {step}: UID {UID} absent'); continue
    i = int(m[0])

    adj = buildVerletList(st, cfg.domain, verletScale=1.0,
                          supportMode=SupportScheme.SuperSymmetric,
                          priorNeighborhood=None, verbose=False)
    drhodt = computeMomentum(st, cfg, scheme, adj)

    xi, vi = st.positions[i], st.velocities[i]
    bnd = st.kinds == 1
    d = (st.positions[bnd] - xi).norm(dim=1)
    inRange = d < float(st.supports[i])
    vw = st.velocities[bnd][inRange]
    # ceiling: inward normal points -y (fluid below the wall above)
    nhat = np.array([0.0, -1.0])
    that = np.array([1.0, 0.0])

    G = gradSum(st, adj, i)
    vrel = (vi.detach().cpu().numpy() - (vw.mean(dim=0).detach().cpu().numpy()
                                         if vw.numel() else np.zeros(2)))

    print(f'\n=== step {step}  t={float(h.attrs["time"]):.4f} ===')
    print(f'  pos=({float(xi[0]):.5f},{float(xi[1]):.5f})  rho={float(st.densities[i]):.5f}  '
          f'drho/dt={float(drhodt[i]):+.4g}')
    print(f'  v_i        = ({float(vi[0]):+8.4f},{float(vi[1]):+8.4f})   '
          f'|v|={float(vi.norm()):.4f}')
    print(f'  v_wall_mean= ({float(vw.mean(dim=0)[0]) if vw.numel() else 0:+8.4f},'
          f'{float(vw.mean(dim=0)[1]) if vw.numel() else 0:+8.4f})   '
          f'spread |v_j - mean| max = '
          f'{float((vw - vw.mean(dim=0)).norm(dim=1).max()) if vw.numel() else 0:.5g}')
    print(f'  v_rel      : tangential {float(np.dot(vrel, that)):+8.4f}   '
          f'normal {float(np.dot(vrel, nhat)):+8.4f}')
    print(f'  SUM grad W : tangential {float(np.dot(G, that)):+10.4g}  '
          f'normal {float(np.dot(G, nhat)):+10.4g}   '
          f'|tan/norm| = {abs(np.dot(G, that) / (np.dot(G, nhat) + 1e-30)):.4g}')
    print(f'  predicted drho/dt = rho (v_rel . SUM) = '
          f'{float(st.densities[i]) * float(np.dot(vrel, G)):+.4g}')
    print(f'    of which tangential part: '
          f'{float(st.densities[i]) * float(np.dot(vrel, that) * np.dot(G, that)):+.4g}'
          f'   normal part: '
          f'{float(st.densities[i]) * float(np.dot(vrel, nhat) * np.dot(G, nhat)):+.4g}')
    print(f'  neighbours in range: {int(inRange.sum())} boundary, '
          f'{int(((st.positions[st.kinds == 0] - xi).norm(dim=1) < float(st.supports[i])).sum()) - 1} fluid')
