"""Per-step state extrema for the triplePoint equalSpacing blow-up (OPEN_PROBLEMS §15)."""
import sys, dataclasses
sys.path.insert(0, '.')
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch
from warpSPH.cases.triplePoint import triplePointCase
from warpSPH.cases.compressible import compressibleDiagnostics
from warpSPH.runner import caseMain

import warpSPHCore.crk.crk_wrapper as _cw
_orig = _cw.computeCRKTermsWarp
_stash = []
def _wrapped(m_0, m_1, m_2, dm0, dm1, dm2, num_nbrs, supports):
    out = _orig(m_0, m_1, m_2, dm0, dm1, dm2, num_nbrs, supports)
    A, B, gA, gB = out
    ev = torch.linalg.eigvalsh(m_2.double())
    cond = (ev[:, -1] / ev[:, 0].clamp_min(1e-300))
    Bh = torch.linalg.norm(B, dim=-1) * supports
    _stash.append((cond.max().item(), int(cond.argmax()), Bh.max().item(), int(Bh.argmax()), A.min().item(), A.max().item(), int(num_nbrs.min()), int(num_nbrs.max())))
    return out
_cw.computeCRKTermsWarp = _wrapped

_prev = None
_dumped = []

def diag(ctx, state):
    d = compressibleDiagnostics(ctx, state)
    s = state.state
    def ext(name, x):
        i = int(torch.argmin(x)); j = int(torch.argmax(x))
        return f"{name} [{x[i].item():.4g}@({s.positions[i,0].item():.2f},{s.positions[i,1].item():.2f}) , {x[j].item():.4g}@({s.positions[j,0].item():.2f},{s.positions[j,1].item():.2f})]"
    speed = torch.linalg.norm(s.velocities, dim=-1)
    print(f"[probe] dt={ctx.config.dt if not torch.is_tensor(ctx.config.dt) else ctx.config.dt.item():.3e} vmax={speed.max().item():.4g} "
          + ext('rho', s.densities) + ' ' + ext('h', s.supports) + ' ' + ext('P', s.pressures) + ' ' + ext('u', s.internalEnergies), flush=True)
    for c in _stash[-2:]:
        print(f"[probe]   CRK cond max {c[0]:.3g} (idx {c[1]}) |B|h max {c[2]:.3g} (idx {c[3]}) A [{c[4]:.3g},{c[5]:.3g}] nbrs [{c[6]},{c[7]}]", flush=True)
    _stash.clear()
    if True:
        j = int(torch.argmax(speed)); i = int(torch.argmin(s.internalEnergies))
        def desc(k):
            return (f"uid={int(s.uids[k]) if hasattr(s,'uids') else k} x=({s.positions[k,0].item():.3f},{s.positions[k,1].item():.3f}) v=({s.velocities[k,0].item():.3g},{s.velocities[k,1].item():.3g}) "
                    f"m={s.masses[k].item():.4g} rho={s.densities[k].item():.4g} h={s.supports[k].item():.4g} P={s.pressures[k].item():.4g} u={s.internalEnergies[k].item():.4g}")
        print('[probe]   vmaxParticle ' + desc(j) + '\n[probe]   uminParticle ' + desc(i), flush=True)
    global _prev
    snap = {k: getattr(s, k).detach().cpu().numpy().copy() for k in ('positions','velocities','densities','supports','pressures','internalEnergies','masses','soundspeeds','alphas','divergence') if getattr(s, k, None) is not None}
    if speed.max().item() > 5 and _prev is not None and not _dumped:
        import numpy as np
        np.savez(sys.argv[1] + '/tp_blow.npz', **{'prev_' + k: v for k, v in _prev.items()}, **{'cur_' + k: v for k, v in snap.items()})
        _dumped.append(1); print('[probe] dumped blow-up snapshot', flush=True)
    _prev = snap
    return dict(d, maxVelocity=speed.max().item())

import os
_cfg0 = triplePointCase.configureScheme
def _cfg(ctx):
    _cfg0(ctx)
    for kv in filter(None, os.environ.get('TPK', '').split(',')):
        k, v = kv.split('=')
        tgt = ctx.schemeConfig.crkViscosityParams
        setattr(tgt, k, (v == '1') if k.startswith(('enable', 'force')) else float(v))
        print('[probe] set crkViscosityParams.%s=%s' % (k, v), flush=True)
case = dataclasses.replace(triplePointCase, diagnostics=diag, configureScheme=_cfg)
caseMain(case, ['--no-equalMass', '--nx', '256', '--caseName', 'tpProbe', '--no-plot', '--no-video', '--no-show',
                '--exportRoot', sys.argv[1], '--nSteps', '300', '--stallProgress', '0', '--velocityAlarmFactor', '1e9'] + sys.argv[2:])
