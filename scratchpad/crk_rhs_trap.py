"""Trap the first CRKSPH right-hand-side evaluation with a crazy acceleration / energy rate (OPEN_PROBLEMS §15):
saves the input state and every pair-summed term of that evaluation, then lets the probe run on to its own stop."""
import copy, os, runpy, sys
import numpy as np, torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import warpSPH.schemes.crkSPH as crk
import warpSPH.schemes.builder as builder
import warpSPH.schemes as schemes

_orig = crk.crkSPH_step
OUT = os.environ.get('TRAP_OUT', 'scripts/out_crk2d/trap.npz')
LIM = float(os.environ.get('TRAP_LIM', '500'))
DONE = [False]; N = [0]
def np_(t): return t.detach().cpu().numpy() if t is not None else None

def wrapped(system, dt, config, schemeConfig, verbose=False):
    st = system.state
    pre = {k: np_(getattr(st, k).clone()) for k in ('positions', 'velocities', 'internalEnergies', 'masses', 'supports', 'densities', 'alphas', 'materials') if getattr(st, k, None) is not None}
    upd, adj, state = _orig(system, dt, config, schemeConfig, verbose)
    N[0] += 1
    dv = upd.dvdt.norm(dim=-1); du = upd.dudt.abs()
    if not DONE[0] and (float(dv.max()) > LIM or float(du.max()) > LIM or not torch.isfinite(dv).all()):
        DONE[0] = True
        out = {'pre_' + k: v for k, v in pre.items()}
        out.update(dict(call=N[0], dt=float(dt), dvdt=np_(upd.dvdt), dudt=np_(upd.dudt), drhodt=np_(upd.drhodt),
                        rho_post=np_(state.densities), P=np_(state.pressures), cs=np_(state.soundspeeds), h=np_(state.supports),
                        alphas_post=np_(state.alphas)))
        for k in ('ap_ij', 'av_ij', 'f_ij', 'divergence'):
            v = getattr(state, k, None)
            if v is not None: out[k] = np_(v)
        print('[trap] adjacency type', type(adj), [a for a in dir(adj) if not a.startswith('_')][:30], flush=True)
        for a in ('i', 'j', 'edgeOffsets', 'numNeighbors'):
            v = getattr(adj, a, None)
            if isinstance(v, torch.Tensor): out['adj_' + a] = np_(v)
        np.savez(OUT, **out)
        i = int(torch.argmax(torch.maximum(dv, du)))
        print(f'[trap] call {N[0]} dt={float(dt):.3e}: max|dvdt|={float(dv.max()):.3e} max|dudt|={float(du.max()):.3e} at particle {i}, x={np_(st.positions[i])}; saved {OUT}', flush=True)
    return upd, adj, state

crk.crkSPH_step = wrapped; builder.crkSPH_step = wrapped; schemes.crkSPH_step = wrapped
sys.argv = ['probe_triplePointPileup.py'] + sys.argv[1:]
runpy.run_path('scripts/probe_triplePointPileup.py', run_name='__main__')
