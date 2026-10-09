import sys; sys.path.insert(0, 'src')
import numpy as np, torch
from scipy.optimize import brentq
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
from warpSPH.cases.sodND import sod2dCase
from warpSPH.runner import run

def exact(xi, t, x0=0.5, g=5/3, rl=1.0, pl=1.0, rr=0.25, pr=0.1795):
    cl = np.sqrt(g*pl/rl); cr = np.sqrt(g*pr/rr)
    def fK(p, rk, pk, ck):
        if p > pk:
            A = 2/((g+1)*rk); B = (g-1)/(g+1)*pk; return (p-pk)*np.sqrt(A/(p+B))
        return 2*ck/(g-1)*((p/pk)**((g-1)/(2*g))-1)
    ps = brentq(lambda p: fK(p, rl, pl, cl)+fK(p, rr, pr, cr), 1e-6, 10)
    us = 0.5*(fK(ps, rr, pr, cr)-fK(ps, rl, pl, cl)); rsl = rl*(ps/pl)**(1/g); cstl = cl*(ps/pl)**((g-1)/(2*g))
    rsr = rr*((ps/pr+(g-1)/(g+1))/((g-1)/(g+1)*ps/pr+1)); S = cr*np.sqrt((g+1)/(2*g)*ps/pr+(g-1)/(2*g))
    out = np.empty_like(xi)
    for i, s in enumerate((xi-x0)/t):
        if s < -cl: out[i] = rl
        elif s < us-cstl: c = (2/(g+1))*(cl+(g-1)/2*(-s)); out[i] = rl*(c/cl)**(2/(g-1))
        elif s < us: out[i] = rsl
        elif s < S: out[i] = rsr
        else: out[i] = rr
    return out

for label, eq in (('equal-mass', True), ('same-lattice', False)):
    r = run(sod2dCase, scheme='CRKSPH', tLimit=0.25, plot=False, store=False, quiet=True, progress=False,
            supportMode='KernelMeanSymmetric', params=dict(equalMass=eq))
    s = r.state.state; t = float(r.state.t)
    x = s.positions[:, 0].cpu().numpy(); rho = s.densities.cpu().numpy()
    o = np.argsort(x); x, rho = x[o], rho[o]
    m = (x > 0.05) & (x < 0.95)
    e = exact(x, t)
    L1 = np.trapezoid(np.abs(rho-e)[m], x[m])/(x[m][-1]-x[m][0])
    print(f'{label:13s} n={len(x)} t={t:.3f}  L1(rho) right interface = {L1:.3e}   min rho {rho.min():.3f} max {rho.max():.3f}', flush=True)
