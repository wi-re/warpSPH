"""EXPERIMENT (OPEN_PROBLEMS §15): Frontiere et al. 2017 Eq. (76) multi-material mass rule for the CRK density,
  rho_i = sum_j m_ij V_j W^R_ij / sum_j V_j^2 W^R_ij,   m_ij = m_j (same material), m_i (different material)
built by linearity out of ordinary density passes with masked masses (the core kernel has no material tag):
  material-a row i:  R[m 1_a] + m_i R[1_b],   R[f] = density pass with per-particle 'mass' f.
Patches warpSPH.schemes.crkSPH.computeCRKFactors, then runs probe_triplePointPileup.py with the given args."""
import copy, os, runpy, sys
import torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import warpSPH.schemes.crkSPH as crk

_orig = crk.computeCRKFactors
STATE = {'on': os.environ.get('MIJ', '1') == '1'}

def patched(state, *a, **kw):
    V, rho, crkState = _orig(state, *a, **kw)
    mat = getattr(state, 'materials', None)
    if not STATE['on'] or mat is None or int(mat.max()) == int(mat.min()):
        return V, rho, crkState
    m = state.masses
    def R(f):
        s = copy.copy(state); s.masses = f
        return _orig(s, *a, **kw)[1]
    a0 = (mat == int(mat.min()))
    ma = torch.where(a0, m, torch.zeros_like(m)); mb = torch.where(~a0, m, torch.zeros_like(m))
    one_a = a0.to(m.dtype); one_b = (~a0).to(m.dtype)
    rho_new = torch.where(a0, R(ma) + m * R(one_b), R(mb) + m * R(one_a))
    return V, rho_new, crkState

crk.computeCRKFactors = patched
sys.argv = ['probe_triplePointPileup.py'] + sys.argv[1:]
runpy.run_path('scripts/probe_triplePointPileup.py', run_name='__main__')
