"""Watch the light-gas pile-up at the Sod-2D contact zone (|x| in [0.5, 1.0], both interfaces): every N calls of the CRKSPH step,
nearest-neighbour spacing statistics of the particles there (cKDTree, periodic-y ignored), plus max|v|. Runs the probe's case as given."""
import os, runpy, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import warpSPH.schemes.crkSPH as crk, warpSPH.schemes.builder as builder, warpSPH.schemes as schemes
_orig = crk.crkSPH_step
N = [0]; ROWS = []
DX = float(os.environ.get('DX', '0.01'))
def wrapped(system, dt, config, schemeConfig, verbose=False):
    out = _orig(system, dt, config, schemeConfig, verbose)
    N[0] += 1
    if N[0] % 40 == 0:
        x = system.state.positions.detach().cpu().numpy(); m = system.state.masses.detach().cpu().numpy()
        sel = (np.abs(x[:, 0]) > 0.5) & (np.abs(x[:, 0]) < 1.0)
        xs = x[sel]; nn = cKDTree(xs).query(xs, k=2)[0][:, 1] / DX
        ROWS.append((N[0] // 2, float(system.t), nn.min(), int((nn < 0.5).sum()), int((nn < 0.1).sum()), int((nn < 0.01).sum())))
    return out
crk.crkSPH_step = wrapped; builder.crkSPH_step = wrapped; schemes.crkSPH_step = wrapped
sys.argv = ['probe_triplePointPileup.py'] + sys.argv[1:]
try:
    runpy.run_path('scripts/probe_triplePointPileup.py', run_name='__main__')
finally:
    print('\n[pilewatch] step   t     min_nn/dx  n<0.5dx  n<0.1dx  n<0.01dx   (|x| in 0.5..1.0)', flush=True)
    for r in ROWS: print('[pilewatch] %4d  %.3f  %8.4f  %6d  %6d  %6d' % r, flush=True)
