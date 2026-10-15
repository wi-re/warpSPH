"""Every iterative linear solver of the repo on ONE synthetic SPD system (2D 5-point Laplacian + small shift, n = 256^2, Jacobi precond),
fixed iteration budget: aten ops / host syncs / wall time per iteration (`warpSPH/utils/microBench.py`). The inventory it feeds is
docs/linalg/SOLVER_INVENTORY.md; rerun it after changing any solver in `modules/shifting` or `linalg`.

    python scripts/bench_solver_inventory.py

The synthetic matvec costs ~13 aten ops per application (included in the ops column). Iteration counts are nominal (30): a solver that
stops early on stagnation or breakdown shows fewer than the per-iteration figures suggest.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')
import torch
from warpSPH.utils.microBench import microBench, formatTable, formatSyncSites

dev = 'cuda'
m = 256
n = m * m
shift = 1e-2


def A(x):
    g = x.view(m, m)
    out = 4.0 * g + shift * g
    out[1:, :] -= g[:-1, :]
    out[:-1, :] -= g[1:, :]
    out[:, 1:] -= g[:, :-1]
    out[:, :-1] -= g[:, 1:]
    return out.reshape(-1)


torch.manual_seed(0)
xs = torch.randn(n, device=dev)
b = A(xs)
x0 = torch.zeros(n, device=dev)
inv = torch.full((n,), 1.0 / (4.0 + shift), device=dev)
IT = 30

from warpSPH.modules.shifting.cg import cgSolve
from warpSPH.modules.shifting.bicg import bicgSolve
from warpSPH.modules.shifting.bicgstab import bicgstabSolve
from warpSPH.modules.shifting.gmres import gmresSolve
from warpSPH.modules.shifting.minres import minresSolve
from warpSPH.modules.shifting.richardson import richardsonSolve
from warpSPH.modules.incompressible.compactProjection import CompactCG

rows, iters = [], {}
kw = dict(tol=0.0, rtol=1e-30, maxiter=IT)


def run(name, fn):
    out = {}

    def call():
        out['r'] = fn()
    r = microBench(call, name, calls=3, timedCalls=5)
    res = out['r']
    its = len(res[2]) if isinstance(res, tuple) and len(res) == 3 and isinstance(res[2], list) else IT
    iters[name] = its
    rows.append(r)


run('cg', lambda: cgSolve(A, b, x0, precond=inv, dim=1, **kw))
run('bicg', lambda: bicgSolve(A, A, b, x0, precond=inv, dim=1, **kw))
run('bicgstab', lambda: bicgstabSolve(A, b, x0, precond=inv, dim=1, **kw))
run('gmres(30)', lambda: gmresSolve(A, b, x0, precond=inv, dim=1, restart=30, **kw))
run('minres', lambda: minresSolve(A, b, x0, precond=inv, dim=1, **kw))
run('richardson', lambda: richardsonSolve(A, b, x0, dim=1, **kw))

# integrators' JFNK gmres (flat vector, 1-arg precond)
from warpSPHIntegrators.solvers import gmres as jgmres
run('integrators.gmres(30)', lambda: (jgmres(A, b, x0, tol=1e-30, maxiter=IT, restart=30, preconditioner=lambda v: inv * v)[0], 0, [None] * IT))

# compact CG: needs a pair list; build the same 5-point graph (w = -1 off-diagonals, D = shift)
idx = torch.arange(n, device=dev).view(m, m)
pi, pj = [], []
for a, c in ((idx[1:, :], idx[:-1, :]), (idx[:-1, :], idx[1:, :]), (idx[:, 1:], idx[:, :-1]), (idx[:, :-1], idx[:, 1:])):
    pi.append(a.reshape(-1)); pj.append(c.reshape(-1))
pi, pj = torch.cat(pi), torch.cat(pj)
w = -torch.ones(pi.shape[0], device=dev)
D = torch.full((n,), -shift, device=dev)           # CompactCG convention: sum w (Pi-Pj) + D P = rhs, w<=0
cg = CompactCG(warmStart=False)
rhs = -b
run('CompactCG', lambda: (cg.solve(pi, pj, w, rhs.clone(), tol=1e-30, maxIterations=IT, diag=D)[0], 0, [None] * IT))

text = formatTable(rows)
print(text)
print()
print('per iteration (counts / IT):')
for r in rows:
    k = IT
    print(f'{r.label:<24} ops/it {r.ops / k:6.1f}  launches/it {r.launches / k:5.1f}  syncs/it {r.syncs / k:5.2f}  ({r.syncs:5.1f} per solve)  wall us/it {r.wallUs / k:7.0f}')
print()
print(formatSyncSites(rows, top=6))
