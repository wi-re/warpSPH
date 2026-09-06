"""Audit: the sign of the mDBC linear-extrapolation term (English et al. 2022
Eq. 12), used by `modules/mdbc/density2025.py` and
`modules/incompressible/wallPressure.py`.

English Eq. (12):   f_b = f_g + (r_b - r_g) . grad(f)_g
with g the ghost node inside the fluid and b the boundary particle it belongs
to. Three links have to be right for the code to realise that:

  1. `interpolateLiuLiu` must return grad(f) with the STANDARD sign
     (+df/dx), not DualSPHysics' negated-in-the-solve convention.
  2. ghost placement: the repo builds `r_g = r_b - ghostOffset`
     (`rigidBody/ghostParticles.py`), so `ghostOffset == r_b - r_g` and
     English's `(r_b - r_g)` is `+ghostOffset`.
  3. `density2025.py` must combine them so the net is `+ghostOffset . grad`.
     It computes `relPos = -ghostOffset`, then `drho = -relPos . grad`
     (= `+ghostOffset . grad`), `rho_proj = rho_interp + drho`. Algebraically
     English's `+`.

Part 1 (synthetic) nails link 1 directly: interpolate a known linear field
`f = a . x + b` and check the recovered gradient is `+a`.
Part 2 (hydrostatic column) checks the assembled result: which of
`+ghostOffset` / `-ghostOffset` the stored boundary density actually tracks.

Usage:  python scripts/probe_mdbcExtrapolationSign.py [--nx 48] [--steps 100]
"""
from __future__ import annotations
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--nx', type=int, default=48)
ap.add_argument('--steps', type=int, default=100)
args = ap.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import torch
from warpSPHCore import OperationDirection, DomainDescription
from warpSPH.modules.liu import interpolateLiuLiu


# ------------------------------------------------------------------ Part 1
# interpolateLiuLiu on a known linear field: does it return +grad or -grad?
print("=" * 68)
print("Part 1 -- interpolateLiuLiu gradient sign on a synthetic linear field")
print("=" * 68)

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dt = torch.float32
n = 40
dx = 1.0 / n
xs = torch.arange(n, device=dev, dtype=dt) * dx
gx, gy = torch.meshgrid(xs, xs, indexing='ij')
pos = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
# small jitter so it is not a perfect lattice
torch.manual_seed(0)
pos = pos + (torch.rand_like(pos) - 0.5) * 0.25 * dx

h = 2.0 * dx
a = torch.tensor([3.0, -1.7], device=dev, dtype=dt)     # the true gradient
b0 = 0.5
field = pos @ a + b0

class _S:                                                # minimal ParticleState
    pass
st = _S()
st.positions = pos
st.supports = torch.full((pos.shape[0],), h, device=dev, dtype=dt)
st.masses = torch.full((pos.shape[0],), dx * dx, device=dev, dtype=dt)
st.densities = torch.ones(pos.shape[0], device=dev, dtype=dt)
st.kinds = torch.zeros(pos.shape[0], device=dev, dtype=torch.int32)

class _C:
    pass
cfg = _C()
from warpSPHCore import KernelFunctions
cfg.kernel = KernelFunctions.Wendland2
cfg.domain = DomainDescription(
    torch.tensor([-1.0, -1.0], device=dev, dtype=dt),
    torch.tensor([2.0, 2.0], device=dev, dtype=dt),
    torch.zeros(2, dtype=torch.bool, device=dev), 2)
cfg.dim = 2

q = torch.tensor([[0.5, 0.5], [0.3, 0.7]], device=dev, dtype=dt)
val, grad, nnbr, A_g, b = interpolateLiuLiu(
    q, referenceParticles=st, referenceQuantities=field, config=cfg,
    neighbor_threshold=4, direction=OperationDirection.AllToAll, supportScale=1.0)

print(f"true gradient a           = {a.tolist()}")
print(f"interpolateLiuLiu grad    = {grad[0].tolist()}   (query 0, nnbr={int(nnbr[0])})")
print(f"                            {grad[1].tolist()}   (query 1, nnbr={int(nnbr[1])})")
err_plus = float((grad - a).abs().mean())
err_minus = float((grad + a).abs().mean())
print(f"|grad - a| = {err_plus:.3e}   |grad + a| = {err_minus:.3e}")
sign = '+grad (STANDARD)' if err_plus < err_minus else '-grad (NEGATED, DualSPHysics-solve convention)'
print(f"=> interpolateLiuLiu returns {sign}")
val_ok = float((val - (q @ a + b0)).abs().mean())
print(f"value recovery |f - (a.x+b)| = {val_ok:.3e}")

linExtrapSignOK = err_plus < err_minus


# ------------------------------------------------------------------ Part 2
print()
print("=" * 68)
print("Part 2 -- assembled mDBC boundary density on a hydrostatic column")
print("=" * 68)
print("(hydrostaticColumn's deltaSPH path is not well-tuned -- c0 is soft and")
print(" the sampled bulk sits ~5% below rho0, so absolute agreement with the")
print(" analytic profile is poor; the useful signal is which of +/-ghostOffset")
print(" the stored value TRACKS.)")

from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as case
from warpSPH.runner import run

r = run(case, scheme='deltaSPH', nx=args.nx, nSteps=args.steps,
        quiet=True, plot=False, store=False, progress=False)
s = r.state.state
ctx = r.ctx
rho0 = ctx.schemeConfig.fluid.restDensity
c0 = ctx.schemeConfig.fluid.fixedSoundSpeed
g_vec = torch.tensor(ctx.schemeConfig.gravityConfig.direction, dtype=s.positions.dtype,
                     device=s.positions.device) * ctx.schemeConfig.gravityConfig.magnitude
gmag = float(torch.linalg.norm(g_vec)); gdir = g_vec / gmag

bnd = s.kinds == 1
ghost = s.kinds == 2
fluid = s.kinds == 0
print(f"\nnx={args.nx} steps={args.steps}  rho0={rho0}  c0={c0:.4g}  |g|={gmag:.4g}")
print(f"counts: fluid={int(fluid.sum())} boundary={int(bnd.sum())} ghost={int(ghost.sum())}")

proj = s.positions @ gdir
fs_proj = proj[fluid].min()
depth = (proj - fs_proj).clamp(min=0.0)

rho_g, grad_g, nnbr, A_g, bb = interpolateLiuLiu(
    s.positions[ghost], referenceParticles=s, referenceQuantities=s.densities,
    config=ctx.config, neighbor_threshold=4,
    direction=OperationDirection.FluidToGhost, supportScale=1.0)

ghostAbs = torch.nonzero(ghost, as_tuple=False).flatten()
firstGhost = int(ghostAbs.min())
gpos = s.positions[ghost]
rowOf = (s.ghostIndices[bnd] - firstGhost).long()

off = s.ghostOffsets[bnd]                 # r_b - r_g  (verified below)
rb = s.positions[bnd]; rg = gpos[rowOf]
print(f"ghost placement check: max|r_b - r_g - ghostOffset| = "
      f"{float((rb - rg - off).abs().max()):.2e}   (0 => ghostOffset == r_b - r_g)")

rho_g_b = rho_g[rowOf]; grad_g_b = grad_g[rowOf]; nnbr_b = nnbr[rowOf]
english = rho_g_b + torch.einsum('nu,nu->n', off, grad_g_b)     # +ghostOffset
flipped = rho_g_b - torch.einsum('nu,nu->n', off, grad_g_b)     # -ghostOffset
code_rho = s.densities[bnd]
analytic = rho0 * (1.0 + gmag * depth[bnd] / c0 ** 2)

ok = (nnbr_b > 4) & (depth[bnd] > 0)
print(f"\nwetted boundary particles with ghost support: {int(ok.sum())} / {int(bnd.sum())}")
d_eng = float((code_rho[ok] - english[ok]).abs().mean())
d_flp = float((code_rho[ok] - flipped[ok]).abs().mean())
print(f"stored code density vs English [+ghostOffset]: mean|Δ| = {d_eng:.3e}")
print(f"stored code density vs flipped  [-ghostOffset]: mean|Δ| = {d_flp:.3e}")
assembledSignOK = d_eng < d_flp
print(f"=> the assembled code path uses "
      f"{'+ghostOffset  (English Eq. 12)' if assembledSignOK else '-ghostOffset  (FLIPPED)'}")

for nm, pred in [('code', code_rho), ('English[+]', english),
                 ('flipped[-]', flipped), ('rho_g only', rho_g_b)]:
    e = (pred[ok] - analytic[ok]).abs()
    print(f"   |{nm:11s} - analytic|  mean={float(e.mean()):.3e}  max={float(e.max()):.3e}")

print()
print("=" * 68)
ok_all = linExtrapSignOK and assembledSignOK
print(f"VERDICT: interpolateLiuLiu grad sign {'OK' if linExtrapSignOK else 'NEGATED'}; "
      f"assembled mDBC extrapolation {'= English Eq. 12 (+)' if assembledSignOK else 'FLIPPED (-)'}")
print(f"         mDBC linear-extrapolation sign is {'CORRECT' if ok_all else 'WRONG -- fix density2025.py / wallPressure.py'}")
print("=" * 68)
