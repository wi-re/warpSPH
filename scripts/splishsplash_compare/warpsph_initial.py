"""warpSPH hydrostaticColumn initial state (nx=128) for cross-library compare."""
import json, numpy as np
from warpSPH.cases import hydrostaticColumn
from warpSPH.runner import run

r = run(hydrostaticColumn.hydrostaticColumnCase, nx=128, nSteps=0,
        scheme='omniIncompressible', quiet=True, plot=False, store=False,
        progress=False, integrationScheme='semiImplicitEuler')
ctx = r.ctx
st = r.state.state if hasattr(r.state, "state") else r.state
cfg, sc = ctx.config, ctx.schemeConfig

k = st.kinds.cpu().numpy()
pos = st.positions.detach().cpu().numpy()
vel = st.velocities.detach().cpu().numpy()
mass = st.masses.detach().cpu().numpy()
sup = st.supports.detach().cpu().numpy()
den = st.densities.detach().cpu().numpy()
fluid = k == 0
pf = pos[fluid]

# Real SPH summation density on the initial config (st.densities at nSteps=0 is
# just the rho0 init constant; the first step is what computes the sum).
try:
    from warpSPH.modules.density import computeDensities
    den = computeDensities(st, cfg, sc, r.state.adjacency).detach().cpu().numpy()
except Exception as e:
    print("computeDensities failed:", e)

info = dict(
    nTotal=int(len(pos)), nFluid=int(fluid.sum()), nBoundary=int((k == 1).sum()),
    rho0=float(sc.fluid.restDensity), dx=float(cfg.dx),
    support_h=float(np.median(sup[fluid])),
    support_over_dx=float(np.median(sup[fluid]) / cfg.dx),
    kernel=str(getattr(cfg, "kernel", "?")),
    gravity=float(sc.gravityConfig.magnitude),
    massFluid_mean=float(mass[fluid].mean()),
    massFluid_min=float(mass[fluid].min()), massFluid_max=float(mass[fluid].max()),
    density_t0_min=float(den[fluid].min()), density_t0_max=float(den[fluid].max()),
    density_t0_mean=float(den[fluid].mean()), density_t0_median=float(np.median(den[fluid])),
)
try:
    from warpSPH.modules.util import countNeighbors
    nn = countNeighbors(st, cfg, sc, r.state.adjacency).detach().cpu().numpy()
    info["neighbors_fluid_mean"] = float(nn[fluid].mean())
    mid = int(np.argmin(np.abs(pf[:, 0]) + np.abs(pf[:, 1] + 0.25)))
    info["neighbors_midColumn_t0"] = int(nn[fluid][mid])
except Exception as e:
    info["neighbors_err"] = repr(e)

print("=== warpSPH hydrostaticColumn initial state (nx=128) ===")
print(json.dumps(info, indent=2))
print(f"fluid pos bbox x[{pf[:,0].min():.6f},{pf[:,0].max():.6f}] "
      f"y[{pf[:,1].min():.6f},{pf[:,1].max():.6f}]")
xs = np.unique(np.round(pf[:, 0], 6))
print(f"fluid spacing (median dx along x): {np.median(np.diff(xs)):.6f}")
if (k == 1).any():
    pb = pos[k == 1]
    print(f"boundary bbox x[{pb[:,0].min():.4f},{pb[:,0].max():.4f}] "
          f"y[{pb[:,1].min():.4f},{pb[:,1].max():.4f}]  nBoundary={len(pb)}")

np.savez("/home/lu26029/dev/warpSPH/scripts/splishsplash_compare/warpsph_initial.npz",
         positions=pf, velocities=vel[fluid], masses=mass[fluid],
         support=sup[fluid], density=den[fluid],
         boundary=pos[k == 1], info=json.dumps(info))
print("saved warpsph_initial.npz")
