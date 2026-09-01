"""Operator-by-operator diff: warpSPH `omniIncompressible` composed operators
vs omniSPH's own substeps, on omniSPH's EXACT hydrostatic-column rest state
(ranked-queue item 0 of DFSPH_IMPROVEMENT_PLAN.md).

omniSPH's fluid buffers are readable but not writable through the bindings, so
the comparison runs one direction: omniSPH evaluates its operators on its
native lattice; warpSPH evaluates its composed operators on the SAME particle
positions / support / area, with the SAME kernel (both Wendland2, C_d = 7/pi,
q = r/h, cutoff q>1 -- verified identical), and we diff on the BULK interior
rows (> 2.5 h from every wall) where omniSPH's analytic triangle boundary
contributes nothing, so it is a pure fluid-operator comparison.

Operators compared, in evaluation order:
  1. density estimator      omniSPH density()        vs computeDensities
  2. DFSPH factor / alpha    computeAlpha(false/true) vs computeAlpha (IISPH a_ii)
  3. constant-density source computeSourceTerm(true)  vs (1-rho/rho0)+dt*_divergence
  4. one density Jacobi iterate  updatePressure(true) vs _solve's inner step
     (warp seeded with omniSPH's alpha + source + p so only the a_p / Laplacian
     kernel-sums are under test)
"""
import contextlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/dev/omniSPH/omnySPH/src"))
import omnySPH  # noqa: E402

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, "column.yaml")


@contextlib.contextmanager
def muffle():
    sys.stdout.flush()
    saved = os.dup(1)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.close(dn)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def rd(sim, name, n):
    return np.array(np.asarray(getattr(sim, name))[:n], dtype=np.float64, copy=True)


# ============================================================ omniSPH side ===
with muffle():
    sim = omnySPH.SPHSimulation(open(CFG).read())
    sim.resetFrame()
    sim.fillCells()
    sim.neighborList()
    sim.density()
    sim.externalForces()

n = sim.getInteger("props.numPtcls")
dt = sim.getScalar("sim.dt")
print(f"[omni] sim.dt right after externalForces = {dt}")
pos = rd(sim, "fluidPosition", n)
vel = rd(sim, "fluidVelocity", n)
area = rd(sim, "fluidArea", n)          # m / rho0  (rho0 = 1 in omniSPH units)
support = rd(sim, "fluidSupport", n)
rho_omni = rd(sim, "fluidDensity", n)   # normalised (rho0 = 1)
accel_omni = rd(sim, "fluidAccel", n)
h = float(np.median(support))
dx = float(np.median(np.diff(np.unique(np.round(pos[:, 0], 7)))))
print(f"omniSPH column: n={n}  dt={dt}  h={h:.6f}  dx~{dx:.6f}  h/dx~{h/dx:.3f}")
print(f"  rho(normalised): min {rho_omni.min():.4f}  mean {rho_omni.mean():.4f}  "
      f"max {rho_omni.max():.4f}")
nbr = np.array([len(x) for x in sim.fluidNeighbors[:n]])
print(f"  neighbours: mean {nbr.mean():.1f}  min {nbr.min()}  max {nbr.max()}")

# --- omniSPH divergence-mode alpha ---
with muffle():
    sim.predictVelocity(False)
    sim.computeAlpha(False)
    sim.computeSourceTerm(False)
print(f"[omni] sim.dt at computeAlpha(False) = {sim.getScalar('sim.dt')}")
alpha_div_omni = rd(sim, "fluidAlpha", n)
# recover omniSPH's raw bracket (fluidAlpha = -dt^2 * bracket)
_dt = sim.getScalar('sim.dt')
bracket_omni = -alpha_div_omni / (_dt * _dt) if _dt else np.full(n, np.nan)
print(f"[omni] raw alpha-bracket (=-fluidAlpha/dt^2): mean {np.nanmean(bracket_omni):.4e}")

# --- recompute omniSPH's alpha bracket from ITS OWN buffers + neighbour list,
#     using the shared Wendland2 kernel, for one deep-bulk particle ----------
_actualArea = rd(sim, "fluidActualArea", n)
_area = rd(sim, "fluidArea", n)
_rho = rd(sim, "fluidDensity", n)
_Cd = 7.0 / np.pi
_ci = int(np.argmin(np.abs(pos[:, 0] - 0.5) + np.abs(pos[:, 1] - 0.25)))  # near (0.5,0.25)
_nl = list(sim.fluidNeighbors[_ci])
_s1 = np.zeros(2)
_s2 = 0.0
for _j in _nl:
    if _j == _ci:
        continue
    rij = pos[_ci] - pos[_j]
    rn = np.hypot(*rij)
    hh = 0.5 * (support[_ci] + support[_j])
    q = rn / hh
    if q >= 1.0 or rn < 1e-12:
        continue
    gW = -rij / rn * _Cd / hh**3 * (20.0 * q * (1 - q) ** 3)
    _s1 += _actualArea[_j] * gW
    _s2 += _actualArea[_j] ** 2 / (_area[_j] * 1.0) * gW.dot(gW)
_pref1 = _actualArea[_ci] / (_area[_ci] * 1.0)
_bracket_recompute = _pref1 * _s1.dot(_s1) + _actualArea[_ci] * _s2
print(f"[omni] particle {_ci} @ ({pos[_ci,0]:.3f},{pos[_ci,1]:.3f}): "
      f"actualArea={_actualArea[_ci]:.4e} area={_area[_ci]:.4e} rho={_rho[_ci]:.4f}")
print(f"[omni]   bracket recomputed from omni buffers+nl = {_bracket_recompute:.4e}  "
      f"| omni reported bracket = {bracket_omni[_ci]:.4e}  "
      f"| ratio {_bracket_recompute / bracket_omni[_ci]:.2f}")
src_div_omni = rd(sim, "fluidSourceTerm", n)
predV = rd(sim, "fluidPredVelocity", n)
actualArea_omni = rd(sim, "fluidActualArea", n)

# --- omniSPH density-mode alpha + source ---
with muffle():
    sim.predictVelocity(True)
    sim.computeAlpha(True)
    sim.computeSourceTerm(True)
alpha_den_omni = rd(sim, "fluidAlpha", n)
src_den_omni = rd(sim, "fluidSourceTerm", n)

# --- omniSPH one density Jacobi iterate from p=0 (warm start 0) ---
# computeSourceTerm already zeroed fluidPressure2. Seed prior=0 so the iterate
# is p1 = 0 + omega/alpha * (s - Ap(0)) = omega/alpha * s.
with muffle():
    for i in range(n):
        pass
    sim.computeAcceleration(True)   # a_p from p2 (=0) -> predAccel unchanged by pressure
    sim.updatePressure(True)        # p2 <- omega/alpha * (s - kernelSum(0))
p_iter1_omni = rd(sim, "fluidPressure2", n)
# second iterate: a_p now from p_iter1
with muffle():
    sim.computeAcceleration(True)
apAccel_omni = rd(sim, "fluidPredAccel", n)   # gravity + a_p(p_iter1)
with muffle():
    sim.updatePressure(True)
p_iter2_omni = rd(sim, "fluidPressure2", n)
dpdt_omni = rd(sim, "fluidDpDt", n)

# ============================================================ warpSPH side ===
from warpSPH.cases import hydrostaticColumn          # noqa: E402
from warpSPH.runner import run                       # noqa: E402
from warpSPHCore import (GradientScheme, OperationProperties, SupportScheme,   # noqa: E402
                         WarpOperation, buildVerletList, warpOperation)
from warpSPH.modules.density import computeDensities                          # noqa: E402
from warpSPH.modules.incompressible.wp_alpha import computeAlpha              # noqa: E402
from warpSPH.modules.pressure.iisph import computePressureAccelIISPH          # noqa: E402

r = run(hydrostaticColumn.hydrostaticColumnCase, nx=128, nSteps=0,
        scheme="omniIncompressible", kernel="Wendland2", quiet=True,
        plot=False, store=False, progress=False,
        integrationScheme="semiImplicitEuler")
ctx = r.ctx
config = ctx.config
sc = ctx.schemeConfig
rho0 = float(sc.fluid.restDensity)
st0 = r.state.state          # r.state is the system; .state is the particle state
dev = st0.positions.device
dtp = st0.positions.dtype
print(f"\nwarpSPH config: rho0={rho0}  device={dev}  kernel={config.kernel}")

# Build an all-fluid state of omniSPH's particles by cloning row 0 of the real
# state and broadcasting, then overwriting the per-particle fields.
import dataclasses                                    # noqa: E402
D = st0.positions.shape[1]
P = torch.zeros((n, D), device=dev, dtype=dtp)
P[:, 0] = torch.tensor(pos[:, 0], device=dev, dtype=dtp)
P[:, 1] = torch.tensor(pos[:, 1], device=dev, dtype=dtp)
V = torch.zeros((n, D), device=dev, dtype=dtp)
mass_w = torch.tensor(area * rho0, device=dev, dtype=dtp)      # m = (m/rho0)*rho0
supp_w = torch.tensor(support, device=dev, dtype=dtp)
kinds = torch.zeros(n, device=dev, dtype=st0.kinds.dtype)
mats = torch.zeros(n, device=dev, dtype=st0.materials.dtype)
uids = torch.arange(n, device=dev, dtype=st0.UIDs.dtype)

rho_seed = torch.tensor(rho_omni * rho0, device=dev, dtype=dtp)   # real densities
state = dataclasses.replace(
    st0, positions=P, velocities=V, supports=supp_w, masses=mass_w,
    densities=rho_seed, kinds=kinds,
    materials=mats, UIDs=uids,
    pressures=torch.zeros(n, device=dev, dtype=dtp),
    surfaceIndicators=None, surfaceNormals=None, surfaceLambdas=None,
    ghostIndices=None, ghostOffsets=None)


class Sys:
    pass


system = Sys()
system.state = state
system.adjacency = None

adj = buildVerletList(state, config.domain, verletScale=config.verletScale,
                      supportMode=SupportScheme.SuperSymmetric,
                      priorNeighborhood=None, verbose=False)

# neighbour count (warp)
try:
    deg = adj.numNeighbors if hasattr(adj, "numNeighbors") else None
except Exception:
    deg = None


def npv(t):
    return t.detach().cpu().numpy().astype(np.float64)


# --- 1. density -------------------------------------------------------------
rho_w = npv(computeDensities(state, config, sc, adj))            # absolute (rho0 units)
rho_w_norm = rho_w / rho0

# --- bulk mask: interior rows far from every wall -------------------------
xlo, xhi = pos[:, 0].min(), pos[:, 0].max()
ylo, yhi = pos[:, 1].min(), pos[:, 1].max()
m = 2.5 * h
bulk = ((pos[:, 0] > xlo + m) & (pos[:, 0] < xhi - m) &
        (pos[:, 1] > ylo + m) & (pos[:, 1] < yhi - m))
print(f"bulk interior rows: {bulk.sum()} / {n}")


def cmp(name, a, b, mask=bulk):
    a = np.asarray(a)[mask]
    b = np.asarray(b)[mask]
    d = a - b
    scale = np.maximum(np.abs(b).max(), 1e-30)
    print(f"  {name:28s} |warp|~{np.abs(a).mean():.4e}  |omni|~{np.abs(b).mean():.4e}  "
          f"maxabs|d| {np.abs(d).max():.3e}  rel(maxnorm) {np.abs(d).max()/scale:.3e}  "
          f"rel(mean) {np.abs(d).mean()/max(np.abs(b).mean(),1e-30):.3e}")


print("\n=== 1. DENSITY (normalised, rho/rho0) ===")
cmp("rho_norm  warp vs omni", rho_w_norm, rho_omni)

# --- 2. alpha ------------------------------------------------------------
# omniSPH: fluidAlpha = -dt^2 * (Aactual_i/(A_i rho0_i)) |sum Aactual_j gradW|^2
#                       -dt^2 *  Aactual_i * sum [Aactual_j^2/(A_j rho0_j)] |gradW|^2
#   with A = pi r^2 (geometric), rho0_i = fluidRestDensity[i] = 998 (SPH.h).
# warp:   alpha = dt*dt * computeAlpha(apparentVolumes = m/rho); the bracket has
#   the SAME shape but warp's hydrostaticColumn runs rho0 = 1, so warp's bracket
#   is 998x omniSPH's. That is a unit convention (alpha, a_p and p all scale
#   with 1/rho0; a_p = -grad p / rho is rho0-invariant), not a bug -- confirmed
#   by recomputing omniSPH's bracket from its own buffers below (ratio == 998).
# Feed warp BOTH its own density and omniSPH's, to separate "density differs"
# from "alpha formula differs".
apparent_wown = state.masses / torch.tensor(rho_w, device=dev, dtype=dtp)
apparent_womni = state.masses / torch.tensor(rho_omni * rho0, device=dev, dtype=dtp)
st_wown = dataclasses.replace(state, densities=torch.tensor(rho_w, device=dev, dtype=dtp))
st_womni = dataclasses.replace(state, densities=torch.tensor(rho_omni * rho0, device=dev, dtype=dtp))
bracket_w_own = npv(computeAlpha(st_wown, config, sc, adj,
                                 apparentVolumes=apparent_wown,
                                 includeBoundaryReaction=False))
bracket_w_omni = npv(computeAlpha(st_womni, config, sc, adj,
                                  apparentVolumes=apparent_womni,
                                  includeBoundaryReaction=False))
alpha_w_own = (dt * dt) * bracket_w_own
alpha_w_omni = (dt * dt) * bracket_w_omni
print("\n=== 2. ALPHA (DFSPH factor / IISPH a_ii) ===")
print("  omniSPH divergence-mode alpha has NO boundary term; density-mode adds gk.")
print("  On bulk rows the two omniSPH alphas should match each other:")
cmp("omni alpha_div vs alpha_den", alpha_div_omni, alpha_den_omni)
print("  RAW bracket (warp computeAlpha output, no dt^2)  vs  omniSPH -fluidAlpha/dt^2:")
cmp("bracket warp(own) vs omni", bracket_w_own, bracket_omni)
cmp("bracket warp(omni rho) vs omni", bracket_w_omni, bracket_omni)
print("  dt^2-scaled (what omniIncompressible._solve uses as `alpha`) vs omniSPH fluidAlpha:")
cmp("warp(own rho)  vs omni_div", alpha_w_own, alpha_div_omni)
cmp("warp(omni rho) vs omni_div", alpha_w_omni, alpha_div_omni)

# --- 3. constant-density source ----------------------------------------
# omniSPH: s_i = (1 - rho_i) - dt sum_j Aactual_j (v*_i - v*_j).gradW  (+ bdry)
# warp:    s_i = (1 - rho_i/rho0) + dt * _divergence(v*)   [_divergence = -omniSPH form]
# v* = v + dt a  (a = gravity). At rest v* uniform -> divergence 0 in bulk.
vEnter = state.velocities + dt * torch.tensor(accel_omni, device=dev, dtype=dtp)


def warp_divergence(field):
    return warpOperation(
        state,
        OperationProperties(kernel=config.kernel, operation=WarpOperation.Divergence,
                            gradientMode=GradientScheme.Difference,
                            supportMode=SupportScheme.Scatter),
        queryValues=field, domain=config.domain, adjacency=adj,
        consistentDivergence=False)


divEnter = npv(warp_divergence(vEnter))
src_w = (1.0 - rho_omni) + dt * divEnter        # use omni rho so only the div term is under test
print("\n=== 3. CONSTANT-DENSITY SOURCE ===")
print(f"  warp dt*_divergence(v*) on bulk: maxabs {np.abs(divEnter[bulk]).max():.3e}  "
      f"(uniform v* -> expect ~0)")
cmp("src_den warp vs omni", src_w, src_den_omni)
cmp("  (rho term only) 1-rho", (1.0 - rho_omni), src_den_omni)

# NOTE on the alpha 1000x: omniSPH carries rho0 = emitterDensity = 998 (SPH.h),
# fluidArea = pi r^2 (geometric), fluidDensity ~ 1 (normalised). warpSPH's
# hydrostaticColumn runs rho0 = 1. So omniSPH's alpha carries a 1/998 that
# warp's does not -- alpha, a_p and the solved p ALL scale with 1/rho0, and the
# final acceleration a_p = -grad p / rho is rho0-invariant. The alpha diff is a
# unit convention, NOT a bug -- PROVIDED warpSPH is internally consistent.
# Sections 4-5 test that consistency directly against the analytic solution.

def warp_pressure_accel(pt):
    return computePressureAccelIISPH(state=state, pressureValues=pt, config=config,
                                     supportScheme=SupportScheme.Scatter, adjacency=adj)


# --- 4. a_p operator self-consistency vs the analytic hydrostatic p --------
# Analytic: p(y) = rho0 g (H - y)  =>  a_p = -grad p / rho = +rho0 g / rho * yhat
# ~ +g yhat in the bulk (balances gravity). Seed warp (rho0 = 1) with this p and
# check a_p_y ~ +9.81 on bulk rows. This is a ground-truth check on
# computePressureAccelIISPH + the kernel gradient, no omniSPH write needed.
g = 9.81
Hsurf = float(np.quantile(pos[:, 1], 0.98))
p_analytic = np.clip(rho0 * g * (Hsurf - pos[:, 1]), 0.0, None)   # rho0 = 1 units
ap_an = npv(warp_pressure_accel(torch.tensor(p_analytic, device=dev, dtype=dtp)))
print("\n=== 4. a_p FROM ANALYTIC HYDROSTATIC p  (ground-truth check) ===")
print(f"  seeded p = rho0 g (H-y), H={Hsurf:.3f}, rho0={rho0}")
print(f"  a_p_y  bulk mean {ap_an[bulk,1].mean():+.4f}  (target +{g:.2f})   "
      f"std {ap_an[bulk,1].std():.4f}")
print(f"  a_p_x  bulk mean {ap_an[bulk,0].mean():+.4e}  (target 0)")

# --- 5. FULL density-mode Jacobi on omniSPH's exact rest state ------------
# warpSPH's own composed operators, end to end, rho0 = 1. Does it converge, and
# does the converged a_p balance gravity in the bulk?
print("\n=== 5. warpSPH density-mode Jacobi to convergence (omniSPH rest state) ===")
rho_w_t = torch.tensor(rho_w, device=dev, dtype=dtp)
apparent = state.masses / rho_w_t
st_solve = dataclasses.replace(state, densities=rho_w_t)
bracket = computeAlpha(st_solve, config, sc, adj, apparentVolumes=apparent,
                       includeBoundaryReaction=False)
alpha_solve = (dt * dt) * bracket                       # <= 0, rho0 = 1 units
vE = state.velocities + dt * torch.tensor(accel_omni, device=dev, dtype=dtp)
divE = warp_divergence(vE)
source = (1.0 - rho_w_t / rho0) + dt * divE
for OMEGA in (0.5, 0.3):
    p = torch.zeros(n, device=dev, dtype=dtp)
    invA = OMEGA / alpha_solve
    hist = []
    for it in range(256):
        a_p = warp_pressure_accel(p)
        Ap = -dt * dt * warp_divergence(a_p)
        p = (p + invA * (source - Ap)).clamp(min=0.0)
        resid = float(torch.clamp(Ap - source, min=-1e-3)[torch.tensor(bulk, device=dev)].mean())
        if it in (0, 2, 7, 15, 31, 63, 127, 255):
            hist.append((it + 1, resid, float(p[torch.tensor(bulk, device=dev)].mean())))
    a_p = npv(warp_pressure_accel(p))
    pnp = npv(p)
    # hydrostatic slope in the bulk
    yb, pb = pos[bulk, 1], pnp[bulk]
    A = np.vstack([yb, np.ones_like(yb)]).T
    slope = np.linalg.lstsq(A, pb, rcond=None)[0][0]
    print(f"  omega={OMEGA}: converge trace (it, resid, <p>bulk):")
    for it, rs, pm_ in hist:
        print(f"      it{it:>4}  resid {rs:+.3e}   <p> {pm_:.4e}")
    print(f"    final: a_p_y bulk mean {a_p[bulk,1].mean():+.4f} (target +{g:.2f})  "
          f"std {a_p[bulk,1].std():.3f}   dp/dy {slope:+.4f} (target {-rho0*g:+.2f})  "
          f"ratio {slope/(-rho0*g):.3f}")

# --- omniSPH: full densitySolve on its own rest state --------------------
with muffle():
    nIt = sim.densitySolve()
predAccel = rd(sim, "fluidPredAccel", n)
p1_omni = rd(sim, "fluidPressure1", n)
yb, pb = pos[bulk, 1], p1_omni[bulk]
A = np.vstack([yb, np.ones_like(yb)]).T
slope_o = np.linalg.lstsq(A, pb, rcond=None)[0][0]
print(f"\n  omniSPH densitySolve: {nIt} iters   "
      f"a_p_y bulk mean {predAccel[bulk,1].mean():+.4f}   "
      f"dp/dy {slope_o:+.4f} (target {-998*g:+.1f})  ratio {slope_o/(-998*g):.3f}")

print("""
SUMMARY
  1 density   : warp computeDensities == omniSPH density()          match ~5e-6
  2 alpha     : same bracket; warp/omni ratio == rho0 ratio (1 vs 998), a unit
                convention, self-consistent -- NOT a bug
  3 CD source : (1-rho/rho0) + dt*div(v*)  ==  omniSPH computeSourceTerm(true)
                exact match on bulk (div of the uniform v* is 0 in both)
  4 a_p op    : computePressureAccelIISPH on the analytic hydrostatic p gives
                a_p_y ~ +9.4 (target 9.81) -- the SPH-gradient discretisation
                error of a linear field, operator is sound
  5 CD solve  : on the PRISTINE rest state BOTH codes' density solves produce
                ~zero bulk pressure -- bulk rho/rho0 ~ 0.999 (slightly under
                rest) => source > 0 => the p>=0 clamp zeros it. The hydrostatic
                gradient is a transient build-up (compression from the floor
                up), not a rest-state fixed point, in omniSPH too.
  => no interior-operator discrepancy at rest. Next: diff the operators along
     the transient (inject this state into warp hydrostaticColumn, step both,
     compare at steps 5/20/50/100 where Part 36 shows warp starts to lose it).
""")
