"""The wall viscous term and its booked torque of the Taylor-Couette annulus (`taylorCouette`), warpSPH against the boundaries repo's `DFSPH2D._viscous_accel`, on the IDENTICAL particles with the EXACT profile
`u_theta(r) = A r + B / r` as the velocity (a static evaluation: no stepping, so lattice and dynamics drop out).

    python scripts/probe_taylorCouetteOracle.py [--n 48]

Prints, per body: the booked wall torque (sum of lever x (-m a_wall)), the fluid-only Morris part, the per-particle difference of the wall terms, and the exact torque of the profile (T = -4 pi mu B on the inner
cylinder, + on the outer). The Morris traction correction of the rotating inner wall (-2 mu Omega A) is added to the inner torque in both.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from warpSPHBootstrap import bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--nu', type=float, default=0.0185)
    ap.add_argument('--nh', type=float, default=2.5050, help='support radius in dx: the oracle (DFSPH2D) runs its Taylor-Couette at h = dx / PACKING = 2.505 dx')
    ap.add_argument('--field', default='exact', choices=('exact', 'translation', 'rigid', 'noise'))
    a = ap.parse_args()
    bootstrap(precision='float32')
    from warpSPH.cases import importAll
    importAll()
    from warpSPH.runner import getCase, buildContext
    from warpSPH.runner.caseSpec import CaseSpec
    from warpSPH.modules.density.density import computeDensities
    from warpSPH.modules.analyticBoundary import resolveWall
    from warpSPH.modules.analyticBoundary.wallViscosity import wallNoSlipAcceleration
    from warpSPH.modules.deltaSPH import computeVelocityDiffusion
    from warpSPH.modules.incompressible.compactProjection import morrisCalibration
    from warpSPH.enumTypes import ViscosityTerm
    import warpSPH.schemes.omniIncompressible as O
    from warpSPHBoundaries.scene.implicitBodies import DiskBody
    from warpSPHBoundaries.scene.scene import Body, DiskArrayRep, ImplicitRep, Scene
    from warpSPHBoundaries.sim.dfsph2d import DFSPH2D, DFSPHConfig

    case = getCase('taylorCouette')
    spec = CaseSpec(caseName='tc', scheme='omniIncompressible', params={**case.params, 'wallViscosityClosure': 'noslipMoment', 'fluidViscosity': 'morris', 'closedPreset': True, 'nuReference': a.nu}).merged(
        **case.defaults).merged(scheme='omniIncompressible', n_h=a.nh, nx=a.n, tLimit=1.0, plot=False, video=False, show=False, store=False, progress=False, quiet=True, kernel='Wendland2',
                                integrationScheme='semiImplicitEuler', supportMode='SuperSymmetric', dt=2e-3, adaptiveDt=False)
    ctx = buildContext(case, spec)
    case.configureScheme(ctx)
    system = case.buildSystem(ctx)
    if case.initialConditions is not None:
        case.initialConditions(ctx, system)
    st, sc, cfg = system.state, ctx.schemeConfig, ctx.config
    r1, r2, om = ctx.param('r1'), ctx.param('r2'), ctx.param('omega')
    nu = a.nu
    fl = st.kinds == 0
    x0 = st.positions[fl].double().cpu().numpy()
    m0 = st.masses[fl].double().cpu().numpy()
    h0 = st.supports[fl].double().cpu().numpy()
    rr = np.linalg.norm(x0, axis=1)
    A = -om * r1 ** 2 / (r2 ** 2 - r1 ** 2)
    B = om * r1 ** 2 * r2 ** 2 / (r2 ** 2 - r1 ** 2)
    ut = A * rr + B / rr
    vel = np.stack([-ut * x0[:, 1] / rr, ut * x0[:, 0] / rr], 1)
    if a.field == 'translation':
        vel = np.tile([0.1, 0.0], (len(x0), 1))
    elif a.field == 'rigid':
        vel = np.stack([-om * x0[:, 1], om * x0[:, 0]], 1)
    elif a.field == 'noise':
        vel = 0.05 * np.random.default_rng(3).standard_normal(x0.shape)
    Tex = -4 * math.pi * nu * B
    print(f'N={len(x0)}  mass [{m0.min():.3e}, {m0.max():.3e}]  support {h0.min():.4f}  r in [{rr.min():.4f}, {rr.max():.4f}]  exact inner torque {Tex:+.6f}')

    # ---- warpSPH ----
    adj = O._rebuildAdjacency(st, system, cfg)
    st.densities = computeDensities(st, cfg, sc, adj)
    cal = morrisCalibration(float(st.supports[fl].double().median()), float(cfg.dx), float(st.masses[fl].double().median()) / sc.fluid.restDensity)
    sc.diffusionParams.inviscid = False
    sc.diffusionParams.viscousTerm = ViscosityTerm.morris1997
    sc.diffusionParams.viscidNu = nu / cal
    sc.morrisCalibration = None
    sc.wallViscosityClosure = 'noslipMoment'
    sc.complementMoments = True
    full = np.zeros((len(st.positions), 2))
    full[fl.cpu().numpy()] = vel
    st.velocities = torch.tensor(full, dtype=st.velocities.dtype, device=st.velocities.device)
    wall = resolveWall(st, cfg, sc, adj)
    viscf = computeVelocityDiffusion(st, cfg, sc, adj, wall=False)
    perBody = wallNoSlipAcceleration(st, cfg, sc, adj, wall, viscf, perBody=True).double().cpu().numpy()[:, fl.cpu().numpy()]
    mine = []
    for bi in range(perBody.shape[0]):
        F = -m0[:, None] * perBody[bi]
        mine.append(float((x0[:, 0] * F[:, 1] - x0[:, 1] * F[:, 0]).sum()))
    mine[0] -= 2.0 * nu * om * math.pi * r1 ** 2

    print(f'wall mass (mu) used by warpSPH: {float(torch.as_tensor(wall.wm).double().mean()):.4f}')
    # ---- the oracle on the same particles ----
    dev = 'cuda:0'
    inner = Body(bodyId=0, center=(0.0, 0.0), angularVelocity=om, reps=[DiskArrayRep([(0.0, 0.0)], [r1])])
    outer = Body(bodyId=1, center=(0.0, 0.0), reps=[ImplicitRep(DiskBody(center=(0.0, 0.0), radius=r2, solid='outside'))])
    scene = Scene([inner, outer], dev)
    dcfg = DFSPHConfig(gravity=(0.0, 0.0), viscosity=nu, boundaryFriction=0.0, wallMass=float(torch.as_tensor(wall.wm).double().mean()), maxDt=1e-3, minDt=1e-4, recordForces=False, morrisCalibration=cal, fluidPairs='cells', graphIterations=False)
    sim = DFSPH2D(x0, vel, m0, h0, scene, dcfg, dev)
    sim.dt = 1e-3
    sim._prepare()
    sim.rho = sim._sum(sim.V[sim.pj] * sim.W) + sim.lam
    sim.forceViscous = torch.zeros((sim.nb, 2), dtype=torch.float64, device=dev)
    sim.torque = torch.zeros((3, sim.nb), dtype=torch.float64, device=dev)
    sim.v = torch.tensor(vel, dtype=torch.float64, device=dev)
    acc = sim._viscous_accel()
    orc = [float(sim.torque[2, bi]) for bi in range(sim.nb)]

    # per-particle wall term (the oracle's total minus its fluid part is not separable here: compare the totals)
    vfl = viscf.double().cpu().numpy()[fl.cpu().numpy()]
    tot_mine = perBody.sum(0) + vfl
    d = np.abs(tot_mine - acc.cpu().double().numpy())
    print(f'total viscous acceleration (fluid + walls) warpSPH vs DFSPH2D: max|d| {d.max():.3e}  rms|d| {np.sqrt((d ** 2).mean()):.3e}  rms(ref) {np.sqrt((acc.cpu().double().numpy() ** 2).mean()):.3e}')
    from warpSPH.modules.analyticBoundary.wallViscosity import curvature as curvMine
    xt = torch.tensor(x0, dtype=torch.float64, device=dev)
    Hh = float(wall.support)
    for bi in range(len(scene.bodies)):
        d1, n1, h1 = sc.boundaryProvider.scene.signed_distance(xt, body=bi, supportMax=Hh)
        d2, n2, h2 = scene.signed_distance(xt, body=bi, supportMax=Hh)
        t1 = torch.stack([-n1[:, 1], n1[:, 0]], 1)
        t2 = torch.stack([-n2[:, 1], n2[:, 0]], 1)
        k1 = curvMine(sc.boundaryProvider.scene, bi, xt, n1, t1, Hh, float(cfg.dx))
        k2 = sim._wcObj.curvature(bi, xt, n2, t2) if getattr(sim, '_wcObj', None) is not None else None
        near = (d1 < Hh)
        print(f'body {bi}: signed distance max|d| {float((d1 - d2).abs()[near].max()):.2e}  normal max|d| {float((n1 - n2).abs()[near].max()):.2e}  hit equal {bool((h1 == h2).all())}  curvature warpSPH [{float(k1[near].min()):+.3f},{float(k1[near].max()):+.3f}]' + ('' if k2 is None else f'  oracle [{float(k2[near].min()):+.3f},{float(k2[near].max()):+.3f}]  max|d| {float((k1 - k2).abs()[near].max()):.2e}'))
    wallMine = perBody.sum(0)
    wallOrc = acc.cpu().double().numpy() - vfl                       # the oracle's total minus the (identical, checked in the interior) fluid-only Morris part
    for lab, sel in (('inner row(s) r < 0.275', rr < 0.275), ('outer row(s) r > 0.425', rr > 0.425), ('bulk', (rr >= 0.275) & (rr <= 0.425))):
        dd = np.abs(wallMine[sel] - wallOrc[sel])
        print(f'wall term, field={a.field}, {lab:24s}: max|d| {dd.max():.3e}  rms|d| {np.sqrt((dd ** 2).mean()):.3e}  rms(oracle) {np.sqrt((wallOrc[sel] ** 2).mean()):.3e}  rms(warpSPH) {np.sqrt((wallMine[sel] ** 2).mean()):.3e}')
    aT = lambda q: (x0[:, 0] * q[:, 1] - x0[:, 1] * q[:, 0]) / rr
    ao = acc.cpu().double().numpy()
    bins = np.linspace(r1, r2, 9)
    print('azimuthal viscous acceleration by radial bin (inner -> outer): [mean warpSPH total | mean DFSPH2D total | mean warpSPH fluid-only | warpSPH wall inner | wall outer]')
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (rr >= b0) & (rr < b1)
        print(f'  r [{b0:.3f},{b1:.3f}) n={int(m.sum()):4d}   {aT(tot_mine)[m].mean():+.4f} | {aT(ao)[m].mean():+.4f} | {aT(vfl)[m].mean():+.4f} | {aT(perBody[0])[m].mean():+.4f} | {aT(perBody[1])[m].mean():+.4f}')
    print(f'inner torque / exact:  warpSPH {mine[0] / Tex:.4f}   DFSPH2D {orc[0] / Tex:.4f}')
    print(f'outer torque / (-exact): warpSPH {mine[1] / -Tex:.4f}   DFSPH2D {orc[1] / -Tex:.4f}')
    print(f'rho  warpSPH [{st.densities[fl].min():.4f}, {st.densities[fl].max():.4f}]  DFSPH2D [{float(sim.rho.min()):.4f}, {float(sim.rho.max()):.4f}]')


if __name__ == '__main__':
    main()
