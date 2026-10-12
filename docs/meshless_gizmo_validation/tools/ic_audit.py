#!/usr/bin/env python
"""IC audit for the MFM / MFV closure A/B (COMPRESSIBLE_SPH_PLAN E2): is the IC consistent with the scheme's
own geometry?  Per case, original IC vs scheme-consistent IC:

  rho_geo / rho_target - 1   (rho_geo = m / V_geo, V_geo from the moments; the step uses rho_geo, not the
                              sampler's rho, so a mismatch is an IC pressure / density error)
  |a| at t = 0 (rates at dt ~ 0) for closure 'project' and 'none'; exact zero for the steady vortices.

`consistentIC(system, ...)`: m_i = rho_target_i V_geo_i with the supports iterated to the fixed point
h = (nngb V / C)^(1/nu) the step re-derives every step, then Q rebuilt from (m, rho_target, v, P).
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / ".tmp"))
import torch

torch.set_grad_enabled(False)

from warpSPHCore.mfm import MeshlessWarp, primitives, conserved
from warpSPH.schemes.meshless import supports_from_volume


def geometry(system, cfg, sc, closure, supports=None):
    st = system.state
    return MeshlessWarp.build(
        st.positions, st.supports if supports is None else supports, system.domain, cfg.kernel,
        cond_max=sc.condMax, closure=closure, closure_power=sc.closurePower, rtol=None, cg_tol=sc.cgTol,
        cg_iters=sc.cgIters, lam0=None, guards=True, face_cond_max=sc.faceCondMax, centred_weights=True,
        area_cap=False, closure_solver=sc.closureSolver, mg_min_particles=sc.mgMinParticles,
        mg_max_coarse=sc.mgMaxCoarse)


def consistentIC(system, cfg, sc, rho, vel, P, iters=4):
    """Rebuild the state so that the scheme's own primitives are (rho, vel, P) exactly."""
    st = system.state
    dim = st.positions.shape[1]
    sup = st.supports
    for _ in range(iters):
        g = geometry(system, cfg, sc, 'none', supports=sup)
        sup = supports_from_volume(g.volume, sc.nngb, dim)
    g = geometry(system, cfg, sc, 'none', supports=sup)
    m = rho * g.volume
    st.supports = sup
    st.masses = m
    st.Q = conserved(m, rho, vel, P, float(sc.gamma))
    st.densities, st.velocities, st.pressures = rho, vel, P
    return system


def audit(label, system, cfg, sc, rho, vel, P, vortex=False, mask=None):
    st = system.state
    gamma = float(sc.gamma)
    out = {}
    for closure in ('project', 'none'):
        g = geometry(system, cfg, sc, closure)
        r_, v_, p_ = primitives(st.Q, g.volume, gamma)
        kw = dict(dt=1e-9, mode='MFM', order=sc.order, beta_min=sc.betaMin, beta_max=sc.betaMax,
                  cond_crit=sc.condCrit, psi1=sc.psi1, psi2=sc.psi2, timeCentredFrame=sc.timeCentredFrame,
                  retry=sc.retry, massFluxLimit=sc.massFluxLimit, faceGuard=0)
        dQ = g.rates(r_, v_, p_, gamma, **kw)
        a = dQ[:, 1:-1] / st.Q[:, 0:1]
        if vortex:
            # exact Lagrangian acceleration of a steady azimuthal vortex: centripetal, -v^2 / r rhat
            r = st.positions.norm(dim=-1).clamp_min(1e-9)
            a = a + (vel * vel).sum(-1, keepdim=True) / r[:, None] * (st.positions / r[:, None])
        if mask is not None:
            a = a[mask]
        out[closure] = (float(a.double().pow(2).mean().sqrt()), float(a.norm(dim=-1).max()))
    g = geometry(system, cfg, sc, 'none')
    r_ = st.Q[:, 0] / g.volume
    e = (r_ / rho - 1)
    if mask is not None:
        e = e[mask]
    print(f'{label:34s} rho_geo/rho-1 rms {float(e.double().pow(2).mean().sqrt()):.2e} max {float(e.abs().max()):.2e} | '
          f'|a(t=0)-a_exact| rms project {out["project"][0]:.2e} none {out["none"][0]:.2e} '
          f'(max {out["project"][1]:.2e} / {out["none"][1]:.2e})', flush=True)


def main():
    import dataclasses
    from warpSPH.runner.runner import buildContext
    from warpSPH.runner.caseSpec import CaseSpec
    from warpSPH.cases.compressible.greshoVortex import greshoVortexCase
    from warpSPH.cases.compressible.sod import sodCase
    from warpSPH.caseUtils.compressible.greshoVortex.sample import greshoProfile
    import yee_valid
    from warpSPH.cases.compressible.yeeVortex import yeeVortexCase

    def make(case, scheme='MFM', **kw):
        spec = CaseSpec(caseName=case.name, scheme=scheme, params=dict(case.params)).merged(**case.defaults).merged(
            scheme=scheme, precision='float32', **kw)
        ctx = buildContext(case, spec)
        case.configureScheme(ctx)
        return ctx, case.buildSystem(ctx)

    # ---- Gresho (the steady vortex: exact a = 0) ----
    for nx in (48, 64, 96):
        for jit in (0.0, 0.1):
            c = dataclasses.replace(greshoVortexCase)
            ctx, system = make(greshoVortexCase, nx=nx, params=dict(greshoVortexCase.params, jitter=jit))
            cfg, sc, st = ctx.config, ctx.schemeConfig, system.state
            rho0 = st.densities.clone()
            vel0 = st.velocities.clone()
            P0 = st.pressures.clone()
            audit(f'gresho nx{nx} j{jit} original', system, cfg, sc, rho0, vel0, P0, vortex=True)
            consistentIC(system, cfg, sc, rho0, vel0, P0)
            audit(f'gresho nx{nx} j{jit} consistent', system, cfg, sc, rho0, vel0, P0, vortex=True)

    # ---- valid Yee (steady: exact a = 0) ----
    for nx in (64, 96):
        for jit in (0.0, 0.1, 0.25):
            yee_valid.OPT.update(jitter=jit, bg=0.0)
            case = dataclasses.replace(yeeVortexCase, buildSystem=yee_valid.buildSystem)
            ctx, system = make(case, nx=nx)
            cfg, sc, st = ctx.config, ctx.schemeConfig, system.state
            rho0, vel0, P0 = st.densities.clone(), st.velocities.clone(), st.pressures.clone()
            audit(f'yeeValid nx{nx} j{jit} original', system, cfg, sc, rho0, vel0, P0, vortex=True)
            consistentIC(system, cfg, sc, rho0, vel0, P0)
            audit(f'yeeValid nx{nx} j{jit} consistent', system, cfg, sc, rho0, vel0, P0, vortex=True)

    # ---- Sod (not steady: report away from the contacts |x -+ 0.5| > 0.2, where the exact a = 0) ----
    for jit in (0.0, 0.1):
        ctx, system = make(sodCase, nx=200, params=dict(sodCase.params, jitter=jit))
        cfg, sc, st = ctx.config, ctx.schemeConfig, system.state
        rho0, vel0, P0 = st.densities.clone(), st.velocities.clone(), st.pressures.clone()
        x = st.positions[:, 0]
        far = ((x - 0.5).abs() > 0.2) & ((x + 0.5).abs() > 0.2)
        audit(f'sod nx200 j{jit} original (far)', system, cfg, sc, rho0, vel0, P0, mask=far)
        near = ~far
        audit(f'sod nx200 j{jit} original (near)', system, cfg, sc, rho0, vel0, P0, mask=near)
        consistentIC(system, cfg, sc, rho0, vel0, P0)
        audit(f'sod nx200 j{jit} consistent (far)', system, cfg, sc, rho0, vel0, P0, mask=far)
        audit(f'sod nx200 j{jit} consistent (near)', system, cfg, sc, rho0, vel0, P0, mask=near)


if __name__ == '__main__':
    main()
