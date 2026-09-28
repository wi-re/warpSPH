"""Probe (`MDBC_CONTACT_LINE_PLAN.md` step 1): measure where the negative
pressure at a wall/free-surface contact comes from, on synthetic toys with a
known answer, before choosing a fix.

Toys (all `hydrostaticColumn`, no new geometry -- only gravity is turned):

  film   gravity UP (+y). The column sits on the floor, which is now a
         *ceiling* relative to gravity. Exact answer: the whole column free-
         falls away from it (p == 0, centroid displacement 0.5 g t^2). A
         wall that pulls shows up as fluid left behind on the floor.
         NB: full width = the inverted-glass equilibrium (p = -rho g H at
         the ceiling is a genuine static solution, only RT-unstable; air
         can reach the ceiling only up the side walls). Kept as a secondary.
  drop   the same, but a finite block (`blockWidth` x L wide, centred) under
         the ceiling: air at both of its ceiling contact lines, so no static
         hanging solution exists -- the exact answer is free fall.
  peel   gravity sideways (+x): the column peels off the left wall with a
         contact line at the top-left corner; the free surface tilts.
  sealed fillRatio = 1, gravity UP: no free surface anywhere, nothing may
         move, and the bottom wall legitimately carries p < 0 (no air can
         reach it). The negative control: a fix must leave this alone.

The per-step split (ACSPH, `pressureForceTerm = nonConservative`, i.e. the
linear `A(p)_i = -sum_j V_j (p_i + p_j) gradW_ij`), for every fluid row:

  F_wallPj  = A(p * 1_wall)_i            the wall's extrapolated pressure
  F_wallPi  = p_i * A(1_wall)_i          the particle's own p against the wall
  F_fluid   = A(p * 1_fluid)_i - F_wallPi   everything fluid-fluid

reported as the component opposing gravity, in units of g, averaged over the
fluid rows still in wall contact ("hold" = 1 means the wall is carrying the
particle's weight, i.e. it sticks).

Usage:
  python scripts/probe_contactLine.py --toy film --nx 32 --tLimit 0.4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# toy initial pressure (uniform); nonzero only to probe the closed-box gauge
# drift (--p0, MDBC_CONTACT_LINE_PLAN.md §9.3)
P0_OFFSET = 0.0
DEFAULT_OUT = os.path.join(HERE, 'out_contactLine')

TOYS = {
    'film':   dict(gravityDirection=[0.0, 1.0], fillRatio=0.5),
    'drop':   dict(gravityDirection=[0.0, 1.0], fillRatio=0.25, blockWidth=0.4),
    'peel':   dict(gravityDirection=[1.0, 0.0], fillRatio=0.5),
    'sealed': dict(gravityDirection=[0.0, 1.0], fillRatio=1.0),
    'column': dict(gravityDirection=[0.0, -1.0], fillRatio=0.5, keepIC=True),
}


def pressureSplit(system, ctx):
    """Per-row tensors of the wall/fluid decomposition at the current state.
    Returns a dict; see the module docstring for the definitions."""
    import torch
    from warpSPH.modules.gravity import computeGravity
    from warpSPH.modules.pressure import computePressureForceSurfaceAware
    from warpSPH.schemes.artificialCompressible import _workingState, wallPressures
    from warpSPH.modules.incompressible.wallPressure import _shepardValue

    state = system.state
    config, sc = ctx.config, ctx.schemeConfig
    adj = system.adjacency
    fluid = state.kinds == 0
    wall = state.kinds == 1
    g = computeGravity(state, config, sc, adj)
    wallG = g if bool(g.abs().any()) else None

    view = _workingState(state, state.positions, state.velocities, state.pressures)
    if not hasattr(sc, 'acParams'):
        # WC schemes: the step's own EOS pressure (wall rows = mDBC density)
        pAll = state.pressures
    else:
        pAll = wallPressures(view, config, adj, wallG)
    if not hasattr(sc, 'acParams'):
        pass
    elif sc.acParams.cavitationProjection == 'vicinity' and state.surfaceIndicators is not None:
        pAll = torch.where(wall & (state.surfaceIndicators > 0), pAll.clamp_min(0.0), pAll)
    elif sc.acParams.cavitationProjection in ('fluid', 'contact'):
        pAll = torch.where(wall, pAll.clamp_min(0.0), pAll)
    _, den = _shepardValue(view, config, adj, pAll)

    def A(q):
        v = _workingState(state, state.positions, state.velocities, q)
        return computePressureForceSurfaceAware(v, config, sc, adj) / state.densities.unsqueeze(-1)

    zero = torch.zeros_like(pAll)
    gWall = A(wall.to(pAll.dtype))                       # -sum_wall V gradW
    fFluid = A(torch.where(fluid, pAll, zero)) - pAll.unsqueeze(-1) * gWall
    # 'wall': the fluid-solid pair terms use max(0, .) on both sides
    pPair = (pAll.clamp_min(0.0) if hasattr(sc, 'acParams')
             and sc.acParams.cavitationProjection == 'wall' else pAll)
    fWallPj = A(torch.where(wall, pPair, zero))
    fWallPi = pPair.unsqueeze(-1) * gWall
    return dict(pAll=pAll, den=den, g=g, fWallPj=fWallPj, fWallPi=fWallPi,
                fFluid=fFluid, gWall=gWall, fluid=fluid, wall=wall)


def withWcSwitches(case, args):
    """delta-SPH switches of MDBC_CONTACT_LINE_PLAN.md §12 on top of a case's
    own configureScheme: `loneDensityReset`, `mdbcOneSidedHydrostatic`."""
    if not (args.loneReset or args.oneSidedHydro):
        return case
    _cs = case.configureScheme

    def configureScheme(ctx):
        _cs(ctx)
        if not hasattr(ctx.schemeConfig, 'loneDensityReset'):
            raise SystemExit('--loneReset/--oneSidedHydro are weakly-compressible only')
        ctx.schemeConfig.loneDensityReset = bool(args.loneReset)
        ctx.schemeConfig.mdbcOneSidedHydrostatic = bool(args.oneSidedHydro)
    return dataclasses.replace(case, configureScheme=configureScheme)


def patchWallMode(mode):
    """Swap `artificialCompressible.wallPressures` (looked up by module global
    inside the step) for an A/B variant. Probe-only; nothing in src changes."""
    import warpSPH.schemes.artificialCompressible as ac
    orig = ac.wallPressures
    if mode == 'eq61':
        return
    if mode == 'noHydro':
        ac.wallPressures = lambda view, config, adj, bf, clampNonNeg=False: orig(view, config, adj, None)
    elif mode == 'blanketClamp':
        ac.wallPressures = lambda view, config, adj, bf, clampNonNeg=False: orig(view, config, adj, bf, clampNonNeg=True)
    elif mode == 'dtsph':
        # Dual-time SPH (Ramachandran, Muta & Ramakrishna; gitlab.com/pypr/dtsph,
        # dtsph.py): SetPressureSolid = Adami Eq. 27 incl. the hydrostatic term,
        # then max(0, p_w) (Hughes & Graham 2010), recomputed every pseudo-
        # iteration; PredictPressure's VIJ uses the wall's OWN velocity (0 for
        # a static tank), not the mirrored ghost velocity (that one, ug, only
        # enters SolidWallNoSlipBC, and their dam break runs nu = 0). Static
        # walls only.
        import torch
        ac.wallPressures = lambda view, config, adj, bf, clampNonNeg=False: orig(view, config, adj, bf, clampNonNeg=True)
        ac.computeBoundaryVelocities = lambda view, config, sc, adj: torch.where(
            (view.kinds == 1).unsqueeze(-1), torch.zeros_like(view.velocities), view.velocities)
    elif mode == 'dtsphDiv':
        # 'dtsph' with the static wall velocity in the pressure (continuity)
        # row only; the viscosity keeps the case's free-slip mirror. Isolates
        # the clamp + kinematics from the no-slip drag that zeroing the wall
        # velocity everywhere adds to ACSPH's viscous term (Marrone 3.1 is
        # free-slip; 'dtsph' slowed the whole surge).
        import torch
        ac.wallPressures = lambda view, config, adj, bf, clampNonNeg=False: orig(view, config, adj, bf, clampNonNeg=True)
        origBV, origVisc = ac.computeBoundaryVelocities, ac._viscosity
        last = {}

        def staticWalls(view, config, sc, adj):
            last['sc'] = sc
            return torch.where((view.kinds == 1).unsqueeze(-1),
                               torch.zeros_like(view.velocities), view.velocities)

        def mirroredViscosity(view, config, adj, nu):
            v = ac._workingState(view, view.positions, view.velocities, view.pressures)
            v.velocities = origBV(v, config, last['sc'], adj)
            return origVisc(v, config, adj, nu)
        ac.computeBoundaryVelocities = staticWalls
        ac._viscosity = mirroredViscosity


def makeDiagnostics(baseDiagnostics, toy, record):
    import torch

    def diagnostics(ctx, system):
        out = dict(baseDiagnostics(ctx, system)) if baseDiagnostics else {}
        s = pressureSplit(system, ctx)
        st = system.state
        fluid, wall = s['fluid'], s['wall']
        gVec = s['g'][fluid]
        gMag = float(gVec.norm(dim=-1).max()) or 1.0
        up = -gVec[0] / gMag                             # unit vector opposing gravity
        dx = float(ctx.config.dx)

        # wall rows that actually see fluid
        seen = wall & (s['den'] > 1e-6)
        pw = s['pAll'][seen]
        out['pwMin'] = float(pw.min()) if pw.numel() else 0.0
        out['pwNegFrac'] = float((pw < 0).float().mean()) if pw.numel() else 0.0
        out['pfMin'] = float(st.pressures[fluid].min())
        out['pfMax'] = float(st.pressures[fluid].max())

        # contact rows: fluid with a non-trivial wall kernel sum
        contact = fluid & (s['gWall'].norm(dim=-1) > 1e-3 * float(s['gWall'].norm(dim=-1).max() + 1e-30))
        out['nContact'] = int(contact.sum())
        if int(contact.sum()):
            for k in ('fWallPj', 'fWallPi', 'fFluid'):
                out['hold_' + k[1:]] = float((s[k][contact] @ up).mean()) / gMag

        x0 = ctx.scratch.get('initialPositions')
        if x0 is not None:
            t = float(system.t)
            disp = float(((st.positions[fluid] - x0[fluid]) @ (-up)).mean())
            ff = 0.5 * gMag * t * t
            out['fallRatio'] = disp / ff if ff > 0 else 0.0
            # rows still within one spacing of where they started (stuck)
            moved = (st.positions[fluid] - x0[fluid]) @ (-up)
            out['stuckFrac'] = float((moved < 0.5 * dx).float().mean()) if t > 0 else 1.0
        # forensics on the most negative fluid row
        from warpSPH.modules.surfaceDetection import detectFreeSurface
        raw, dil, _, _, lam = detectFreeSurface(st, ctx.config, ctx.schemeConfig,
                                                ctx.schemeConfig.surfaceDetectionConfig,
                                                system.adjacency, returnNormals=True)
        fIdx = torch.nonzero(fluid).squeeze(-1)
        k = fIdx[torch.argmin(st.pressures[fluid])]
        r = (st.positions - st.positions[k]).norm(dim=-1)
        H = float(st.supports[k])
        out['worst_p'] = float(st.pressures[k])
        out['worst_raw'] = float(raw[k] > 0.5)
        out['worst_dil'] = float(dil[k] > 0.5)
        out['worst_lam'] = float(lam[k]) if lam is not None else float('nan')
        out['worst_nF'] = int(((r < H) & fluid).sum()) - 1
        out['worst_nW'] = int(((r < H) & wall).sum())
        out['worst_dW'] = float(r[wall].min()) / dx
        neg = fluid & (st.pressures < 0)
        out['negRaw'] = int((neg & (raw > 0.5)).sum())
        out['negNotRaw'] = int((neg & ~(raw > 0.5)).sum())
        out['nRaw'] = int((fluid & (raw > 0.5)).sum())
        tmp = {}
        forensics(ctx, system, tmp, None)
        out['nCeiling'] = tmp.get('nCeiling', 0)
        out['vCeiling'] = tmp.get('vCeiling', 0.0)
        record.append(dict(t=float(system.t), **out))
        # make the plotted wall rows show the pressure the scheme actually uses
        st.pressures = torch.where(wall, s['pAll'], st.pressures)
        return out
    return diagnostics


def buildCase(toy: str, dambreakLikeAC: bool, cav: str = 'off', nuOff=False, shiftOff=False, ddt=None):
    import torch
    from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as base
    from warpSPH.cases.plotting import Field, particlePlot
    from warpSPH.configurations.moduleConfigurations.shifting import ShiftingProjectionScheme, ShiftingScheme

    def configureScheme(ctx):
        base.configureScheme(ctx)
        if hasattr(ctx.schemeConfig, 'acParams'):
            ctx.schemeConfig.acParams.cavitationProjection = cav
        elif cav != 'off':
            raise SystemExit('--cav is ACSPH-only: the delta-SPH variants were tried '
                             'and removed (MDBC_CONTACT_LINE_PLAN.md §7.5)')
        else:
            # hydrostaticColumn is a DFSPH case and forces a physical nu = 0,
            # which strips delta-SPH's artificial viscosity -- unstable on its
            # own (a free-falling block blows up by t = 0.09, §8). Run the WC
            # toys in sloshingTank's configuration instead: alpha = 0.02,
            # Michel shifting.
            sc = ctx.schemeConfig
            sc.diffusionParams.inviscid = True
            sc.diffusionParams.inviscidAlpha = 0.02
            if ddt is not None:
                from warpSPH.enumTypes import DensityDiffusionScheme
                sc.diffusionParams.densityDiffusionTerm = DensityDiffusionScheme[ddt]
            sc.shiftProperties.active = True
            sc.shiftProperties.scheme = ShiftingScheme.michel2022
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.michel2022
        if dambreakLikeAC:
            # the dam-break ACSPH configuration the failures were seen under:
            # Michel shifting + the paper's alpha_nu viscosity + eps_v = -5
            sc = ctx.schemeConfig
            sc.shiftProperties.active = True
            sc.shiftProperties.scheme = ShiftingScheme.michel2022
            sc.shiftProperties.projectionScheme = ShiftingProjectionScheme.michel2022
            depth = ctx.param('fillRatio') * ctx.spec.L
            sc.acParams.alphaNu = 0.01
            sc.acParams.referenceSoundSpeedForViscosity = float(
                50.0 * (ctx.param('gravityMagnitude') * depth) ** 0.5)
            sc.acParams.epsilonV = -5.0
            if nuOff:
                sc.acParams.referenceSoundSpeedForViscosity = None
            if shiftOff:
                sc.shiftProperties.active = False

    def isWC(ctx):
        return not hasattr(ctx.schemeConfig, 'acParams')

    def initialConditions(ctx, system):
        base.initialConditions(ctx, system)
        if isWC(ctx):
            # weakly compressible: Sun et al. 2017 Eq. (2), Ma = 0.1 on the
            # free-fall speed over the box height; dt from the acoustic CFL
            from warpSPH.cases.weaklyCompressible import setupTimestep
            g = ctx.param('gravityMagnitude')
            # c0 at Ma 0.1, but dt pinned at an acoustic Courant number
            # c0 dt / dx = 0.3: the machTarget route's CFL gave 0.84 here, and
            # even the at-rest column blew up by t = 0.1 (§8; sloshingTank
            # runs at 0.42)
            c0 = 10.0 * (2.0 * g * ctx.spec.L) ** 0.5
            ctx.spec.params['soundSpeed'] = c0
            ctx.spec.params['targetDt'] = 0.3 * float(ctx.config.dx) / c0
            setupTimestep(ctx, system)
        if ctx.param('keepIC'):
            return
        # the exact solution of every toy here starts from p = 0 (free fall /
        # an unbalanced column); the base IC seeds downward hydrostatics
        system.state.pressures = torch.full_like(system.state.pressures, P0_OFFSET)

    def buildSystem(ctx):
        from warpSPH.cases.hydrostaticColumn import columnSdf
        from warpSPH.cases.weaklyCompressible import (boundaryRegion, buildRegionSystem,
                                                      domainBoundarySdf, fluidRegion, shapeSdf)
        from warpSPH.configurations.region import BCType
        if not ctx.param('blockWidth'):
            return base.buildSystem(ctx)
        if not ctx.param('band'):
            from warpSPH.cases.randomFlow import BOUNDED_BAND
            ctx.spec.params['band'] = BOUNDED_BAND
        L = ctx.spec.L
        halfH = 0.5 * ctx.param('fillRatio') * L
        sdf = shapeSdf('box', args=[[0.5 * ctx.param('blockWidth') * L, halfH]],
                       offset=[0.0, -0.5 * L + halfH])
        return buildRegionSystem(ctx, [fluidRegion(ctx, sdf),
                                       boundaryRegion(ctx, domainBoundarySdf(ctx),
                                                      kind=BCType[ctx.param('wallBC')])])

    setupPlot, updatePlot = particlePlot([
        Field('velocities', '|v|', colorMap='viridis', mapping='L2Norm', boundary='Visualize'),
        Field('pressures', 'p (walls: Eq. 61 value)', colorMap='RdBu', flip=True,
              colorMapKind='diverging', boundary='Visualize'),
    ])
    def timestep(ctx, state):
        if isWC(ctx):
            # adaptive, but never above the pinned acoustic Courant 0.3 set in
            # `initialConditions` -- the bare adaptive step ran the toys at
            # ~0.8 while the probe claimed 0.3 (OPEN_PROBLEMS.md §11)
            from warpSPH.modules.timestep import computeTimestep
            adaptive = computeTimestep(state, ctx.config, ctx.schemeConfig, dt=ctx.config.dt)
            return min(adaptive, ctx.param('targetDt')) if ctx.param('targetDt') else adaptive
        return base.timestep(ctx, state)

    return base, dataclasses.replace(base, name=f'contactLine_{toy}', timestep=timestep,
                                     configureScheme=configureScheme,
                                     initialConditions=initialConditions,
                                     buildSystem=buildSystem,
                                     setupPlot=setupPlot, updatePlot=updatePlot)


class Stalled(RuntimeError):
    pass


def forensics(ctx, system, out, cav):
    """Most-negative fluid row + wall closure summary, shared by toys and the
    dam break. Also a sim-time stall watchdog: `stallDtSteps` only fires at
    dt == minDt exactly, and ACSPH's [0.8, 1.2] step clamp hovers above it."""
    import torch
    from warpSPH.modules.surfaceDetection import detectFreeSurface
    if system.adjacency is None:
        return None
    st = system.state
    s = pressureSplit(system, ctx)
    fluid, wall = s['fluid'], s['wall']
    dx = float(ctx.config.dx)
    seen = wall & (s['den'] > 1e-6)
    out['pwMin'] = float(s['pAll'][seen].min()) if bool(seen.any()) else 0.0
    out['pfMin'] = float(st.pressures[fluid].min())
    raw, dil, _, _, lam = detectFreeSurface(st, ctx.config, ctx.schemeConfig,
                                            ctx.schemeConfig.surfaceDetectionConfig,
                                            system.adjacency, returnNormals=True)
    fIdx = torch.nonzero(fluid).squeeze(-1)
    k = fIdx[torch.argmin(st.pressures[fluid])]
    r = (st.positions - st.positions[k]).norm(dim=-1)
    Hs = float(st.supports[k])
    out['worst_p'] = float(st.pressures[k])
    out['worst_uid'] = int(st.UIDs[k]) if getattr(st, 'UIDs', None) is not None else int(k)
    out['worst_raw'] = float(raw[k] > 0.5)
    out['worst_dil'] = float(dil[k] > 0.5)
    out['worst_lam'] = float(lam[k]) if lam is not None else float('nan')
    out['worst_nF'] = int(((r < Hs) & fluid).sum()) - 1
    out['worst_nW'] = int(((r < Hs) & wall).sum())
    out['worst_dW'] = float(r[wall].min()) / dx if bool(wall.any()) else float('nan')
    out['worst_v'] = float(st.velocities[k].norm())
    # force split on the worst row, in units of g
    gMag = float(s['g'][fluid].norm(dim=-1).max()) or 1.0
    for key in ('fWallPj', 'fWallPi', 'fFluid'):
        out['worst_' + key[1:]] = float(s[key][k].norm()) / gMag
    # the wall rows it sees
    wk = wall & (r < Hs)
    out['worst_pwMin'] = float(s['pAll'][wk].min()) if bool(wk.any()) else 0.0
    # the fastest fluid row: where the dt-pinning kicks come from
    speed = st.velocities[fluid].norm(dim=-1)
    kv = fIdx[torch.argmax(speed)]
    rv = (st.positions - st.positions[kv]).norm(dim=-1)
    out['fast_v'] = float(speed.max())
    out['fast_p'] = float(st.pressures[kv])
    out['fast_raw'] = float(raw[kv] > 0.5)
    out['fast_dil'] = float(dil[kv] > 0.5)
    out['fast_lam'] = float(lam[kv]) if lam is not None else float('nan')
    out['fast_nF'] = int(((rv < Hs) & fluid).sum()) - 1
    out['fast_nW'] = int(((rv < Hs) & wall).sum())
    out['fast_dW'] = float(rv[wall].min()) / dx if bool(wall.any()) else float('nan')
    # ceiling riders, geometry-free (works for a rolling tank): fluid rows
    # with a wall row within 1.5 dx on their gravity-up side, i.e. hanging
    # under a wall; vCeiling = their mean velocity along `up`
    up = -s['g'][fluid][0] / gMag
    xf, xw = st.positions[fluid], st.positions[wall]
    d = xw.unsqueeze(0) - xf.unsqueeze(1)                    # (nF, nW)
    dUp = d @ up
    dSide = (d - dUp.unsqueeze(-1) * up).norm(dim=-1)
    close = d.norm(dim=-1) < 1.5 * dx
    above = (dUp > 0.5 * dx) & (dSide < 0.75 * dx)       # directly above, not a side wall
    nearCeil = (close & above).any(dim=1)
    out['nCeiling'] = int(nearCeil.sum())
    out['vCeiling'] = float((st.velocities[fluid][nearCeil] @ up).mean()) if bool(nearCeil.any()) else 0.0
    return s


class FlierTracker:
    """Sparse-support fluid rows, split by wall contact (MDBC_CONTACT_LINE_PLAN.md §9).

    Counts come straight from the step's own adjacency on the GPU
    (`countNeighborsWarp`, FluidToFluid and FluidToBoundary, kernel support):
    nF = fluid neighbours, nW = wall neighbours. A fluid row with nF <= `maxF`
    (after removing itself) is *sparse*, and then either

      wall-sparse  nW > 0   a thinly supported row hanging on a wall -- the
                            population the contact-line fix is meant to remove
      free flier   nW = 0   a lone particle in free space -- the population a
                            bad fix creates

    with `iso*` the nF == 0 subsets. `lastWallT` (per UID) remembers when each
    row last had a wall neighbour; every row that becomes a free flier is kept
    in `births` with `sinceWall` = t - lastWallT (inf: never touched a wall),
    its speed, pressure and density, so the origin of the spray can be read off.
    """

    def __init__(self, maxF=2):
        self.maxF = maxF
        self.prevFree = None
        self.lastWallT = None
        self.births = []
        # §11.1 per-flier trace: every UID ever born is followed every step
        # (own p, speed, nF, nW, how many of its fluid neighbours sit in the
        # wall-adjacent band that 'contact' clamps, their p, and vn = the
        # velocity relative to the neighbours' mean along centroid -> row,
        # > 0 departing). Separates "partner clamped" from "own frozen p".
        self.tracked = set()
        self.trace = []

    def _traceTracked(self, st, config, adjacency, uid, fluid, nF, nW, t):
        import torch
        from warpSPH.schemes.artificialCompressible import _wallAdjacent
        rows = torch.nonzero(fluid & torch.isin(uid, torch.tensor(sorted(self.tracked), device=uid.device))).squeeze(-1)
        if rows.numel() == 0:
            return
        band = fluid & _wallAdjacent(st, config, adjacency)
        fIdx = torch.nonzero(fluid).squeeze(-1)
        xs, xf = st.positions[rows], st.positions[fIdx]
        d = torch.cdist(xs, xf)
        H = st.supports[rows].unsqueeze(-1)
        nb = (d < H) & (fIdx.unsqueeze(0) != rows.unsqueeze(-1))
        for a, i in enumerate(rows.tolist()):
            j = fIdx[nb[a]]
            rec = dict(t=t, uid=int(uid[i]), p=float(st.pressures[i]),
                       v=float(st.velocities[i].norm()), nF=int(nF[i]), nW=int(nW[i]),
                       inBand=bool(band[i]), nNbr=int(j.numel()))
            if j.numel() > 0:
                rec['nBand'] = int(band[j].sum())
                rec['pNbrMean'] = float(st.pressures[j].mean())
                rec['pNbrMin'] = float(st.pressures[j].min())
                n = st.positions[i] - st.positions[j].mean(0)
                n = n / n.norm().clamp_min(1e-12)
                rec['vn'] = float(((st.velocities[i] - st.velocities[j].mean(0)) * n).sum())
            self.trace.append(rec)

    @staticmethod
    def counts(st, config, adjacency, mode):
        from warpSPH.modules.util.wp_numNeighbors import countNeighborsWarp
        from warpSPHCore import GradientScheme, OperationDirection, OperationProperties, SupportScheme, WarpOperation
        return countNeighborsWarp(
            st, OperationProperties(kernel=config.kernel, operation=WarpOperation.Gradient,
                                    supportMode=SupportScheme.SuperSymmetric,
                                    operationMode=mode, gradientMode=GradientScheme.Naive),
            config.domain, adjacency=adjacency)

    def __call__(self, system, ctx, out):
        import torch
        from warpSPHCore import OperationDirection
        st, config = system.state, ctx.config
        fluid = st.kinds == 0
        # FluidToBoundary also gates on the *query* kind (returns 0 for every
        # fluid row), so the wall count is AllToAll - FluidToFluid; both
        # include the row itself (no i == j skip, W(0) > 0)
        nFs = self.counts(st, config, system.adjacency, OperationDirection.FluidToFluid).to(torch.int64)
        nAll = self.counts(st, config, system.adjacency, OperationDirection.AllToAll).to(torch.int64)
        nF = nFs - self.selfCount
        nW = nAll - nFs
        sparse = fluid & (nF <= self.maxF)
        iso = fluid & (nF == 0)
        onWall = nW > 0
        free, wallSparse = sparse & ~onWall, sparse & onWall
        speed = st.velocities.norm(dim=-1)
        out['nFree'] = int(free.sum())
        out['nFreeIso'] = int((iso & ~onWall).sum())
        out['nWallSparse'] = int(wallSparse.sum())
        out['nWallIso'] = int((iso & onWall).sum())
        out['freeVmax'] = float(speed[free].max()) if bool(free.any()) else 0.0
        out['wallSparseVmax'] = float(speed[wallSparse].max()) if bool(wallSparse.any()) else 0.0
        out['minDensity'] = float(st.densities[fluid].min())
        uid = st.UIDs if getattr(st, 'UIDs', None) is not None else torch.arange(fluid.numel(), device=fluid.device)
        uid = uid.to(torch.int64)
        t = float(system.t)
        if self.lastWallT is None:
            self.lastWallT = torch.full((int(uid.max()) + 1,), float('inf'), device=uid.device)
            self.prevFree = torch.zeros_like(self.lastWallT, dtype=torch.bool)
        touching = fluid & onWall
        self.lastWallT[uid[touching]] = t
        freeNow = torch.zeros_like(self.prevFree)
        freeNow[uid[free]] = True
        born = free & ~self.prevFree[uid]
        out['nFreeBirths'] = int(born.sum())
        if bool(born.any()):
            idx = torch.nonzero(born).squeeze(-1).tolist()
            for i in idx:
                lw = float(self.lastWallT[uid[i]])
                self.births.append(dict(t=t, uid=int(uid[i]), v=float(speed[i]),
                                        sinceWall=(t - lw) if lw != float('inf') else None,
                                        nF=int(nF[i]), p=float(st.pressures[i]),
                                        rho=float(st.densities[i])))
                self.tracked.add(int(uid[i]))
        # also every row that is ever wall-sparse (a lone / thin row hanging on
        # a wall: the ceiling riders), so its carried state is on record too
        if bool(wallSparse.any()):
            self.tracked.update(uid[wallSparse].tolist())
        if self.tracked:
            self._traceTracked(st, config, system.adjacency, uid, fluid, nF, nW, t)
        self.prevFree = freeNow

    selfCount = 1   # the count includes the row itself (no i == j skip)


def runMarrone(args):
    """Marrone et al. 2011 Sec. 3.1 dam break, the exact configuration of
    `probe_deltaSPHMarrone.py` (FREESLIP_DAMBREAK_FINDINGS.md Sec. 9), with the
    contact-line forensics streamed every step."""
    import time
    import torch
    sys.path.insert(0, HERE)
    import probe_deltaSPHMarrone as M
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.cases.sloshingTank import sloshingTankCase
    from warpSPH.runner import run
    sloshing = args.toy == 'sloshing'
    baseCase = sloshingTankCase if sloshing else dambreakCase
    patchWallMode(args.wallMode)
    baseCase = withWcSwitches(baseCase, args)
    if args.jitter > 0.0:
        # independent realisations (MDBC_CONTACT_LINE_PLAN.md §12): after the
        # case's own initial conditions, move every fluid particle by a seeded
        # uniform offset of at most `jitter * dx` per component
        _ic = baseCase.initialConditions

        def _jitteredIC(ctx, system):
            if _ic is not None:
                _ic(ctx, system)
            st = system.state
            gen = torch.Generator(device=st.positions.device).manual_seed(args.seed)
            off = (torch.rand(st.positions.shape, generator=gen, device=st.positions.device,
                              dtype=st.positions.dtype) * 2.0 - 1.0) * args.jitter * float(ctx.config.dx)
            st.positions = torch.where((st.kinds == 0).unsqueeze(-1), st.positions + off, st.positions)
        baseCase = dataclasses.replace(baseCase, initialConditions=_jitteredIC)

    record = []
    baseDiag = baseCase.diagnostics
    hist = []
    fliers = FlierTracker()

    def diagnostics(ctx, system):
        out = dict(baseDiag(ctx, system)) if baseDiag else {}
        if len(hist) % args.forensicsEvery == 0:
            forensics(ctx, system, out, args.cav)
            fliers(system, ctx, out)
        t = float(system.t)
        hist.append(t)
        out['wall_s'] = time.perf_counter() - t0
        record.append(dict(t=t, **{k: v for k, v in out.items()
                                   if isinstance(v, (int, float))}))
        if len(record) % 500 == 0:
            with open(os.path.join(runRoot, 'record.json'), 'w') as f:
                json.dump(record, f)
            with open(os.path.join(runRoot, 'births.json'), 'w') as f:
                json.dump(fliers.births, f)
            with open(os.path.join(runRoot, 'trace.json'), 'w') as f:
                json.dump(fliers.trace, f)
        if len(hist) > 300 and hist[-1] - hist[-301] < args.stallSimTime:
            raise Stalled(f'sim time advanced {hist[-1] - hist[-301]:.2e} s over 300 steps')
        return out

    case = dataclasses.replace(baseCase, diagnostics=diagnostics)
    tag = (f'{args.toy}_{args.scheme}_nx{args.nx}_cav{args.cav}'
           + ('_iso0' if args.isoZero else '')
           + ('_lone0' if args.loneReset else '')
           + ('_hyd1' if args.oneSidedHydro else '')
           + (f'_j{args.seed}' if args.jitter > 0.0 else '')
           + (f'_wall{args.wallMode}' if args.wallMode != 'eq61' else '')
           + (f'_{args.tag}' if args.tag else ''))
    runRoot = os.path.join(args.out, tag)
    os.makedirs(runRoot, exist_ok=True)
    params = dict(W=M.TANK_W, fillRatio=M.H / M.TANK_L, fluidWidth=M.COL_W / M.TANK_W,
                  gravityMagnitude=M.G, pressureProbeHeights=M.PROBE_HEIGHTS,
                  pressureProbeInset=0.0, pressureProbeDiscRadius=M.PROBE_DISC_RADIUS,
                  referenceVelocity=M.U_MAX, machTarget=1.95 / 40.0,
                  acCavitationProjection=args.cav, acIsolatedZeroPressure=args.isoZero,
                  epsilonV=-5.0)
    if args.scheme != 'artificialCompressible' and args.cav != 'off':
        raise SystemExit('--cav is ACSPH-only: the delta-SPH variants were tried '
                         'and removed (MDBC_CONTACT_LINE_PLAN.md §7.5)')
    kw = dict(scheme=args.scheme, L=M.TANK_L, nx=args.nx, tLimit=args.tLimit,
              quiet=False, progress=True, store=args.store, params=params)
    if sloshing:
        # the case's own defaults (diffSPH TC10 reproduction); only the
        # contact treatment and the run length change
        kw = dict(tLimit=args.tLimit, quiet=False, progress=True, store=args.store)
        if args.scheme != 'default':
            kw['scheme'] = args.scheme
        if args.nx:
            kw['nx'] = args.nx
    if args.store:
        kw.update(storeMode='states', storeInterval=args.storeInterval, exportRoot=runRoot)
    if args.resume:
        kw.update(resumeFrom=args.resume)
    if args.video:
        kw.update(plot=True, video=True, plotInterval=args.plotInterval, exportRoot=runRoot)
    from _runWatch import watchOverrides
    kw.update(watchOverrides(args))
    print(f'[{tag}] tLimit {args.tLimit}', flush=True)
    t0 = time.perf_counter()
    status = 'ok'
    try:
        r = run(case, **kw)
        status = f'diverged={r.diverged} steps={r.nSteps}'
    except Stalled as e:
        status = f'STALLED: {e}'
    with open(os.path.join(runRoot, 'record.json'), 'w') as f:
        json.dump(record, f)
    with open(os.path.join(runRoot, 'births.json'), 'w') as f:
        json.dump(fliers.births, f)
    with open(os.path.join(runRoot, 'trace.json'), 'w') as f:
        json.dump(fliers.trace, f)
    last = record[-1] if record else {}
    print(f'[{tag}] {status}  t={last.get("t", 0):.4f} t*={last.get("tStar", 0):.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--toy', choices=sorted(TOYS) + ['marrone31', 'sloshing'], default='film')
    ap.add_argument('--forensicsEvery', type=int, default=1)
    ap.add_argument('--stallSimTime', type=float, default=1e-3)
    ap.add_argument('--store', action='store_true')
    ap.add_argument('--storeInterval', type=int, default=200)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--scheme', default='artificialCompressible')
    ap.add_argument('--nx', type=int, default=32)
    ap.add_argument('--tLimit', type=float, default=0.4)
    ap.add_argument('--gravity', type=float, default=9.81)
    ap.add_argument('--plotInterval', type=int, default=5)
    ap.add_argument('--no-video', dest='video', action='store_false')
    ap.add_argument('--paperAC', action='store_true',
                    help='dam-break ACSPH extras (Michel + alpha_nu + eps_v=-5)')
    ap.add_argument('--wallMode', default='eq61',
                    choices=['eq61', 'noHydro', 'blanketClamp', 'dtsph', 'dtsphDiv'],
                    help='A/B of the ACSPH wall closure (probe-side patch only)')
    ap.add_argument('--cav', default='off', choices=['off', 'wall', 'vicinity', 'contact', 'fluid'])
    ap.add_argument('--isoZero', action='store_true',
                    help='ACSPH acParams.isolatedZeroPressure (p = 0 on empty supports, §11.1)')
    ap.add_argument('--loneReset', action='store_true',
                    help='delta-SPH schemeConfig.loneDensityReset: rho = rho0 on rows with no fluid neighbour (§12)')
    ap.add_argument('--oneSidedHydro', action='store_true',
                    help='delta-SPH schemeConfig.mdbcOneSidedHydrostatic (english2025 max(0, .), §12)')
    ap.add_argument('--jitter', type=float, default=0.0,
                    help='seeded uniform IC perturbation of fluid positions, in units of dx (marrone31/sloshing)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--nuOff', action='store_true')
    ap.add_argument('--shiftOff', action='store_true')
    ap.add_argument('--ddt', default=None,
                    help='WC density diffusion term (DensityDiffusionScheme name)')
    ap.add_argument('--integrator', default=None, help='integrationScheme override')
    ap.add_argument('--p0', type=float, default=0.0,
                    help='uniform initial toy pressure (closed-box gauge-drift control)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=DEFAULT_OUT)
    # velocity alarm (+ stall flags; this probe keeps its own sim-time watchdog,
    # `--stallSimTime`, so the runner's stallProgress defaults off here)
    from _runWatch import addWatchArguments
    addWatchArguments(ap, stallProgress=0)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(HERE))
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float32')
    from warpSPH.runner import run

    if args.toy in ('marrone31', 'sloshing'):
        return runMarrone(args)
    patchWallMode(args.wallMode)
    global P0_OFFSET
    P0_OFFSET = args.p0
    base, case = buildCase(args.toy, args.paperAC, args.cav, args.nuOff, args.shiftOff, args.ddt)
    case = withWcSwitches(case, args)
    record = []
    case = dataclasses.replace(case, diagnostics=makeDiagnostics(base.diagnostics, args.toy, record))

    tag = (f'{args.toy}_{args.scheme}_nx{args.nx}'
           + ('_lone0' if args.loneReset else '') + ('_hyd1' if args.oneSidedHydro else '')
           + (f'_{args.tag}' if args.tag else ''))
    runRoot = os.path.join(args.out, tag)
    os.makedirs(runRoot, exist_ok=True)
    params = dict(TOYS[args.toy], gravityMagnitude=args.gravity, epsilonV=-5.0)
    kw = dict(scheme=args.scheme, nx=args.nx, tLimit=args.tLimit, quiet=False,
              progress=True, params=params, stallDtSteps=200)
    if args.integrator:
        kw['integrationScheme'] = args.integrator
    elif args.scheme == 'deltaSPH':
        kw['integrationScheme'] = 'symplecticEuler'   # sloshingTank's default
    if args.video:
        kw.update(plot=True, video=True, plotInterval=args.plotInterval,
                  exportRoot=runRoot)
    from _runWatch import watchOverrides
    kw.update(watchOverrides(args))
    print(f'[{tag}] {params}', flush=True)
    r = run(case, **kw)
    with open(os.path.join(runRoot, 'record.json'), 'w') as f:
        json.dump(record, f)
    last = record[-1] if record else {}
    print(f'[{tag}] done diverged={r.diverged} steps={r.nSteps} '
          f'video={getattr(r, "videoPath", None)}', flush=True)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in last.items()}), flush=True)


if __name__ == '__main__':
    main()
