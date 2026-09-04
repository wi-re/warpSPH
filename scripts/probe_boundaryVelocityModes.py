"""Probe (`DFSPH_IMPROVEMENT_PLAN.md` Part 9, 2026-08-28): what the mDBC
boundary-velocity conditions actually put on a boundary particle, and what that
costs the incompressible solvers.

Part 9 found that `BoundaryOperatorTerms.staticBoundary` (dropping a static
neighbour's reaction terms, as [BK]/SPlisHSPlasH do) is a 5.9x win in the
constant-density solve and *diverges* in the divergence-free one, and proposed
the reason: these are mDBC walls, so a boundary particle has zero position
change but a non-zero extrapolated *velocity*, which makes "static" false for
the solve that projects velocities. In DFSPH the boundary velocity is the
rigid body's -- constant zero for a static wall -- and the question does not
arise. This measures both halves of that.

  --mode verify   For one captured state: which `BCType` each boundary
      particle actually gets, and what each `BCType`'s velocity is, decomposed
      against the wall normal. Reported as least-squares slopes of the boundary
      velocity against the Shepard-interpolated fluid velocity at the ghost
      point, separately for the normal and tangential components -- so the
      published forms are exact integers:

          no-slip   (u_b = 2 u_wall - u_f)         normal -1, tangential -1
          free-slip (u_b = u_f - 2 (u_f.n) n)      normal -1, tangential +1
          zeros/constant (rigid body at rest)      normal  0, tangential  0

      Anything else is the implementation deviating from the formula in its
      own comment.

  --mode ab       End-to-end cross of `BCType` against `BoundaryOperatorTerms`,
      with per-step solver iteration counts, which is where "does it change
      convergence behaviour" gets answered.

Usage:
  python scripts/probe_boundaryVelocityModes.py --mode verify
  python scripts/probe_boundaryVelocityModes.py --mode ab --bcs zeros freeSlip
"""
from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='verify', choices=['verify', 'ab'])
parser.add_argument('--case', default='randomFlowIncompressible')
parser.add_argument('--extra', nargs='*', default=['--bounded'])
parser.add_argument('--nx', type=int, default=128)
parser.add_argument('--cflFactor', type=float, default=0.4,
                    help="0.4 is Bender & Koschier's published constant, and since Part 12 `cflFactor` multiplies the particle diameter, so the number here is theirs. Numbers recorded in DFSPH_IMPROVEMENT_PLAN.md before Part 12 say `cflFactor=0.1`, which was the same timestep under the old support-radius convention")
parser.add_argument('--warmup', type=int, default=120)
# --mode ab
parser.add_argument('--nsteps', type=int, default=900)
parser.add_argument('--bcs', nargs='*', default=['freeSlip', 'zeros', 'noSlip'],
                    help="BCType names; `freeSlipReflect`/`noSlipReflect` run the "
                         "published reflecting form instead of the shipped "
                         "projecting one (patched in, `src/` unchanged)")
parser.add_argument('--terms', nargs='*', default=['full', 'staticBoundary'])
parser.add_argument('--noPen', default='on', choices=['on', 'off'],
                    help="`mdbcNoPenetrationShift`, the ad-hoc normal-approach damper. "
                         "It supplies the wall-normal response that the slip conditions "
                         "drop, so the reflecting variants are only interpretable "
                         "against both settings")
parser.add_argument('--solvers', default='both',
                    choices=['both', 'divergenceFree', 'incompressible'])
args = parser.parse_args()

from warpSPHBootstrap import bootstrap
bootstrap(precision='float32')

import importlib
import math

import torch

from warpSPH.runner.cli import caseMain
from warpSPHCore import (OperationDirection, OperationProperties, SupportScheme,
                         WarpOperation, warpOperation)
from warpSPH.configurations import BoundaryOperatorTerms
from warpSPH.configurations.moduleConfigurations.boundaryConditions import BCType
from warpSPH.cases.weaklyCompressible import domainBoundarySdf
import warpSPH.modules.mdbc.velocity as velmod
import warpSPH.schemes.divergenceFree as dfsphmod
import warpSPH.systems.incompressible as sysmod
import warpSPH.cases.randomFlow as rfmod

mod = importlib.import_module(f'warpSPH.cases.{args.case}')
case = getattr(mod, f'{args.case}Case')


def _interpolatedFluidVelocity(currentState, config, adjacency):
    """The Shepard-normalised fluid velocity at the ghost points -- byte for
    byte what `velocity.py`'s `noSlip`/`freeSlip` compute as `qVel`."""
    def interp(values):
        return warpOperation(
            currentState,
            OperationProperties(kernel=config.kernel, operation=WarpOperation.Interpolate,
                                supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.FluidToGhost),
            domain=config.domain, adjacency=adjacency, queryValues=values)

    qVel = interp(currentState.velocities)
    shep = interp(torch.ones_like(currentState.densities))
    return qVel / (shep.view(-1, 1) + 1e-7)


def _reflecting(flipTangential: bool):
    """The published mDBC slip conditions, which `velocity.py` does not
    implement: both *reflect* the normal component (`-2 (u.n) n`) where the
    shipped code projects it out (`-1 (u.n) n`, i.e. the boundary particle ends
    up with no normal velocity at all). `freeSlip`'s own comment states the
    reflecting form; its code does not. Returns a drop-in replacement for the
    module function, for A/B only -- nothing in `src/` changes."""
    def fn(currentState, config, schemeConfig, adjacency):
        qVel = _interpolatedFluidVelocity(currentState, config, adjacency)
        ghost = currentState.kinds == 2
        bIndices = currentState.ghostIndices[ghost].long()
        off = currentState.ghostOffsets
        n_b = off / (off.norm(dim=-1, keepdim=True) + 1e-7)
        if flipTangential:
            # no-slip: u_b = 2 u_wall - u_f, with u_wall the body velocity
            # (zero on these rows) -- reverses both components.
            u = 2.0 * currentState.velocities - qVel
        else:
            # free-slip: u_b = u_f - 2 (u_f.n) n -- keeps the tangential part,
            # reverses the normal one.
            u = qVel - 2.0 * torch.einsum('nd,nd->n', qVel, n_b).view(-1, 1) * n_b
        out = currentState.velocities.clone()
        out[bIndices, :] = u[ghost, :]
        return out
    return fn


def _forcedFull(fn):
    """Force `boundaryOperatorTerms` back to `full` around one solver's call,
    so a run isolates the other one. Both solvers read the setting off
    `schemeConfig` when called, so wrapping the call is enough."""
    def wrapped(particles, config, schemeConfig, adjacency, dvdt, dt, verbose=False):
        saved = schemeConfig.solverConfig.boundaryOperatorTerms
        schemeConfig.solverConfig.boundaryOperatorTerms = BoundaryOperatorTerms.full
        try:
            return fn(particles=particles, config=config, schemeConfig=schemeConfig,
                      adjacency=adjacency, dvdt=dvdt, dt=dt, verbose=verbose)
        finally:
            schemeConfig.solverConfig.boundaryOperatorTerms = saved
    return wrapped


def forceBC(bc: BCType):
    """Override the boundary region's `BCType`. The case builds its wall with
    `boundaryRegion(ctx, sdf)` and takes the default (`freeSlip`), so patching
    the name in `randomFlow`'s namespace is the whole override."""
    _orig = rfmod.boundaryRegion

    def _wrapped(ctx, sdf, kind=None, **kwargs):
        return _orig(ctx, sdf, kind=bc, **kwargs)

    rfmod.boundaryRegion = _wrapped
    return lambda: setattr(rfmod, 'boundaryRegion', _orig)


def runVerify():
    cap = {}
    _real = velmod.computeBoundaryVelocities

    def _capture(currentState, config, schemeConfig, adjacency):
        out = _real(currentState, config, schemeConfig, adjacency)
        # Capture the *pre-BC* velocities: the mode functions read
        # `currentState.velocities` both as the interpolation source (fluid
        # rows) and as the body velocity (boundary rows), so re-running them on
        # the post-BC state would apply the condition twice.
        cap.update(state=currentState, config=config, schemeConfig=schemeConfig,
                   adjacency=adjacency, preVel=currentState.velocities.clone(),
                   applied=out.clone())
        return out

    import warpSPH.schemes.deltaSPH as deltamod
    holders = [velmod, dfsphmod, deltamod]
    originals = [getattr(h, 'computeBoundaryVelocities', None) for h in holders]
    for h in holders:
        if hasattr(h, 'computeBoundaryVelocities'):
            h.computeBoundaryVelocities = _capture
    try:
        r = caseMain(case, argv=['--nx', str(args.nx), '--nSteps', str(args.warmup),
                                 '--tLimit', '1000.0', '--cflFactor', str(args.cflFactor),
                                 '--quiet', '--no-store', '--no-plot'] + args.extra)
    finally:
        for h, o in zip(holders, originals):
            if o is not None:
                h.computeBoundaryVelocities = o

    state, config = cap['state'], cap['config']
    schemeConfig, adjacency = cap['schemeConfig'], cap['adjacency']
    state.velocities = cap['preVel'].clone()

    kinds = state.kinds
    fluid, boundary, ghost = kinds == 0, kinds == 1, kinds == 2
    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
          f"after {args.warmup} steps ===")
    print(f"{int(fluid.sum())} fluid, {int(boundary.sum())} boundary, "
          f"{int(ghost.sum())} ghost")

    # --- which BCType each boundary particle is actually assigned ------------
    from warpSPH.configurations.region import RegionType
    boundaryRegions = [rg for rg in config.regions if rg.type == RegionType.Boundary]
    print(f"\nboundary regions: {[rg.kind.name for rg in boundaryRegions]}")
    mats = state.materials[boundary]
    print(f"boundary materials present: {sorted(set(mats.tolist()))} "
          f"(indexes into that list -- `computeBoundaryVelocities` maps "
          f"material -> BCType by position)")
    for m in sorted(set(mats.tolist())):
        n = int((mats == m).sum())
        name = boundaryRegions[m].kind.name if m < len(boundaryRegions) else "OUT OF RANGE"
        print(f"  material {m}: {n} particles -> {name}")

    # --- the body velocity the conditions read -------------------------------
    # `noSlip` computes `2 * currentState.velocities - u_f` and then takes the
    # result at the *ghost* rows, so the "u_wall" it uses is the ghost row's
    # stored velocity, not the boundary particle's. On a moving wall those two
    # can disagree, and only `BCType.constant` bodies get their ghost rows
    # refreshed (`rigidBody/update.py:51-56`).
    preV = cap['preVel']
    for label, m in (('fluid', fluid), ('boundary', boundary), ('ghost', ghost)):
        v = preV[m].norm(dim=-1)
        print(f"pre-BC |v| {label:>9}: mean={float(v.mean()):.4e} max={float(v.max()):.4e}")

    # --- the wall normal, from two independent sources -----------------------
    gIdx = state.ghostIndices[ghost].long()          # boundary row of each ghost
    off = state.ghostOffsets[ghost]
    nGhost = off / (off.norm(dim=-1, keepdim=True) + 1e-7)
    ctx = r.ctx
    sdf = domainBoundarySdf(ctx)
    _, nSdf = sdf(state.positions.detach().clone().requires_grad_(True))
    nSdf = nSdf.detach()[gIdx]
    cosang = torch.einsum('nd,nd->n', nGhost, nSdf)
    print(f"\nghostOffset direction vs SDF normal at the boundary particle: "
          f"cos mean={float(cosang.mean()):+.4f} min={float(cosang.min()):+.4f} "
          f"({int((cosang < 0).sum())} of {len(cosang)} anti-aligned)")

    # --- the Shepard-interpolated fluid velocity at the ghost points ---------
    def interp(values):
        return warpOperation(
            state,
            OperationProperties(kernel=config.kernel, operation=WarpOperation.Interpolate,
                                supportMode=SupportScheme.Gather,
                                operationMode=OperationDirection.FluidToGhost),
            domain=config.domain, adjacency=adjacency, queryValues=values)

    uF = interp(state.velocities) / (interp(torch.ones_like(state.densities)).view(-1, 1) + 1e-7)
    uF = uF[ghost]
    fN = torch.einsum('nd,nd->n', uF, nGhost)
    fT = uF - fN.view(-1, 1) * nGhost

    modes = {
        'zeros': velmod.zeroVelocity, 'constant': velmod.constantVelocity,
        'noSlip': velmod.noSlip, 'freeSlip': velmod.freeSlip,
        'extended': velmod.extendedVelocity,
    }
    print(f"\nboundary velocity against the fluid velocity at the ghost point,")
    print(f"as least-squares slopes over the {len(gIdx)} boundary particles that have a ghost")
    print(f"{'BCType':>10} {'normal':>9} {'tangential':>11} {'published':>18} "
          f"{'|u_b| mean':>11} {'rows != 0':>10}")
    published = {'zeros': '0 / 0', 'constant': '0 / 0', 'noSlip': '-1 / -1',
                 'freeSlip': '-1 / +1', 'extended': '(extrapolated)'}
    for name, fn in modes.items():
        try:
            v = fn(state, config, schemeConfig, adjacency)
        except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
            print(f"{name:>10}  RAISED {type(exc).__name__}: {exc}")
            continue
        uB = v[gIdx]
        bN = torch.einsum('nd,nd->n', uB, nGhost)
        bT = uB - bN.view(-1, 1) * nGhost
        slopeN = float((bN * fN).sum() / ((fN * fN).sum() + 1e-30))
        slopeT = float((bT * fT).sum() / ((fT * fT).sum() + 1e-30))
        vb = v[boundary]
        print(f"{name:>10} {slopeN:+9.4f} {slopeT:+11.4f} {published[name]:>18} "
              f"{float(vb.norm(dim=-1).mean()):11.4e} "
              f"{int((vb.norm(dim=-1) > 1e-6).sum()):5d}/{int(boundary.sum())}")

    print("\nA slope of 0 in the normal column means the condition *projects out* the")
    print("normal component instead of reflecting it -- the wall neither pushes back")
    print("nor follows, it just has no normal velocity of its own.")


def runAB():
    rows = []
    for bcName in args.bcs:
        for termName in args.terms:
            # `<mode>Reflect` runs the published formula in place of the
            # shipped one, patched in for the duration of the run.
            base = bcName[:-len('Reflect')] if bcName.endswith('Reflect') else bcName
            restoreBC = forceBC(BCType[base])
            restoreFn = lambda: None
            if bcName.endswith('Reflect'):
                _origFn = getattr(velmod, base)
                setattr(velmod, base, _reflecting(flipTangential=(base == 'noSlip')))
                restoreFn = lambda _o=_origFn, _b=base: setattr(velmod, _b, _o)
            mode = BoundaryOperatorTerms[termName]
            _origCfg = case.configureScheme
            iters = {'df': [], 'ps': [], 'dfRes': [], 'psRes': []}

            def _wrapped(ctx, _orig=_origCfg, mode=mode):
                _orig(ctx)
                ctx.schemeConfig.solverConfig.boundaryOperatorTerms = mode
                ctx.schemeConfig.solverConfig.mdbcNoPenetrationShift = (args.noPen == 'on')

            _df, _ps = dfsphmod.solveDivergenceFree, sysmod.solveIncompressible

            def countDF(*a, _f=_df, **k):
                out = _f(*a, **k)
                iters['df'].append(len(out[2]))
                # The iteration count saturates at `maxIterations` in every
                # configuration (both solvers run their full budget every step
                # and never hit their tolerance -- Part 8 recorded the same),
                # so it does not discriminate. The residual the last iteration
                # actually reached does.
                iters['dfRes'].append(out[2][-1] if out[2] else float('nan'))
                return out

            def countPS(*a, _f=_ps, **k):
                out = _f(*a, **k)
                iters['ps'].append(len(out[2]))
                iters['psRes'].append(out[2][-1] if out[2] else float('nan'))
                return out

            if args.solvers == 'divergenceFree':
                countPS = _forcedFull(countPS)
            elif args.solvers == 'incompressible':
                countDF = _forcedFull(countDF)

            case.configureScheme = _wrapped
            dfsphmod.solveDivergenceFree = countDF
            sysmod.solveIncompressible = countPS
            try:
                r = caseMain(case, argv=[
                    '--nx', str(args.nx), '--nSteps', str(args.nsteps),
                    '--tLimit', '1000.0', '--cflFactor', str(args.cflFactor),
                    '--quiet', '--no-store', '--no-plot'] + args.extra)
            finally:
                case.configureScheme = _origCfg
                dfsphmod.solveDivergenceFree = _df
                sysmod.solveIncompressible = _ps
                restoreFn()
                restoreBC()
            rows.append((bcName, termName, r, iters))

    print(f"\n=== {args.case} {' '.join(args.extra)} nx={args.nx} "
          f"nSteps={args.nsteps} cflFactor={args.cflFactor} solvers={args.solvers} ===")
    print(f"mdbcNoPenetrationShift = {args.noPen}")
    print(f"{'BCType':>16} {'terms':>16} {'steps':>7} {'div':>5} {'minRho':>9} "
          f"{'maxRho':>9} {'|rho-1|':>10} {'t_final':>8} {'DF it':>6} {'DF resid':>10} "
          f"{'PS it':>6} {'PS resid':>10}")
    for bcName, termName, r, iters in rows:
        tr = [row for row in r.trajectory if all(math.isfinite(v) for v in row.values())]
        tail = tr[len(tr) // 2:] or tr
        band = [max(abs(row['maxDensity'] - 1.0), abs(row['minDensity'] - 1.0))
                for row in tail]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else float('nan')
        half = lambda xs: xs[len(xs) // 2:] or xs
        print(f"{bcName:>16} {termName:>16} {len(tr):7d} {str(r.diverged):>5} "
              f"{min(row['minDensity'] for row in tr):9.5f} "
              f"{max(row['maxDensity'] for row in tr):9.5f} "
              f"{sum(band) / max(1, len(band)):10.3e} "
              f"{(tr[-1].get('t', float('nan')) if tr else float('nan')):8.4f} "
              f"{mean(half(iters['df'])):6.1f} {mean(half(iters['dfRes'])):10.3e} "
              f"{mean(half(iters['ps'])):6.1f} {mean(half(iters['psRes'])):10.3e}")


if args.mode == 'verify':
    runVerify()
else:
    runAB()
