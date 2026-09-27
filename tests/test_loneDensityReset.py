"""`WeaklyCompressibleSPHConfig.loneDensityReset` and its detector
(`detectNoFluidNeighbours`), MDBC_CONTACT_LINE_PLAN.md §12."""
import dataclasses

import torch

from test_artificialCompressible import _walledLattice


def test_noFluidNeighboursIsExactlyTheRowsWithoutAFluidPartner(runtime):
    """Brute force: a fluid row is lone iff no OTHER fluid row lies strictly
    inside its support; wall rows do not count. The walled lattice has none,
    so a copy is thinned until some rows keep only wall neighbours."""
    from warpSPH.modules.surfaceDetection import detectNoFluidNeighbours
    st, config, _, wall = _walledLattice()
    fluid = ~wall
    # make the first fluid row above the wall lone: turn every other fluid row
    # within its support into wall so only wall partners remain
    i = int(torch.nonzero(fluid)[0])
    d = (st.positions - st.positions[i]).norm(dim=-1)
    H = float(st.supports[i])
    near = fluid & (d < H + 1e-3) & (torch.arange(d.numel(), device=d.device) != i)
    st.kinds = torch.where(near, torch.ones_like(st.kinds), st.kinds)
    fluid = st.kinds == 0
    lone = detectNoFluidNeighbours(st, config, None)
    dd = torch.cdist(st.positions, st.positions[fluid])
    selfPair = dd == 0.0
    edge = ((dd - st.supports.unsqueeze(-1)).abs() < 1e-5).any(-1)
    ref = ~((dd < st.supports.unsqueeze(-1)) & ~selfPair).any(-1)
    rows = fluid & ~edge
    assert bool(lone[i]) and bool(rows[i])
    assert torch.equal(lone[rows], ref[rows])


def test_loneResetIsANoOpWithoutLoneParticles(runtime):
    """A resting delta-SPH column has no lone particle: the flag must leave
    the run bit-identical."""
    from warpSPH.runner import run
    from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as base

    def withFlag(flag):
        def configureScheme(ctx):
            base.configureScheme(ctx)
            ctx.schemeConfig.loneDensityReset = flag
        return dataclasses.replace(base, configureScheme=configureScheme)

    kw = dict(scheme='deltaSPH', nx=16, nSteps=5, progress=False, plot=False,
              store=False, quiet=True)
    off = run(withFlag(False), **kw)
    on = run(withFlag(True), **kw)
    assert torch.equal(off.state.state.densities, on.state.state.densities)
    assert torch.equal(off.state.state.velocities, on.state.state.velocities)


def test_oneSidedHydrostaticOnlyEverRaisesTheWallDensity(runtime):
    """max(0, .) on the ghost -> wall hydrostatic increment: every wall row's
    english2025 density is >= the two-sided value, and rows whose increment
    was already >= 0 (the floor) are unchanged. Not a run-level no-op: the
    side-wall ghosts sit slightly off level, so a few side-wall increments are
    small negatives even in an open column."""
    from warpSPH.runner import run
    from warpSPH.cases.hydrostaticColumn import hydrostaticColumnCase as base
    from warpSPH.modules.mdbc.english2025 import computeMdbcDensityEnglish2025
    r = run(base, scheme='deltaSPH', nx=16, nSteps=1, progress=False, plot=False,
            store=False, quiet=True)
    st, sc, config = r.state.state, r.ctx.schemeConfig, r.ctx.config
    wall = st.kinds == 1
    sc.mdbcOneSidedHydrostatic = False
    two = computeMdbcDensityEnglish2025(st, config, sc, r.state.adjacency)[wall]
    sc.mdbcOneSidedHydrostatic = True
    one = computeMdbcDensityEnglish2025(st, config, sc, r.state.adjacency)[wall]
    sc.mdbcOneSidedHydrostatic = False
    assert bool((one >= two).all())
    assert bool((one > two).any()) and bool((one == two).any())
