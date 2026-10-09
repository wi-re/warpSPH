"""Stage 3 E5 (ANALYTIC_BOUNDARIES_PLAN.md): the pinned band (`modules/boundaryConditions/pinned.py`), a prescribed-velocity region of fluid particles in a periodic box; the oracle's `test_pinned_frame.py`.

(a) a uniform stream in a wall-free periodic box with a band is stationary (velocity exactly the stream, density 1, the raw positions advance by u t through the seam, nothing created or removed);
(b) fluid at rest with the band at a stream: the band holds the stream exactly, the rest is carried along, everything stays finite;
(c) the shift is off inside the band (`pinnedKeepWeight`) and untouched elsewhere / without a band.
"""
import pytest
import torch

from warpSPH.cases import importAll
from warpSPH.modules.boundaryConditions import PinnedBand, pinnedBandBC, pinnedKeepWeight
from warpSPH.runner import getCase, run
from warpSPH.runner.caseSpec import CaseSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA')

STREAM = 0.5


def pinnedRun(streamInit, nSteps, nx=24, scheme='deltaSPH', caseName='tgv-wc'):
    """TGV box (no walls) with the velocity replaced by `streamInit` and a band of half width 0.6 around the seam x = 0; returns (result, band)."""
    importAll()
    case = getCase(caseName)
    L = float(case.defaults['L'])
    band = PinnedBand(slabs=((0, 0.0, 0.6),), velocity=(STREAM, 0.0), lo=(0.0, 0.0), hi=(L, L))
    original = case.initialConditions

    def initialConditions(ctx, system):
        original(ctx, system)
        system.state.velocities[:] = 0.0
        system.state.velocities[:, 0] = streamInit
        ctx.schemeConfig.boundaryConditions = list(ctx.schemeConfig.boundaryConditions or []) + [pinnedBandBC(band)]

    case.initialConditions = initialConditions
    try:
        spec = CaseSpec(caseName='pin', scheme=scheme, params={**case.params, **({'inviscid': False, 'nu': 0.05} if caseName == 'tgv-wc' else {'nu': 0.05})}).merged(**case.defaults).merged(
            nx=nx, nSteps=nSteps, plot=False, store=False, progress=False, video=False, show=False, quiet=True)
        return run(case, spec), band
    finally:
        case.initialConditions = original


def test_a_uniform_stream_with_a_band_is_stationary():
    r, band = pinnedRun(STREAM, 40)
    st = r.state.state
    fl = st.kinds == 0
    v = st.velocities[fl]
    assert float((v[:, 0] - STREAM).abs().max()) < 1e-6 and float(v[:, 1].abs().max()) < 1e-6
    assert float((st.densities[fl] - 1.0).abs().max()) < 1e-3
    assert int(fl.sum()) == 24 * 24


def test_the_band_holds_the_stream_and_drives_the_rest():
    r, band = pinnedRun(0.0, 120)
    st = r.state.state
    fl = st.kinds == 0
    from warpSPH.math import getPeriodicPositions
    inside = band.inside(getPeriodicPositions(st.positions, r.ctx.config.domain)) & fl
    assert int(inside.sum()) > 30
    assert float((st.velocities[inside, 0] - STREAM).abs().max()) < 1e-5            # the band IS the stream (start-of-step reset; the finalize corrections are shift-free there)
    assert bool(torch.isfinite(st.velocities).all()) and bool(torch.isfinite(st.densities).all())
    free = fl & ~inside
    assert float(st.velocities[free, 0].mean()) > 0.02                              # the viscosity carries the stream into the rest
    assert int(fl.sum()) == 24 * 24


def test_the_shift_weight_is_zero_inside_the_band_only():
    r, band = pinnedRun(STREAM, 1)
    st, ctx = r.state.state, r.ctx
    keep = pinnedKeepWeight(st, ctx.config, ctx.schemeConfig)
    from warpSPH.math import getPeriodicPositions
    inside = band.inside(getPeriodicPositions(st.positions, ctx.config.domain))
    assert keep.shape == (len(st.positions), 1)
    assert float(keep[inside].abs().max()) == 0.0 and float((keep[~inside] - 1.0).abs().max()) == 0.0
    ctx.schemeConfig.boundaryConditions = []
    assert pinnedKeepWeight(st, ctx.config, ctx.schemeConfig) is None


def test_the_band_holds_the_stream_in_the_incompressible_loop():
    """the same band on `divergenceFree` (projection + VD+PS shift): the end-of-step reset keeps the band exactly at the stream although the projection and the shift touch every fluid particle."""
    r, band = pinnedRun(0.0, 30, nx=24, scheme='divergenceFree', caseName='tgv')
    st = r.state.state
    fl = st.kinds == 0
    from warpSPH.math import getPeriodicPositions
    inside = band.inside(getPeriodicPositions(st.positions, r.ctx.config.domain)) & fl
    assert int(inside.sum()) > 30
    assert float((st.velocities[inside, 0] - STREAM).abs().max()) < 1e-6 and float(st.velocities[inside, 1].abs().max()) < 1e-6
    assert bool(torch.isfinite(st.velocities).all())
