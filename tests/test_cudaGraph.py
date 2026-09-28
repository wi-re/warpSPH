"""CUDA-graph replay of the delta-SPH right-hand side (`utils/cudaGraph.py`)
and the sync-free mDBC helpers it relies on (`utils/syncFree.py`)."""

import pytest
import torch

from warpSPH.utils.syncFree import deviceConstant, ghostSourceIndex, ghostTargetIndex, scatterRows

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


def _ghostLayout(device):
    # 3 fluid, 3 boundary (-> ghosts 6..8), 3 ghosts (-> boundaries 3..5)
    kinds = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], device=device, dtype=torch.int32)
    ghostIndices = torch.tensor([-1, -1, -1, 8, 6, 7, 4, 5, 3], device=device, dtype=torch.int64)
    return kinds, ghostIndices


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=cuda)])
def test_syncFreeHelpersMatchMaskedIndexing(device):
    kinds, gi = _ghostLayout(device)
    ghost = kinds == 2
    values = torch.arange(9 * 2, device=device, dtype=torch.float32).view(9, 2)

    # gather: values[ghostIndices[ghost]] on the ghost rows
    gathered = values[ghostSourceIndex(kinds, gi)]
    assert torch.equal(gathered[ghost], values[gi[ghost]])

    # scatter: out[ghostIndices[ghost]] = rowValues[ghost]
    base = -torch.ones(9, 2, device=device)
    rowValues = values * 10.0
    expected = base.clone()
    expected[gi[ghost]] = rowValues[ghost]
    assert torch.equal(scatterRows(base, ghostTargetIndex(kinds, gi), rowValues), expected)

    # no ghost rows at all: a no-op
    noGhost = torch.zeros_like(kinds)
    assert torch.equal(scatterRows(base, ghostTargetIndex(noGhost, gi), rowValues), base)


def test_deviceConstantIsCached():
    a = deviceConstant([0.0, -1.0], torch.float32, 'cpu')
    assert a is deviceConstant([0.0, -1.0], torch.float32, 'cpu')
    assert a.tolist() == [0.0, -1.0]


def _marrone(nx, nSteps, cudaGraph, **kw):
    from warpSPH.cases.dambreak import dambreakCase
    from warpSPH.runner import run
    H, L = 0.6, 1.0
    params = dict(W=5.366 * H, fillRatio=H / L, fluidWidth=2.0 * H / (5.366 * H),
                  gravityMagnitude=9.81, pressureProbeHeights=[0.16, 0.584],
                  pressureProbeDiscRadius=0.045, referenceVelocity=1.95 * (9.81 * H) ** 0.5,
                  machTarget=1.95 / 40.0)
    kw = dict(dict(quiet=True, store=False, progress=False, plot=False), **kw)
    return run(dambreakCase, scheme='sun2017DeltaSPH', L=L, nx=nx, nSteps=nSteps,
               params=params, cudaGraph=cudaGraph, **kw)


@cuda
def test_graphedStepIsBitwiseEager():
    """Marrone 3.1 (the probe's configuration, coarse): graph replay must
    reproduce the eager run bit for bit -- state and every diagnostic -- and
    must actually have replayed (not silently fallen back to eager)."""
    steps = 40
    eager = _marrone(24, steps, cudaGraph=False)
    graph = _marrone(24, steps, cudaGraph=True)

    # Whole steps are replayed from one graph (runner-level
    # GraphedIntegratorStep); steps it hands back to eager (the first, warm-up
    # step and any Verlet-rebuild step) replay their RHS from the per-RHS graph.
    sg = graph.ctx.scratch['stepGraph']
    assert sg.disabled is None, sg.disabled
    assert sg.captures >= 1
    assert sg.replays - sg.fallbacks >= steps // 2
    rhs = getattr(graph.ctx.schemeConfig, '_rhsGraph', None)
    assert rhs is None or rhs.disabled is None, rhs.disabled

    for name in ('positions', 'velocities', 'densities', 'pressures'):
        a = getattr(eager.state.state, name)
        b = getattr(graph.state.state, name)
        assert torch.equal(a, b), name
    for ra, rb in zip(eager.trajectory, graph.trajectory):
        for k, v in ra.items():
            if k != 'stepTime_ms' and isinstance(v, float) and v == v:
                assert rb[k] == v, (k, ra['step'])


@cuda
def test_pipelinedOutputsMatchSequentialLoop():
    """`CaseSpec.pipelineOutputs` (step n's diagnostics on a side stream while
    step n+1 replays) must record the same rows, stop at the same step (here a
    time-limited run, whose last step is diagnosed before the loop exits) and
    end in the same state as the one-thing-at-a-time loop."""
    seq = _marrone(24, None, cudaGraph=True, pipelineOutputs=False, tLimit=0.02)
    pipe = _marrone(24, None, cudaGraph=True, pipelineOutputs=True, tLimit=0.02)
    assert pipe.ctx.scratch['stepGraph'].replays > 0
    assert len(seq.trajectory) == len(pipe.trajectory) > 2
    assert seq.trajectory[-1]['t'] >= 0.02
    for name in ('positions', 'velocities', 'densities', 'pressures'):
        assert torch.equal(getattr(seq.state.state, name), getattr(pipe.state.state, name)), name
    for ra, rb in zip(seq.trajectory, pipe.trajectory):
        assert ra.keys() == rb.keys()
        for k, v in ra.items():
            if k != 'stepTime_ms' and isinstance(v, float) and v == v:
                assert rb[k] == v, (k, ra['step'])


@cuda
def test_sortedQuantilesMatchTorch():
    """`utils/syncFree.py:sortedQuantiles` (one sort, several quantiles, no
    sync) must be bitwise `torch.quantile` / `torch.nanquantile`, including
    NaN and inf inputs and a positively rescaled sorted array."""
    from warpSPH.utils.syncFree import sortedQuantiles
    gen = torch.Generator(device='cuda').manual_seed(3)
    qs = (0.01, 0.05, 0.5, 0.95, 0.99)
    for trial in range(60):
        n = 1 + trial * 37
        x = torch.randn(n, device='cuda', generator=gen) * 3
        if trial % 3 == 1:
            x[::7] = float('nan')
        if trial % 3 == 2:
            x[::5] = float('inf')
        for ignoreNan, ref in ((False, torch.quantile), (True, torch.nanquantile)):
            got = sortedQuantiles(x, qs, ignoreNan=ignoreNan)
            for q, g in zip(qs, got):
                r = ref(x, q)
                assert torch.equal(r, g) or (r.isnan() and g.isnan()), (trial, q, ignoreNan)
        xn = torch.where(torch.isfinite(x), x.abs(), torch.full_like(x, float('nan')))
        scale = torch.nanquantile(xn, 0.5).clamp_min(1e-30)
        got = sortedQuantiles(None, (0.01, 0.5), ignoreNan=True, sortedX=torch.sort(xn)[0] / scale)
        for q, g in zip((0.01, 0.5), got):
            r = torch.nanquantile(xn / scale, q)
            assert torch.equal(r, g) or (r.isnan() and g.isnan()), (trial, q)


@cuda
def test_probeVerletHashMatchesFreshHash():
    """The dam-break pressure probes' fast path (the Verlet list's hash map,
    `inv_ex` instead of `pinv`; graph-capturable) must give the fresh-hash
    fit up to float rounding -- at query points inside the fluid, where the
    fit is non-trivial (the wall probes read 0 until the wave arrives)."""
    from warpSPH.cases import dambreak as DB
    r = _marrone(24, 30, cudaGraph=False)
    ctx, st = r.ctx, r.state
    setup = DB._probeSetup(ctx, st)
    hashMap = DB._probeHashMap(ctx, st)
    assert setup is not None and hashMap is not None
    fluid = st.state.kinds == 0
    pos = st.state.positions[fluid]
    pts = pos[torch.randperm(pos.shape[0], device=pos.device)[:64]] + 0.1 * ctx.config.dx
    fresh = DB._mlsPressureDevice(setup, st.state, pts)
    fast = DB._mlsPressureDevice(setup, st.state, pts, hashMap)
    assert torch.equal(fresh[1], fast[1]) and torch.equal(fresh[2], fast[2])  # counts, conditioning
    scale = fresh[0].abs().max().clamp_min(1e-30)
    assert float((fresh[0] - fast[0]).abs().max() / scale) < 1e-4
    assert float((fresh[3] - fast[3]).abs().max() / fresh[3].abs().max().clamp_min(1e-30)) < 1e-5


def test_compileGlueIsPassThroughWhenOff():
    from warpSPHCore import compileGlue, compileGlueEnabled, setCompileGlue
    setCompileGlue(False)
    try:
        calls = []

        @compileGlue
        def f(x):
            calls.append(x)
            return x + 1

        assert not compileGlueEnabled()
        assert f(1) == 2 and calls == [1] and f.eager(2) == 3
    finally:
        setCompileGlue(None)


@cuda
def test_renderThreadFramesMatchMainThread(tmp_path):
    """`CaseSpec.asyncPlot` (frames on a worker thread through EGL) must
    write the same frames as plotting on the loop's thread."""
    pytest.importorskip('vispy')
    import glob
    import os
    import numpy as np
    imageio = pytest.importorskip('imageio.v3')
    frames = {}
    for asyncPlot in (False, True):
        root = tmp_path / f'async{int(asyncPlot)}'
        _marrone(24, 12, cudaGraph=True, plot=True, show=False, plotBackend='vispy',
                 plotInterval=5, exportRoot=str(root), asyncPlot=asyncPlot,
                 plotBackendOptions={'app_backend': 'egl'})
        frames[asyncPlot] = sorted(glob.glob(os.path.join(str(root), '*', 'images', 'frame_*.png')))
    assert len(frames[False]) == len(frames[True]) >= 3
    for a, b in zip(frames[False], frames[True]):
        x, y = imageio.imread(a).astype(int), imageio.imread(b).astype(int)
        diff = np.abs(x - y).max(-1)
        rows = np.nonzero(diff.any(1))[0]
        # the title carries the run's start time (minute resolution)
        assert rows.size == 0 or rows.max() < 60, (a, b)


@cuda
def test_velocityAlarmOnThePipelinedLoop():
    """The graphed, pipelined loop reads the same one `max |v|` for its
    non-finite check and the velocity alarm: flag without stopping, stop when
    the hook asks. The dam break declares its own scale (`referenceVelocity`)."""
    seen = []
    flagged = _marrone(24, 8, cudaGraph=True, velocityScale=1e-9,
                       onVelocityAlarm=lambda ctx, state, event: seen.append(event))
    assert flagged.ctx.scratch.get('stepGraph') is not None
    assert not flagged.diverged and flagged.nSteps == 8
    assert flagged.velocityAlarms and flagged.velocityAlarms[0]['kind'] == 'raised'
    assert seen == flagged.velocityAlarms

    stopped = _marrone(24, 8, cudaGraph=True, velocityScale=1e-9,
                       onVelocityAlarm=lambda *args: True)
    assert stopped.stopReason == 'onVelocityAlarm' and stopped.nSteps < 8

    healthy = _marrone(24, 8, cudaGraph=True)
    assert healthy.ctx.scratch['velocityAlarmMonitor'].source == 'referenceVelocity'
    assert healthy.velocityAlarms == []
