"""The Antuono switch's surface mask across the stages of a step (OPEN_PROBLEMS.md section 1, step 1): measured (`surfaceMaskDiagnostics`) and
optionally frozen (`freezeSurfaceMaskAcrossStages`). Eager only (it keeps Python-side per-step state)."""

__all__ = ['surfaceMaskAcrossStages']


def _isFirstStage(t, dt, schemeConfig, stageIndex) -> bool:
    """Whether this RHS call is the first stage of a real step. From
    `stageIndex` where the integrator passes one (RungeKuttaB); otherwise
    (symplectic Euler: k0 at t^n, k1 at t^n + dt/2, both called with the
    stage `dt/2` and no index) from the time: a later stage sits exactly one
    stage-`dt` after the step's first one."""
    if stageIndex is not None:
        return stageIndex == 0
    t = float(t)
    t0 = getattr(schemeConfig, '_surfaceMaskStepT', None)
    if t0 is not None and abs(t - (t0 + float(dt))) <= 1e-9 * max(1.0, abs(t)):
        return False
    return True


def surfaceMaskAcrossStages(currentState, dt, schemeConfig, stageIndex, t) -> None:
    """OPEN_PROBLEMS.md §1 step 1: the Antuono switch's surface mask,
    measured (`surfaceMaskDiagnostics`) and optionally frozen across the
    stages of a step (`freezeSurfaceMaskAcrossStages`). Eager only."""
    freeze = getattr(schemeConfig, 'freezeSurfaceMaskAcrossStages', False)
    diag = getattr(schemeConfig, 'surfaceMaskDiagnostics', False)
    if not (freeze or diag):
        return
    mask = currentState.surfaceIndicators
    if _isFirstStage(t, dt, schemeConfig, stageIndex):
        stats = getattr(schemeConfig, '_surfaceMaskStats', None)
        if diag:
            prev = getattr(schemeConfig, '_surfaceMaskStepStart', None)
            stats = {'maskSurface': int(mask.sum()), 'maskFlipsIntraStep': 0,
                     'maskFlipsStepToStep': (int((prev != mask).sum())
                                             if prev is not None and prev.shape == mask.shape else -1)}
            schemeConfig._surfaceMaskStats = stats
        schemeConfig._surfaceMaskStepStart = mask.clone()
        schemeConfig._surfaceMaskStepT = float(t)
        return
    start = getattr(schemeConfig, '_surfaceMaskStepStart', None)
    if start is None or start.shape != mask.shape:
        return
    if diag:
        # rows whose classification this stage differs from the step's first
        stats = schemeConfig._surfaceMaskStats
        stats['maskFlipsIntraStep'] = max(stats['maskFlipsIntraStep'], int((start != mask).sum()))
    if freeze:
        currentState.surfaceIndicators = start.clone()
