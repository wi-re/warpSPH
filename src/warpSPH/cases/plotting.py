"""Plot hooks the converted example cases share.

Every 2D example notebook built the same `warpSPHPlotting.visualize` call --
one or two particle fields, a mosaic, `export` to `frame_NNNNN.png` -- differing
only in which fields and which colour maps. :func:`particlePlot` is that call as
data, so a case declares the fields and gets a matching `setupPlot` /
`updatePlot` pair back. :func:`buildFieldPlotter`/:func:`refreshFieldPlotter`
are the window/event-loop-free core of that pair -- what a notebook calls
directly instead of the `Case` hooks, the same way `profilePlot`'s `draw` is
its `setupPlot` minus `openWindow`/`pumpEvents`.

The 1D compressible examples instead scattered a few state fields against `x`;
:func:`profilePlot` is that, with an optional analytic overlay.

The plotting backend is chosen by dimension -- **vispy for 2D**, matplotlib for
1D -- because a matplotlib scatter of a large 2D particle set costs more per
frame than the step it is drawing. ``--plotBackend`` overrides, and a vispy
canvas that cannot start (no GL context over ssh, in a container) falls back to
matplotlib rather than taking the run down with it.

Plots are *live* by default when run from a console: :func:`openWindow` puts
matplotlib into interactive mode and shows the figure, and :func:`pumpEvents`
gives the GUI toolkit a chance to repaint after every redraw. Without those two
calls a script builds the figure, writes its PNGs and never opens a window --
which is what a notebook gets away with, because the notebook frontend displays
the figure for it. ``--no-show`` turns the window off and keeps the frames.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess as _sp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..runner import RunContext

__all__ = ['Field', 'particlePlot', 'buildFieldPlotter', 'refreshFieldPlotter',
           'profilePlot', 'ProfileAxis', 'figureTitle',
           'openWindow', 'pumpEvents', 'holdWindow', 'closeWindow', 'figureOf',
           'resolvePlotBackend', 'visualizeWithFallback']


# -- live display -----------------------------------------------------------
# Implemented in `warpSPH.runner.display` so the runner can tear a figure down
# without importing from `warpSPH.cases`; re-exported here because that is
# where a case looks for them.

from ..runner.display import (closeWindow, figureOf,  # noqa: E402
                              holdWindow, openWindow, pumpEvents,
                              resolvePlotBackend, visualizeWithFallback)


@dataclass
class Field:
    """One panel of a particle plot: what to colour by, and how."""

    #: Attribute of the particle state, e.g. ``'densities'``.
    quantity: str
    title: str
    colorMap: str = 'viridis'
    #: Which colour-map family `colorMap` names.
    colorMapKind: str = 'uniform'          # 'uniform' | 'diverging' | 'cyclic'
    flip: bool = False
    scaling: str = 'Linear'                # 'Linear' | 'Logarithmic'
    mapping: str = 'none'                  # 'none' | 'L2Norm' | 'x' | 'y'
    vMin: Optional[float] = None
    vMax: Optional[float] = None
    midPoint: Optional[float] = 0.0
    #: Resolution of the interpolated-grid rendering; `None` scatters particles.
    gridResolution: Optional[int] = None
    #: Title-to-plot spacing: matplotlib title padding in points, vispy a
    #: fraction of domain height. `None` keeps the backend default.
    plotTitleGap: Optional[float] = None
    #: How to draw the boundary particles: 'Passive' (grey crosses, excluded
    #: from the colour normalisation), 'Visualize' (coloured by the quantity
    #: like the fluid) or 'Hide'.
    #:
    #: `PlottingOptions` defaults to `Hide`, which leaves a walled case with no
    #: indication of where its walls are -- the lid-driven cavity plots as a
    #: square of fluid with nothing around it. `Passive` is the default here
    #: instead: a case with no boundary particles is unaffected, and one with
    #: them shows where they are without letting them set the colour range.
    boundary: str = 'Passive'                # 'Passive' | 'Visualize' | 'Hide'
    #: The same, for the fluid particles. Only worth changing to look at a
    #: boundary in isolation.
    fluid: str = 'Visualize'

    def tensor(self, state) -> torch.Tensor:
        value = getattr(state.state, self.quantity)
        # Integer state fields (`UIDs`, `kinds`) are plotted as floats: the
        # grid path interpolates the quantity with a warp kernel typed off the
        # particle positions, and an int32 input does not match that kernel.
        if not value.is_floating_point():
            value = value.to(state.state.positions.dtype)
        return value


def _plotOptions(field: Field, markerSize: float):
    from warpSPHPlotting import (CyclicColorMap, DivergingColorMap, GridVisualization,
                                 Mapping, PlotScaling, PlottingOptions, UniformColorMap,
                                 VisualizeOptions)

    families = {'uniform': UniformColorMap, 'diverging': DivergingColorMap,
                'cyclic': CyclicColorMap}
    family = families[field.colorMapKind]
    return PlottingOptions(
        colorMap=getattr(family, field.colorMap),
        flipColorMap=field.flip,
        markerSize=markerSize,
        midPoint=field.midPoint,
        quantityScaling=getattr(PlotScaling, field.scaling),
        mapping=getattr(Mapping, field.mapping),
        plotTitle=field.title,
        plotTitleGap=field.plotTitleGap,
        vMin=field.vMin,
        vMax=field.vMax,
        fluidVisualization=getattr(VisualizeOptions, field.fluid),
        boundaryVisualization=getattr(VisualizeOptions, field.boundary),
        gridVisualization=(GridVisualization(resolution=field.gridResolution)
                           if field.gridResolution else None),
    )


#: Wall-clock time this process started, stamped into every figure title so a
#: frame on disk can be traced back to the run that produced it.
_LAUNCH_TIME = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')


def _gitHash() -> str:
    """`git describe`-style short hash of the working tree, or `'nogit'`.

    Resolved once per process and cached -- the title is rebuilt every frame and
    a `subprocess` per frame would dominate the render. A dirty tree gets a
    `-dirty` suffix, because a frame produced from uncommitted edits is not
    reproducible from the hash alone and the title should say so.
    """
    global _GIT_HASH
    if _GIT_HASH is not None:
        return _GIT_HASH
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    try:
        head = _sp.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                       capture_output=True, text=True, timeout=5)
        if head.returncode != 0:
            _GIT_HASH = 'nogit'
            return _GIT_HASH
        h = head.stdout.strip()
        dirty = _sp.run(['git', '-C', repo, 'status', '--porcelain'],
                        capture_output=True, text=True, timeout=5)
        if dirty.returncode == 0 and dirty.stdout.strip():
            h += '-dirty'
        _GIT_HASH = h
    except Exception:
        _GIT_HASH = 'nogit'
    return _GIT_HASH


_GIT_HASH: Optional[str] = None


def _boundaryConditions(ctx: RunContext) -> str:
    """The wall treatment, as `BC <kinds>` (+ the mDBC no-penetration mode).

    Two cases differing only in wall BC produce visually similar fields and
    wildly different physics -- `DELTASPH_VALIDATION_PLAN.md` 5.7 -- so the
    frame should say which one it is.
    """
    sc = getattr(ctx, 'schemeConfig', None)
    bits = []
    try:
        from ..configurations.region import RegionType
        kinds = {r.kind.name for r in (getattr(sc, 'regions', None) or [])
                 if getattr(r, 'type', None) == RegionType.Boundary
                 and getattr(r, 'kind', None) is not None}
        if kinds:
            bits.append('/'.join(sorted(kinds)))
    except Exception:
        pass
    mode = getattr(sc, 'mdbcNoPenShiftMode', None)
    if mode:
        bits.append(f'noPen:{mode}')
    return ' | '.join(bits)


def _gravityString(ctx: RunContext) -> str:
    """`g = (gx, gy)` -- direction times magnitude, the vector actually applied.

    Recomputed per frame rather than cached with the rest of the provenance:
    `sloshingTank` rotates gravity every step to solve in the tank frame, so a
    cached value would be wrong for all but the first frame.
    """
    gc = getattr(getattr(ctx, 'schemeConfig', None), 'gravityConfig', None)
    if gc is None or not getattr(gc, 'active', False):
        return ''
    try:
        mag = float(gc.magnitude)
        d = gc.direction
        d = [float(v) for v in (d.tolist() if hasattr(d, 'tolist') else d)]
        return 'g(' + ','.join(f'{mag * v:.3g}' for v in d) + ')'
    except Exception:
        return ''


def _provenance(ctx: RunContext) -> str:
    """The static half of the title: what was run, with what, when, from which
    commit. Cached per `RunContext` -- none of it changes during a run.

    Gravity is deliberately NOT here (it can rotate per step); see
    `_gravityString`.
    """
    cached = ctx.scratch.get('_titleProvenance') if hasattr(ctx, 'scratch') else None
    if cached is not None:
        return cached
    spec = ctx.spec
    kernel = getattr(ctx.config, 'kernel', None)
    kernelName = getattr(kernel, 'name', None) or str(spec.kernel)
    # Bare values, no labels: the scheme / integrator / kernel names are each
    # unambiguous on their own, and spelling out "scheme ... integrator ...
    # kernel ..." pushed the line past the figure width, where the backends
    # clip rather than wrap.
    bits = [str(spec.scheme), str(spec.integrationScheme), kernelName]
    # The *absolute* sound speed, whatever route set it: cases disagree on
    # whether the knob is `soundSpeed` (absolute) or `c0Ratio` (a multiple of
    # sqrt(gH)), so the echoed input alone does not tell you what ran.
    cs = getattr(getattr(ctx, 'schemeConfig', None), 'fluid', None)
    cs = getattr(cs, 'fixedSoundSpeed', None) if cs is not None else None
    if cs is not None:
        try:
            csv = float(cs.detach().cpu().item() if hasattr(cs, 'detach') else cs)
            if csv > 0:
                bits.append(f'c_s {csv:.4g}')
        except Exception:
            pass
    bc = _boundaryConditions(ctx)
    if bc:
        bits.append(bc)
    bits += [_LAUNCH_TIME, _gitHash()]
    text = ' | '.join(bits)
    if hasattr(ctx, 'scratch'):
        ctx.scratch['_titleProvenance'] = text
    return text


def figureTitle(ctx: RunContext, state, row: Optional[Dict[str, float]] = None) -> str:
    """Two-line banner: the live `t / dt / ptcls` state over a provenance line.

    The second line (scheme, time integrator, kernel, launch time, git hash)
    exists so a frame or a video answers "what produced this?" on its own. Runs
    of the same case differing only in integrator or scheme were otherwise
    indistinguishable once the PNG left its directory -- which is exactly the
    comparison this repo's validation work spends its time on.
    """
    parts = [f'{ctx.case.name}  t = {float(state.t):.4g}',
             f'dt = {float(ctx.config.dt):.3g}',
             f'ptcls = {len(state.state.positions)}']
    if row:
        parts += [f'{k} = {v:.4g}' for k, v in row.items()]
    tail = _provenance(ctx)
    g = _gravityString(ctx)
    if g:
        tail = tail + ' | ' + g
    return ' | '.join(parts) + '\n' + tail


def _mosaicKeys(fields: Sequence[Field]) -> List[str]:
    return [chr(ord('A') + i) for i in range(len(fields))]


def buildFieldPlotter(ctx: RunContext, state, fields: Sequence[Field],
                      figsize: Tuple[float, float] = (11, 5), dpi: int = 300,
                      exportFrame0: bool = True):
    """Build the `fields` plotter and export its frame 0 -- no window calls.

    This is `particlePlot`'s `setupPlot` minus `openWindow`; a notebook calls
    it directly instead of a case's `setupPlot` hook, the same reason
    `profilePlot` exports its `draw`.

    `exportFrame0=False` skips the export here -- for a caller that is about
    to call `openWindow` on the returned plotter before exporting frame 0
    itself (`particlePlot`, `dambreak.setupPlot`). The vispy canvas has not
    settled to its real physical size until `openWindow` runs (the first
    `show()`/layout pass resizes it), so exporting here unconditionally
    captured frame 0 at a *different* resolution than every later frame
    (measured: 2688x768 vs the run's steady 1440x768) -- a real, if cosmetic,
    per-run glitch: the exported video's first frame renders squished/at the
    wrong aspect relative to the rest.
    """
    keys = _mosaicKeys(fields)
    markerSize = ctx.param('markerSize', 2)
    plotter = visualizeWithFallback(
        ctx, resolvePlotBackend(ctx),
        particleState=state.state,
        domain=ctx.config.domain,
        quantities={k: f.tensor(state) for k, f in zip(keys, fields)},
        plotOptions={k: _plotOptions(f, markerSize) for k, f in zip(keys, fields)},
        figTitle=figureTitle(ctx, state),
        mosaic=''.join(keys),
        figsize=figsize,
        backendOptions=getattr(ctx.spec, 'plotBackendOptions', None),
    )
    if exportFrame0:
        _export(ctx, plotter, 0, dpi)
    return plotter


def refreshFieldPlotter(ctx: RunContext, state, plotter, fields: Sequence[Field],
                        step: int = 0, dpi: int = 300) -> None:
    """Update an existing `fields` plotter in place -- no event-pump calls."""
    keys = _mosaicKeys(fields)
    plotter.updateQuantities(
        {k: f.tensor(state) for k, f in zip(keys, fields)},
        newParticleState=state.state,
    )
    # The notebooks never refreshed the title, so every frame after the
    # first showed t = 0 -- which makes the encoded video misleading about
    # what it is showing.
    plotter.updateTitle(figureTitle(ctx, state))
    _export(ctx, plotter, step, dpi)


def particlePlot(fields: Sequence[Field], figsize: Tuple[float, float] = (11, 5),
                 dpi: int = 300) -> Tuple[Callable, Callable]:
    """`(setupPlot, updatePlot)` rendering `fields` side by side.

    Panels are keyed 'A', 'B', ... in order, which is exactly the mosaic string
    the notebooks passed.
    """

    def setupPlot(ctx: RunContext, state):
        plotter = buildFieldPlotter(ctx, state, fields, figsize, dpi, exportFrame0=False)
        openWindow(ctx, plotter)
        _export(ctx, plotter, 0, dpi)
        return plotter

    def updatePlot(ctx: RunContext, state, plotter, step: int) -> None:
        refreshFieldPlotter(ctx, state, plotter, fields, step, dpi)
        pumpEvents(plotter)

    return setupPlot, updatePlot


def _export(ctx: RunContext, plotter, step: int, dpi: int) -> None:
    if ctx.imagePath:
        plotter.export(os.path.join(ctx.imagePath, f'frame_{step:07d}.png'), dpi=dpi)


@dataclass
class ProfileAxis:
    """One panel of a 1D profile plot."""

    #: Attribute of the particle state to plot against ``x``.
    quantity: str
    title: str
    #: Component to take when the quantity is a vector (velocities).
    component: Optional[int] = None
    yscale: str = 'linear'
    ylim: Optional[Tuple[float, float]] = None
    #: Horizontal reference lines, as ``(ctx, state) -> [y, ...]``.
    hlines: Optional[Callable[[RunContext, Any], List[float]]] = None
    #: Vertical reference lines, same shape.
    vlines: Optional[Callable[[RunContext, Any], List[float]]] = None
    #: Analytic overlay, as ``(ctx, state) -> (x, y)`` or `None`.
    reference: Optional[Callable[[RunContext, Any], Optional[Tuple[Any, Any]]]] = None


def profilePlot(axes: Sequence[ProfileAxis], shape: Tuple[int, int],
                figsize: Tuple[float, float] = (10, 6), dpi: int = 150,
                xlim: Optional[Tuple[float, float]] = None) -> Tuple[Callable, Callable, Callable]:
    """`(setupPlot, updatePlot, draw)` scattering 1D state fields against position.

    `draw(ctx, state, (fig, axis))` is the self-contained per-frame redraw
    (clears and re-populates every axis, no `openWindow`/`pumpEvents`) --
    exported alongside `setupPlot`/`updatePlot` so a notebook can call it
    directly for live updates, the same way `sod.py` calls `plotSod`/`plotSod_`
    directly instead of going through the `Case` hooks.

    At `dim==1` this scatters every particle against its raw (signed) `x`, and
    a vector quantity is read at `spec.component`, matching Sod/Noh/Kidder/
    Woodward-Colella, all of which only ever run at `dim==1`. A radially
    symmetric problem (Sedov) that also runs at `dim>1` instead wants every
    particle against its distance from the origin `r = |x|`, unsigned, with a
    vector quantity read as its magnitude rather than one component -- the
    same "collapse, don't average" idea `PORTING_EXAMPLES.md` describes for
    Sod's `x`, just with the invariant coordinate a radially symmetric problem
    actually has. That branch only triggers when `ctx.spec.dim > 1`, so it is
    inert for every existing `dim==1` caller.
    """
    rows, cols = shape

    def draw(ctx: RunContext, state, handle):
        import matplotlib.pyplot as plt  # noqa: F401  (backend already chosen)

        fig, axis = handle
        flat = axis.flatten()
        radial = ctx.spec.dim > 1
        positionsFull = state.state.positions.detach().cpu().numpy()
        positions = np.linalg.norm(positionsFull, axis=-1) if radial else positionsFull[:, 0]
        for ax, spec in zip(flat, axes):
            ax.clear()
            values = getattr(state.state, spec.quantity).detach().cpu().numpy()
            if values.ndim > 1:
                values = (np.linalg.norm(values, axis=-1) if radial
                          else values[:, spec.component] if spec.component is not None
                          else values)
            values = values.reshape(len(positions), -1)[:, 0]
            ax.scatter(positions, values, s=1)
            if spec.reference is not None:
                reference = spec.reference(ctx, state)
                if reference is not None:
                    ax.plot(reference[0], reference[1], color='black', ls=':',
                            label='analytic')
            for y in (spec.hlines(ctx, state) if spec.hlines else []):
                ax.axhline(y, color='red', ls='--', alpha=0.4)
            for x in (spec.vlines(ctx, state) if spec.vlines else []):
                ax.axvline(x, color='black', ls=':', alpha=0.4)
            ax.set_title(spec.title)
            ax.set_yscale(spec.yscale)
            if spec.ylim:
                ax.set_ylim(*spec.ylim)
            if xlim:
                ax.set_xlim(*xlim)
        fig.suptitle(figureTitle(ctx, state))
        fig.tight_layout()

    def setupPlot(ctx: RunContext, state):
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        ctx.scratch['plotBackend'] = 'matplotlib'   # profile plots are axes-based
        handle = (fig, axis)
        draw(ctx, state, handle)
        _save(ctx, fig, 0, dpi)
        openWindow(ctx, handle)
        return handle

    def updatePlot(ctx: RunContext, state, handle, step: int) -> None:
        fig, _ = handle
        draw(ctx, state, handle)
        _save(ctx, fig, step, dpi)
        pumpEvents(handle)

    return setupPlot, updatePlot, draw


def _save(ctx: RunContext, fig, step: int, dpi: int) -> None:
    if ctx.imagePath:
        fig.savefig(os.path.join(ctx.imagePath, f'frame_{step:07d}.png'), dpi=dpi)
