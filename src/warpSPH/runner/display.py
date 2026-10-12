"""Live plot windows, and their teardown.

Separate from :mod:`warpSPH.cases.common.plotting` for one reason: the runner has to
tear a figure down at the end of a run, and the runner must not import from
`warpSPH.cases` -- it is generic over cases, not built on them. The case-facing
names are re-exported from `warpSPH.cases.common.plotting`, so a case still says
``from .plotting import openWindow``.

Why any of this is needed: a notebook frontend displays a figure for you, so
the notebooks these cases came from never had to. A console script does, and
without :func:`openWindow` it builds the figure, writes its PNGs, and shows
nothing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # avoids a cycle: `case` imports nothing from here at runtime
    from .case import RunContext

__all__ = ['openWindow', 'pumpEvents', 'holdWindow', 'closeWindow', 'figureOf',
           'resolvePlotBackend', 'visualizeWithFallback']

#: Matplotlib backends that render to a file rather than to a window. Anything
#: else is assumed to have a GUI event loop worth pumping.
_HEADLESS_BACKENDS = frozenset({'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'})

_warned = False


def _runningUnderIPython() -> bool:
    """True inside a live IPython/Jupyter kernel (classic Jupyter, JupyterLab,
    or VS Code's notebook extension -- all three run an IPython kernel).

    `False` for a plain `python script.py` process, which is the only case
    that actually needs `openWindow`'s "pop a window" dance below: a notebook
    frontend already displays a newly-created figure on its own (`%matplotlib
    widget`/ipympl included -- that's the whole point of the magic), so
    calling `Figure.show()`/`plt.ion()` again on top of that is not just
    redundant, it visibly duplicates the display and appears to freeze,
    because whichever copy someone is looking at may not be the one
    `pumpEvents` keeps redrawing during the loop.
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def figureOf(handle) -> Optional[Any]:
    """The matplotlib `Figure` behind a plot handle, or `None`.

    Handles come in three shapes: a `warpSPHPlotting` plotter (`.fig`), the
    ``(fig, axis)`` tuple :func:`profilePlot` uses, or a bare figure. The vispy
    and pyVista backends own their window and have no matplotlib figure at all,
    which is what `None` means here.
    """
    if isinstance(handle, tuple):
        handle = handle[0]
    figure = getattr(handle, 'fig', handle)
    return figure if hasattr(figure, 'canvas') else None


def openWindow(ctx: 'RunContext', handle) -> None:
    """Show the figure live, if the run asked for it and a display exists.

    Called once from a case's `setupPlot`. Safe to call headless: it reports
    once and leaves the run writing frames as before. A no-op inside a
    Jupyter/IPython kernel (see `_runningUnderIPython`) -- the notebook
    frontend already displays a freshly-created figure on its own, so this
    would only duplicate it and confuse which copy `pumpEvents` is actually
    redrawing.
    """
    global _warned
    if not getattr(ctx.spec, 'show', True):
        return
    if _runningUnderIPython():
        return

    figure = figureOf(handle)
    if figure is None:
        # vispy / pyVista opened their own window; `show` is theirs to honour.
        if hasattr(handle, 'show'):
            handle.show()
        return

    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower().removeprefix('module://')
    if backend in _HEADLESS_BACKENDS:
        if not _warned:
            _warned = True
            print(f'matplotlib is using the non-interactive {backend!r} backend, so no '
                  f'window will open; frames are still written. Set MPLBACKEND (e.g. '
                  f'TkAgg, QtAgg) for a live plot, or pass --no-show to silence this.')
        return

    plt.ion()
    figure.show()
    pumpEvents(handle)


def holdWindow(ctx: 'RunContext', handle) -> None:
    """Block on the finished figure so the last frame stays up to be read.

    Only for a human at a console -- `caseMain` turns it on, programmatic
    `run()` leaves it off so a sweep or a test never stalls waiting for a
    window to be closed.
    """
    if not getattr(ctx.spec, 'holdPlot', False):
        return

    figure = figureOf(handle)
    if figure is None:
        # vispy / pyVista: block on their own app loop. The canvas is reached
        # through the backend instance because the library exposes no public
        # "run until closed"; if that shape ever changes, losing the hold is
        # harmless -- the run is already finished.
        app = getattr(getattr(getattr(handle, '_backend_instance', None),
                              '_canvas', None), 'app', None)
        if app is not None:
            print('Run finished. Close the plot window to exit (--no-holdPlot to skip).')
            try:
                app.run()
            except Exception:
                pass
        return

    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower().removeprefix('module://') in _HEADLESS_BACKENDS:
        return
    print('Run finished. Close the plot window to exit (--no-holdPlot to skip).')
    plt.ioff()
    try:
        plt.show()
    except Exception:
        pass


def closeWindow(handle) -> None:
    """Release the window, whichever backend owns it.

    Every run must give its window back, and *both* backends leak without
    help. matplotlib figures are registered with pyplot's global manager by
    `subplots`/`subplot_mosaic`; a vispy `SceneCanvas` holds a live GL window.
    Either way a process that runs several cases -- a sweep, the test suite, a
    notebook -- accumulates one window per case until it exits. A single
    example script never noticed, because exiting closed them all.
    """
    figure = figureOf(handle)
    if figure is not None:
        try:
            import matplotlib.pyplot as plt
            plt.close(figure)
        except Exception:
            pass
        return

    # vispy / pyVista: no public teardown, so close the canvas the backend
    # instance holds. Guarded because losing the close is not worth an
    # exception at the very end of a completed run.
    canvas = getattr(getattr(handle, '_backend_instance', None), '_canvas', None)
    for target in (canvas, getattr(handle, '_backend_instance', None), handle):
        close = getattr(target, 'close', None)
        if callable(close):
            try:
                close()
                return
            except Exception:
                pass


#: ipympl (`%matplotlib widget`) and its nbAgg ancestor redraw over a Jupyter
#: Comm, not a native GUI event loop -- `draw_idle`'s "redraw whenever the
#: toolkit is next idle" only actually fires once the toolkit's own loop gets
#: a turn, which a tight synchronous Python loop never yields to. A forced,
#: immediate `draw()` is what these backends' own examples use for exactly
#: this "update every iteration of a loop" shape, and what this repo's own
#: pre-`Case` notebooks always called here.
_FORCE_DRAW_BACKENDS_SUBSTR = ('ipympl', 'nbagg', 'webagg')


def pumpEvents(handle) -> None:
    """Repaint, and let the GUI toolkit process its events.

    `draw_idle` alone only marks the canvas dirty; without `flush_events` the
    window never actually repaints during a tight step loop, and the toolkit
    treats it as unresponsive. ipympl/nbAgg-style backends need a forced
    `draw()` instead -- see `_FORCE_DRAW_BACKENDS_SUBSTR`.
    """
    figure = figureOf(handle)
    if figure is None:
        if hasattr(handle, 'show'):
            handle.show()
        return
    try:
        import matplotlib
        backend = matplotlib.get_backend().lower().removeprefix('module://')
        if any(name in backend for name in _FORCE_DRAW_BACKENDS_SUBSTR):
            figure.canvas.draw()
        else:
            figure.canvas.draw_idle()
        figure.canvas.flush_events()
    except Exception:
        # A closed window, or a backend without an event loop -- neither is a
        # reason to take the simulation down.
        pass


# -- backend selection -------------------------------------------------------

def resolvePlotBackend(ctx: 'RunContext') -> str:
    """Which `warpSPHPlotting` backend this run should draw with.

    An explicit `--plotBackend` wins. Otherwise it is chosen by dimension:
    **2D and up go to vispy**, because a matplotlib scatter redraw of a
    hundred thousand particles takes longer than the integrator step it is
    illustrating -- with a plot every few steps, matplotlib becomes the
    simulation's bottleneck rather than an observer of it. 1D stays on
    matplotlib, where the point count is small and the axes are the content.
    """
    explicit = getattr(ctx.spec, 'plotBackend', None) or ctx.param('plotBackend')
    if explicit:
        return explicit
    return 'vispy' if ctx.spec.dim >= 2 else 'matplotlib'


def visualizeWithFallback(ctx: 'RunContext', backend: str, **kwargs):
    """`visualize(...)`, degrading to matplotlib if `backend` cannot start.

    vispy needs a GL context; over ssh without X forwarding, or in a container,
    creating the canvas raises. On headless Linux (no `DISPLAY`/
    `WAYLAND_DISPLAY`), `warpSPHPlotting`'s vispy backend already routes
    itself to a headless EGL context instead of a windowed one *before* that
    first attempt -- vispy's backend choice is a one-shot, process-global
    decision, so retrying with a different `app_backend` here after a failed
    call would not work (`RuntimeError: Can only select a backend once`).
    What still reaches this `except` is a case EGL itself can't cover (no GPU,
    EGL not installed, a non-Linux headless host, ...); falling back to
    matplotlib there keeps the run alive and still writing frames -- losing
    the plot is not a reason to lose the simulation.
    """
    global _warned
    from warpSPHPlotting import visualize

    try:
        plotter = visualize(backend=backend, **kwargs)
    except Exception as exc:
        if backend == 'matplotlib':
            raise
        if not _warned:
            _warned = True
            print(f'the {backend!r} plotting backend failed to start '
                  f'({type(exc).__name__}: {exc}); falling back to matplotlib. '
                  f'Pass --plotBackend matplotlib to select it directly.')
        backend = 'matplotlib'
        plotter = visualize(backend=backend, **kwargs)

    # What the banner and the report quote -- the backend that actually
    # started, which after a fallback is not the one that was asked for.
    ctx.scratch['plotBackend'] = backend
    return plotter
