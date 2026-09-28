"""Flag a run whose fastest particle is far above the case's expected velocity.

A runaway particle (a wall escaper, a kicked free-surface fragment) shows up as
one `|v|` orders of magnitude above the flow's own scale long before anything
goes non-finite -- and while it lasts it pins the global CFL `dt`, so simulated
time crawls (`FREESLIP_DAMBREAK_FINDINGS.md` §1: ~1e5 m/s, `dt` at 1e-8,
`diverged=False`). The runner only *stops* on non-finite values and on the
stall watchdogs; this *flags* the finite case and lets the run go on, so
whoever is watching can look at the dynamics and decide (`OPEN_PROBLEMS.md`
§10):

* a person at the terminal gets a line starting with :data:`ALARM_TAG`,
  repeated each time the ratio grows by another :data:`ESCALATION` and once
  more when it clears;
* a log monitor can grep for :data:`ALARM_TAG`;
* two declarative reactions, plain `CaseSpec` fields (so CLI flags and spec
  files carry them): `velocityAlarmPlotInterval` draws a frame every N steps
  while the alarm is active (needs `plot=True`; restored when it clears), and
  `velocityAlarmStopRatio` stops the run once the ratio passes that ceiling;
* hooks can read ``ctx.scratch['velocityAlarm']`` (the active event, or
  `None`), and ``run(..., onVelocityAlarm=fn)`` is called on every raise,
  escalation and clear -- returning `True` from it stops the run.

How to use it from a probe or a CLI: `README.md` "Watching a run"; probe
scripts get the flags from `scripts/_runWatch.py`.

The expected velocity is resolved once, before the first step (see
:func:`resolveVelocityScale`).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from warpSPHIntegrators import get_tagged_attr

__all__ = ['ALARM_TAG', 'ESCALATION', 'VelocityAlarm', 'maxSpeed', 'resolveVelocityScale']

#: Prefix of every alarm line, for anything that watches a run's log.
ALARM_TAG = '[warpSPH] VELOCITY ALARM'
#: A raised alarm is reported again each time the ratio grows by this factor.
ESCALATION = 10.0
#: A raised alarm clears once the ratio falls below `factor * CLEAR_FRACTION`
#: (hysteresis, so a flow hovering at the threshold does not flood the log).
CLEAR_FRACTION = 0.5


def maxSpeed(state) -> torch.Tensor:
    """`max |v|` over all particles, as a 0-d device tensor (no host sync).

    `max` propagates NaN and inf, so the one value also answers the runner's
    non-finite check."""
    velocities = get_tagged_attr(state.state, tag='velocity')
    return torch.linalg.vector_norm(velocities, dim=-1).max()


def _scalar(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().max().cpu().item()
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def resolveVelocityScale(ctx, state) -> Tuple[Optional[float], str]:
    """The velocity the run is expected to stay near, and where it came from.

    In order: `spec.velocityScale` (the user's), `ctx.velocityScale` (set by
    the case during setup, e.g. `dambreak`'s `U_max`), else the largest of the
    generic estimates available at t = 0 -- the largest, because a flag at
    100x must never fire on a healthy flow:

    * the initial `max |v|` (TGV, random flow, droplet, patch, ...);
    * compressible schemes: the initial maximum sound speed;
    * weakly compressible schemes: `c0 / 10` (the design rule `c0 >= 10 U_max`);
    * artificial compressibility: `acParams.uChar`.

    Incompressible schemes carry a `fixedSoundSpeed` that means nothing for
    them, so a flow started from rest there needs the case (or the user) to
    declare the scale. `(None, reason)` when nothing is available.
    """
    from ..enumTypes import (ArtificialCompressibleSPHScheme, CompressibleSPHScheme,
                             WeaklyCompressibleSPHScheme)
    spec = ctx.spec
    if spec.velocityScale is not None:
        return float(spec.velocityScale), 'spec.velocityScale'
    if ctx.velocityScale is not None:
        return float(ctx.velocityScale), ctx.velocityScaleSource or 'case'

    candidates = []
    vmax0 = _scalar(maxSpeed(state))
    if vmax0 is not None:
        candidates.append((vmax0, 'initial |v|max'))
    if isinstance(ctx.scheme, CompressibleSPHScheme):
        cs = _scalar(getattr(state.state, 'soundspeeds', None))
        if cs is not None:
            candidates.append((cs, 'initial max sound speed'))
    elif isinstance(ctx.scheme, WeaklyCompressibleSPHScheme):
        fluid = getattr(ctx.schemeConfig, 'fluid', None)
        c0 = _scalar(getattr(fluid, 'fixedSoundSpeed', None))
        if c0 is not None:
            candidates.append((c0 / 10.0, 'c0 / 10'))
    elif isinstance(ctx.scheme, ArtificialCompressibleSPHScheme):
        acParams = getattr(ctx.schemeConfig, 'acParams', None)
        uChar = _scalar(getattr(acParams, 'uChar', None))
        if uChar is not None:
            candidates.append((uChar, 'acParams.uChar'))
    if not candidates:
        return None, 'no velocity scale (set velocityScale)'
    return max(candidates, key=lambda c: c[0])


class VelocityAlarm:
    """Per-run alarm state; `check` is called once per step with that step's
    `max |v|` (already on the host -- the runner reads it for its non-finite
    check anyway, so the alarm costs no extra sync on a quiet step)."""

    def __init__(self, ctx, state, callback: Optional[Callable] = None,
                 write: Callable[[str], None] = print):
        # `None` or `0` (the CLI's way to say off) disables it
        self.factor = ctx.spec.velocityAlarmFactor or None
        self.callback = callback
        self.write = write
        self.events: List[Dict[str, Any]] = []
        self.level: Optional[float] = None  # ratio last reported; None = not raised
        self.peak: Optional[Dict[str, Any]] = None
        if self.factor is None:
            self.scale, self.source = None, 'disabled (velocityAlarmFactor unset)'
        else:
            self.scale, self.source = resolveVelocityScale(ctx, state)
        # A compressible IC whose sound speed is only filled in by the first
        # EOS evaluation (woodwardColella) gets one more try after step 1 --
        # from the sound speed only: velocities one step into a flow started
        # from rest are no scale at all.
        self._retry = (self.factor is not None and self.scale is None
                       and ctx.spec.velocityScale is None and ctx.velocityScale is None)
        self.stopRatio = ctx.spec.velocityAlarmStopRatio
        self.plotInterval = ctx.spec.velocityAlarmPlotInterval
        self._savedPlotInterval: Optional[int] = None
        #: Set when the alarm stopped the run (`RunResult.stopReason`).
        self.stopReason: Optional[str] = None
        ctx.scratch['velocityAlarm'] = None

    @property
    def enabled(self) -> bool:
        return self.factor is not None and self.scale is not None

    def describe(self) -> str:
        if not self.enabled:
            return f'off -- {self.source}'
        text = f'flag |v|max > {self.factor:g} x {self.scale:.4g} ({self.source})'
        if self.plotInterval:
            text += f'; frames every {self.plotInterval} step(s) while active'
        if self.stopRatio is not None:
            text += f'; stop above {self.stopRatio:g} x'
        return text + ('; otherwise the run continues' if self.stopRatio is not None
                       else '; the run continues')

    def _retryScale(self, ctx, state) -> None:
        self._retry = False
        from ..enumTypes import CompressibleSPHScheme
        if isinstance(ctx.scheme, CompressibleSPHScheme):
            cs = _scalar(getattr(state.state, 'soundspeeds', None))
            if cs is not None:
                self.scale, self.source = cs, 'max sound speed after step 1'
                self.write(f'[warpSPH] velocity alarm: {self.describe()}')

    def check(self, ctx, state, vmax: float, step: int, t: float) -> bool:
        """Raise / escalate / clear on this step's `vmax`. Returns `True` when
        the run should stop (`velocityAlarmStopRatio` passed, or the callback
        asked); `stopReason` then says which."""
        if self._retry:
            self._retryScale(ctx, state)
        if not self.enabled or not math.isfinite(vmax):
            return False
        ratio = vmax / self.scale
        if ratio > self.factor:
            if self.peak is None or ratio > self.peak['ratio']:
                self.peak = {'step': step, 't': t, 'vmax': vmax, 'ratio': ratio}
            if self.stopRatio is not None and ratio > self.stopRatio:
                # checked every step, not only at the 10x report points
                self.level = ratio
                self._emit('stop', ctx, state, vmax, ratio, step, t)
                self.stopReason = (f'velocityAlarmStopRatio: |v|max {ratio:.3g} x expected '
                                   f'> {self.stopRatio:g} x')
                return True
            if self.level is not None and ratio < self.level * ESCALATION:
                return False
            kind = 'raised' if self.level is None else 'escalated'
            self.level = ratio
            return self._emit(kind, ctx, state, vmax, ratio, step, t)
        if self.level is not None and ratio < self.factor * CLEAR_FRACTION:
            self.level = None
            return self._emit('cleared', ctx, state, vmax, ratio, step, t)
        return False

    def _emit(self, kind, ctx, state, vmax, ratio, step, t) -> bool:
        event = {'kind': kind, 'step': step, 't': t, 'vmax': vmax, 'ratio': ratio,
                 'scale': self.scale, 'dt': _scalar(ctx.config.dt)}
        where = ''
        if kind != 'cleared':
            # the rare path: a few extra syncs to say *which* particle
            speeds = torch.linalg.vector_norm(get_tagged_attr(state.state, tag='velocity'), dim=-1)
            index = int(torch.argmax(speeds))
            event['index'] = index
            for name, key in (('UIDs', 'uid'), ('kinds', 'particleKind'), ('positions', 'position')):
                values = getattr(state.state, name, None)
                if isinstance(values, torch.Tensor) and values.shape[0] == speeds.shape[0]:
                    value = values[index].detach().cpu()
                    event[key] = value.tolist() if value.ndim else value.item()
            where = f' at particle {index}'
            if 'uid' in event:
                where += f' (uid {event["uid"]})'
            if 'position' in event:
                where += ' x=[' + ', '.join(f'{c:.4g}' for c in event['position']) + ']'
        self.events.append(event)
        ctx.scratch['velocityAlarm'] = None if kind == 'cleared' else event
        self._reactPlotInterval(ctx, kind, step)

        dt = f', dt {event["dt"]:.3g}' if event['dt'] is not None else ''
        if kind == 'cleared':
            self.write(f'{ALARM_TAG} cleared at step {step} (t={t:.6g}): |v|max {vmax:.4g} '
                       f'= {ratio:.3g} x expected {self.scale:.4g}{dt}; peak was '
                       f'{self.peak["ratio"]:.3g} x at t={self.peak["t"]:.6g}')
        elif kind == 'stop':
            self.write(f'{ALARM_TAG} stop at step {step} (t={t:.6g}): |v|max {vmax:.4g} '
                       f'= {ratio:.3g} x expected {self.scale:.4g}{where}{dt} -- above '
                       f'velocityAlarmStopRatio = {self.stopRatio:g}; stopping.')
        else:
            self.write(f'{ALARM_TAG} {kind} at step {step} (t={t:.6g}): |v|max {vmax:.4g} '
                       f'= {ratio:.3g} x expected {self.scale:.4g} ({self.source}){where}'
                       f'{dt}. The run continues -- check the dynamics; if simulated '
                       f'time stops advancing, stop it (or set stallProgress).')
        if self.callback is not None and bool(self.callback(ctx, state, event)):
            if self.stopReason is None:
                self.stopReason = 'onVelocityAlarm'
            return True
        return False

    def _reactPlotInterval(self, ctx, kind: str, step: int) -> None:
        """`velocityAlarmPlotInterval`: closer frames while the alarm is active."""
        spec = ctx.spec
        if not self.plotInterval or not spec.plot:
            return
        if kind == 'raised' and self._savedPlotInterval is None:
            self._savedPlotInterval = spec.plotInterval
            spec.plotInterval = int(self.plotInterval)
            self.write(f'{ALARM_TAG}: drawing a frame every {spec.plotInterval} step(s) '
                       f'from step {step} until the alarm clears')
        elif kind == 'cleared':
            self.restore(ctx)

    def restore(self, ctx) -> None:
        """Undo the plot-interval change (on clear, and at the end of the run --
        `run(case, spec)` without overrides hands the caller's own spec in)."""
        if self._savedPlotInterval is not None:
            ctx.spec.plotInterval = self._savedPlotInterval
            self._savedPlotInterval = None

    def summary(self) -> Optional[str]:
        """One line for the post-run report, or `None` when it never fired."""
        if self.peak is None:
            return None
        raised = sum(1 for e in self.events if e['kind'] == 'raised')
        still = ' (still active at the end)' if self.level is not None else ''
        return (f'velocity alarm raised {raised} time(s){still}; peak |v|max '
                f'{self.peak["vmax"]:.4g} = {self.peak["ratio"]:.3g} x expected '
                f'{self.scale:.4g} at t={self.peak["t"]:.6g}')

