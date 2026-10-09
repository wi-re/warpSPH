"""What a run tells the console about itself.

Two blocks, and they exist for different reasons. The **banner** is printed
once setup is finished and `dt` is known, so what it shows is what will
actually run rather than what was asked for -- resolution, the resolved
timestep, the derived step count, the fluid model and dissipation the scheme
ended up configured with, and where output is going. The **report** is
printed at the end, for the case that motivates all of this: a long run left
unattended. Coming back to a finished terminal should answer "did it finish,
did it stay sane, and where did the output go" without re-reading the scrollback
or opening the HDF5.

Both are suppressed by ``--quiet``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch

__all__ = ['describeRun', 'reportRun', 'formatDuration', 'quietedWarp']

_RULE = '-' * 78


def formatDuration(seconds: float) -> str:
    """`1h 04m 12s` / `3m 12s` / `4.21s` / `812ms`."""
    if seconds < 1:
        return f'{seconds * 1000:.0f}ms'
    if seconds < 60:
        return f'{seconds:.2f}s'
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f'{minutes}m {seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m {seconds:02d}s'


def _row(label: str, value: str) -> str:
    return f'  {label:<11} {value}'


def _deviceDescription(device) -> str:
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            return f'{device} ({torch.cuda.get_device_name(device)})'
        except Exception:
            pass
    return str(device)


def _domainDescription(domain) -> str:
    low = ', '.join(f'{v:g}' for v in domain.min.detach().cpu().tolist())
    high = ', '.join(f'{v:g}' for v in domain.max.detach().cpu().tolist())
    periodic = getattr(domain, 'periodic', None)
    if periodic is not None:
        flags = periodic.tolist() if hasattr(periodic, 'tolist') else list(periodic)
        kind = 'periodic' if all(flags) else ('open' if not any(flags) else str(flags))
    else:
        kind = ''
    return f'[{low}] to [{high}]' + (f'  {kind}' if kind else '')


def _enumName(value) -> str:
    return getattr(value, 'name', str(value))


def _scalarValue(value) -> Optional[float]:
    """Config numbers arrive as python floats, warp scalars, or 0-d tensors.

    `fluid.fixedSoundSpeed` in particular is a tensor whenever a case let
    `setupWeaklyCompressibleTimestep` pick it, and `None` is a legal value for
    every `fluidProperties` field.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        return float(flat[0]) if flat.numel() else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number(label: str, value) -> Optional[str]:
    """`label 1.5`, or nothing at all when the value is unset."""
    number = _scalarValue(value)
    return None if number is None else f'{label} {number:g}'


def _joined(*parts, separator: str = ' | ') -> str:
    return separator.join(part for part in parts if part)


def _viscosityTermName(value) -> str:
    """`DiffusionParameters` is a `wp.struct`, so its terms are stored as ints."""
    from ..configurations.moduleConfigurations.diffusionParameters import ViscosityTerms
    if isinstance(value, ViscosityTerms):
        return value.name
    try:
        return ViscosityTerms(int(value)).name
    except (TypeError, ValueError):
        return str(value)


def _fluidDescription(schemeConfig) -> Optional[str]:
    """The pressure model the scheme will actually close the system with.

    Weakly compressible and incompressible schemes carry a `fluid` block: a
    rest density and a fixed sound speed, which between them set the pressure
    scale and the acoustic timestep. Compressible schemes have neither -- their
    pressure comes from the internal energy through `gamma`.

    Only the parameters the selected EOS reads are printed. `fluidProperties`
    defaults every field, so showing `kappa` beside an isothermal EOS would
    suggest it played a part when `isoThermalEOS` never looks at it. DFSPH
    (the config with a `solverConfig`) gets its pressure from the solver and
    never calls the EOS at all, so naming one there would be a lie -- it still
    reads `restDensity` and `fixedSoundSpeed`, via shifting and the boundaries.
    """
    fluid = getattr(schemeConfig, 'fluid', None)
    if fluid is not None:
        eos = _enumName(fluid.eosType)
        incompressible = hasattr(schemeConfig, 'solverConfig')
        # rho0 and c_s stay on the line whatever the EOS is: the timestep,
        # the dissipation terms and the shifting all read them directly.
        parts = ['incompressible (pressure from solver)' if incompressible
                 else f'EOS {eos}',
                 _number('rho0', fluid.restDensity),
                 _number('c_s', fluid.fixedSoundSpeed)]
        if not incompressible:
            if eos in ('stiffTait', 'Polytropic', 'Murnaghan'):
                parts.append(_number('exponent', fluid.polytropicExponent))
            if eos in ('Tait', 'Polytropic', 'Murnaghan'):
                parts.append(_number('kappa', fluid.kappa))
        return _joined(*parts)

    if hasattr(schemeConfig, 'gamma'):
        background = _scalarValue(getattr(schemeConfig, 'backgroundPressure', None))
        return _joined('ideal gas',
                       _number('gamma', schemeConfig.gamma),
                       _number('rho0', getattr(schemeConfig, 'rho0', None)),
                       # Zero is the default and means "off"; it is only worth
                       # a column when a case has actually dialled it in.
                       _number('p_background', background) if background else None)
    return None


def _viscosityDescriptions(schemeConfig) -> List[str]:
    """Dissipation settings, as one row plus continuations.

    The two families configure this in entirely different places, so there is
    no common line to print: delta-SPH picks between an artificial `alpha` and
    a physical `nu` and adds density diffusion on top, while the compressible
    schemes carry a Monaghan-style viscosity formulation with its own
    coefficients, an artificial conductivity, and optionally a switch that
    rewrites `alpha` per particle as the run goes.
    """
    params = getattr(schemeConfig, 'diffusionParams', None)
    if params is None:
        return []

    if hasattr(params, 'inviscid'):  # WeaklyCompressibleDiffusionParams
        if params.inviscid:
            viscosity = _joined('artificial', _number('alpha', params.inviscidAlpha))
        else:
            viscosity = _joined('physical', _number('nu', params.viscidNu))
        # Density diffusion belongs to the same block but only delta-SPH runs
        # it -- DFSPH (the config with a `solverConfig`) leaves the call out.
        if not hasattr(schemeConfig, 'solverConfig'):
            viscosity = _joined(viscosity,
                                _joined(f'density diffusion '
                                        f'{_enumName(params.densityDiffusionTerm)}',
                                        _number('delta', params.densityDelta),
                                        separator=' '))
        return [viscosity]

    switch = getattr(schemeConfig, 'viscositySwitchParams', None)
    switchName = _enumName(switch.scheme) if switch is not None else 'NoneSwitch'
    rows = [_joined(_viscosityTermName(params.viscosityTerm),
                    _joined(_number('C_l', params.C_l),
                            _number('C_q', params.C_q)
                            + (' (beta fixed)' if getattr(params, 'betaMode', 0) == 1 else ''),
                            separator=', '),
                    _number('K', params.K),
                    f'switch {"none" if switchName == "NoneSwitch" else switchName}')]
    if switch is not None and switchName != 'NoneSwitch':
        # The switch overrides C_l per particle, so its bounds are what the
        # run is really viscous between.
        rows.append(_joined(f'alpha in [{_scalarValue(switch.alpha_min):g}, '
                            f'{_scalarValue(switch.alpha_max):g}]',
                            f'divergence {switch.divergenceScheme}'))
    policy = getattr(params, 'velocityPairPolicy', 0)
    if policy != 0:
        # AV_PLAN Phase 3: the viscosity sees a reconstructed pair velocity
        from ..configurations.moduleConfigurations.diffusionParameters import VelocityPairPolicy
        rows.append(f'pair velocity {VelocityPairPolicy(policy).name} (midpoint reconstruction'
                    + ('' if getattr(params, 'correctReconstructionGradient', True) else ', uncorrected gradient')
                    + ')')
    rows.append(_joined(f'conductivity '
                        f'{_viscosityTermName(params.thermalConductivityTerm)}',
                        _joined(_number('Cu_l', params.Cu_l),
                                _number('Cu_q', params.Cu_q), separator=', '),
                        _number('scaling', params.thermalConductivity)))
    return rows


def describeRun(ctx, state, nSteps: int, timeLimited: bool) -> None:
    """The pre-run banner. Called once `dt` and the step count are final."""
    spec = ctx.spec
    config = ctx.config
    particles = len(state.state.positions)

    print(_RULE)
    print(f'  warpSPH | {ctx.case.name}')
    if ctx.case.description:
        print(f'  {ctx.case.description}')
    print(_RULE)
    # The step function, not just the enum: a comparison run across schemes is
    # judged on which solver actually ran, and the enum name alone does not say.
    solver = getattr(ctx.stepFunction, '__name__', str(ctx.stepFunction))
    print(_row('scheme', f'{_enumName(ctx.scheme)} '
                         f'({type(ctx.scheme).__name__}) | solver {solver}'))
    print(_row('device', f'{_deviceDescription(ctx.device)} | {spec.precision}'))
    print(_row('particles', f'{particles:,} | dim {spec.dim} | nx {spec.nx}'))
    print(_row('domain', _domainDescription(config.domain)))
    print(_row('kernel', f'{_enumName(config.kernel)} | n_h {spec.n_h:g} | '
                         f'targetNeighbors {float(config.targetNeighbors):.1f}'))
    print(_row('integrator', f'{_enumName(config.integrationScheme)} | '
                             f'support {_enumName(config.supportMode)}'))

    # After `configureScheme` and the case's own setup have run, so these are
    # the values the solver will use rather than the scheme defaults -- a case
    # that derives its sound speed from the expected velocity shows the
    # derived one.
    fluid = _fluidDescription(ctx.schemeConfig)
    if fluid:
        print(_row('fluid', fluid))
    for index, viscosity in enumerate(_viscosityDescriptions(ctx.schemeConfig)):
        print(_row('viscosity' if index == 0 else '', viscosity))

    dt = float(config.dt)
    adaptive = (f'adaptive, cfl {config.cflFactor:g}' if config.adaptiveDt else 'fixed')
    print(_row('timestep', f'dt {dt:.4g} | {adaptive}'))
    if timeLimited:
        print(_row('duration', f'until t = {spec.tLimit:g} '
                               f'(dt is recomputed each step)'))
    else:
        print(_row('duration', f'{nSteps:,} steps to t = '
                               f'{spec.tLimit if spec.nSteps is None else nSteps * dt:g}'))

    alarm = ctx.scratch.get('velocityAlarmMonitor')
    if alarm is not None:
        print(_row('watch', f'velocity alarm: {alarm.describe()}'))

    _plannedOutput(ctx)
    print(_RULE, flush=True)


def _plannedOutput(ctx) -> None:
    """What the run is *about* to write. Nothing exists yet at banner time."""
    spec = ctx.spec
    if ctx.exportPath is None:
        print(_row('output', 'nothing written (pass --store and/or --plot)'))
        return

    what = []
    if spec.store:
        what.append('trajectory.h5' if spec.storeMode == 'trajectory'
                    else f'states every {spec.storeInterval} steps')
    if spec.plot:
        what.append(f'frames every {spec.plotInterval} steps '
                    f'({ctx.scratch.get("plotBackend", "?")})')
    if spec.video:
        what.append('video')
    print(_row('output', ctx.exportPath))
    if what:
        print(_row('', ' | '.join(what)))


def _countFiles(directory: Optional[str], suffix: str) -> int:
    if not directory or not os.path.isdir(directory):
        return 0
    return sum(1 for name in os.listdir(directory) if name.endswith(suffix))


def _writtenOutput(ctx, result) -> None:
    """What the run actually wrote.

    Counted from disk rather than from the configured intervals: a run that
    stopped early, or whose store interval never came round, would otherwise
    be reported as having written files it does not have.
    """
    if ctx.exportPath is None:
        print(_row('output', 'nothing written (pass --store and/or --plot)'))
        return

    what = []
    states = _countFiles(os.path.join(ctx.exportPath, 'trajectory'), '.h5')
    if states:
        what.append(f'{states} state file{"s" if states != 1 else ""}')
    if os.path.exists(os.path.join(ctx.exportPath, 'trajectory.h5')):
        what.append('trajectory.h5')
    frames = _countFiles(ctx.imagePath, '.png')
    if frames:
        what.append(f'{frames} frame{"s" if frames != 1 else ""} '
                    f'({ctx.scratch.get("plotBackend", "?")})')

    print(_row('output', ctx.exportPath))
    print(_row('', ' | '.join(what) if what else 'no files written'))
    if result.videoPath:
        print(_row('', f'video -> {result.videoPath}'))


def _series(trajectory: List[Dict[str, float]], key: str) -> List[float]:
    return [row[key] for row in trajectory if key in row and isinstance(row[key], (int, float))]


def reportRun(result, wallTime: float) -> None:
    """The post-run report: did it finish, did it stay sane, where did it go."""
    ctx = result.ctx
    trajectory = result.trajectory

    status = 'DIVERGED' if result.diverged else 'finished'
    print()
    print(_RULE)
    print(f'  {ctx.case.name} {status} in {formatDuration(wallTime)}')
    print(_RULE)

    if result.diverged:
        reason = getattr(result, 'stopReason', None) or 'non-finite velocities'
        print(_row('warning', f'stopped early ({reason}); the results below are '
                              f'not usable.'))
    alarm = ctx.scratch.get('velocityAlarmMonitor')
    summary = alarm.summary() if alarm is not None else None
    if summary:
        print(_row('warning', summary))

    finalT = trajectory[-1]['t'] if trajectory else 0.0
    print(_row('steps', f'{result.nSteps:,} | t = {finalT:.6g} | '
                        f'final dt {float(ctx.config.dt):.4g}'))

    stepTimes = _series(trajectory, 'stepTime_ms')[1:]  # row 0 is the initial state
    if stepTimes:
        total = sum(stepTimes) / 1000.0
        print(_row('step time', f'mean {sum(stepTimes) / len(stepTimes):.1f} ms | '
                                f'min {min(stepTimes):.1f} | max {max(stepTimes):.1f} | '
                                f'{formatDuration(total)} in the loop '
                                f'({total / wallTime * 100:.0f}% of wall)'))

    _reportDiagnostics(trajectory)
    _writtenOutput(ctx, result)
    print(_RULE, flush=True)


def _reportDiagnostics(trajectory: List[Dict[str, float]]) -> None:
    """Initial, final and range for every diagnostic the case recorded.

    Initial-vs-final is what tells an absent user whether the run did what it
    was supposed to; the range catches an excursion that came back before the
    end and would otherwise be invisible in the final value alone.
    """
    if not trajectory:
        return
    keys = [k for k in trajectory[-1]
            if k not in ('step', 't', 'stepTime_ms')
            and isinstance(trajectory[-1][k], (int, float))]
    if not keys:
        return

    width = max(len(k) for k in keys)
    print(f'  {"diagnostics":<11} {"":<{width}}  {"initial":>12} {"final":>12} '
          f'{"min":>12} {"max":>12}')
    for key in keys:
        values = _series(trajectory, key)
        if not values:
            continue
        print(f'  {"":<11} {key:<{width}}  {values[0]:>12.5g} {values[-1]:>12.5g} '
              f'{min(values):>12.5g} {max(values):>12.5g}')


@contextmanager
def quietedWarp(quiet: bool):
    """Silence warp's per-module load chatter for the duration of a quiet run.

    Warp logs a line per kernel module it loads, which is the bulk of what
    lands in an unattended run's log file. `--quiet` is meant to produce a log
    worth reading, so it covers that too. The level is restored on the way out
    -- this is a third-party global, and a library caller may well have set it
    deliberately.
    """
    if not quiet:
        yield
        return

    try:
        import warp as wp
        previous = wp.config.log_level
        wp.config.log_level = wp.LOG_WARNING
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            wp.config.log_level = previous
        except Exception:
            pass
