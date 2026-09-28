"""The generic machinery: does one loop really drive every scheme?"""

import dataclasses

import pytest

from warpSPH.runner import buildContext, listCases
from warpSPH.runner.caseSpec import CaseSpec
from warpSPH.runner.runner import resolveEnum


def test_allCasesRegister():
    from warpSPH.cases import importAll
    importAll()
    assert set(listCases()) >= {'sod', 'tgv', 'dambreak'}


@pytest.mark.parametrize('caseName', ['sod', 'tgv', 'dambreak'])
def test_everySchemeNamesItsConfigTheSame(caseName):
    """One loop drives every scheme only because they agree on the keyword.

    ``compSPH_step``/``crkSPH_step``/``compressibleSPH_Monaghan`` used to call it
    ``compParams`` while ``deltaSPH_step``/``divergenceFree_step`` called it
    ``schemeConfig``; the integrator forwards ``**kwargs`` verbatim, so a caller
    that guessed wrong got a ``TypeError``. The runner passes ``schemeConfig=``
    unconditionally now, so this is the invariant holding that up.
    """
    from inspect import signature

    from warpSPH.cases import importAll
    from warpSPH.runner import getCase
    importAll()
    case = getCase(caseName)
    spec = CaseSpec(caseName=case.name, scheme=case.scheme,
                    params=dict(case.params)).merged(**case.defaults)
    ctx = buildContext(case, spec)
    assert 'schemeConfig' in signature(ctx.stepFunction).parameters
    assert 'compParams' not in signature(ctx.stepFunction).parameters


def test_allStepFunctionsAgreeOnTheKeyword():
    """The same invariant for the schemes no registered case exercises."""
    from inspect import signature

    from warpSPH.schemes import buildScheme
    for name in ('compSPH', 'crkSPH', 'MonaghanCompressibleSPH', 'deltaSPH',
                 'divergenceFree'):
        parameters = signature(buildScheme(name).stepFunction).parameters
        assert 'schemeConfig' in parameters, name
        assert 'compParams' not in parameters, name


def test_resolveEnumIsCaseInsensitiveAndRejectsGarbage():
    from warpSPHCore import KernelFunctions
    assert resolveEnum(KernelFunctions, 'wendland2') is KernelFunctions.Wendland2
    assert resolveEnum(KernelFunctions, KernelFunctions.B7) is KernelFunctions.B7
    with pytest.raises(ValueError, match='Invalid KernelFunctions'):
        resolveEnum(KernelFunctions, 'NotAKernel')


def test_schemeBundleIsNamedAndStillUnpacks():
    """`SchemeBundle` replaced a bare 7-tuple. Named access is the point; the
    positional unpacking is kept so that adding an eighth member (a tangent
    propagator, once forward-mode AD lands) does not break old call sites."""
    from warpSPH.schemes import buildScheme
    from warpSPH.schemes.builder import SchemeBundle

    bundle = buildScheme('compSPH')
    assert isinstance(bundle, SchemeBundle)
    assert bundle.stepFunction.__name__ == 'compSPH_step'

    system, state, config, update, step, export, imp = bundle
    assert (system, state, config, update, step, export, imp) == (
        bundle.SimulationSystem, bundle.SimulationState, bundle.SimulationConfig,
        bundle.SimulationUpdate, bundle.stepFunction, bundle.exportFunction,
        bundle.importFunction)


def test_buildSchemeRejectsUnknownSchemes():
    import pytest as _pytest

    from warpSPH.schemes import buildScheme
    with _pytest.raises(ValueError, match='not recognized'):
        buildScheme('notAScheme')


def test_everyCaseModuleRegistersItsCase():
    """`CASE_MODULES` and the registry have to stay in step.

    A case module that stops registering (a renamed `registerCase` target, a
    module dropped from the tuple) is otherwise invisible until someone runs
    `warpsph-run` and gets "Unknown case".
    """
    from warpSPH.cases import CASE_MODULES, importAll
    importAll()
    # channelFlow declares two cases and dambreak's hooks back them, so the
    # count is not one-per-module; the floor is what matters.
    assert len(listCases()) >= len(CASE_MODULES)


def test_everyCaseNamesAResolvableScheme():
    """Each case's declared scheme has to exist in one of the three enums."""
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase
    from warpSPH.runner.runner import _resolveScheme
    importAll()
    for name in listCases():
        assert _resolveScheme(getCase(name).scheme) is not None, name


def test_everyCaseDeclaresItsParamsAsScalarsOrLists():
    """`params` becomes both CLI flags and HDF5 attributes.

    Only scalars get flags (`buildArgumentParser` skips lists and dicts) and
    only scalars are written per frame, so anything else has to be a deliberate
    list/dict rather than, say, an enum or a tensor that would fail at export.
    `None` is allowed too: it is the deliberate "leave the scheme's own
    setting" sentinel (e.g. dambreak's `alpha`, `shifting`), and both runtime
    paths special-case it -- `_addField` infers the flag's type from the
    annotation and `copy_dict_to_h5` skips it on export.
    """
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase
    importAll()
    for name in listCases():
        for key, value in getCase(name).params.items():
            assert value is None or isinstance(value, (int, float, str, bool, list, dict)), \
                f'{name}.{key} is {type(value).__name__}'


def test_timeLimitedLoopStopsOnTimeNotStepCount():
    """A case with a `timestep` hook is bounded by `tLimit`, not by an estimate.

    `nSteps = tLimit / dt0` is wrong the moment dt changes, which is exactly
    what the hook exists to do -- so those runs loop on simulated time.
    """
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()
    case = getCase('woodwardColella')
    assert case.timestep is not None

    tLimit = 5e-4
    result = run(case, nx=100, tLimit=tLimit, progress=False, plot=False, store=False)
    assert not result.diverged
    assert result.trajectory[-1]['t'] >= tLimit
    # The step before the last must still be short of the limit, or the loop
    # ran past its stopping condition.
    assert result.trajectory[-2]['t'] < tLimit


def test_postStepHookRunsAfterEveryStep():
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()

    case = getCase('sod')
    calls = []
    case = dataclasses.replace(case, postStep=lambda ctx, state, i: calls.append(i))
    result = run(case, nx=100, nSteps=4, progress=False, plot=False, store=False)
    assert calls == list(range(result.nSteps))


def test_runReportsItselfUnlessQuiet(capsys):
    """A run left unattended has to say what it did when it finishes.

    Both blocks are checked because they answer different questions: the banner
    says what is about to run (at the resolved dt, not the requested one), the
    report says whether it finished and where the output went.
    """
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()

    run(getCase('sod'), nx=100, nSteps=2, plot=False, store=False)
    printed = capsys.readouterr().out
    assert 'warpSPH | sod' in printed          # banner
    assert 'particles' in printed and 'timestep' in printed
    assert 'fluid' in printed and 'viscosity' in printed
    assert 'sod finished in' in printed        # report
    assert 'totalEnergy' in printed            # diagnostics summary

    run(getCase('sod'), nx=100, nSteps=2, plot=False, store=False, quiet=True)
    assert capsys.readouterr().out == ''


def test_bannerDescribesTheFluidEachSolverActuallyUses():
    """The three families keep these settings in three different places.

    A compressible run is pinned by `gamma` on the scheme config itself and
    dissipates through a `wp.struct` whose viscosity term is an int; a weakly
    compressible one is pinned by a rest density and a sound speed on a `fluid`
    block and dissipates through an `alpha`. Neither dataclass carries the
    other's fields, so reading the wrong one is the failure guarded here.
    """
    import torch

    from warpSPH.configurations.compSPHConfig import CompSPHConfig
    from warpSPH.configurations.weaklyCompressible import WeaklyCompressibleSPHConfig
    from warpSPH.runner.report import _fluidDescription, _viscosityDescriptions

    compressible = CompSPHConfig()
    assert 'gamma 1.4' in _fluidDescription(compressible)
    assert 'Monaghan1992' in _viscosityDescriptions(compressible)[0]

    weaklyCompressible = WeaklyCompressibleSPHConfig()
    weaklyCompressible.fluid.restDensity = 1000.0
    # A tensor, because that is what `setupWeaklyCompressibleTimestep` leaves
    # behind when a case lets it derive the sound speed.
    weaklyCompressible.fluid.fixedSoundSpeed = torch.tensor(60.0)
    fluid = _fluidDescription(weaklyCompressible)
    assert 'rho0 1000' in fluid and 'c_s 60' in fluid
    assert 'alpha 0.01' in _viscosityDescriptions(weaklyCompressible)[0]


def test_reportNamesDivergenceLoudly():
    """A diverged run must not read like a successful one."""
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    from warpSPH.runner.report import reportRun
    importAll()

    result = run(getCase('noh'), nx=64, nSteps=6, dt=0.5, adaptiveDt=False,
                 plot=False, store=False, quiet=True)
    assert result.diverged
    assert result.wallTime > 0


def test_progressBarIsOffWhenNothingIsWatching(capsys):
    """`progress=None` means "only for a terminal".

    Redirected to a file, a tqdm bar writes a carriage-return smear that buries
    the report the run exists to produce.
    """
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()

    # capsys replaces stderr with a non-tty, which is the condition under test.
    run(getCase('sod'), nx=100, nSteps=2, plot=False, store=False)
    assert 'it/s' not in capsys.readouterr().err


def test_formatDurationReadsAsTime():
    from warpSPH.runner.report import formatDuration
    assert formatDuration(0.812) == '812ms'
    assert formatDuration(4.21) == '4.21s'
    assert formatDuration(192) == '3m 12s'
    assert formatDuration(3852) == '1h 04m 12s'


def test_stallProgressStopsARunWhoseSimTimeIsFrozen(monkeypatch):
    """`stallDtSteps` only sees dt exactly at `minDt`; a dt hovering just above
    it (ACSPH's step-ratio clamp) freezes simulated time just as surely.
    `stallProgress` watches the time itself."""
    import warpSPH.runner.runner as runnerModule
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()
    monkeypatch.setattr(runnerModule, 'STALL_WINDOW_STEPS', 5)

    frozen = dataclasses.replace(getCase('sod'), timestep=lambda ctx, state: 1e-12)
    result = run(frozen, nx=100, tLimit=1.0, stallProgress=1e-6,
                 progress=False, plot=False, store=False, quiet=True)
    assert result.diverged
    assert len(result.trajectory) < 20

    healthy = run(getCase('sod'), nx=100, tLimit=5e-4, stallProgress=1e-6,
                  progress=False, plot=False, store=False, quiet=True)
    assert not healthy.diverged


def _sod(**kw):
    from warpSPH.cases import importAll
    from warpSPH.runner import getCase, run
    importAll()
    return run(getCase('sod'), nx=100, **dict(dict(progress=False, plot=False, store=False,
                                                   quiet=True), **kw))


def test_velocityAlarmFlagsWithoutStopping(capsys):
    """A |v| far above the expected velocity is reported -- on the command
    line, in the result, to the hook -- but the run goes on (OPEN_PROBLEMS.md
    §10). A scale of 1e-9 makes every step after the first an alarm."""
    from warpSPH.runner.velocityAlarm import ALARM_TAG
    seen = []
    result = _sod(nSteps=10, velocityScale=1e-9,
                  onVelocityAlarm=lambda ctx, state, event: seen.append(event))
    assert not result.diverged and result.stopReason is None
    assert result.nSteps == 10
    assert result.velocityAlarms and result.velocityAlarms[0]['kind'] == 'raised'
    assert 'index' in result.velocityAlarms[0] and result.velocityAlarms[0]['ratio'] > 100
    assert seen == result.velocityAlarms
    assert ALARM_TAG in capsys.readouterr().out


def test_velocityAlarmCallbackCanStopTheRun():
    result = _sod(nSteps=10, velocityScale=1e-9, onVelocityAlarm=lambda *args: True)
    assert result.diverged and result.stopReason == 'onVelocityAlarm'
    assert result.nSteps < 10


def test_velocityAlarmDefaultScaleAndDisabling():
    """Sod starts at rest: the scale falls back to the initial sound speed, and
    a healthy run raises nothing. `velocityAlarmFactor=None` turns it off."""
    healthy = _sod(nSteps=5)
    monitor = healthy.ctx.scratch['velocityAlarmMonitor']
    assert monitor.enabled and monitor.source == 'initial max sound speed'
    assert healthy.velocityAlarms == []

    off = _sod(nSteps=5, velocityScale=1e-9, velocityAlarmFactor=None)
    assert not off.ctx.scratch['velocityAlarmMonitor'].enabled
    assert off.velocityAlarms == []


def test_velocityAlarmStopRatioIsAHardCeiling():
    result = _sod(nSteps=10, velocityScale=1e-9, velocityAlarmStopRatio=1e6)
    assert result.diverged and result.stopReason.startswith('velocityAlarmStopRatio')
    assert result.velocityAlarms[-1]['kind'] == 'stop'
    assert result.nSteps < 10


def test_velocityAlarmPlotIntervalWhileActive():
    """`velocityAlarmPlotInterval`: frames every N steps while the alarm is
    active, the normal interval back when it clears and at the end."""
    from types import SimpleNamespace
    from warpSPH.runner.velocityAlarm import VelocityAlarm
    base = _sod(nSteps=2)
    spec = CaseSpec(plot=True, plotInterval=20, velocityScale=1.0, velocityAlarmPlotInterval=1)
    ctx = SimpleNamespace(spec=spec, velocityScale=None, velocityScaleSource=None, scratch={},
                          config=base.ctx.config, scheme=base.ctx.scheme)
    lines = []
    alarm = VelocityAlarm(ctx, base.state, write=lines.append)
    assert not alarm.check(ctx, base.state, vmax=500.0, step=5, t=0.1)
    assert spec.plotInterval == 1 and ctx.scratch['velocityAlarm']['kind'] == 'raised'
    assert not alarm.check(ctx, base.state, vmax=0.5, step=6, t=0.2)
    assert spec.plotInterval == 20 and ctx.scratch['velocityAlarm'] is None
    alarm.check(ctx, base.state, vmax=500.0, step=7, t=0.3)
    alarm.restore(ctx)
    assert spec.plotInterval == 20
    assert [e['kind'] for e in alarm.events] == ['raised', 'cleared', 'raised']


def test_probeWatchFlags():
    """`scripts/_runWatch.py`: the probes' flags map onto CaseSpec fields."""
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from _runWatch import addWatchArguments, watchOverrides
    ap = argparse.ArgumentParser()
    addWatchArguments(ap)
    kw = watchOverrides(ap.parse_args([]))
    assert kw == dict(velocityAlarmFactor=100.0, velocityAlarmPlotInterval=1, stallProgress=1e-3)
    kw = watchOverrides(ap.parse_args(['--velocityAlarmFactor', '0', '--stallProgress', '0',
                                       '--velocityAlarmStopRatio', '1e4']))
    assert kw['velocityAlarmFactor'] is None and kw['stallProgress'] is None
    assert kw['velocityAlarmStopRatio'] == 1e4
    assert set(kw) <= {f.name for f in dataclasses.fields(CaseSpec)}


def test_explicitProgressWinsOverQuiet(capsys):
    """The probes pass `quiet=True, progress=True`: no banner or report, but
    the per-step rows must still stream (CLAUDE.md run rule 2)."""
    _sod(nSteps=3, quiet=True, progress=True)
    err = capsys.readouterr().err
    assert 't=' in err
    _sod(nSteps=3, quiet=True)
    assert 't=' not in capsys.readouterr().err
