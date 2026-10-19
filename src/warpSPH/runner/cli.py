"""The `if __name__ == '__main__'` block shared by every case module."""

from __future__ import annotations

import sys
from typing import List, Optional

from .case import Case
from .caseSpec import buildArgumentParser, specFromArgs, _readMapping, CaseSpec
from .configuration import (RunConfiguration, configurationSchema, parameterReference,
                            resolvedConfiguration, parameterApplies, _write)
from .runner import prepareContext
from dataclasses import fields
from .runner import RunResult, run

__all__ = ['caseMain']


def caseMain(case: Case, argv: Optional[List[str]] = None) -> Optional[RunResult]:
    """Parse `argv` against `case`'s knobs and run it.

    Note that ``--precision`` can only take effect when the process was
    bootstrapped before ``warpSPH`` was imported -- see :mod:`warpSPHBootstrap`.
    Running a case module directly (``python -m warpSPH.cases.compressible.sod``) imports
    ``warpSPH`` first, so use ``python -m warpSPHRun sod`` or set
    ``warpSPHCore_PRECISION`` when a non-default precision is wanted.
    """
    defaults = dict(case.defaults)
    defaults.setdefault('caseName', case.name)
    defaults.setdefault('scheme', case.scheme)
    # A person ran this from a terminal, so leave the final figure up to be
    # read. `--no-holdPlot` overrides, and programmatic `run()` never sets it.
    defaults.setdefault('holdPlot', False)

    # Built after `defaults`, so `--help` can report what this case actually
    # runs rather than the generic CaseSpec value.
    parser = buildArgumentParser(
        description=case.description or f'Run the {case.name} case.',
        caseParams=case.params,
        defaults=defaults,
        parameterDefinitions=case.parameterDefinitions,
    )
    args = parser.parse_args(argv)
    raw = _readMapping(args.config) if args.config else None
    if raw is not None and 'schemaVersion' in raw:
        document = RunConfiguration.fromDict(raw)
        if document.case != case.name:
            raise ValueError(f'configuration.case: expected {case.name!r}, got {document.case!r}')
        known = {f.name for f in fields(CaseSpec)}
        metadata = {'config', 'saveConfig', 'describeConfig', 'parameterReference', 'checkConfig', 'saveResolvedConfig'}
        for name, value in vars(args).items():
            if value is None or name in metadata:
                continue
            if name == 'scheme':
                document.solver['scheme'] = value
            elif name in known:
                document.simulation[name] = value
            else:
                document.params[name] = value
        spec = document.resolve(case)
        if args.saveConfig:
            spec.save(args.saveConfig)
    else:
        spec = specFromArgs(args, caseParams=case.params, defaults=defaults)

    if args.describeConfig:
        _write(args.describeConfig, configurationSchema(case, spec.scheme))
    if args.parameterReference:
        from pathlib import Path
        target = Path(args.parameterReference)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(parameterReference(case, spec.scheme))
    if args.describeConfig or args.parameterReference:
        return None

    from warpSPHBootstrap import activePrecision, bootstrap
    active = activePrecision()
    if active is not None and active != spec.precision:
        print(f'warning: requested precision {spec.precision!r} but warpSPHCore is '
              f'already running at {active!r}; continuing at {active!r}. Use '
              f'`python -m warpSPHRun {case.name} --precision {spec.precision}` to '
              f'select it before import.', file=sys.stderr)
        spec = spec.merged(precision=active)
    bootstrap(precision=spec.precision, verbose=spec.verbose)

    if args.checkConfig:
        if spec.configuration is None:
            # Apply the same typed validation to --checkConfig on a legacy file.
            simulation = {k: v for k, v in spec.toDict().items()
                          if k not in ('params', 'scheme', 'schemeOverrides', 'configuration')}
            params = dict(spec.params)
            explicit = set((raw or {}).get('params', {}))
            explicit.update(k for k, value in vars(args).items() if value is not None)
            explicit.update(k for k in (raw or {}) if k in case.params)
            for name, descriptor in case.parameterDefinitions.items():
                inactive = not parameterApplies(descriptor, spec)
                inactive = inactive or (descriptor.schemes and spec.scheme not in descriptor.schemes)
                derivedDefault = descriptor.derivation and name not in explicit
                if (inactive or derivedDefault) and params.get(name) == case.params.get(name):
                    params.pop(name, None)
            spec = RunConfiguration(case.name, simulation=simulation, params=params,
                                    solver={'scheme': spec.scheme, **spec.schemeOverrides}).resolve(case)
        ctx = prepareContext(case, spec)
        if args.saveResolvedConfig:
            _write(args.saveResolvedConfig, resolvedConfiguration(ctx))
        print(f'Configuration valid: {case.name} / {spec.scheme} (no particles sampled or steps run).')
        return None
    result = run(case, spec)
    if args.saveResolvedConfig:
        _write(args.saveResolvedConfig, resolvedConfiguration(result.ctx))
    return result
