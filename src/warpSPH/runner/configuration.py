"""Versioned run documents and a strict, discoverable configuration contract.

This is the shared Python / file / future GUI interface. It resolves values
without sampling particles; prepareContext additionally runs case configuration
and solver validation, still without constructing or advancing the system.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from enum import Enum
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, get_args, get_origin, get_type_hints

from .case import Case
from .caseSpec import CaseSpec, _FIELD_HELP, _readMapping, enumChoices
from .parameters import Parameter as P

VERSION = 1


def _registeredCase(name):
    from .case import getCase
    try:
        return getCase(name)
    except KeyError:
        from ..cases import importAll
        importAll()
        return getCase(name)


def _sameValue(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=0)
    return left == right


def parameterApplies(descriptor, spec):
    values = {**spec.params, 'solver.scheme': spec.scheme}
    matches = lambda condition: all(values.get(k) == v for k, v in condition.items())
    return matches(descriptor.appliesWhen or {}) and (
        not descriptor.appliesWhenAny or any(matches(c) for c in descriptor.appliesWhenAny))


def _mapping(value, path):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f'{path}: expected a mapping with string keys')
    return value


def _keys(value, allowed, path):
    _mapping(value, path)
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'{path}: unknown fields {sorted(unknown)}')


def simulationParameters():
    """Use CaseSpec types/help and the actual enum registries, not copied defaults."""
    hints, choices, out = get_type_hints(CaseSpec), enumChoices(), {}
    groups = {
        'discretization': {'nx', 'dim', 'L', 'n_h', 'kernel', 'supportMode', 'gradientMode',
                           'laplacianMode', 'samplingScheme', 'periodic', 'verletScale',
                           'densityCorrection', 'calibrateNormalization'},
        'time': {'dt', 'tLimit', 'adaptiveDt', 'cflFactor', 'minDt', 'maxDt', 'nSteps'},
        'runtime': {'precision', 'device', 'execution', 'cudaGraph', 'pipelineOutputs',
                    'graphMaxParticles', 'iterationChecks', 'krylovCheckEvery'},
    }
    positive = {'nx', 'L', 'n_h', 'tLimit', 'dt', 'cflFactor', 'minDt', 'maxDt',
                'exportInterval', 'plotInterval', 'storeInterval', 'verletScale',
                'graphMaxParticles', 'krylovCheckEvery'}
    for f in fields(CaseSpec):
        if f.name in ('params', 'scheme', 'schemeOverrides', 'configuration'):
            continue
        annotation = hints[f.name]
        args = get_args(annotation)
        nullable = type(None) in args
        kind = next((a for a in args if a is not type(None)), annotation) if nullable else annotation
        kind = {int: 'integer', float: 'number', bool: 'boolean', str: 'string', dict: 'object'}.get(
            get_origin(kind) or kind)
        if kind is None:
            raise TypeError(f'No configuration descriptor for CaseSpec.{f.name}')
        allowed = tuple(choices.get(f.name, ()))
        if f.name == 'precision':
            from warpSPHBootstrap import PRECISIONS
            allowed = tuple(PRECISIONS)
        if f.name == 'dim':
            allowed = (1, 2, 3)
        if f.name == 'execution':
            from ..configurations.compute import EXECUTION_MODES
            allowed = tuple(EXECUTION_MODES)
        if f.name == 'iterationChecks':
            from ..configurations.compute import CHECK_SCHEDULES
            allowed = tuple(CHECK_SCHEDULES)
        if f.name == 'storeMode':
            allowed = ('states', 'trajectory')
        if f.name == 'plotBackend':
            allowed = ('matplotlib', 'vispy', 'pyVista')
        group = next((g for g, names in groups.items() if f.name in names), 'output')
        out[f.name] = P(kind, _FIELD_HELP.get(f.name, f.name), group,
                        exclusiveMinimum=0 if f.name in positive else None,
                        minimum=0 if f.name in ('nSteps', 'resumeStepOffset') else None,
                        choices=allowed, nullable=nullable)
    return out


def solverParameters(scheme):
    """Curated supported controls, using native solver paths and enum names.

    Only controls consumed by each formulation are exposed. Additional solver
    families can register descriptors without changing the file or browser format.
    """
    from ..configurations.moduleConfigurations.diffusionParameters import (
        ViscosityTerms, BetaMode, VelocityPairPolicy, RiemannSolver, StateLimiter)
    from ..enumTypes import DensityDiffusionScheme, ViscositySwitch
    def choice(enum, description, group):
        return P('string', description, group, choices=tuple(m.name for m in enum))
    def positive(description, group='diffusion'):
        return P('number', description, group, minimum=0)
    if scheme == 'artificialCompressible':
        return {
            'acParams.cflT': P('number', 'Real-time CFL factor.', 'time', exclusiveMinimum=0, maximum=.4),
            'acParams.cflTau': P('number', 'Pseudo-time CFL factor.', 'pseudoTime', exclusiveMinimum=0),
            'acParams.dtOverDtau': P('number', 'Real/pseudo-time step ratio.', 'pseudoTime', exclusiveMinimum=0),
            'acParams.rkStages': P('integer', 'Pseudo-time Runge-Kutta stage count.', 'pseudoTime', choices=(2, 3, 4)),
            'acParams.epsilonV': P('number', 'Log10 velocity convergence target.', 'convergence', maximum=0),
            'acParams.maxPseudoIterations': P('integer', 'Maximum pseudo-time iterations.', 'convergence', minimum=1),
            'acParams.minPseudoIterations': P('integer', 'Minimum pseudo-time iterations.', 'convergence', minimum=1),
            'acParams.convergenceCheckSchedule': P('string', 'Convergence readback schedule.', 'convergence', choices=('auto', 'every', 'adaptive')),
            'acParams.k2Factor': P('number', 'Pressure-smoothing coefficient factor.', 'diffusion', minimum=0, maximum=.2),
        }
    if scheme in ('MFM', 'MFV'):
        return {
            'order': P('integer', 'Spatial reconstruction order.', 'reconstruction', choices=(1, 2)),
            'cfl': P('number', 'Meshless signal timestep CFL factor.', 'time', exclusiveMinimum=0),
            'cflLength': P('string', 'Length used in the CFL estimate.', 'time', choices=('particle', 'support')),
            'temporalScheme': P('string', 'Meshless time advance.', 'time', choices=('auto', 'ssprk', 'hancock', 'kdk')),
            'closure': P('string', 'Effective-face closure treatment.', 'geometry', choices=('none', 'project')),
        }
    if scheme in ('deltaSPH', 'sun2017DeltaSPH'):
        return {
            'diffusionParams.densityDelta': positive('Density-diffusion strength.'),
            'diffusionParams.densityDiffusionTerm': choice(DensityDiffusionScheme, 'Density-diffusion formulation.', 'diffusion'),
        }
    if scheme in ('GSPH', 'InutsukaGSPH'):
        return {
            'diffusionParams.riemannSolver': choice(RiemannSolver, 'Pair Riemann solver.', 'riemann'),
            'diffusionParams.stateLimiter': choice(StateLimiter, 'Reconstructed-state limiter.', 'reconstruction'),
            'diffusionParams.shockSwitchC': positive('First-order shock-switch coefficient; 0 disables it.', 'reconstruction'),
        }
    if scheme not in ('CompSPH', 'CRKSPH', 'Monaghan'):
        raise ValueError(f'solver.scheme: no version-{VERSION} configuration contract for {scheme!r}')
    out = {f'diffusionParams.{name}': positive(description) for name, description in {
        'K': 'Overall artificial-viscosity strength.', 'C_l': 'Linear viscosity coefficient.',
        'C_q': 'Quadratic viscosity coefficient.', 'thermalConductivity': 'Overall thermal-conductivity strength.',
        'Cu_l': 'Linear conductivity coefficient.', 'Cu_q': 'Quadratic conductivity coefficient.',
    }.items()}
    out.update({
        'diffusionParams.viscosityTerm': choice(ViscosityTerms, 'Artificial-viscosity formulation.', 'diffusion'),
        'diffusionParams.betaMode': choice(BetaMode, 'Quadratic coefficient coupling policy.', 'diffusion'),
        'viscositySwitchParams.scheme': choice(ViscositySwitch, 'Artificial-viscosity switch.', 'viscositySwitch'),
        'viscositySwitchParams.alpha_min': positive('Lower switch coefficient.', 'viscositySwitch'),
        'viscositySwitchParams.alpha_max': positive('Upper switch coefficient.', 'viscositySwitch'),
    })
    # These hosts require Raw; advertising other policies would offer an unusable setting.
    policies = (VelocityPairPolicy.Raw,) if scheme in ('CompSPH', 'CRKSPH') else tuple(VelocityPairPolicy)
    out['diffusionParams.velocityPairPolicy'] = P('string', 'Pair velocity supplied to viscosity.',
                                                 'reconstruction', choices=tuple(m.name for m in policies))
    return out


def _flatten(data, prefix=''):
    _mapping(data, prefix or 'solver')
    out = {}
    for key, value in data.items():
        path = f'{prefix}.{key}' if prefix else key
        if '.' in key or key.startswith('_'):
            raise ValueError(f'solver.{path}: use nested public parameter names')
        if isinstance(value, dict):
            if not value:
                raise ValueError(f'solver.{path}: empty parameter group')
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _nested(properties):
    out = {}
    for path, value in properties.items():
        node = out
        parts = path.split('.')
        for name in parts[:-1]:
            node = node.setdefault(name, {'type': 'object', 'additionalProperties': False,
                                          'properties': {}})['properties']
        node[parts[-1]] = value
    return out


def _validateSolver(scheme, values):
    flat, descriptors = _flatten(values), solverParameters(scheme)
    _keys(flat, descriptors, 'solver')
    for path, value in flat.items():
        descriptors[path].validate(value, f'solver.{path}')
    return flat


@dataclass
class RunConfiguration:
    case: str
    simulation: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    solver: Dict[str, Any] = field(default_factory=dict)
    variant: str = 'default'
    schemaVersion: int = VERSION

    @classmethod
    def fromDict(cls, data):
        _keys(data, ('schemaVersion', 'case', 'simulation', 'params', 'solver', 'variant'), 'configuration')
        if type(data.get('schemaVersion')) is not int or data['schemaVersion'] != VERSION:
            raise ValueError(f'configuration.schemaVersion: expected {VERSION}')
        if not isinstance(data.get('case'), str) or not data['case']:
            raise ValueError('configuration.case: expected a registered case name')
        if not isinstance(data.get('variant', 'default'), str):
            raise ValueError('configuration.variant: expected a named case variant')
        for name in ('simulation', 'params', 'solver'):
            _mapping(data.get(name, {}), name)
        # Also rejects non-JSON objects and non-finite numbers nested in option dictionaries.
        try:
            json.dumps(data, allow_nan=False)
        except (ValueError, TypeError) as e:
            raise ValueError(f'configuration: values must be finite JSON data: {e}') from e
        return cls(**copy.deepcopy(data))

    @classmethod
    def fromOverrides(cls, case, overrides):
        """Convert sparse runner kwargs, e.g. warpSPHCases RunEntry.caseSpecOverrides().

        RunEntry.variant is a recipe label, not a physical variant identifier;
        it is intentionally not forwarded or guessed here.
        """
        known = {f.name for f in fields(CaseSpec)} - {'configuration'}
        _keys(overrides, known, 'runner overrides')
        values = copy.deepcopy(overrides)
        params = values.pop('params', {})
        solver = values.pop('schemeOverrides', {})
        _mapping(params, 'params')
        _mapping(solver, 'solver')
        if 'scheme' in solver:
            raise ValueError('runner overrides.schemeOverrides: put scheme at the top level')
        solver = {'scheme': values.pop('scheme', None) or case.scheme, **solver}
        document = cls(case.name, simulation=values, params=params, solver=solver)
        document.resolve(case)
        return document

    @classmethod
    def load(cls, path):
        return cls.fromDict(_readMapping(str(path)))

    def toDict(self):
        return copy.deepcopy(dict(schemaVersion=self.schemaVersion, case=self.case, variant=self.variant,
                                  simulation=self.simulation, params=self.params, solver=self.solver))

    def save(self, path):
        _write(path, self.toDict())
        return str(path)

    def resolve(self, case=None):
        # Python construction receives exactly the same structural checks as a file.
        self.fromDict(self.toDict())
        if case is None:
            case = _registeredCase(self.case)
        if case.name != self.case:
            raise ValueError(f'configuration.case: expected {case.name!r}, got {self.case!r}')
        if not case.parameterDefinitions or set(case.params) != set(case.parameterDefinitions):
            raise ValueError(f'configuration.case: {case.name!r} has not completed parameter migration')
        variants = {'default': {}, **case.variants}
        if self.variant not in variants:
            raise ValueError(f'configuration.variant: expected one of {list(variants)}')
        base = CaseSpec(caseName=case.name, scheme=case.scheme, params=copy.deepcopy(case.params)).merged(**copy.deepcopy(case.defaults))
        variant = variants[self.variant]
        _keys(variant, ('simulation', 'params', 'solver'), 'variant')
        simulation = {**variant.get('simulation', {}), **self.simulation}
        params = {**variant.get('params', {}), **self.params}
        # Merge nested numerical overrides without replacing untouched sibling values.
        solver = _merge(variant.get('solver', {}), self.solver)
        scheme = solver.pop('scheme', base.scheme)
        if scheme not in case.supportedSchemes:
            raise ValueError(f'solver.scheme: expected one of {list(case.supportedSchemes)}')
        descriptors = {**simulationParameters(), **case.simulationParameters}
        _keys(simulation, descriptors, 'simulation')
        _keys(params, case.parameterDefinitions, 'params')
        spec = base.merged(**simulation, scheme=scheme, params=params, schemeOverrides=solver, configuration=self.toDict())
        for name, descriptor in descriptors.items():
            if name in simulation and descriptor.readOnly and simulation[name] != getattr(base, name):
                raise ValueError(f'simulation.{name}: read-only; {descriptor.derivation or descriptor.description}')
            descriptor.validate(getattr(spec, name), f'simulation.{name}')
        # Validate raw input before a derivation can do arithmetic with it.
        for name, value in spec.params.items():
            case.parameterDefinitions[name].validate(value, f'params.{name}')
        if case.deriveParameters is not None:
            derived = case.deriveParameters(spec)
            _keys(derived, case.parameterDefinitions, 'derived parameters')
            for name, value in derived.items():
                if name in params and not _sameValue(params[name], value):
                    raise ValueError(f'params.{name}: derived from {case.parameterDefinitions[name].derivation}')
            spec = spec.merged(params=derived)
        for name, value in spec.params.items():
            descriptor = case.parameterDefinitions[name]
            descriptor.validate(value, f'params.{name}')
            if name in params and descriptor.readOnly and not descriptor.derivation and params[name] != base.params.get(name):
                raise ValueError(f'params.{name}: read-only; {descriptor.description}')
            if name in params and value is not None and descriptor.schemes and scheme not in descriptor.schemes:
                raise ValueError(f'params.{name}: not used by {scheme}')
            if name in params and not parameterApplies(descriptor, spec):
                raise ValueError(f'params.{name}: inactive for the selected solver/settings')
        if scheme in ('MFM', 'MFV') and not spec.periodic:
            raise ValueError('simulation.periodic: MFM/MFV currently require a periodic domain')
        if spec.minDt is not None and spec.maxDt is not None and spec.minDt > spec.maxDt:
            raise ValueError('simulation.minDt: must not exceed simulation.maxDt')
        if case.validateConfiguration is not None:
            case.validateConfiguration(spec)
        _validateSolver(scheme, solver)
        return spec

    def prepare(self, case=None):
        from .runner import prepareContext
        case = case or _registeredCase(self.case)
        ctx = prepareContext(case, self.resolve(case))
        return ctx

    def run(self, case=None):
        from .runner import run
        case = case or _registeredCase(self.case)
        return run(case, self.resolve(case))


def _merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = _merge(result.get(key, {}), value) if isinstance(value, dict) else copy.deepcopy(value)
    return result


def _solverEnums():
    from ..configurations.moduleConfigurations.diffusionParameters import (
        ViscosityTerms, BetaMode, VelocityPairPolicy, RiemannSolver, StateLimiter)
    from ..enumTypes import DensityDiffusionScheme, ViscositySwitch
    return {'diffusionParams.viscosityTerm': ViscosityTerms, 'diffusionParams.betaMode': BetaMode,
            'diffusionParams.velocityPairPolicy': VelocityPairPolicy, 'diffusionParams.riemannSolver': RiemannSolver,
            'diffusionParams.stateLimiter': StateLimiter, 'diffusionParams.densityDiffusionTerm': DensityDiffusionScheme,
            'viscositySwitchParams.scheme': ViscositySwitch}


def applySchemeOverrides(ctx):
    """Run after case configuration and before solver validation/system construction."""
    scheme = ctx.scheme.name if isinstance(ctx.scheme, Enum) else ctx.scheme
    flat = _validateSolver(scheme, ctx.spec.schemeOverrides) if ctx.spec.schemeOverrides else {}
    enums = _solverEnums()
    for path, value in flat.items():
        owner = ctx.schemeConfig
        parts = path.split('.')
        for name in parts[:-1]:
            if not hasattr(owner, name):
                raise ValueError(f'solver.{path}: unavailable on {scheme}')
            owner = getattr(owner, name)
        name = parts[-1]
        if not hasattr(owner, name):
            raise ValueError(f'solver.{path}: unavailable on {scheme}')
        old = getattr(owner, name)
        enum = enums.get(path)
        if enum:
            member = enum[value]
            value = member if isinstance(old, Enum) else member.value
        setattr(owner, name, value)
    ac = getattr(ctx.schemeConfig, 'acParams', None)
    if ac is not None and ac.minPseudoIterations > ac.maxPseudoIterations:
        raise ValueError('solver.acParams.minPseudoIterations: must not exceed maxPseudoIterations')
    switch = getattr(ctx.schemeConfig, 'viscositySwitchParams', None)
    if switch is not None and switch.alpha_min > switch.alpha_max:
        raise ValueError('solver.viscositySwitchParams.alpha_min: must not exceed alpha_max')


def configurationSchema(case, scheme=None):
    scheme = scheme or case.scheme
    # Includes the migration/completeness checks and authoritative case defaults.
    spec = RunConfiguration(case.name, solver={'scheme': scheme}).resolve(case)
    sim = {**simulationParameters(), **case.simulationParameters}
    from .runner import prepareContext
    ctx = prepareContext(case, spec)
    numerical = {}
    enums = _solverEnums()
    for path, descriptor in solverParameters(scheme).items():
        value = ctx.schemeConfig
        for name in path.split('.'):
            value = getattr(value, name)
        if path in enums:
            value = value.name if isinstance(value, Enum) else enums[path](value).name
        elif descriptor.kind == 'number':
            value = float(value)
        numerical[path] = descriptor.schema(value)
    return {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': f'{case.name} run configuration', 'description': case.description,
        'type': 'object', 'additionalProperties': False, 'required': ['schemaVersion', 'case'],
        'properties': {
            'schemaVersion': {'const': VERSION}, 'case': {'const': case.name},
            'variant': {'type': 'string', 'enum': ['default', *case.variants], 'default': 'default'},
            'simulation': {'type': 'object', 'additionalProperties': False,
                           'properties': {n: p.schema(getattr(spec, n)) for n, p in sim.items()}},
            'params': {'type': 'object', 'additionalProperties': False,
                       'properties': {n: p.schema(spec.params[n]) if n in spec.params else p.schema()
                                      for n, p in case.parameterDefinitions.items()}},
            'solver': {'type': 'object', 'additionalProperties': False,
                       'properties': {'scheme': {'const': scheme, 'default': scheme},
                                      **_nested(numerical)}},
        },
        'x-supported-schemes': list(case.supportedSchemes),
        'x-variants': copy.deepcopy(case.variants),
    }


def parameterReference(case, scheme=None):
    schema = configurationSchema(case, scheme)
    lines = [f'# {case.name} configuration', '', case.description, '',
             f"Solver: `{scheme or case.scheme}`.", '',
             'Generated from the runtime parameter contract. Values use the case\'s chosen units.', '',
             '| Parameter | Type | Default | Description |', '|---|---|---|---|']
    def visit(properties, prefix=''):
        for name, item in properties.items():
            path = f'{prefix}.{name}' if prefix else name
            if 'properties' in item:
                visit(item['properties'], path)
            else:
                default = json.dumps(item.get('default', item.get('const'))) if 'default' in item or 'const' in item else 'solver default / unset'
                description = item.get('description', '')
                if item.get('x-applies-when'):
                    description += f" Applies when {item['x-applies-when']}."
                if item.get('x-supported-schemes'):
                    description += f" Used by {', '.join(item['x-supported-schemes'])}."
                if item.get('x-applies-when-any'):
                    description += f" Applies under any of {item['x-applies-when-any']}."
                if item.get('readOnly'):
                    description += ' Read-only.'
                if item.get('x-derived-from'):
                    description += f" Derived from {item['x-derived-from']}."
                if item.get('deprecated'):
                    description += ' Legacy compatibility field.'
                if item.get('x-units'):
                    description += f" Units: {item['x-units']}."
                bounds = {k: item[k] for k in ('minimum', 'exclusiveMinimum', 'maximum', 'exclusiveMaximum', 'minItems', 'maxItems', 'enum') if k in item}
                if bounds:
                    description += f' Constraints: {bounds}.'
                kind = item.get('type', 'constant')
                kind = ' / '.join(kind) if isinstance(kind, list) else kind
                lines.append(f'| `{path}` | {kind} | `{default}` | {description.replace("|", "/")} |')
                if item.get('items', {}).get('properties'):
                    visit(item['items']['properties'], path + '[]')
    visit(schema['properties'])
    return '\n'.join(lines) + '\n'


def resolvedConfiguration(ctx):
    """Actual setup values (before or after sampling), suitable for a run manifest."""
    from ..configurations import configurationToDict
    versions = {}
    import importlib.metadata
    import importlib.util
    for package in ('warpSPH', 'warpSPHCore', 'warpSPHIntegrators', 'warpSPHBoundaries'):
        info = {}
        try:
            info['version'] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
        module = importlib.util.find_spec(package)
        if module and module.origin:
            try:
                folder = str(Path(module.origin).parent)
                revision = subprocess.run(['git', '-C', folder, 'rev-parse', 'HEAD'],
                                          capture_output=True, text=True, timeout=3)
                if revision.returncode == 0:
                    info['commit'] = revision.stdout.strip()
                    dirty = subprocess.run(['git', '-C', folder, 'status', '--porcelain'],
                                           capture_output=True, text=True, timeout=3)
                    info['dirty'] = bool(dirty.stdout.strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
        if info:
            versions[package] = info
    return {'schemaVersion': VERSION, 'case': ctx.case.name,
            'requested': ctx.spec.configuration,
            'spec': ctx.spec.toDict(), 'simulationConfig': configurationToDict(ctx.config),
            'solverConfig': ctx.exportFunction(ctx.schemeConfig), 'versions': versions,
            'stage': 'setup (no particle sampling)' if 'initialParticleCount' not in ctx.scratch else 'initialized'}


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def configurationCatalogue(cases=None):
    """List contract coverage without constructing a context or sampling particles."""
    if cases is None:
        from ..cases import importAll
        from .case import getCase, listCases
        importAll()
        cases = [getCase(name) for name in listCases()]
    entries = []
    for case in sorted(cases, key=lambda c: c.name):
        missing = sorted(set(case.params) - set(case.parameterDefinitions))
        stale = sorted(set(case.parameterDefinitions) - set(case.params))
        ready = bool(case.parameterDefinitions) and not missing and not stale and bool(case.supportedSchemes)
        status = 'ready' if ready else 'unmigrated'
        error = None
        if ready:
            try:
                RunConfiguration(case.name).resolve(case)
            except (ValueError, TypeError) as failure:
                status, error = 'invalid', str(failure)
        entries.append({'id': case.name, 'description': case.description,
                        'status': status, 'error': error,
                        'parameterCount': len(case.params), 'missingParameters': missing,
                        'staleParameters': stale, 'schemes': list(case.supportedSchemes),
                        'variants': ['default', *case.variants]})
    return {'schemaVersion': VERSION, 'cases': entries}
