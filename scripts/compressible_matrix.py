"""Fixed-resolution compressible comparison contracts; no simulator imports.

Plans describe inputs, runs are append-only, selections describe figures.
Importing this module never initializes Torch, Warp or a CUDA context.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / '.tmp/compressible-matrix'
VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def identifier():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:10]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, 'item'):
        return clean(value.item())
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def merge(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def case(name, source, nx, end, dim, label, metric, scale=1., params=None, precision=None, unsupported=(),
         variant=None, metric_note=None, boundaries=None):
    definition = dict(id=name, case=source, label=label, simulation=dict(nx=nx, dim=dim, tLimit=end),
                      params=params or {}, precision=precision, unsupportedSchemes=list(unsupported),
                      quality=dict(metric=metric, scale=scale, lowerIsBetter=True))
    # Optional additions leave the original twelve case definitions unchanged.
    if variant is not None:
        definition['variant'] = variant
    if metric_note is not None:
        definition['metricNote'] = metric_note
    if boundaries is not None:
        definition['boundaryTreatments'] = boundaries
    return definition


# One fixed resolution per column. Sedov's native sampler derives its endpoint
# from goalRadius; the effective endpoint is saved in each run's resolved input.
CASES = {c['id']: c for c in (
    case('linearWave', 'linearWave', 128, 1., 1, 'Linear wave\n1D · nx=128', 'normalizedVelocityL1',
         params={'A': 1e-3}),
    case('sod', 'sod', 800, .15, 1, 'Sod\n1D · nx=800', 'densityVolumeWeightedL1',
         params={'gamma': 1.4, 'right_rho': .125, 'right_pressure': .1, 'samplingRatio': 8}),
    case('sod2d', 'sod2d', 512, .15, 2, 'Sod strip\n2D · nx=512', 'densityVolumeWeightedL1'),
    case('noh', 'noh', 256, .6, 1, 'Noh\n1D · nx=256', 'postShockDensityAbsoluteRelativeError',
         params={'uAmbient': 1e-8}, precision='float64'),
    case('nohOpen', 'noh', 256, .6, 1, 'Noh open edges\n1D · nx=256', 'postShockDensityAbsoluteRelativeError',
         params={'uAmbient': 1e-8, 'outerBoundary': 'open'}, precision='float64',
         metric_note='Open edges; MFM/MFV use an analytic Lagrangian inflow reservoir, SPH uses its native open boundary.',
         boundaries={'MFM': 'analytic Lagrangian inflow reservoir',
                     'MFV': 'analytic Lagrangian inflow reservoir', 'default': 'native open edges'}),
    case('kidder', 'kidder', 200, None, 1, 'Kidder\n1D · nx=200', 'densityVolumeWeightedL1',
         params={'tauFraction': .99}, precision='float64'),
    case('sedov1d', 'sedov', 800, None, 1, 'Sedov\n1D · nx=800', 'densityVolumeWeightedL1',
         params={'uAmbient': 1e-8, 'goalRadius': .8}),
    case('sedov2d', 'sedov', 256, None, 2, 'Sedov\n2D · nx=256', 'densityVolumeWeightedL1',
         params={'uAmbient': 1e-8, 'goalRadius': .8}),
    case('sedov3d', 'sedov', 64, None, 3, 'Sedov\n3D · nx=64', 'densityVolumeWeightedL1',
         params={'uAmbient': 1e-8, 'goalRadius': .8}),
    case('gresho', 'gresho', 256, 3., 2, 'Gresho\n2D · nx=256', 'tangentialVelocityL1'),
    case('hydrostatic', 'hydrostatic', 100, 7., 2, 'Pressure equilibrium\n2D · nx=100', 'pressureError',
         params={'rho_high': 4., 'gamma': 1.5}),
    case('hydrostaticTwoGas', 'hydrostatic', 100, 7., 2, 'Pressure equilibrium: two gases\n2D · nx=100',
         'pressureError', variant='twoGas', unsupported=('PESPH', 'GSPH', 'InutsukaGSPH'),
         metric_note='Stationary square with rho=1 and 4, gamma=1.4 and 5/3, and common P=1; no gravity.'),
    case('hydrostaticAdvectedEllipse', 'hydrostatic', 100, 1., 2, 'Advected two-gas ellipse\n2D · nx=100',
         'pressureError', variant='advectedEllipse', unsupported=('PESPH', 'GSPH', 'InutsukaGSPH'),
         metric_note='Ellipse with rho=1 and 4, gamma=1.4 and 5/3, common P=1 and x velocity=1; one periodic crossing, no gravity.'),
    case('kelvinHelmholtz', 'kelvinHelmholtz', 256, 2.5, 2, 'KH sharp density\n2D · nx=256', None),
    case('khSmooth', 'kelvinHelmholtz', 256, 2.5, 2, 'KH smooth density\n2D · nx=256', None,
         params={'smoothDensity': 1}),
)}

METHODS = {
    'Monaghan': dict(label='Monaghan (constant AV)', solver=dict(scheme='Monaghan',
        viscositySwitchParams=dict(scheme='NoneSwitch'))),
    'Monaghan-CD': dict(label='Monaghan + Cullen–Dehnen', solver=dict(scheme='Monaghan',
        viscositySwitchParams=dict(scheme='CullenDehnen2010'))),
    'CompSPH': dict(label='CompSPH', solver=dict(scheme='CompSPH')),
    'CRKSPH': dict(label='CRKSPH', solver=dict(scheme='CRKSPH')),
    'PESPH-E': dict(label='PESPH-E', solver=dict(scheme='PESPH', variant='PressureEnergy', dissipation=True)),
    'PESPH-A': dict(label='PESPH-A (SwiftLocal)*', solver=dict(scheme='PESPH', variant='PressureEntropy',
        dissipation=True, entropyHeating='SwiftLocal'),
        note='Experimental local heat-to-entropy conversion; conservation defects are measured, not repaired.'),
    'MFM': dict(label='MFM', solver=dict(scheme='MFM')),
    'MFV': dict(label='MFV', solver=dict(scheme='MFV')),
    'GSPH': dict(label='Godunov SPH', solver=dict(scheme='GSPH')),
    'InutsukaGSPH': dict(label='Inutsuka Godunov SPH', solver=dict(scheme='InutsukaGSPH')),
}

RECIPES = {
    'linearWave': 'Mean absolute velocity error / imposed travelling-wave velocity amplitude; analytic linear wave.',
    'sod': 'Volume-weighted density L1 against the exact Riemann profile (density units).',
    'sod2d': 'Volume-weighted density L1 against the exact Riemann profile (density units).',
    'noh': 'Absolute relative mean post-shock density error in 0.2–0.8 of the exact front radius; excludes centre/seam.',
    'kidder': 'Volume-weighted density L1 against analytic compression at 0.99 collapse time; analytic boundary bands excluded. Boundary work means total energy is not a closed-system invariant.',
    'sedov1d': 'Volume-weighted density L1 inside 1.15 exact shock radii (density units); initial energy is also reported.',
    'sedov2d': 'Volume-weighted density L1 inside 1.15 exact shock radii (density units); initial energy is also reported.',
    'sedov3d': 'Volume-weighted density L1 inside 1.15 exact shock radii (density units); initial energy is also reported.',
    'gresho': 'Tangential velocity L1 within radius 0.5 against the stationary piecewise Gresho profile (velocity units).',
    'hydrostatic': 'Maximum absolute pressure error from P=1 in the stationary dense-square contact test (also a relative error here); no gravity. Displacement and spurious velocity are retained separately.',
    'kelvinHelmholtz': 'Mode amplitude is a diagnostic, not accuracy; no matched continuum reference is assumed.',
    'khSmooth': 'Mode amplitude is a diagnostic, not accuracy; density smoothing alone does not match McNally ICs.',
}


def method_catalog(overrides=None):
    catalog = deepcopy(METHODS)
    for name, patch in (overrides or {}).items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*', name):
            raise ValueError(f'Unsafe method ID {name!r}')
        patch = deepcopy(patch)
        base = patch.pop('base', name)
        if base not in catalog:
            raise ValueError(f'Unknown base method {base!r}')
        unknown = set(patch) - {'label', 'solver', 'simulation', 'params', 'note'}
        if unknown:
            raise ValueError(f'Unknown preset fields: {sorted(unknown)}')
        catalog[name] = merge(catalog[base], patch)
        if name not in METHODS and 'label' not in patch:
            catalog[name]['label'] = name
    return catalog


def select_rows(plan, cases=None, methods=None):
    if plan.get('schemaVersion') != VERSION:
        raise ValueError('Unsupported matrix manifest version')
    for row in plan['rows']:
        if row['inputSHA256'] != digest({k: v for k, v in row.items() if k != 'inputSHA256'}):
            raise ValueError(f"Planned input was edited: {row['id']}; create a new plan/preset instead")
    for requested, allowed, kind in ((cases, plan['cases'], 'case'), (methods, plan['methods'], 'method')):
        unknown = set(requested or ()) - set(allowed)
        if unknown:
            raise ValueError(f'Unknown {kind}(s): {sorted(unknown)}; available: {list(allowed)}')
    return [r for r in plan['rows'] if (not cases or r['caseId'] in cases) and (not methods or r['method'] in methods)]


def make_plan(cases=None, methods=None, presets=None, precision='float32', device='cuda:0', smoke=False):
    catalog = method_catalog(presets)
    case_ids, method_ids = list(cases or CASES), list(methods or catalog)
    for ids, allowed in ((case_ids, CASES), (method_ids, catalog)):
        if len(ids) != len(set(ids)) or set(ids) - set(allowed):
            raise ValueError(f'Unknown or duplicate selection: {ids}')
    rows = []
    for cid in case_ids:
        definition = deepcopy(CASES[cid])
        for mid in method_ids:
            preset = catalog[mid]
            case_controls = {'nx', 'dim', 'L', 'periodic', 'tLimit', 'nSteps', 'precision'} & set(preset.get('simulation', {}))
            if case_controls:
                raise ValueError(f'{mid}: presets cannot change fixed case controls {sorted(case_controls)}; '
                                 'define a separate case/study instead')
            simulation = dict(device=device, precision=definition.get('precision') or precision, nSteps=None, kernel='B7', n_h=4.,
                integrationScheme='rungeKutta2', supportMode='KernelMeanSymmetric',
                adaptiveDt=True, cflFactor=.15, minDt=1e-10, maxDt=.01,
                calibrateNormalization=True, execution='eager', cudaGraph=False,
                plot=False, video=False, store=False, show=False, progress=True, quiet=True,
                stallDtSteps=1000, stallProgress=1e-4, velocityAlarmStopRatio=100.)
            simulation.update({k: v for k, v in definition['simulation'].items() if v is not None})
            simulation.update(preset.get('simulation', {}))
            if smoke:
                simulation.update(nx=12 if simulation['dim'] == 3 else 24, nSteps=3)
            configuration = dict(schemaVersion=1, case=definition['case'], simulation=simulation,
                params=merge(definition['params'], preset.get('params', {})), solver=deepcopy(preset['solver']))
            if 'variant' in definition:
                configuration['variant'] = definition['variant']
            row = dict(id=cid+'__'+mid, caseId=cid, method=mid, configuration=configuration,
                       caseDefinition=definition, preset=deepcopy(preset))
            if preset['solver']['scheme'] in definition.get('unsupportedSchemes', ()):
                row['unsupportedReason'] = f"Native {definition['case']} {definition.get('variant', '')} case does not support {preset['solver']['scheme']}"
            row['inputSHA256'] = digest(row)
            rows.append(row)
    return dict(schemaVersion=VERSION, id=identifier(), created=now(), smoke=smoke,
                precision=precision, device=device, warmupSteps=10, seed=1234,
                casePrecisions={cid: CASES[cid].get('precision') or precision for cid in case_ids},
                cases={c: deepcopy(CASES[c]) for c in case_ids},
                methods={m: catalog[m] for m in method_ids}, rows=rows,
                protocol=dict(resolution='One fixed nx per case; actual particle counts and initial-array hashes retained.',
                    initialization='Native case samplers; linear wave uses physical analytic masses for all methods.',
                    timesteps='Native method CFL policies, with the first and final steps capped. No failure-driven retries.',
                    timing='Headless: no rendering, video or trajectory export. Independent ten-step warmup, then fresh seeded measured state. Loop wall includes diagnostics and timestep selection; runner integrator times are separate.',
                    limitations=['PESPH-A SwiftLocal is explicitly experimental.',
                        'Cold Noh/Sedov use common uAmbient=1e-8; failures are retained.',
                        'KH has no accuracy rank without an explicitly matched reference.',
                        'Fixed-resolution overview; no convergence or equal-compute claim.']))


def complete(status, actual, target, smoke=False):
    return (not smoke and status == 'finished' and actual is not None and target is not None
            and abs(actual-target) <= 2e-6 * max(abs(target), 1e-30))


def classify_outcome(record, *, smoke=False):
    """Keep endpoint completion, early termination and admissibility distinct.

    Also interprets earlier records without rewriting their saved evidence.
    An endpoint overshoot cannot be repaired by relabelling it as complete.
    """
    status = record.get('status', 'pending')
    if status in ('pending', 'running', 'error', 'timeout', 'interrupted', 'unsupported', 'rejected'):
        return status
    if record.get('stopReason'):
        return 'stopped'
    if record.get('executionStatus') == 'stopped':
        return 'stopped'
    if status == 'stopped' and 'admissibility' not in record:
        return 'stopped'
    if smoke or status == 'smoke':
        return 'smoke'
    actual, target = record.get('actualTime'), record.get('targetTime')
    if not complete('finished', actual, target):
        return 'endpoint_mismatch' if actual is not None and target is not None and actual >= target else 'stopped'
    admissibility = record.get('admissibility', {})
    if admissibility.get('positive') is False or admissibility.get('finite') is False:
        return 'completed_invalid'
    return 'completed'


def case_precisions(plan):
    """Infer per-column precision from frozen rows, including older manifests."""
    precisions = {}
    for row in plan['rows']:
        cid = row['caseId']
        value = row['configuration']['simulation'].get('precision', plan['precision'])
        if cid in precisions and precisions[cid] != value:
            raise ValueError(f'Mixed precisions within case {cid}')
        precisions[cid] = value
    return precisions


def snap_endpoint(system, target, precision):
    """Remove only a few time-dtype ULPs; never skip a physical remainder."""
    actual = float(system.t)
    epsilon = 2.**(-23 if precision == 'float32' else -52)
    if abs(actual-target) <= 4*epsilon*max(abs(target), abs(actual), 1e-30):
        system.t = float(target)
        return True
    return False


def rank(values):
    """Competition ranks, retaining ties and raw values (no aggregate score)."""
    ordered = sorted((key, v) for key, v in values.items() if v is not None and math.isfinite(v) and v >= 0)
    return {key: 1 + sum(other < value for _, other in ordered) for key, value in ordered}


def comparison(root, cases=None, methods=None, includes=()):
    """Freeze explicit selected result paths; never select a numerical winner."""
    root = Path(root).resolve()
    plan = read_json(root/'manifest.json')
    rows = select_rows(plan, cases, methods)
    out = dict(schemaVersion=VERSION, id=identifier(), created=now(), smoke=plan['smoke'],
               casePrecisions=case_precisions(plan),
               cases={r['caseId']: plan['cases'][r['caseId']] for r in rows},
               methods={r['method']: plan['methods'][r['method']] for r in rows}, cells=[])
    sources = [(root, plan, rows)]
    for other in includes:
        other = Path(other).resolve()
        other_plan = read_json(other/'manifest.json')
        select_rows(other_plan)  # validate frozen row hashes even for partial trials
        if other_plan['smoke'] != plan['smoke']:
            raise ValueError('Cannot merge smoke/full runs')
        other_precisions = case_precisions(other_plan)
        # A trial may cover only a subset of the displayed cases. Keep its
        # other cells pending instead of requiring another full matrix.
        extra = [r for r in other_plan['rows'] if r['caseId'] in out['cases']]
        for r in extra:
            cid, mid = r['caseId'], r['method']
            if cid not in out['cases']:
                continue
            if other_precisions[cid] != out['casePrecisions'][cid]:
                raise ValueError(f'Cannot merge different precisions within case {cid}')
            if other_plan['cases'][cid] != out['cases'][cid]:
                raise ValueError(f'Case definitions differ: {cid}')
            if mid in out['methods']:
                raise ValueError(f'Duplicate method {mid}; give a parameter trial a distinct preset ID')
        out['methods'].update({r['method']: other_plan['methods'][r['method']] for r in extra})
        sources.append((other, other_plan, extra))
    for source, _, selected in sources:
        selection = read_json(source/'selection.json') if (source/'selection.json').exists() else {}
        for row in selected:
            if row['caseId'] not in out['cases']:
                continue
            run_id = selection.get(row['id'])
            out['cells'].append(dict(caseId=row['caseId'], method=row['method'], inputSHA256=row['inputSHA256'],
                result=str(source/'runs'/run_id/'result.json') if run_id else None,
                root=str(source)))
            boundaries = row['caseDefinition'].get('boundaryTreatments', {})
            boundary = boundaries.get(row['preset']['solver']['scheme'], boundaries.get('default'))
            if boundary:
                out['cells'][-1]['boundaryTreatment'] = boundary
    return out
