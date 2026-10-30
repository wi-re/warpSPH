"""Declared cumulative Lobovsky controls. Pure data; no simulation imports."""
from dataclasses import asdict,dataclass,replace


@dataclass(frozen=True)
class Options:
    static_geometry: bool=False
    accepted_fields: bool=False
    source_shift: bool=False
    pressure_remap: bool=False
    velocity_remap: bool=False
    warmup: bool=False
    source_operators: bool=False
    last_stage_metric: bool=False
    pressure_gate: bool=False
    plateau_trim: bool=False


CHANGES=(
    ('baseline',None,'Previous consistent-closure solver with Michel shift'),
    ('static-geometry','static_geometry','Preserve stationary wall/ghost positions and offsets'),
    ('accepted-fields','accepted_fields','Direct accepted p/v handoff, avoiding derivative roundtrip'),
    ('source-shift','source_shift','Compiled-source shift geometry, surface inputs and required BDF history'),
    ('pressure-remap','pressure_remap','Compiled-source pressure remap and pressure safeguards'),
    ('velocity-remap','velocity_remap','Compiled-source diagonal velocity remap'),
    ('warmup','warmup','First two half pseudo-time clocks with consistent beta and destination RK factors'),
    ('source-operators','source_operators','Coordinated source wall/moment/surface/residual/frozen-cell operator block'),
    ('last-stage-metric','last_stage_metric','Source last-RK-stage residual and double-precision norm'),
    ('pressure-gate','pressure_gate','Require pressure residual below epsilonV+1 as well'),
    ('plateau-trim','plateau_trim','Source plateau/trim policy with stable fluid-ID selection; current endpoint'))


def rows():
    options=Options();result=[]
    for index,(name,key,description) in enumerate(CHANGES):
        if key:options=replace(options,**{key:True})
        result.append(dict(id=f'{index:02d}-{name}',index=index,change=key,
            description=description,options=asdict(options),reuse=index in (0,10),
            coupledGroup=index in (3,6,7)))
    return result


def select(identifier):
    for row in rows():
        if identifier in (row['id'],str(row['index'])):return row
    raise ValueError('Unknown ablation row: '+identifier)


def validate(options):
    values=list(asdict(options).values())
    if values!=sorted(values,reverse=True):
        raise ValueError('This campaign requires a cumulative prefix of declared changes')
    if options.pressure_remap and not options.source_shift:raise ValueError('Pressure remap requires source shift')
    if options.velocity_remap and not options.source_shift:raise ValueError('Velocity remap requires source shift')


def notable(previous,current):
    """Descriptive screening thresholds, not confidence tests or model knobs."""
    flags=[]
    if current.get('physicalFailure'):flags.append('physical run stopped')
    for name in ('P1RmsPa','P4RmsPa'):
        a,b=previous.get(name),current.get(name)
        if a is not None and b is not None and b>2*a and b-a>1000:
            flags.append(name+': >2x and >1kPa absolute increase')
    a,b=previous.get('caps'),current.get('caps')
    if a is not None and b is not None and b>2*a and b-a>=10:flags.append('caps: >2x and at least10 extra')
    return flags
