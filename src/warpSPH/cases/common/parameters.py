"""Semantic descriptors reused by cases; values still come from their defaults."""
from ...runner.parameters import Parameter as P
from ...enumTypes import AdaptiveSupportScheme, ViscositySwitch

COMPRESSIBLE_PARAMETERS = {
    'gamma': P('number', 'Ideal-gas ratio of specific heats.', 'fluid', exclusiveMinimum=1),
    'viscositySwitch': P('string', 'Viscosity switch; null inherits the solver default.',
                         'solver.viscosity', schemes=('CompSPH', 'CRKSPH', 'Monaghan'), nullable=True, choices=tuple(m.name for m in ViscositySwitch)),
    'adaptiveSupportScheme': P('string', 'Adaptive smoothing-support prescription.', 'discretization', schemes=('CompSPH', 'CRKSPH', 'Monaghan', 'GSPH', 'InutsukaGSPH'), choices=tuple(m.name for m in AdaptiveSupportScheme)),
    'adaptiveSupportCorrections': P('boolean', 'Include adaptive-support (grad-H) corrections.', 'discretization', schemes=('CompSPH', 'CRKSPH', 'Monaghan', 'GSPH', 'InutsukaGSPH')),
}

WEAKLY_COMPRESSIBLE_PARAMETERS = {
    'rho0': P('number', 'Reference fluid density.', 'fluid', units='density', exclusiveMinimum=0),
    'targetDt': P('number', 'Target step used to derive sound speed when machTarget is null.',
                  'time', units='time', exclusiveMinimum=0, appliesWhen={'machTarget': None}),
    'band': P('integer', 'Extra particle layers outside each side of the domain.', 'domain', minimum=0),
    'freeSurface': P('boolean', 'Enable free-surface detection.', 'surface'),
    'inviscid': P('boolean', 'Use artificial viscosity instead of physical kinematic viscosity.', 'viscosity', schemes=('deltaSPH', 'sun2017DeltaSPH')),
    'nu': P('number', 'Physical kinematic viscosity.', 'viscosity', units='length^2/time', minimum=0,
            appliesWhenAny=({'inviscid': False}, {'solver.scheme': 'artificialCompressible'})),
    'alpha': P('number', 'Artificial-viscosity coefficient.', 'viscosity', minimum=0, schemes=('deltaSPH', 'sun2017DeltaSPH'),
               appliesWhen={'inviscid': True}),
    'backgroundPressure': P('number', 'Uniform EOS background-pressure offset.', 'fluid', units='pressure', schemes=('deltaSPH', 'sun2017DeltaSPH')),
    'markerSize': P('number', 'Legacy local plotting marker size.', 'appearance', exclusiveMinimum=0),
}

# These parameter sets match the existing shared case defaults. They deliberately
# do not extend COMPRESSIBLE_PARAMETERS: Sod has no independent rho0/markerSize.
SPH_SCHEMES = ('CompSPH', 'Monaghan', 'CRKSPH', 'GSPH', 'InutsukaGSPH')
COMPRESSIBLE_SCHEMES = (*SPH_SCHEMES, 'MFM', 'MFV')
COMPRESSIBLE_CASE_PARAMETERS = {
    **COMPRESSIBLE_PARAMETERS,
    'rho0': P('number', 'Reference density used in initialization/EOS setup.', 'fluid',
              units='density', exclusiveMinimum=0),
    'markerSize': P('number', 'Legacy local plotting marker size.', 'appearance', exclusiveMinimum=0),
}
RIEMANN_STATE_PARAMETERS = {
    f'{side}_{name}': P('number', f'{side.title()} initial {label}.', f'initial.{side}',
                       units=label, exclusiveMinimum=0 if name in ('rho', 'pressure') else None)
    for side in ('left', 'right')
    for name, label in (('rho', 'density'), ('pressure', 'pressure'), ('velocity', 'velocity'))
}

NOISE_PARAMETERS = {
    'octaves': P('integer', 'Number of noise octaves.', 'initial.noise', minimum=1),
    'lacunarity': P('integer', 'Frequency multiplier between octaves.', 'initial.noise', minimum=1),
    'persistence': P('number', 'Amplitude multiplier between octaves.', 'initial.noise', minimum=0),
    'baseFrequency': P('integer', 'First octave spatial frequency.', 'initial.noise', minimum=1),
    'tileable': P('boolean', 'Generate a tileable noise field.', 'initial.noise'),
    'kind': P('string', 'Noise generator.', 'initial.noise', choices=('perlin', 'simplex')),
    'seed': P('integer', 'Deterministic noise seed.', 'initial.noise', minimum=0),
}


def obstacleParameters(shapes):
    """SDF shape names come from the runtime SHAPE_PRESETS registry."""
    condition = {'obstacle': True}
    return {
        'obstacleShape': P('string', 'Obstacle SDF primitive.', 'obstacle', choices=tuple(shapes), appliesWhen=condition),
        'obstacleSize': P('number', 'Characteristic obstacle half-size.', 'obstacle', units='length', exclusiveMinimum=0, appliesWhen=condition),
        'obstacleAspect': P('number', 'Shape-specific obstacle aspect ratio.', 'obstacle', exclusiveMinimum=0, appliesWhen=condition),
        'obstacleRotation': P('number', 'Counter-clockwise obstacle rotation.', 'obstacle', units='degree', appliesWhen=condition),
        'obstacleOffset': P('array', 'Obstacle centre offset in the domain.', 'obstacle', minItems=2, maxItems=2,
                            items=P('number', 'Coordinate offset.', units='length'), appliesWhen=condition),
    }
