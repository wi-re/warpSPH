"""Saving a delta-plus configuration must preserve its selected numerical policies."""
import json
from dataclasses import fields
from enum import Enum
import pytest
from warpSPH.configurations.weaklyCompressible import (WeaklyCompressibleSPHConfig,
    Sun2017DeltaSPHConfig, weaklyCompressibleConfigToDict, dictToWeaklyCompressibleConfig)
from warpSPH.configurations.moduleConfigurations.shifting import ShiftProperties


@pytest.mark.parametrize('cls', [WeaklyCompressibleSPHConfig, Sun2017DeltaSPHConfig])
def test_deltaConfigurationRoundTripPreservesPresetAndExplicitPolicies(cls):
    config = cls()
    config.fluid.fixedSoundSpeed = 10.
    config.shiftProperties.correctdrhodt = True
    config.shiftProperties.correctdvdt = True
    config.shiftProperties.maxShiftVelocityFraction = .17
    config.shiftProperties.surfaceLambdaTaper = .03
    config.shiftProperties.implicitSolverThreshold = .02
    config.timeCentredContinuity = True
    config.loneDensityReset = True
    config.mdbcOneSidedHydrostatic = True
    config.mdbcNoPenShiftMode = 'off'
    config.mdbcDensityScheme = 'band'
    config.bandwith = 12.5
    restored = dictToWeaklyCompressibleConfig(json.loads(json.dumps(weaklyCompressibleConfigToDict(config))))
    assert restored.shiftProperties == config.shiftProperties
    assert json.loads(json.dumps(weaklyCompressibleConfigToDict(restored))) == json.loads(json.dumps(weaklyCompressibleConfigToDict(config)))
    assert restored.freezeDiffusionAcrossStages == config.freezeDiffusionAcrossStages
    for name in ('timeCentredContinuity', 'loneDensityReset', 'mdbcOneSidedHydrostatic',
                 'mdbcNoPenShiftMode', 'mdbcDensityScheme', 'bandwith'):
        assert getattr(restored, name) == getattr(config, name)


def test_allShiftingOptionsRoundTripIncludingEnumAndOptionalSettings():
    from warpSPH.configurations.moduleConfigurations.shifting import shiftPropertiesToDict, dictToShiftProperties
    config = ShiftProperties()
    for f in fields(config):
        value = getattr(config, f.name)
        if isinstance(value, Enum):
            others = [m for m in type(value) if m != value]
            if others:
                setattr(config, f.name, others[0])
        elif isinstance(value, bool):
            setattr(config, f.name, not value)
        elif value is None:
            setattr(config, f.name, .123)
        else:
            setattr(config, f.name, value + 1)
    encoded = json.loads(json.dumps(shiftPropertiesToDict(config)))
    assert set(encoded) == {f.name for f in fields(config)}
    assert dictToShiftProperties(encoded) == config
    assert dictToShiftProperties({}) == ShiftProperties()


def test_missingNewKeysRetainLegacyDecoderDefaults():
    config = WeaklyCompressibleSPHConfig()
    config.fluid.fixedSoundSpeed = 10.
    old = weaklyCompressibleConfigToDict(config)
    legacyShiftKeys = ('iterations', 'CFL', 'computeMach', 'maxC', 'active', 'scheme',
                      'projectionScheme', 'summationDensity', 'surfaceScaling', 'threshold', 'projectQuantities')
    old['shiftProperties'] = {k:v for k,v in old['shiftProperties'].items() if k in legacyShiftKeys}
    for name in ('timeCentredContinuity', 'loneDensityReset', 'mdbcOneSidedHydrostatic',
                 'mdbcNoPenShiftMode', 'mdbcDensityScheme', 'bandwith'):
        old.pop(name, None)
    restored = dictToWeaklyCompressibleConfig(old)
    assert restored.shiftProperties.sun2017Eq7Shift is False
    assert restored.shiftProperties.correctdrhodt is False
    assert restored.shiftProperties.correctdvdt is False
    assert restored.mdbcNoPenShiftMode == config.mdbcNoPenShiftMode
    assert restored.mdbcDensityScheme == config.mdbcDensityScheme
