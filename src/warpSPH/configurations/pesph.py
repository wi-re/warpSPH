"""PESPH-specific controls, reusing the compressible configuration contract."""

from dataclasses import dataclass, field
import math

from .compressibleConfig import CompressibleSPHConfig, compressibleConfigToDict, dictToCompressibleConfig
from .moduleConfigurations.viscositySwitchParameters import ViscositySwitchConfig
from ..enumTypes import AdaptiveSupportScheme, PESPHVariant, ViscositySwitch

__all__ = ['PESPHConfig', 'pesphConfigToDict', 'dictToPESPHConfig']


@dataclass
class PESPHConfig(CompressibleSPHConfig):
    variant: PESPHVariant = PESPHVariant.PressureEnergy
    dissipation: bool = True
    entropyHeating: str = 'Guarded'
    entropyInitialization: str = 'PressureVolume'
    viscositySwitchParams: ViscositySwitchConfig = field(default_factory=lambda: ViscositySwitchConfig(
        scheme=ViscositySwitch.CullenDehnen2010, correctVelocityGradient=True))
    adaptiveSupportScheme: AdaptiveSupportScheme = AdaptiveSupportScheme.NumberDensity
    adaptiveSupportIterations: int = 16
    adaptiveSupportThreshold: float = 1e-5
    schemeName: str = 'PESPH'

    def validate(self, config=None):
        super().validate(config)
        if not isinstance(self.variant, PESPHVariant):
            raise ValueError('PESPH variant must be a PESPHVariant')
        if self.entropyHeating not in ('Guarded', 'SwiftLocal', 'NonlocalPower'):
            raise ValueError('PESPH entropyHeating must be Guarded, SwiftLocal or NonlocalPower')
        if self.variant is PESPHVariant.PressureEntropy and self.dissipation and self.entropyHeating == 'Guarded':
            raise ValueError('PESPH pressure-entropy dissipation is not validated; set dissipation=False for adiabatic A')

        if self.entropyHeating == 'NonlocalPower':
            if self.variant is not PESPHVariant.PressureEntropy:
                raise ValueError('PESPH NonlocalPower is an experimental entropy source only')
            if self.diffusionParams.thermalConductivity != 0:
                raise ValueError('PESPH NonlocalPower requires conductivity off; heat transport is not validated')
        if self.entropyInitialization not in ('PressureVolume', 'ThermalEnergy'):
            raise ValueError('PESPH entropyInitialization must be PressureVolume or ThermalEnergy')
        if self.entropyInitialization == 'ThermalEnergy':
            if self.variant is not PESPHVariant.PressureEntropy or not 1 < self.gamma < 2:
                raise ValueError('PESPH ThermalEnergy initialization requires entropy SPH with 1 < gamma < 2')
        if not math.isfinite(self.gamma) or self.gamma <= 1:
            raise ValueError('PESPH gamma must be finite and greater than one')
        if self.materials is not None:
            raise ValueError('PESPH currently requires one ideal gas with a shared gamma')
        if self.backgroundPressure != 0:
            raise ValueError('PESPH does not support backgroundPressure')
        if self.adaptiveSupportScheme not in (AdaptiveSupportScheme.NumberDensity, AdaptiveSupportScheme.NoScheme):
            raise ValueError('PESPH requires NumberDensity support, or NoScheme for fixed supports')
        adaptive = self.adaptiveSupportScheme is AdaptiveSupportScheme.NumberDensity
        if self.adaptiveSupportCorrections != adaptive:
            raise ValueError('PESPH grad-h must be enabled for NumberDensity and disabled for fixed supports')
        if adaptive and (self.adaptiveSupportIterations < 1 or self.adaptiveSupportThreshold <= 0):
            raise ValueError('PESPH support solve requires positive iterations and convergence threshold')
        if self.supportSolveOnGrid or self.stepMode not in (None, 'eager'):
            raise ValueError('PESPH currently supports eager list-based support solves')
        if config is not None:
            correction = getattr(config, 'densityCorrection', False)
            if getattr(correction, 'enabled', correction):
                raise ValueError('PESPH does not support post-summation density corrections')
            if getattr(getattr(config, 'compute', None), 'execution', None) == 'graph':
                raise ValueError('PESPH CUDA graph execution is not implemented')


def pesphConfigToDict(config):
    return {**compressibleConfigToDict(config), 'variant': config.variant.name,
            'dissipation': config.dissipation, 'entropyHeating': config.entropyHeating,
            'entropyInitialization': config.entropyInitialization}


def dictToPESPHConfig(values):
    base = dictToCompressibleConfig(values)
    variant = values.get('variant', PESPHVariant.PressureEnergy.name)
    if isinstance(variant, str):
        variant = PESPHVariant[variant]
    config = PESPHConfig(**base.__dict__, variant=variant,
                         dissipation=values.get('dissipation', variant is PESPHVariant.PressureEnergy),
                         entropyHeating=values.get('entropyHeating', 'Guarded'),
                         entropyInitialization=values.get('entropyInitialization', 'PressureVolume'))
    config.validate()
    return config
