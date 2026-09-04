"""String -> enum parsers for CLI/config values."""

from ..enumTypes import *
from warpSPHIntegrators.integration import IntegrationSchemeType


def parseKernelFunctions(kernelName):
    for kernel in KernelFunctions:
        if kernel.name.lower() == kernelName.lower():
            return kernel
    raise ValueError(f"Invalid kernel name: {kernelName}. Valid options are: {[k.name for k in KernelFunctions]}")

def parseIntegrationScheme(integrationSchemeName):
    for scheme in IntegrationSchemeType:
        if scheme.name.lower() == integrationSchemeName.lower():
            return scheme
    raise ValueError(f"Invalid integration scheme name: {integrationSchemeName}. Valid options are: {[s.name for s in IntegrationSchemeType]}")

def parseViscositySwitch(viscositySwitchName):
    for switch in ViscositySwitch:
        if switch.name.lower() == viscositySwitchName.lower():
            return switch
    raise ValueError(f"Invalid viscosity switch name: {viscositySwitchName}. Valid options are: {[s.name for s in ViscositySwitch]}")

def parseAdaptiveSupportScheme(adaptiveSupportSchemeName):
    for scheme in AdaptiveSupportScheme:
        if scheme.name.lower() == adaptiveSupportSchemeName.lower():
            return scheme
    raise ValueError(f"Invalid adaptive support scheme name: {adaptiveSupportSchemeName}. Valid options are: {[s.name for s in AdaptiveSupportScheme]}")

def parseCompressibleSPHScheme(schemeName):
    for scheme in CompressibleSPHScheme:
        if scheme.name.lower() == schemeName.lower():
            return scheme
    raise ValueError(f"Invalid compressible SPH scheme name: {schemeName}. Valid options are: {[s.name for s in CompressibleSPHScheme]}")

def parseIncompressibleSPHScheme(schemeName):
    for scheme in IncompressibleSPHScheme:
        if scheme.name.lower() == schemeName.lower():
            return scheme
    raise ValueError(f"Invalid incompressible SPH scheme name: {schemeName}. Valid options are: {[s.name for s in IncompressibleSPHScheme]}")

def parseArtificialCompressibleSPHScheme(schemeName):
    for scheme in ArtificialCompressibleSPHScheme:
        if scheme.name.lower() == schemeName.lower():
            return scheme
    raise ValueError(f"Invalid artificial compressible SPH scheme name: {schemeName}. Valid options are: {[s.name for s in ArtificialCompressibleSPHScheme]}")

def parseWeaklyCompressibleSPHScheme(schemeName):
    for scheme in WeaklyCompressibleSPHScheme:
        if scheme.name.lower() == schemeName.lower():
            return scheme
    raise ValueError(f"Invalid weakly compressible SPH scheme name: {schemeName}. Valid options are: {[s.name for s in WeaklyCompressibleSPHScheme]}")


__all__ = [
    'parseKernelFunctions', 'parseIntegrationScheme', 'parseViscositySwitch',
    'parseAdaptiveSupportScheme', 'parseCompressibleSPHScheme',
    'parseIncompressibleSPHScheme', 'parseWeaklyCompressibleSPHScheme',
    'parseArtificialCompressibleSPHScheme',
]
