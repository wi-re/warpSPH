"""Convert prescribed case pressure/density to PESPH's authoritative E or A."""

import torch

from ..eos import idealGasEOS
from ..adaptiveSupport import solveStageSupport
from .operators import computePressureSums
from .stage import refreshPESPHState
from .thermodynamics import requirePositive
from .chunked import checkPESPHChunks, pesphInputs
from warpSPHCore.radiusSearch import ChunkedAdjacency

# All integration/switch history required to resume, including scalar/optional
# history. Generic trajectory I/O handles optional fields without rebuilding A/E.
CHECKPOINT_FIELDS = (
    'supports', 'internalEnergies', 'totalEnergies', 'entropies', 'pressures', 'soundspeeds',
    'divergence', 'alpha0s', 'alphas', 'entropiesPrev', 'entropiesPrevTime', 'entropyRates',
    'divergencePrevStep', 'divergencePrevStepTime', 'wadsleyD', 'wadsleyXi', 'wadsleyVsig',
    'wadsleyDPrev', 'wadsleyDPrevTime', 'previousTimestep',
)


def initializePESPHState(system, config, schemeConfig):
    """Call once after all IC hooks; never use this on a resumed state.

    Preserve masses and form thermal targets from prescribed pressure at
    the geometric particle volume 1/n. PressureVolume uses the historical
    local entropy conversion; opt-in ThermalEnergy solves for A to reproduce
    each of these specific-energy targets, making smoothed pressure an outcome.
    A constant-pressure regular lattice remains balanced for unequal masses.
    """
    state = system.state
    checkPESPHChunks(system, config, schemeConfig)
    def validate():
        requirePositive(state.pressures, 'initial pressure')
        requirePositive(state.densities, 'initial density')
        schemeConfig.validate(config)
    pesphInputs(system.adjacency, validate)
    # Samplers can return a Gather/mean-support list. The Verlet cache does
    # not record that mode, so it cannot certify the union-support list our
    # pair force needs. Chunked caches record their mode and rebuild through
    # solveStageSupport when it changes; keep their policy and communication.
    if not isinstance(system.adjacency, ChunkedAdjacency):
        system.adjacency = None
    solveStageSupport(system, state, config, schemeConfig)
    sums = computePressureSums(state, config, state.masses, system.adjacency)
    rhoVolume = state.masses*sums.numberDensity
    A, u, _, _ = idealGasEOS(A=None, u=None, P=state.pressures,
                             rho=rhoVolume, gamma=schemeConfig.gamma)
    state.entropies = A
    if schemeConfig.entropyInitialization == 'ThermalEnergy':
        from .entropyInitialization import matchEntropyThermalEnergy
        matchEntropyThermalEnergy(state, config, schemeConfig, u, system.adjacency)
    state.totalEnergies = state.masses * (u+0.5*state.velocities.square().sum(-1))
    for name in ('alpha0s', 'alphas', 'divergence'):
        if getattr(state, name) is None:
            setattr(state, name, torch.zeros_like(state.masses) if name == 'divergence'
                    else torch.ones_like(state.masses))
    return refreshPESPHState(system, config, schemeConfig, solveSupport=False)
