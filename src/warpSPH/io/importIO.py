"""Reading a run back from disk: the inverse of :mod:`.export`."""

import json
import os
from typing import Any, Optional

import h5py
import torch

from ..configurations import dictToConfig
from ..enumTypes import (ArtificialCompressibleSPHScheme, CompressibleSPHScheme,
                         WeaklyCompressibleSPHScheme, IncompressibleSPHScheme,
                         WaveEquationScheme)
from ..modules import idealGasEOS
from ..schemes import buildScheme
from .hdf5 import loadAdjacency, loadState, loadStage, hdfDtypeToTorchDtype


def schemeNameToSimulationScheme(name: str) -> CompressibleSPHScheme:
    for scheme in CompressibleSPHScheme:
        # print(f'Comparing {scheme.name.lower()} to {name.lower()}')
        if scheme.name.lower() == name.lower():
            return scheme
    for scheme in WeaklyCompressibleSPHScheme:
        if scheme.name.lower() == name.lower():
            return scheme
    # Mirrors `schemeAttribute`: whatever can be written must be readable back.
    for scheme in IncompressibleSPHScheme:
        if scheme.name.lower() == name.lower():
            return scheme
    for scheme in ArtificialCompressibleSPHScheme:
        if scheme.name.lower() == name.lower():
            return scheme
    for scheme in WaveEquationScheme:
        if scheme.name.lower() == name.lower():
            return scheme
    raise ValueError(f'Unsupported scheme name: {name}')


def importSimulationSystem(
    importPath,
    device,
    dtype,
    SimulationSystem : Optional[Any] = None,
    SimulationState: Optional[Any] = None,
    SimulationUpdate: Optional[Any] = None,
):
    inFile = h5py.File(importPath, 'r')

    t = inFile.attrs['time']
    scheme = inFile.attrs['scheme']
    schemeEnum = schemeNameToSimulationScheme(scheme)
    if SimulationSystem is None or SimulationState is None or SimulationUpdate is None:
        bundle = buildScheme(schemeEnum)
        SimulationSystem = bundle.SimulationSystem
        SimulationState = bundle.SimulationState
        SimulationUpdate = bundle.SimulationUpdate

    adjacency = loadAdjacency(inFile['adjacency'], device) if 'adjacency' in inFile else None
    state = loadState(inFile['state'], device, SimulationState)

    stageGroups = inFile['stages'] if 'stages' in inFile else None
    stages = []
    if stageGroups is not None:
        for stageKey in stageGroups.keys():
            stages.append(loadStage(stageGroups[stageKey], device, SimulationState, SimulationUpdate))

    extraData = {}
    for key, value in inFile.items():
        if key in ['adjacency', 'state', 'stages']:
            continue
        if key.startswith('dict_'):
            continue
        else:
            extraData[key] = torch.from_numpy(value[:]).to(device).to(hdfDtypeToTorchDtype(value.dtype)) if value.shape else torch.tensor(value[()], device=device, dtype=hdfDtypeToTorchDtype(value.dtype))

    for key, value in inFile.attrs.items():
        if key in ['scheme', 'time']:
            continue
        if value == 'dict':
            dictKey = key[len('dict_'):] if key.startswith('dict_') else key
            extraData[dictKey] = {}
            dictGroup = inFile[f'dict_{dictKey}'] if f'dict_{dictKey}' in inFile else inFile[dictKey]
            for subKey, subValue in dictGroup.items():
                if isinstance(subValue, h5py.Dataset):
                    extraData[dictKey][subKey] = torch.from_numpy(subValue[:]).to(device).to(hdfDtypeToTorchDtype(subValue.dtype)) if subValue.shape else torch.tensor(subValue[()], device=device, dtype=hdfDtypeToTorchDtype(subValue.dtype))
                else:
                    extraData[dictKey][subKey] = subValue
        elif value == 'list':
            listKey = key[len('dict_'):] if key.startswith('dict_') else key
            extraData[listKey] = []
            listGroup = inFile[f'dict_{listKey}'] if f'dict_{listKey}' in inFile else inFile[listKey]
            for subIndex, subValue in listGroup.items():
                if isinstance(subValue, h5py.Dataset):
                    extraData[listKey].append(torch.from_numpy(subValue[:]).to(device).to(hdfDtypeToTorchDtype(subValue.dtype)) if subValue.shape else torch.tensor(subValue[()], device=device, dtype=hdfDtypeToTorchDtype(subValue.dtype)))
                elif isinstance(subValue, h5py.Group):
                    subDict = {}
                    for subSubKey, subSubValue in subValue.items():
                        if isinstance(subSubValue, h5py.Dataset):
                            subDict[subSubKey] = torch.from_numpy(subSubValue[:]).to(device).to(hdfDtypeToTorchDtype(subSubValue.dtype)) if subSubValue.shape else torch.tensor(subSubValue[()], device=device, dtype=hdfDtypeToTorchDtype(subSubValue.dtype))
                        else:
                            subDict[subSubKey] = subSubValue
                    extraData[listKey].append(subDict)
                else:
                    extraData[listKey].append(subValue)
        else:
            extraData[key] = value

    inFile.close()

    return SimulationSystem(state=state, adjacency=adjacency, t = t), stages, schemeEnum, extraData


def importConfigs(configPath, import_fn):
    # configPath = f'{path}/config.json'
    with open(configPath, 'r') as f:
        configDict = json.load(f)

    return dictToConfig(configDict['config']), import_fn(configDict['schemeConfig'])


def _toTensor(dataset, device):
    return torch.from_numpy(dataset[:]).to(device).to(hdfDtypeToTorchDtype(dataset.dtype))


def loadTrajectory(
    exportPath,
    device,
    SimulationSystem: Optional[Any] = None,
    SimulationState: Optional[Any] = None,
    SimulationUpdate: Optional[Any] = None,
    extraFields=(),
):
    """Open a ``storeMode='trajectory'`` export (the inverse of ``writeInitialData``/``writeFrame``).

    Returns ``(trajectoryFile, meta)``. ``trajectoryFile`` is the open
    ``h5py.File`` -- keep it open and pass it to :func:`loadTrajectoryFrame` to
    materialise individual frames, and to append further frames when resuming
    (matching how the file was written in the first place: one growing
    ``trajectory.h5``, not one file per frame). ``meta['static']`` holds the
    fields written once at t=0 (``masses``/``kinds``/``materials``/``UIDs``,
    plus ``supports`` as the fallback for schemes that don't re-export it every
    frame); ``meta['frameKeys']`` is the sorted list of per-frame keys actually
    present in ``positions``/``velocities``/``densities``/``times`` and any of
    `extraFields` that the file was written with (a case may have requested
    fewer than `extraFields` lists, e.g. an older export).
    """
    trajectoryFile = h5py.File(os.path.join(exportPath, 'trajectory.h5'), 'r')

    schemeName = trajectoryFile.attrs['scheme']
    schemeEnum = schemeNameToSimulationScheme(schemeName)
    # Always built, even when the caller overrides the three classes above --
    # `bundle.stepFunction`/`exportFunction` are what a resume loop needs to
    # keep stepping, and building it is cheap class resolution, not a kernel
    # compile.
    bundle = buildScheme(schemeEnum)
    SimulationSystem = SimulationSystem or bundle.SimulationSystem
    SimulationState = SimulationState or bundle.SimulationState
    SimulationUpdate = SimulationUpdate or bundle.SimulationUpdate

    static = {
        'masses': _toTensor(trajectoryFile['combinedMasses'], device),
        'supports': _toTensor(trajectoryFile['combinedSupports'], device),
        'kinds': _toTensor(trajectoryFile['combinedKinds'], device),
        'materials': _toTensor(trajectoryFile['combinedMaterials'], device),
        'UIDs': _toTensor(trajectoryFile['combinedUIDs'], device),
    }

    meta = dict(
        SimulationSystem=SimulationSystem,
        SimulationState=SimulationState,
        SimulationUpdate=SimulationUpdate,
        bundle=bundle,
        scheme=schemeEnum,
        static=static,
        frameKeys=sorted(trajectoryFile['positions'].keys()),
        extraFields=tuple(name for name in extraFields if name in trajectoryFile),
    )
    return trajectoryFile, meta


def loadTrajectoryFrame(trajectoryFile, meta, frameIndex: int, schemeConfig=None, gamma: Optional[float] = None):
    """Reconstruct one full particle state at ``meta['frameKeys'][frameIndex]``.

    ``positions``/``velocities``/``densities`` and any of ``meta['extraFields']``
    present in the file come from that frame; ``masses``/``kinds``/
    ``materials``/``UIDs`` (and ``supports``, unless it was exported every
    frame) come from the static initial-condition snapshot `loadTrajectory`
    already read. ``pressures``/``soundspeeds``/``entropies`` are not stored --
    they are recomputed from ``densities``/``internalEnergies`` via the same
    ``idealGasEOS`` call the case's own IC builder makes, using ``gamma`` (or
    ``schemeConfig.gamma`` when ``gamma`` is not given).
    """
    key = meta['frameKeys'][frameIndex]
    static = meta['static']
    device = static['masses'].device

    positions = _toTensor(trajectoryFile['positions'][key], device)
    velocities = _toTensor(trajectoryFile['velocities'][key], device)
    densities = _toTensor(trajectoryFile['densities'][key], device)
    t = float(trajectoryFile['times'][key][0])

    frameFields = {name: _toTensor(trajectoryFile[name][key], device) for name in meta['extraFields']}
    supports = frameFields.get('supports', static['supports'])
    internalEnergies = frameFields.get('internalEnergies')

    stateKwargs = dict(
        positions=positions, velocities=velocities, densities=densities, supports=supports,
        masses=static['masses'], kinds=static['kinds'], materials=static['materials'], UIDs=static['UIDs'],
        UIDcounter=int(static['UIDs'].max().item()) + 1,
        # Matches the case IC builders (e.g. buildSod1D): a fresh state always
        # starts divergence at zero and alpha0s/alphas at one -- a resumed
        # state needs the same starting point, not the `None` a scheme
        # otherwise reads as "never computed" (which, for divergence, is only
        # handled as a one-off fallback; alpha0s/alphas have no such fallback).
        divergence=torch.zeros_like(densities),
        alpha0s=torch.ones_like(densities), alphas=torch.ones_like(densities),
    )

    if internalEnergies is not None:
        gammaValue = gamma if gamma is not None else schemeConfig.gamma
        A_, u_, P_, c_s = idealGasEOS(A=None, u=internalEnergies, P=None, rho=densities, gamma=gammaValue)
        # Same formula the case's own IC builder (e.g. buildSod1D) uses --
        # totalEnergies isn't in extraFields (it's a pure function of
        # velocities/internalEnergies/masses, all already reconstructed
        # above), but it is still a required field for a scheme's `finalize`
        # step, so it has to be filled in here rather than left None.
        kineticEnergy = torch.linalg.norm(velocities, dim=-1) ** 2 / 2
        totalEnergies = (u_ + kineticEnergy) * static['masses']
        stateKwargs.update(internalEnergies=u_, pressures=P_, soundspeeds=c_s, entropies=A_,
                          totalEnergies=totalEnergies)

    state = meta['SimulationState'](**stateKwargs)
    system = meta['SimulationSystem'](state=state, adjacency=None, t=t)
    return system, t


__all__ = [
    'schemeNameToSimulationScheme', 'importSimulationSystem', 'importConfigs',
    'loadTrajectory', 'loadTrajectoryFrame',
]
