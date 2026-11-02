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


def _bundleForStoredState(scheme, variable=None):
    bundle = buildScheme(scheme)
    if scheme is CompressibleSPHScheme.PESPH:
        from ..enumTypes import PESPHVariant
        choices = {'energy': PESPHVariant.PressureEnergy, 'entropy': PESPHVariant.PressureEntropy}
        if variable not in choices:
            raise ValueError('PESPH checkpoint is missing a valid thermodynamicVariable')
        config = bundle.SimulationConfig(variant=choices[variable], dissipation=False)
        bundle = bundle.configured(config)
    return bundle


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
        bundle = _bundleForStoredState(schemeEnum, inFile['state'].attrs.get('thermodynamicVariable'))
        SimulationSystem = bundle.SimulationSystem
        SimulationState = bundle.SimulationState
        SimulationUpdate = bundle.SimulationUpdate

    adjacency = loadAdjacency(inFile['adjacency'], device) if 'adjacency' in inFile else None
    state = loadState(inFile['state'], device, SimulationState)
    uidLayout = inFile.attrs.get('particleLayoutVersion') == 1

    stageGroups = inFile['stages'] if 'stages' in inFile else None
    stages = []
    if stageGroups is not None:
        stageKeys = sorted(stageGroups, key=lambda k:int(stageGroups[k].attrs['index'])) if uidLayout else stageGroups.keys()
        for stageKey in stageKeys:
            stages.append(loadStage(stageGroups[stageKey], device, SimulationState, SimulationUpdate))

    extraData = {}
    for key, value in inFile.items():
        if key in ['adjacency', 'state', 'stages']:
            continue
        if key.startswith('dict_'):
            continue
        else:
            extraData[key] = torch.from_numpy(value[:]).to(device).to(hdfDtypeToTorchDtype(value.dtype)) if value.shape else torch.tensor(value[()], device=device, dtype=hdfDtypeToTorchDtype(value.dtype))

    layoutKeys = {'particleLayoutVersion','particleLayoutFormat','particleOrder','particlePrecision',
                  'particleDomain_min','particleDomain_max','particleDomain_periodic','particleUIDcounter'}
    for key, value in inFile.attrs.items():
        if key in ['scheme', 'time'] or (uidLayout and key in layoutKeys):
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

    system = SimulationSystem(state=state, adjacency=adjacency, t=t)
    if uidLayout and hasattr(system,'canonicalExport'):
        from .particleLayout import readLayoutDomain
        system.canonicalExport = True
        system.domain = readLayoutDomain(inFile.attrs,device)
    inFile.close()

    return system, stages, schemeEnum, extraData


def importConfigs(configPath, import_fn):
    # configPath = f'{path}/config.json'
    with open(configPath, 'r') as f:
        configDict = json.load(f)

    return dictToConfig(configDict['config']), import_fn(configDict['schemeConfig'])


def _toTensor(dataset, device):
    value = dataset[()] if dataset.shape == () else dataset[:]
    return torch.as_tensor(value, device=device, dtype=hdfDtypeToTorchDtype(dataset.dtype))


def _frameValue(group, key, device):
    if key in group:
        return _toTensor(group[key], device)
    if group.attrs.get(key) == 'None':
        return None
    raise ValueError(f'Trajectory field {group.name} is missing frame {key}')


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
    bundle = _bundleForStoredState(schemeEnum, trajectoryFile.attrs.get('thermodynamicVariable'))
    SimulationSystem = SimulationSystem or bundle.SimulationSystem
    SimulationState = SimulationState or bundle.SimulationState
    SimulationUpdate = SimulationUpdate or bundle.SimulationUpdate
    if schemeEnum is CompressibleSPHScheme.PESPH and SimulationState is not bundle.SimulationState:
        trajectoryFile.close()
        raise ValueError('PESPH trajectory state override does not match its stored variant')

    static = {
        'masses': _toTensor(trajectoryFile['combinedMasses'], device),
        'supports': _toTensor(trajectoryFile['combinedSupports'], device),
        'kinds': _toTensor(trajectoryFile['combinedKinds'], device),
        'materials': _toTensor(trajectoryFile['combinedMaterials'], device),
        'UIDs': _toTensor(trajectoryFile['combinedUIDs'], device),
    }
    uidLayout = trajectoryFile.attrs.get('particleLayoutVersion') == 1
    for field, dataset in (('emitters', 'combinedEmitters'), ('bodyIDs', 'combinedBodyIDs')):
        value = _toTensor(trajectoryFile[dataset], device) if dataset in trajectoryFile else None
        static[field] = value if uidLayout or value is None else value.to(torch.int32)

    storedFields = ('crkMassDensities', 'crkVolumes', 'boundaryProjectionBudget')
    if schemeEnum is CompressibleSPHScheme.PESPH:
        from ..modules.pesph.initialization import CHECKPOINT_FIELDS
        missing = set(CHECKPOINT_FIELDS)-set(trajectoryFile)
        if missing:
            trajectoryFile.close()
            raise ValueError(f'PESPH trajectory is missing restart fields: {sorted(missing)}')
        storedFields = (*storedFields, *CHECKPOINT_FIELDS)
    meta = dict(
        SimulationSystem=SimulationSystem,
        SimulationState=SimulationState,
        SimulationUpdate=SimulationUpdate,
        bundle=bundle,
        scheme=schemeEnum,
        static=static,
        frameKeys=sorted(trajectoryFile['positions'].keys(), key=lambda name: int(name.rsplit('_', 1)[-1])),
        extraFields=tuple(dict.fromkeys(name for name in (*extraFields, *storedFields) if name in trajectoryFile)),
    )
    if uidLayout:
        from .particleLayout import readLayoutDomain, validateFields
        validateFields(meta['extraFields'])
        meta['particleLayoutVersion'] = 1
        meta['UIDcounter'] = int(trajectoryFile.attrs['particleUIDcounter'])
        meta['domain'] = readLayoutDomain(trajectoryFile.attrs,device)
    return trajectoryFile, meta


def loadTrajectoryFrame(trajectoryFile, meta, frameIndex: int, schemeConfig=None, gamma: Optional[float] = None):
    """Reconstruct one full particle state at ``meta['frameKeys'][frameIndex]``.

    ``positions``/``velocities``/``densities`` and any of ``meta['extraFields']``
    present in the file come from that frame; ``masses``/``kinds``/
    ``materials``/``UIDs`` (and ``supports``, unless it was exported every
    frame) come from the static initial-condition snapshot `loadTrajectory`
    already read. ``pressures``/``soundspeeds``/``entropies`` are not stored --
    they are recomputed for legacy schemes from ``densities``/``internalEnergies`` via the same
    ``idealGasEOS`` call the case's own IC builder makes, using ``gamma`` (or
    ``schemeConfig.gamma`` when ``gamma`` is not given).
    PESPH instead restores its authoritative thermal fields and switch history
    directly, selecting the integrated-field layout from the stored variant.
    """
    key = meta['frameKeys'][frameIndex]
    static = meta['static']
    device = static['masses'].device

    positions = _toTensor(trajectoryFile['positions'][key], device)
    velocities = _toTensor(trajectoryFile['velocities'][key], device)
    densities = _toTensor(trajectoryFile['densities'][key], device)
    t = float(trajectoryFile['times'][key][0])

    frameFields = {name: _frameValue(trajectoryFile[name], key, device) for name in meta['extraFields']}
    supports = frameFields.get('supports', static['supports'])
    internalEnergies = frameFields.get('internalEnergies')

    stateKwargs = dict(
        positions=positions, velocities=velocities, densities=densities, supports=supports,
        masses=static['masses'], kinds=static['kinds'], materials=static['materials'], UIDs=static['UIDs'],
        emitters=static.get('emitters'), bodyIDs=static.get('bodyIDs'),
        UIDcounter=meta.get('UIDcounter',int(static['UIDs'].max().item()) + 1),
        # Matches the case IC builders (e.g. buildSod1D): a fresh state always
        # starts divergence at zero and alpha0s/alphas at one -- a resumed
        # state needs the same starting point, not the `None` a scheme
        # otherwise reads as "never computed" (which, for divergence, is only
        # handled as a one-off fallback; alpha0s/alphas have no such fallback).
        divergence=torch.zeros_like(densities),
        alpha0s=torch.ones_like(densities), alphas=torch.ones_like(densities),
    )
    for name in ('crkMassDensities', 'crkVolumes', 'boundaryProjectionBudget'):
        if name in frameFields:
            stateKwargs[name] = frameFields[name]

    if meta.get('particleLayoutVersion') == 1:
        # Axes were declared by the UID writer; restore all explicitly requested
        # fields, including global vectors and per-particle switch observations.
        stateKwargs.update(frameFields)
        ids = trajectoryFile['frameUIDs'][key][:]
        if not torch.equal(torch.as_tensor(ids,device=device),static['UIDs']):
            raise ValueError('trajectory frame UID association differs from its header')
    if meta['scheme'] is CompressibleSPHScheme.PESPH:
        if gamma is not None:
            raise ValueError('PESPH authoritative E/A cannot be loaded with a gamma override')
        if schemeConfig is not None:
            configured = meta['bundle'].configured(schemeConfig)
            if configured.SimulationState is not meta['SimulationState']:
                raise ValueError('PESPH trajectory variant does not match schemeConfig')
        stateKwargs.update(frameFields)
        state = meta['SimulationState'](**stateKwargs)
        return meta['SimulationSystem'](state=state, adjacency=None, t=t), t

    if internalEnergies is not None:
        from ..modules.eos import evaluateEOS
        if getattr(schemeConfig, 'materials', None) is not None and gamma is not None:
            raise ValueError('gamma override is ambiguous with a material EOS table')
        gammaValue = gamma if gamma is not None else schemeConfig.gamma
        if getattr(schemeConfig, 'materials', None) is not None:
            ids = stateKwargs['materials']
            if not torch.equal(ids, ids.to(torch.int32).to(ids.dtype)):
                raise ValueError('materials: trajectory IDs must be integral')
            stateKwargs['materials'] = ids.to(torch.int32)
            A_, u_, P_, c_s = evaluateEOS(densities, stateKwargs['materials'], schemeConfig, u=internalEnergies)
        else:
            A_, u_, P_, c_s = idealGasEOS(A=None, u=internalEnergies, P=None, rho=densities, gamma=gammaValue)
        # Same formula the case's own IC builder (e.g. buildSod1D) uses --
        # totalEnergies isn't in extraFields (it's a pure function of
        # velocities/internalEnergies/masses, all already reconstructed
        # above), but it is still a required field for a scheme's `finalize`
        # step, so it has to be filled in here rather than left None.
        kineticEnergy = torch.linalg.norm(velocities, dim=-1) ** 2 / 2
        masses = stateKwargs['masses'] if meta.get('particleLayoutVersion') == 1 else static['masses']
        totalEnergies = (u_ + kineticEnergy) * masses
        stateKwargs.update(internalEnergies=u_, pressures=P_, soundspeeds=c_s, entropies=A_,
                          totalEnergies=totalEnergies)

    state = meta['SimulationState'](**stateKwargs)
    if getattr(schemeConfig, 'materials', None) is not None:
        from ..modules.eos import validateMaterialState
        validateMaterialState(state, schemeConfig)
    system = meta['SimulationSystem'](state=state, adjacency=None, t=t)
    if meta.get('particleLayoutVersion') == 1 and hasattr(system,'canonicalExport'):
        system.canonicalExport, system.domain = True,meta['domain']
    return system, t


__all__ = [
    'schemeNameToSimulationScheme', 'importSimulationSystem', 'importConfigs',
    'loadTrajectory', 'loadTrajectoryFrame',
]
