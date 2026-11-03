"""Writing a run to disk: per-frame HDF5 export, ``config.json``, and run-folder layout."""

import json
import os
from contextlib import contextmanager
from uuid import uuid4
from typing import Any

import h5py
import numpy as np
import torch

from ..configurations import *
from ..enumTypes import (ArtificialCompressibleSPHScheme, CompressibleSPHScheme,
                         WeaklyCompressibleSPHScheme, IncompressibleSPHScheme,
                         WaveEquationScheme)
from ..utils import getCurrentTimestamp
from .hdf5 import dumpAdjacency, dumpState, dumpStage, copy_dict_to_h5, _encode_callable


def schemeAttribute(scheme) -> str:
    """The scheme's name, as an HDF5 attribute can store it.

    Every scheme enum has to be named here: an enum member that falls through
    lands in `attrs` as a Python object, and h5py rejects it with "Object
    dtype has no native HDF5 equivalent" (and `json.dump` rejects it outright
    for `config.json`). `IncompressibleSPHScheme` was missing from two of the
    three sites, so `--store` had never worked for the incompressible family;
    `WaveEquationScheme` was missing here entirely, breaking `--plot`/`--store`
    for the wave case with the same `TypeError`.
    """
    return scheme.name if isinstance(
        scheme, (CompressibleSPHScheme, WeaklyCompressibleSPHScheme,
                 IncompressibleSPHScheme, ArtificialCompressibleSPHScheme,
                 WaveEquationScheme)) else scheme


def validateSortedExtraData(extraData):
    """Scalars have no particle axis; arrays/nested data require an explicit adapter."""
    reserved = {'scheme','time','integration','nativeMemory','particleLayoutVersion','particleLayoutFormat','particleOrder','particlePrecision',
                'particleDomain_min','particleDomain_max','particleDomain_periodic','particleUIDcounter'}
    if reserved.intersection(extraData or {}):
        raise ValueError('extraData cannot replace reserved layout metadata')
    if any(value is not None and not isinstance(value, (str, bool, int, float, np.number, np.bool_, np.str_))
               and not (isinstance(value,torch.Tensor) and value.ndim==0)
           for value in (extraData or {}).values()):
        raise ValueError('sorted frame extraData needs explicit axes; use canonical_view and transform extras first')


@contextmanager
def _stateFile(path,*,atomic=False):
    """A failed numerical checkpoint must not replace an earlier complete one."""
    temporary=f'{path}.tmp-{uuid4().hex}' if atomic else path
    try:
        with h5py.File(temporary,'w') as outFile:
            yield outFile
        if atomic:
            os.replace(temporary,path)
    finally:
        if atomic and os.path.exists(temporary):
            os.unlink(temporary)


def exportSimulationSystem(
    exportPath,
    tag,
    scheme: CompressibleSPHScheme,
    system: Any,
    exportAdjacency = False,
    stages = None,
    exportStagesAdjacency = True,
    extraData = None,
    continuation = None,
    integrator = None,
    config = None,
    schemeConfig = None,
    last_dt = None,
    next_dt = None,
):
    from warpSPHCore.radiusSearch import ChunkedAdjacency
    if isinstance(system.adjacency, ChunkedAdjacency):
        raise NotImplementedError('Chunked neighbor checkpoint export is not implemented; pair data must remain chunk-local.')
    if continuation is not None:
        from .continuation import validateFrame,signature
        validateFrame(continuation,system)
        signature(config,schemeConfig)
    uidLayout = getattr(system, 'spatialContext', None) is not None or getattr(system, 'canonicalExport', False)
    from ..systems.compSPH import CompSPHSystem
    nativeMemory=uidLayout and type(system) is not CompSPHSystem and config is not None
    nativeSource=system
    if uidLayout:
        validateSortedExtraData(extraData)
        # Optional private sorting is invisible to the particle/stage file order.
        from ..evaluation import canonical_view
        system, stages = canonical_view(system, stages, include_adjacency=exportAdjacency,
                                         include_stage_adjacency=exportStagesAdjacency)
    outFolderPath = f'{exportPath}/trajectory/'
    os.makedirs(outFolderPath, exist_ok=True)
    with _stateFile(f'{exportPath}/trajectory/{tag}.h5',atomic=continuation is not None or nativeMemory) as outFile:

        outFile.attrs['scheme'] = schemeAttribute(scheme)
        if uidLayout:
            from .particleLayout import writeLayoutMetadata
            writeLayoutMetadata(outFile, system, kind='state')
        outFile.attrs['time'] = system.t if isinstance(system.t, float) else system.t.cpu().item()

        if system.adjacency is not None:
            adjacencyGroup = outFile.create_group('adjacency')
            dumpAdjacency(system.adjacency, adjacencyGroup)

        stateGroup = outFile.create_group('state')
        dumpState(system.state, stateGroup)

        if continuation is not None:
            from .continuation import writeContinuation
            writeContinuation(outFile.create_group('integration'),continuation,
                integrator=integrator,config=config,schemeConfig=schemeConfig,
                last_dt=last_dt,next_dt=config.dt if next_dt is None else next_dt)
        if nativeMemory:
            from .nativeMemory import writeNativeMemory
            writeNativeMemory(outFile.create_group('nativeMemory'),nativeSource,config=config,
                schemeConfig=schemeConfig,integrator=integrator,next_dt=config.dt if next_dt is None else next_dt)

        if stages is not None:
            stageGroups = outFile.create_group('stages')
            stageGroupList = []
            for stageIndex, stage in enumerate(stages):
                stageGroupList.append(stageGroups.create_group(f'stage_{stageIndex}'))
                # dumpState(stage, stageGroup)
                dumpStage(stages[stageIndex], stageGroupList, stageIndex, type(system.state), exportStagesAdjacency)

        for key, value in extraData.items() if extraData is not None else {}:
            if isinstance(value, dict):
                extraGroup = outFile.create_group(f'dict_{key}')
                for subKey, subValue in value.items():
                    # print(f'Exporting extra data key {key}[{subKey}] with value {subValue} of type {type(subValue)}')
                    if isinstance(subValue, torch.Tensor):
                        extraGroup.create_dataset(subKey, data=subValue.cpu().numpy())
                    else:
                        extraGroup.attrs[subKey] = subValue
                outFile.attrs[key] = 'dict'
            elif isinstance(value, list):
                extraGroup = outFile.create_group(f'dict_{key}')
                for subIndex, subValue in enumerate(value):
                    if isinstance(subValue, torch.Tensor):
                        extraGroup.create_dataset(f'{subIndex}', data=subValue.cpu().numpy())
                    elif isinstance(subValue, dict):
                        subGroup = extraGroup.create_group(f'{subIndex}')
                        for subSubKey, subSubValue in subValue.items():
                            if isinstance(subSubValue, torch.Tensor):
                                subGroup.create_dataset(subSubKey, data=subSubValue.cpu().numpy())
                            else:
                                # print(f'Exporting extra data key {key}[{subIndex}][{subSubKey}] with value {subSubValue} of type {type(subSubValue)}')
                                subGroup.attrs[subSubKey] = subSubValue
                    else:
                        # print(f'Exporting extra data key {key}[{subIndex}] with value {subValue} of type {type(subValue)}')
                        extraGroup.attrs[f'{subIndex}'] = subValue
                outFile.attrs[key] = 'list'
            elif isinstance(value, torch.Tensor):
                outFile.create_dataset(key, data=value.cpu().numpy())
            elif value is None:
                # An unset optional case parameter. `None` maps to numpy object
                # dtype, which h5py rejects ("no native HDF5 equivalent"), and that
                # aborted the *whole* state write -- so any case carrying an unset
                # optional in `extraData` could not be checkpointed at all
                # (`sloshingTank`'s `noPenShift`). Record the key as an explicit
                # sentinel instead of dropping it, so a reader can tell "unset"
                # apart from "absent".
                outFile.attrs[key] = 'None'
            else:
                # print(f'Exporting extra data key {key} with value {value} of type {type(value)}')
                outFile.attrs[key] = value




def exportDirName(caseName, timestamp):
    """Directory name for one run: ``{caseName}_{timestamp}``."""
    return f'{caseName}_{timestamp}'


def resolveExportRoot(exportRoot=None):
    """Parent directory that holds per-run folders."""
    if exportRoot is None:
        exportRoot = os.environ.get('WARPSPH_EXPORT_ROOT', 'export')
    return exportRoot


def findExportRuns(caseName, exportRoot=None):
    """Every run directory for ``caseName``, oldest first.

    Relies on the timestamp format sorting lexicographically, so a plain sort
    is chronological.
    """
    root = resolveExportRoot(exportRoot)
    if not os.path.isdir(root):
        return []
    prefix = f'{caseName}_'
    runs = [
        os.path.join(root, entry)
        for entry in os.listdir(root)
        if entry.startswith(prefix) and os.path.isdir(os.path.join(root, entry))
    ]
    return sorted(runs)


def latestExportPath(caseName, exportRoot=None):
    """Most recent run directory for ``caseName``.

    Use this to pick up a run whose folder name you do not know -- resuming,
    post-processing, plotting. Falls back to an untimestamped
    ``{root}/{caseName}`` so trees written before run folders were timestamped
    still resolve.

    Raises ``FileNotFoundError`` when nothing matches.
    """
    runs = findExportRuns(caseName, exportRoot)
    if runs:
        return runs[-1]
    legacy = os.path.join(resolveExportRoot(exportRoot), caseName)
    if os.path.isdir(legacy):
        return legacy
    raise FileNotFoundError(
        f'No export directory for case {caseName!r} under '
        f'{resolveExportRoot(exportRoot)!r}'
    )


def prepExport(caseName, config, schemeConfig, scheme, export_fn, exportRoot=None,
               timestamped=None):
    """Write ``config.json`` for a run and return its output directory.

    ``exportRoot`` selects the parent directory for ``caseName``. It defaults to
    the ``WARPSPH_EXPORT_ROOT`` environment variable, falling back to ``export``
    relative to the CWD -- the historical behaviour. Overriding it lets parallel
    sweeps write to separate trees instead of colliding on ``export/{caseName}``.

    Run folders are named ``{caseName}_{YYYY-MM-DD_HH-MM-SS}`` so repeated runs
    of the same case accumulate side by side instead of overwriting each other.
    Set ``timestamped=False`` (or ``WARPSPH_EXPORT_TIMESTAMP=0``) to get the old
    ``{exportRoot}/{caseName}`` behaviour. Use :func:`latestExportPath` to find
    the newest run of a case afterwards.
    """
    currentTime = getCurrentTimestamp()

    cfg = configurationToDict(config)
    schemeCfg = export_fn(schemeConfig)

    exportDict = {
        'scheme': schemeAttribute(scheme),
        'config': cfg,
        'schemeConfig': schemeCfg,
        'timestamp': currentTime,
    }

    exportRoot = resolveExportRoot(exportRoot)

    if timestamped is None:
        timestamped = os.environ.get('WARPSPH_EXPORT_TIMESTAMP', '1') not in ('0', 'false', 'False')

    if timestamped:
        exportPath = os.path.join(exportRoot, exportDirName(caseName, currentTime))
        # Two runs launched inside the same second must not share a folder.
        if os.path.exists(exportPath):
            suffix = 1
            while os.path.exists(f'{exportPath}-{suffix}'):
                suffix += 1
            exportPath = f'{exportPath}-{suffix}'
    else:
        exportPath = os.path.join(exportRoot, caseName)

    os.makedirs(exportPath, exist_ok=True)
    configPath = os.path.join(exportPath, 'config.json')

    with open(configPath, 'w') as f:
        json.dump(exportDict, f, indent=4)

    return exportPath


def writeInitialData(
    exportPath: str,
    outFile: h5py.File,
    scheme: Any,
    config: SimulationConfig,
    schemeConfig: Any,
    args: Any,
    runningState: Any,
    extraData: dict | None = None,
    extraFields: tuple = (),
):
    uidLayout = getattr(runningState, 'spatialContext', None) is not None or getattr(runningState, 'canonicalExport', False)
    if uidLayout:
        from .particleLayout import initialView
        runningState = initialView(runningState,extraData,extraFields,outFile)
    if extraData is None:
        extraData = {}

    outFile.attrs['scheme'] = schemeAttribute(scheme)
    if hasattr(runningState.state, 'thermodynamicVariable'):
        outFile.attrs['thermodynamicVariable'] = runningState.state.thermodynamicVariable
    outFile.attrs['time'] = runningState.t if isinstance(runningState.t, float) else runningState.t.cpu().item()
    outFile.create_group('states')
    outFile.create_group('stages')

    uniqueParticles = True

    def _hostValue(value):
        # a case may stash a device tensor in its extra data (tgv-wc: a sound speed); h5py cannot take one
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value

    for key, value in extraData.items():
        if value is None:
            continue   # an unset case param (sod: viscositySwitch=None); HDF5 has no None attribute
        if not isinstance(value, dict):
            outFile.attrs[key] = _hostValue(value)
        else:
            for subkey, subvalue in value.items():
                if subvalue is None:
                    continue
                if isinstance(subvalue, (list, np.ndarray)):
                    outFile.attrs[f'{key}_{subkey}'] = np.array(subvalue)
                else:
                    outFile.attrs[f'{key}_{subkey}'] = _hostValue(subvalue)

    outFile.attrs['uniqueParticles'] = uniqueParticles
    outFile.attrs['exportInterval'] = args.exportInterval
    outFile.attrs['original_dt'] = config.dt if isinstance(config.dt, float) else config.dt.cpu().item()
    outFile.attrs['exportRatio'] = args.exportInterval / (config.dt if isinstance(config.dt, float) else config.dt.cpu().item())

    outFile.create_group('config')
    loadedConfig = json.load(open(f'{exportPath}/config.json', 'r'))
    copy_dict_to_h5(outFile['config'], loadedConfig)

    rigidBodyGroup = outFile.create_group('rigidBodies')
    bodies=getattr(schemeConfig, 'rigidBodies', []) or []
    if uidLayout and bodies:
        from ..evaluation.nativeLayout import bodies_view
        bodies=bodies_view(bodies,runningState.state)
    for r, rigidBody in enumerate(bodies):
        cGroup = rigidBodyGroup.create_group(f'rigidBody_{r:02d}')
        cGroup.attrs['centerOfMass'] = rigidBody.centerOfMass.cpu().numpy()
        cGroup.attrs['orientation'] = rigidBody.orientation.cpu().numpy()
        cGroup.attrs['angularVelocity'] = rigidBody.angularVelocity.cpu().numpy()
        cGroup.attrs['linearVelocity'] = rigidBody.linearVelocity.cpu().numpy()
        cGroup.attrs['mass'] = rigidBody.mass.cpu().numpy()
        cGroup.attrs['inertia'] = rigidBody.inertia.cpu().numpy()

        cGroup.create_dataset('particlePositions', data=rigidBody.particlePositions.cpu().numpy())
        cGroup.create_dataset('particleVelocities', data=rigidBody.particleVelocities.cpu().numpy())
        cGroup.create_dataset('particleMasses', data=rigidBody.particleMasses.cpu().numpy())
        cGroup.create_dataset('particleUIDs', data=rigidBody.particleUIDs.cpu().numpy())
        cGroup.create_dataset('particleIndices', data=rigidBody.particleIndices.cpu().numpy())
        cGroup.create_dataset('particleBoundaryDistances', data=rigidBody.particleBoundaryDistances.cpu().numpy())
        cGroup.create_dataset('particleBoundaryNormals', data=rigidBody.particleBoundaryNormals.cpu().numpy())

        # the ghost datasets reused the body's names, so any rigid body with ghosts failed with "name already exists"
        cGroup.create_dataset('ghostParticlePositions', data=rigidBody.ghostParticlePositions.cpu().numpy())
        cGroup.create_dataset('ghostParticleIndices', data=rigidBody.ghostParticleIndices.cpu().numpy())
        cGroup.create_dataset('ghostParticleUIDs', data=rigidBody.ghostParticleUIDs.cpu().numpy())
        cGroup.create_dataset('ghostParticleBoundaryDistances', data=rigidBody.ghostParticleBoundaryDistances.cpu().numpy())
        cGroup.create_dataset('ghostParticleBoundaryNormals', data=rigidBody.ghostParticleBoundaryNormals.cpu().numpy())

        cGroup.attrs['bodyID'] = rigidBody.bodyID
        cGroup.attrs['kind'] = rigidBody.kind.value
        cGroup.attrs['sdf'] = _encode_callable(rigidBody.sdf)

    initialState = runningState.state
    uidDtype = np.int64 if uidLayout else np.int32
    particleFloat = initialState.positions.detach().cpu().numpy().dtype if uidLayout else np.float32
    storedFloat = np.float64 if (hasattr(initialState, 'thermodynamicVariable')
                                 and initialState.positions.dtype == torch.float64) else np.float32
    if uidLayout:
        storedFloat = particleFloat
    kinds = initialState.kinds

    boundaryMask = kinds == 1
    boundaryPositions = initialState.positions[boundaryMask]
    boundaryMasses = initialState.masses[boundaryMask]
    boundarySupports = initialState.supports[boundaryMask]
    boundaryUIDs = initialState.UIDs[boundaryMask]
    boundaryKinds = initialState.kinds[boundaryMask]
    stateGhostOffsets = getattr(initialState, 'ghostOffsets', None)
    boundaryOffsets = stateGhostOffsets[boundaryMask] if stateGhostOffsets is not None else None

    outFile.create_dataset('boundaryPositions', data=boundaryPositions.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('boundaryMasses', data=boundaryMasses.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('boundarySupports', data=boundarySupports.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('boundaryUIDs', data=boundaryUIDs.detach().cpu().numpy(), dtype=uidDtype)
    outFile.create_dataset('boundaryKinds', data=boundaryKinds.detach().cpu().numpy(), dtype=np.int32)
    if boundaryOffsets is not None:
        outFile.create_dataset('boundaryOffsets', data=boundaryOffsets.detach().cpu().numpy(), dtype=particleFloat if uidLayout else np.int32)

    fluidMask = kinds == 0
    fluidPositions = initialState.positions[fluidMask]
    fluidMasses = initialState.masses[fluidMask]
    fluidSupports = initialState.supports[fluidMask]
    fluidUIDs = initialState.UIDs[fluidMask]
    fluidKinds = initialState.kinds[fluidMask]

    outFile.create_dataset('fluidPositions', data=fluidPositions.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('fluidMasses', data=fluidMasses.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('fluidSupports', data=fluidSupports.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('fluidUIDs', data=fluidUIDs.detach().cpu().numpy(), dtype=uidDtype)
    outFile.create_dataset('fluidKinds', data=fluidKinds.detach().cpu().numpy(), dtype=np.int32)

    ghostMask = kinds == 2
    ghostPositions = initialState.positions[ghostMask]
    ghostMasses = initialState.masses[ghostMask]
    ghostSupports = initialState.supports[ghostMask]
    ghostUIDs = initialState.UIDs[ghostMask]
    ghostKinds = initialState.kinds[ghostMask]
    outFile.create_dataset('ghostPositions', data=ghostPositions.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('ghostMasses', data=ghostMasses.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('ghostSupports', data=ghostSupports.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('ghostUIDs', data=ghostUIDs.detach().cpu().numpy(), dtype=uidDtype)
    outFile.create_dataset('ghostKinds', data=ghostKinds.detach().cpu().numpy(), dtype=np.int32)

    outFile.create_dataset('combinedPositions', data=initialState.positions.detach().cpu().numpy(), dtype=storedFloat)
    outFile.create_dataset('combinedMasses', data=initialState.masses.detach().cpu().numpy(), dtype=storedFloat)
    outFile.create_dataset('combinedSupports', data=initialState.supports.detach().cpu().numpy(), dtype=storedFloat)
    outFile.create_dataset('combinedUIDs', data=initialState.UIDs.detach().cpu().numpy(), dtype=uidDtype)
    outFile.create_dataset('combinedKinds', data=initialState.kinds.detach().cpu().numpy(), dtype=np.int32)
    outFile.create_dataset('combinedDensities', data=initialState.densities.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('combinedVelocities', data=initialState.velocities.detach().cpu().numpy(), dtype=particleFloat)
    outFile.create_dataset('combinedMaterials', data=initialState.materials.detach().cpu().numpy(), dtype=np.int32)
    # Provenance and body membership are independent of physical materials.
    for field, dataset in (('emitters', 'combinedEmitters'), ('bodyIDs', 'combinedBodyIDs')):
        ids = getattr(initialState, field, None)
        if ids is not None:
            outFile.create_dataset(dataset, data=ids.detach().cpu().numpy(),
                                   dtype=ids.detach().cpu().numpy().dtype if uidLayout else np.int32)
    if stateGhostOffsets is not None:
        outFile.create_dataset('combinedGhostOffsets', data=stateGhostOffsets.detach().cpu().numpy(), dtype=particleFloat)
        outFile.create_dataset('combinedGhostIndices', data=initialState.ghostIndices.detach().cpu().numpy(), dtype=np.int32)

    positionGroup = outFile.create_group('positions')
    velocityGroup = outFile.create_group('velocities')
    densityGroup = outFile.create_group('densities')
    timeGroup = outFile.create_group('times')
    rigidBodyGroup = outFile.create_group('rigidBodyTrajectories')
    extraGroups = tuple(outFile.create_group(name) for name in extraFields)
    for name in ('crkMassDensities', 'crkVolumes'):
        if getattr(initialState, name, None) is not None and name not in outFile:
            outFile.create_group(name)

    if uidLayout:
        from .particleLayout import stateAxes, writeLayoutMetadata
        particleFields=stateAxes(initialState)[0]
        writeLayoutMetadata(outFile,runningState,kind='trajectory')
        outFile.create_group('frameUIDs')
        for group in (positionGroup,velocityGroup,densityGroup,*extraGroups):
            name = group.name.rsplit('/',1)[-1]
            group.attrs['axis'] = 'particle' if name in particleFields else 'global'
            if name in particleFields:
                group.attrs['uidDataset'] = '/combinedUIDs'
    return (positionGroup, velocityGroup, densityGroup, timeGroup, rigidBodyGroup) + extraGroups


def _writeFrameData(group, values):
    for name, value in values.items():
        if isinstance(value, dict):
            _writeFrameData(group.create_group(name), value)
        elif isinstance(value, torch.Tensor):
            group.create_dataset(name, data=value.detach().cpu().numpy())
        elif isinstance(value, (np.ndarray, list, tuple)):
            group.create_dataset(name, data=np.asarray(value))
        else:
            group.attrs[name] = 'None' if value is None else value


def writeFrame(groups, i, state, stages, config, schemeConfig, uniqueParticles=True, writeStages=False,
              extraFields: tuple = (), extraData: dict | None = None):
    uidLayout = getattr(state, 'spatialContext', None) is not None or getattr(state, 'canonicalExport', False)
    uidLayout = uidLayout or bool(groups and groups[0].file.attrs.get('particleLayoutVersion'))
    if uidLayout:
        from .particleLayout import trajectoryView
        state, stages = trajectoryView(groups,state,stages,extraFields,extraData,writeStages)
    positionGroup, velocityGroup, densityGroup, timeGroup, rigidBodyGroup, *extraGroups = groups
    if uidLayout:
        positionGroup.file['frameUIDs'][f'frame_{i:05d}'] = positionGroup.file['combinedUIDs']
        if writeStages and stages:
            group = positionGroup.file['stages'].create_group(f'frame_{i:05d}')
            stageGroups = [group.create_group(f'stage_{j}') for j in range(len(stages))]
            for j,stage in enumerate(stages):
                dumpStage(stage,stageGroups,j,type(state.state),True)

    positionGroup.create_dataset(f'frame_{i:05d}', data=state.state.positions.cpu().numpy())
    velocityGroup.create_dataset(f'frame_{i:05d}', data=state.state.velocities.cpu().numpy())
    densityGroup.create_dataset(f'frame_{i:05d}', data=state.state.densities.cpu().numpy())
    timeGroup.create_dataset(
        f'frame_{i:05d}',
        data=np.array([state.t.cpu().item() if isinstance(state.t, torch.Tensor) else state.t],
                      dtype=np.float64),
    )
    for name, group in zip(extraFields, extraGroups):
        value = getattr(state.state, name)
        key = f'frame_{i:05d}'
        if value is None:
            group.attrs[key] = 'None'
        else:
            group.create_dataset(key, data=value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
    for name in ('crkMassDensities', 'crkVolumes'):
        value = getattr(state.state, name, None)
        if value is not None and name not in extraFields:
            positionGroup.file[name].create_dataset(f'frame_{i:05d}', data=value.detach().cpu().numpy())

    if extraData is not None:
        # Time-varying forcing/budgets cannot live only in initial file attrs.
        frame = positionGroup.file.require_group('frameData').create_group(f'frame_{i:05d}')
        _writeFrameData(frame, extraData)

    rbg = rigidBodyGroup.create_group(f'frame_{i:05d}')
    for r, rigidBody in enumerate(getattr(schemeConfig, 'rigidBodies', []) or []):
        rbg.attrs[f'rigidBody_{r:02d}_centerOfMass'] = rigidBody.centerOfMass.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_orientation'] = rigidBody.orientation.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_angularVelocity'] = rigidBody.angularVelocity.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_linearVelocity'] = rigidBody.linearVelocity.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_mass'] = rigidBody.mass.cpu().numpy()
        rbg.attrs[f'rigidBody_{r:02d}_inertia'] = rigidBody.inertia.cpu().numpy()


__all__ = [
    'schemeAttribute', 'exportSimulationSystem', 'writeInitialData', 'writeFrame',
    'exportDirName', 'resolveExportRoot', 'findExportRuns', 'latestExportPath', 'prepExport',
]
