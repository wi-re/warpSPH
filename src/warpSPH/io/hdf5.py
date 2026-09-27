"""Low-level HDF5 read/write primitives shared by :mod:`.export` and :mod:`.importIO`."""

import codecs
import os

import dill
import h5py
import pickle
import torch

from warpSPHCore import *
from warpSPHIntegrators.specs import StageResult


def dumpAdjacency(adjacency, adjacencyGroup):
    adjacencyGroup.create_dataset('i', data=adjacency.i.cpu().numpy())
    adjacencyGroup.create_dataset('j', data=adjacency.j.cpu().numpy())
    adjacencyGroup.create_dataset('numNeighbors', data=adjacency.numNeighbors.cpu().numpy())
    adjacencyGroup.create_dataset('edgeOffsets', data=adjacency.edgeOffsets.cpu().numpy())

    adjacencyGroup.create_dataset('queryPositions', data=adjacency.queryPositions.cpu().numpy())
    adjacencyGroup.create_dataset('querySupports', data=adjacency.querySupports.cpu().numpy())
    adjacencyGroup.create_dataset('referencePositions', data=adjacency.referencePositions.cpu().numpy())
    adjacencyGroup.create_dataset('referenceSupports', data=adjacency.referenceSupports.cpu().numpy())

    adjacencyGroup.attrs['numRows'] = adjacency.numRows
    adjacencyGroup.attrs['numCols'] = adjacency.numCols

def dumpState(state, stateGroup):
    for fieldName in state.__dict__.keys():
        if fieldName.startswith('_'):
            continue
        fieldValue = getattr(state, fieldName)
        # print(f'Dumping field {fieldName}... -> {type(fieldValue)}')
        if isinstance(fieldValue, torch.Tensor):
            stateGroup.create_dataset(fieldName, data=fieldValue.cpu().numpy())
        elif fieldValue is None:
            stateGroup.attrs[fieldName] = 'None'
        else:
            stateGroup.attrs[fieldName] = fieldValue

def dumpStage(stage, stageGroups, index, SimulationState, exportStagesAdjacency):
    stageGroups[index].attrs['index'] = index

    auxGroup = stageGroups[index].create_group('aux')
    for auxIndex, aux in enumerate(stage.aux):
        if isinstance(aux, AdjacencyList):
            if exportStagesAdjacency:
                auxAdjacencyGroup = auxGroup.create_group(f'{auxIndex}_adjacency')
                dumpAdjacency(aux, auxAdjacencyGroup)
        elif isinstance(aux, SimulationState):
            auxStateGroup = auxGroup.create_group(f'{auxIndex}_state')
            dumpState(aux, auxStateGroup)
        else:
            print(f'Unsupported aux type: {type(aux)}')

    for updateKey, updateValue in stage.update.__dict__.items():
        if isinstance(updateValue, torch.Tensor):
            stageGroups[index].create_dataset(updateKey, data=updateValue.cpu().numpy())
        else:
            continue


def hdfDtypeToTorchDtype(hdfDtype):
    if hdfDtype == "<i4":
        return torch.int32
    elif hdfDtype == "<i8":
        return torch.int64
    elif hdfDtype == "<f4":
        return torch.float32
    elif hdfDtype == "<f8":
        return torch.float64
    elif hdfDtype == 'bool':
        return torch.bool
    else:
        raise ValueError(f'Unsupported HDF dtype: {hdfDtype}')



def loadAdjacency(adjacencyGroup, device):

    i = torch.from_numpy(adjacencyGroup['i'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['i'].dtype))
    j = torch.from_numpy(adjacencyGroup['j'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['j'].dtype))
    numNeighbors = torch.from_numpy(adjacencyGroup['numNeighbors'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['numNeighbors'].dtype))
    edgeOffsets = torch.from_numpy(adjacencyGroup['edgeOffsets'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['edgeOffsets'].dtype))

    queryPositions = torch.from_numpy(adjacencyGroup['queryPositions'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['queryPositions'].dtype))
    querySupports = torch.from_numpy(adjacencyGroup['querySupports'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['querySupports'].dtype))
    referencePositions = torch.from_numpy(adjacencyGroup['referencePositions'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['referencePositions'].dtype))
    referenceSupports = torch.from_numpy(adjacencyGroup['referenceSupports'][:]).to(device).to(hdfDtypeToTorchDtype(adjacencyGroup['referenceSupports'].dtype))

    numRows = adjacencyGroup.attrs['numRows']
    numCols = adjacencyGroup.attrs['numCols']

    return AdjacencyList(i, j, numNeighbors, edgeOffsets, numRows, numCols, queryPositions, referencePositions, querySupports, referenceSupports)

def loadState(stateGroup, device, SimulationState):
    stateDict = {}
    for fieldName, fieldValue in stateGroup.items():
        # print(f'Loading field {fieldName}... -> {type(fieldValue)}', fieldValue)
        if isinstance(fieldValue, h5py.Dataset):
            stateDict[fieldName] = torch.from_numpy(fieldValue[:]).to(device).to(hdfDtypeToTorchDtype(fieldValue.dtype)) if fieldValue.shape else torch.tensor(fieldValue[()], device=device, dtype=hdfDtypeToTorchDtype(fieldValue.dtype))
        else:
            stateDict[fieldName] = fieldValue
    # `UIDcounter` is a plain int, not a per-particle array, so the writer --
    # which emits datasets -- never stores it, and rebuilding the dataclass
    # then fails on the missing required argument. That made every
    # `storeMode='states'` checkpoint unloadable. Derive it from the UIDs, the
    # same reconstruction the trajectory importer already uses (`importIO.py`).
    if 'UIDcounter' not in stateDict and 'UIDs' in stateDict:
        stateDict['UIDcounter'] = int(stateDict['UIDs'].max().item()) + 1
    return SimulationState(**stateDict)

def loadStage(stageGroup, device, SimulationState, SimulationUpdate):
    index = stageGroup.attrs['index']

    aux = []
    for auxKey, auxValue in stageGroup['aux'].items():
        if '_adjacency' in auxKey:
            aux.append(loadAdjacency(auxValue, device))
        elif '_state' in auxKey:
            aux.append(loadState(auxValue, device, SimulationState))
        else:
            print(f'Unsupported aux type for key {auxKey}')

    updateDict = {}
    for updateKey, updateValue in stageGroup.items():
        if updateKey == 'aux':
            continue
        updateDict[updateKey] = torch.from_numpy(updateValue[:]).to(device).to(hdfDtypeToTorchDtype(updateValue.dtype)) if updateValue.shape else torch.tensor(updateValue[()], device=device, dtype=hdfDtypeToTorchDtype(updateValue.dtype))

    return StageResult(aux=aux, update=SimulationUpdate(**updateDict))


def _encode_callable(fn):
    return codecs.encode(dill.dumps(fn), 'base64').decode()


def _decode_callable(encoded_fn):
    raw = codecs.decode(encoded_fn.encode(), 'base64')
    try:
        return dill.loads(raw)
    except Exception:
        return pickle.loads(raw)


def copy_dict_to_h5(group, d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            subgroup.attrs['taggedType'] = 'dict'
            copy_dict_to_h5(subgroup, value, indent + 1)
        elif isinstance(value, list):
            subgroup = group.create_group(key)
            subgroup.attrs['taggedType'] = 'list'
            copy_dict_to_h5(subgroup, {f'item_{i}': v for i, v in enumerate(value)}, indent + 1)
        else:
            if value is None:
                continue
            group.attrs[key] = value


def restore_config_from_h5(group, indent=0):
    config = {}
    for key, value in group.attrs.items():
        if key != 'taggedType':
            config[key] = value
    for key, subgroup in group.items():
        if subgroup.attrs['taggedType'] == 'dict':
            config[key] = restore_config_from_h5(subgroup, indent + 1)
        elif subgroup.attrs['taggedType'] == 'list':
            config[key] = [restore_config_from_h5(subgroup[f'item_{i}'], indent + 1) for i in range(len(subgroup))]
            subkeys = [k for k in list(subgroup.attrs.keys()) if k.startswith('item_')]
            for i in range(len(subkeys)):
                item_key = f'item_{i}'
                if item_key in subgroup.attrs:
                    config[key].append(subgroup.attrs[item_key])
        else:
            raise ValueError(f'Unknown type for subgroup {key}: {subgroup.attrs["taggedType"]}')
    return config


# Backward-compatible alias used by existing notebooks/scripts.
def restoreConfig_from_h5(group, indent=0):
    return restore_config_from_h5(group, indent=indent)


def createOutFile(exportPath: str):
    os.makedirs(exportPath, exist_ok=True)
    return h5py.File(f'{exportPath}/trajectory.h5', 'w')


__all__ = [
    'dumpAdjacency', 'dumpState', 'dumpStage', 'hdfDtypeToTorchDtype',
    'loadAdjacency', 'loadState', 'loadStage',
    'copy_dict_to_h5', 'restore_config_from_h5', 'restoreConfig_from_h5', 'createOutFile',
    '_encode_callable', '_decode_callable',
]
