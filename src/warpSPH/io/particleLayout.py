"""Explicit UID axes for fixed-roster trajectories and cache-free checkpoints."""
import math

import numpy as np
import torch

from ..evaluation import canonical_view
from ..evaluation.layout import EDGE_FIELDS, GLOBAL_FIELDS, PARTICLE_FIELDS, validate_state_schema

VERSION = 1
INTEGER_FIELDS=frozenset(('UIDs','kinds','materials','emitters','bodyIDs'))


def stateAxes(state):
    from ..systems.compSPH import CompSPHState
    if type(state) is CompSPHState:
        validate_state_schema(state)
        return PARTICLE_FIELDS,GLOBAL_FIELDS,EDGE_FIELDS
    from ..evaluation.boundaryLayout import _rows
    return _rows(state),frozenset({'UIDcounter'}),frozenset()


def validatePrecision(state,config):
    """Declared numeric types, including integer-corrupted floating datasets."""
    rows,globals_,_=stateAxes(state)
    for name in rows:
        value=getattr(state,name)
        if value is None:continue
        if name in INTEGER_FIELDS or name == 'ghostIndices':
            if value.dtype not in (torch.int32,torch.int64):
                raise ValueError(f'checkpoint {name} must have integer particle labels')
        elif name == 'surfaceIndicators' and value.dtype == torch.int32:
            if bool(((value != 0)&(value != 1)).any()):raise ValueError('invalid surface indicator')
        elif value.dtype != config.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError(f'checkpoint {name} has invalid precision or non-finite data')
    for name in globals_:
        value=getattr(state,name)
        if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError(f'checkpoint {name} has non-finite global data')


def validateRoster(state):
    _,_,edges=stateAxes(state)
    ids = state.UIDs
    if (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64)
            or bool((ids < 0).any()) or torch.unique(ids).numel() != ids.numel()):
        raise ValueError('UID trajectory/checkpoint requires unique nonnegative integer particle IDs')
    if bool(((state.kinds < 0)|(state.kinds > (1 if edges else 2))).any()):
        raise ValueError('UID checkpoint contains an unsupported particle kind')


def validateFields(names,state=None):
    rows,globals_,_=stateAxes(state) if state is not None else (PARTICLE_FIELDS,GLOBAL_FIELDS,EDGE_FIELDS)
    allowed = rows | globals_
    if len(set(names)) != len(names) or any(n not in allowed or n in ('positions','velocities','densities') for n in names):
        raise ValueError('trajectory extraFields must declare unique audited particle/global fields')


def initialView(system, extraData, extraFields, outFile):
    if outFile is None:
        raise ValueError('UID trajectory ID/schema adapter requires an initialized HDF5 file')
    from .export import validateSortedExtraData
    validateSortedExtraData(extraData)
    validateFields(extraFields,system.state)
    validateRoster(system.state)
    return canonical_view(system, include_adjacency=False)[0]


def writeLayoutMetadata(outFile, system, *, kind):
    outFile.attrs['particleLayoutVersion'] = VERSION
    outFile.attrs['particleLayoutFormat'] = kind
    outFile.attrs['particleOrder'] = 'ascending_UID'
    outFile.attrs['particlePrecision'] = str(system.state.positions.dtype)
    outFile.attrs['particleUIDcounter'] = int(system.state.UIDcounter)
    if system.domain is not None:
        for name in ('min','max','periodic'):
            outFile.attrs['particleDomain_' + name] = getattr(system.domain,name).detach().cpu().numpy()


def trajectoryView(groups, system, stages, extraFields, extraData, writeStages):
    if not groups or groups[0].file.attrs.get('particleLayoutVersion') != VERSION:
        raise ValueError('UID trajectory ID/schema adapter requires a UID-roster header')
    from .export import validateSortedExtraData
    validateSortedExtraData(extraData)
    validateFields(extraFields,system.state)
    validateRoster(system.state)
    if [g.name.rsplit('/',1)[-1] for g in groups[5:]] != list(extraFields):
        raise ValueError('trajectory extraFields changed from the declared header axes')
    system, stages = canonical_view(system, stages if writeStages else None,
        include_adjacency=False, include_stage_adjacency=writeStages)
    roster = groups[0].file['combinedUIDs'][:]
    ids = system.state.UIDs.detach().cpu().numpy()
    if not np.array_equal(ids,roster):
        raise ValueError('trajectory UID roster changed; population events need a topology schema')
    staticFields = {'masses':'combinedMasses','kinds':'combinedKinds','materials':'combinedMaterials',
                    'emitters':'combinedEmitters','bodyIDs':'combinedBodyIDs'}
    for name,dataset in staticFields.items():
        if name in extraFields:
            continue
        value = getattr(system.state,name)
        present = dataset in groups[0].file
        if (value is None) != (not present) or (value is not None and not np.array_equal(
                value.detach().cpu().numpy(),groups[0].file[dataset][:])):
            raise ValueError(f'trajectory {name} changed; declare it in extraFields')
    if stages:
        for stage in stages:
            if not np.array_equal(stage.aux[1].UIDs.detach().cpu().numpy(),roster):
                raise ValueError('trajectory stage UID roster differs from its frame')
    return system, stages


def restoreCheckpoint(outFile, template, config, scheme):
    """Restore full physical state, never portable pointers or a saved search cache."""
    from .export import schemeAttribute
    from .hdf5 import loadState
    if (outFile.attrs.get('particleLayoutVersion') != VERSION
            or outFile.attrs.get('particleLayoutFormat') != 'state' or 'state' not in outFile):
        raise ValueError('Morton restart requires a UID-layout state checkpoint')
    if outFile.attrs.get('scheme') != schemeAttribute(scheme):
        raise ValueError('checkpoint scheme does not match the requested solver')
    if outFile.attrs.get('particlePrecision') != str(config.dtype):
        raise ValueError('checkpoint precision does not match the requested configuration')
    for name in ('min','max','periodic'):
        expected = getattr(config.domain,name).detach().cpu().numpy()
        if not np.array_equal(outFile.attrs.get('particleDomain_' + name),expected):
            raise ValueError('checkpoint domain does not match the requested configuration')
    time = float(outFile.attrs['time'])
    if not math.isfinite(time):
        raise ValueError('checkpoint time must be finite')
    restored = loadState(outFile['state'],config.device,type(template.state))
    validateRoster(restored)
    expectedIDs, actualIDs = torch.argsort(template.state.UIDs),torch.argsort(restored.UIDs)
    if not torch.equal(template.state.UIDs[expectedIDs],restored.UIDs[actualIDs]):
        raise ValueError('checkpoint UID roster differs from the sampled configuration')
    for name in ('masses','kinds','materials','emitters','bodyIDs'):
        before, after = getattr(template.state,name),getattr(restored,name)
        a,b=(before[expectedIDs],after[actualIDs]) if before is not None and after is not None else (before,after)
        if name == 'masses' and a is not None and b is not None:
            # Solid masses follow the native fixed-volume wall density update.
            fluid=template.state.kinds[expectedIDs] == 0
            a,b=a[fluid],b[fluid]
        if (a is None) != (b is None) or (a is not None and not torch.equal(a,b)):
            raise ValueError(f'checkpoint {name} differs from the sampled configuration')
    validatePrecision(restored,config)
    if not bool((restored.supports > 0).all()):
        raise ValueError('checkpoint supports must be positive')
    if int(restored.UIDcounter) <= int(restored.UIDs.max()):
        raise ValueError('checkpoint UID counter does not cover its particle roster')
    for name in stateAxes(restored)[2]:
        setattr(restored,name,None)
    template.state, template.t, template.domain = restored,time,config.domain
    template.adjacency = None
    for name in ('spatialContext','pairWork','supportGrid','listGrid','rerunRequest'):
        if hasattr(template,name):setattr(template,name,None)
    template.canonicalExport = True
    return template



def readLayoutDomain(attrs,device):
    """Reconstruct portable domain values for observation imports, without caches."""
    if not all('particleDomain_'+name in attrs for name in ('min','max','periodic')):
        return None
    from warpSPHCore import DomainDescription
    values=[torch.as_tensor(attrs['particleDomain_'+name],device=device) for name in ('min','max','periodic')]
    return DomainDescription(*values,len(values[0]))
