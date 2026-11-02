"""Versioned numerical continuation, with explicit UID axes and no executable payloads.

Spatial search/edge caches are deliberately absent. The physical state remains
the ordinary UID checkpoint; this group carries its numerical trajectory memory.
"""
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math

import numpy as np
import torch

from ..evaluation.continuation import history_view, stage_view, uid_order, reusable_prior, reuse_stamp, EndpointDerivative
from ..evaluation.layout import EDGE_FIELDS, PARTICLE_FIELDS, GLOBAL_FIELDS, UPDATE_FIELDS, GLOBAL_UPDATE_FIELDS, ParticleLayout, permute_state, validate_update_schema
from .hdf5 import dumpState, loadState

VERSION = 1


def portable(value):
    """Numerical configuration identity independent of device/pointers/process."""
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,Enum):
        return [type(value).__module__,type(value).__qualname__,value.name]
    if isinstance(value,torch.Tensor):
        return dict(dtype=str(value.dtype),shape=list(value.shape),values=value.detach().cpu().tolist())
    if isinstance(value,(torch.dtype,torch.device)):
        return str(value)
    from ..evaluation.boundaryInputs import UIDValues
    if type(value) is UIDValues:
        p=torch.argsort(value.ids,stable=True)
        return ['UIDValues',portable(value.ids[p]),portable(value.values[p])]
    from ..modules.boundaryConditions.pinned import PinnedBandSdf
    if type(value) is PinnedBandSdf:
        return ['PinnedBandSdf',portable(value.band)]
    from ..configurations.moduleConfigurations.diffusionParameters import DiffusionParameters
    from ..configurations.crkSPH import CRKViscosity
    for schema in (DiffusionParameters,CRKViscosity):
        if type(value) is schema.instance_type:
            if set(vars(value)) != {'_ctype',*schema.vars}:
                raise ValueError('checkpoint has undeclared parameter fields')
            return [schema.key,{name:portable(getattr(getattr(value,name),'value',getattr(value,name))) for name in schema.vars}]
    if is_dataclass(value):
        # Placement/execution policy and next dt are portable run controls.
        excluded = {'device','compute','dt','integrationScheme'} if type(value).__name__ == 'SimulationConfig' else set()
        return [type(value).__module__,type(value).__qualname__,
                {f.name:portable(getattr(value,f.name)) for f in fields(value) if f.name not in excluded}]
    if isinstance(value,(tuple,list)):
        return [portable(v) for v in value]
    if isinstance(value,dict):
        return {k:portable(v) for k,v in value.items()}
    raise ValueError(f'checkpoint configuration requires a declared portable adapter for {type(value).__name__}')


def signature(config,schemeConfig):
    encoded=json.dumps([portable(config),portable(schemeConfig)],sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _number(value,name,*,nonzero=False):
    result=float(value)
    if not math.isfinite(result) or (nonzero and result == 0):
        raise ValueError(f'checkpoint {name} must be finite'+(' and nonzero' if nonzero else ''))
    return result


def _state(group,state):
    # All pair work is keyed to an omitted spatial cache.
    order=torch.argsort(state.UIDs,stable=True)
    dumpState(permute_state(state,order),group)


def _update(group,update):
    validate_update_schema(update,group.parent['state/UIDs'].shape[0])
    for name in UPDATE_FIELDS:
        value=getattr(update,name)
        if value is None:
            group.attrs[name]='None'
        else:
            group.create_dataset(name,data=value.detach().cpu().numpy())


def _stage(group,stage,ids):
    stage=stage_view(stage,ids)
    _state(group.create_group('state'),stage.aux[1])
    _update(group.create_group('update'),stage.update)


def writeContinuation(group,memory,*,integrator,config,schemeConfig,last_dt,next_dt):
    """Write accepted memory in ascending UID order; never serialize callbacks."""
    from warpSPHIntegrators.runtime import Continuation
    from warpSPHIntegrators.state.history import StepHistory
    if not isinstance(memory,Continuation):
        raise TypeError('checkpoint requires an accepted Continuation')
    state=memory.state
    ids=torch.sort(state.state.UIDs).values
    history=memory.history if memory.history is not None else StepHistory(max(1,integrator.metadata.history_length))
    history=history_view(history,ids)
    if history.maxlen < integrator.metadata.history_length:
        raise ValueError('checkpoint history capacity is insufficient for its method')
    group.attrs.update(version=VERSION,method=integrator.identifier.name,time=_number(state.t,'time'),
                       configuration=signature(config,schemeConfig),maxlen=history.maxlen,count=len(history),
                       next_dt=_number(next_dt,'next dt',nonzero=True),last_dt_none=last_dt is None)
    if last_dt is not None:
        group.attrs['last_dt']=_number(last_dt,'last dt',nonzero=True)
    entries=group.create_group('history')
    if history.context is not None:
        context=group.create_group('context')
        scalars={}
        for name,value in history.context.items():
            if isinstance(value,torch.Tensor):
                context.create_dataset(name,data=value.detach().cpu().numpy())
            elif name=='order_defects':
                defects=context.create_group(name)
                for key,tensor in value.items():defects.create_dataset(key,data=tensor.detach().cpu().numpy())
            else:scalars[name]=value
        context.attrs['scalars']=json.dumps(scalars,allow_nan=False)
    for index,entry in enumerate(history):
        item=entries.create_group(str(index))
        item.attrs.update(t=_number(entry.t,'entry time'),dt=_number(entry.dt,'entry dt',nonzero=True),snapshot=entry.state is not None,stage=entry.update is not None)
        from warpSPHIntegrators.state import StageResult
        if entry.update is not None:
            _stage(item,StageResult(aux=entry.aux,update=entry.update),ids)
        if entry.state is not None:
            snapshot=item.create_group('snapshot')
            snapshot.attrs['time']=_number(entry.state.t,'snapshot time')
            _state(snapshot.create_group('state'),entry.state.state)
    group.attrs['prior']=memory.prior_step is not None
    if memory.prior_step is not None:
        prior=memory.prior_step
        raw=None if last_dt is None else reusable_prior(stage_view(prior,state.state.UIDs),integrator,state,last_dt,config,schemeConfig)
        item=group.create_group('prior')
        item.attrs['reusable']=raw is not None
        _stage(item,raw if raw is not None else prior,ids)


def _read_state(group,ids,config):
    from ..systems.compSPH import CompSPHState
    from .particleLayout import validateRoster,validatePrecision
    # Complete schema, including explicit nulls: truncated files cannot silently
    # replace history fields with dataclass defaults.
    if set(group.keys())&set(group.attrs) or set(group.keys())|set(group.attrs) != PARTICLE_FIELDS|GLOBAL_FIELDS|EDGE_FIELDS:
        raise ValueError('checkpoint continuation state schema is incomplete or unknown')
    state=loadState(group,config.device,CompSPHState)
    validateRoster(state)
    if not torch.equal(state.UIDs,ids):
        raise ValueError('checkpoint continuation UID roster/order differs from the frame')
    validatePrecision(state,config)
    if not bool((state.supports > 0).all()) or any(getattr(state,n) is not None for n in EDGE_FIELDS):
        raise ValueError('checkpoint continuation has invalid support or persisted edge cache')
    return state


def _read_stage(group,ids,config):
    from warpSPHIntegrators.state import StageResult
    from ..systems.compressibleMonaghan import CompressibleSystemUpdate
    state=_read_state(group['state'],ids,config)
    values=group['update']
    if set(values.keys())&set(values.attrs) or set(values.keys())|set(values.attrs) != UPDATE_FIELDS:
        raise ValueError('checkpoint continuation update schema is incomplete or unknown')
    data={}
    for name in UPDATE_FIELDS:
        if name in values:
            value=torch.as_tensor(np.array(values[name]),device=config.device)
            allowed=(torch.bool,) if name=='passive' else (torch.float32,torch.float64) if name in GLOBAL_UPDATE_FIELDS else (config.dtype,)
            if value.dtype not in allowed or not bool(torch.isfinite(value).all()):
                raise ValueError('checkpoint continuation has invalid derivative precision/values')
            data[name]=value
        elif values.attrs[name] == 'None':
            data[name]=None
        else:
            raise ValueError('checkpoint continuation has invalid derivative null marker')
    update=CompressibleSystemUpdate(**data)
    validate_update_schema(update,len(ids))
    return StageResult(aux=(None,state),update=update)


def readContinuation(group,state,*,integrator,config,schemeConfig):
    """Restore owned numerical memory, with fresh layout handles and no CSR."""
    from warpSPHIntegrators.runtime import Continuation
    from warpSPHIntegrators.state.history import HistoryEntry,StepHistory
    from ..evaluation.sortedStep import SpatialStageContext,_system_view
    if group.attrs.get('version') != VERSION or group.attrs.get('method') != integrator.identifier.name:
        raise ValueError('checkpoint continuation version/method does not match')
    if group.attrs.get('configuration') != signature(config,schemeConfig):
        raise ValueError('checkpoint numerical configuration does not match')
    if _number(group.attrs['time'],'time') != float(state.t):
        raise ValueError('checkpoint continuation time differs from its frame')
    ids=state.state.UIDs
    context=None
    if 'context' in group:
        from ..evaluation.solverInputs import context_view,CONTEXT_KEYS
        saved=group['context']
        context=json.loads(saved.attrs['scalars'])
        if set(context)&set(saved):raise ValueError('checkpoint context has duplicate descriptors')
        for name,value in saved.items():
            if name=='order_defects':
                context[name]={key:torch.as_tensor(np.array(tensor),device=config.device) for key,tensor in value.items()}
            else:
                context[name]=torch.as_tensor(np.array(value),device=config.device)
        description=context.get('layout',())
        context['layout']=tuple((name,tuple(shape),dtype) for name,shape,dtype in description)
        for name in ('current','differences'):
            value=context.get(name)
            if value is not None and (value.dtype != config.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('checkpoint method context has invalid precision/values')
        for value in context.get('order_defects',{}).values():
            if value.dtype != config.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('checkpoint Adams defects have invalid precision/values')
        context=context_view(context,ids)
        from ..evaluation.solverInputs import integrated_description
        from warpSPHIntegrators.state.fields import flatten_integrated
        if context['layout'] != integrated_description(state):
            raise ValueError('checkpoint method layout differs from the frame')
        if context.get('current') is not None and not torch.equal(context['current'],flatten_integrated(state)):
            raise ValueError('checkpoint method current vector differs from its frame')
        method=integrator.identifier.name
        expected='ndf.v1' if method.startswith(('ndf_','adaptive_ndf_')) else 'variable_adams.v1' if method.startswith(('variable_adams_','adaptive_adams_')) else None
        if context['method'] != expected:
            raise ValueError('checkpoint method-owned context differs from the selected method')
    maxlen,count=int(group.attrs['maxlen']),int(group.attrs['count'])
    if maxlen < max(1,integrator.metadata.history_length) or not 0 <= count <= maxlen:
        raise ValueError('checkpoint continuation history capacity/count is invalid')
    if set(group['history']) != {str(i) for i in range(count)}:
        raise ValueError('checkpoint continuation history entries are incomplete')
    next_dt=_number(group.attrs['next_dt'],'next dt',nonzero=True)
    last_dt=None if bool(group.attrs['last_dt_none']) else _number(group.attrs['last_dt'],'last dt',nonzero=True)
    entries=[]
    for index in range(count):
        item=group['history'][str(index)]
        stage=_read_stage(item,ids,config) if bool(item.attrs.get('stage',True)) else None
        if stage is None and (context is None or context['method'] != 'ndf.v1'):
            raise ValueError('checkpoint is missing derivative history')
        t,dt=_number(item.attrs['t'],'entry time'),_number(item.attrs['dt'],'entry dt',nonzero=True)
        snapshot=None
        if bool(item.attrs['snapshot']):
            snapshot=_system_view(state,_read_state(item['snapshot/state'],ids,config))
            snapshot.t=_number(item['snapshot'].attrs['time'],'snapshot time')
            snapshot.retainStageDerivatives=True
            if snapshot.t != t:
                raise ValueError('checkpoint BDF snapshot time differs from its entry')
        if integrator.identifier.name.startswith('bdf') and snapshot is None:
            raise ValueError('checkpoint BDF history is missing a state snapshot')
        if (dt > 0 and t > float(state.t)) or (dt < 0 and t < float(state.t)):
            raise ValueError('checkpoint continuation history is ahead of its frame')
        entries.append(HistoryEntry(t=t,dt=dt,update=None if stage is None else stage.update,aux=None if stage is None else stage.aux,state=snapshot))
    history=StepHistory(maxlen,entries,context=context)
    prior=None
    if bool(group.attrs['prior']):
        prior=_read_stage(group['prior'],ids,config)
        if bool(group['prior'].attrs['reusable']):
            if last_dt is None:
                raise ValueError('checkpoint reusable prior is missing last dt')
            stamp=reuse_stamp(integrator,state,last_dt,config,schemeConfig)
            if stamp is not None:
                layout=ParticleLayout.create(ids,torch.arange(len(ids),device=ids.device))
                metadata=SpatialStageContext(layout,None,{name:None for name in EDGE_FIELDS},
                    EndpointDerivative(stamp,ids.clone(),prior.update))
                prior=prior._replace(aux=(*prior.aux,metadata))
    return Continuation(state,history,prior),last_dt,next_dt


def validateFrame(memory,state):
    """A history carrier cannot quietly supply a different accepted frame."""
    if float(memory.state.t) != float(state.t):
        raise ValueError('checkpoint continuation and frame times differ')
    before=memory.state.state
    order=uid_order(before.UIDs,state.state.UIDs)
    for name in PARTICLE_FIELDS|GLOBAL_FIELDS:
        a,b=getattr(before,name),getattr(state.state,name)
        if isinstance(a,torch.Tensor):
            a=a[order] if name in PARTICLE_FIELDS else a
            if not isinstance(b,torch.Tensor) or not torch.equal(a,b):
                raise ValueError(f'checkpoint continuation/frame differ in {name}')
        elif isinstance(b,torch.Tensor) or a != b:
            raise ValueError(f'checkpoint continuation/frame differ in {name}')
