"""Explicit integrated-field axes for solver vectors and method-owned memory."""
from dataclasses import dataclass
import math

import torch

from .layout import PARTICLE_FIELDS,permute_state
from .continuation import uid_order

CONTEXT_KEYS = {
    'ndf.v1': frozenset('method order layout uid phase differences current constant_steps adaptive_ndf last_order particle_uids'.split()),
    'variable_adams.v1': frozenset('method layout uid adaptive_adams order last_order age current t order_defects particle_uids'.split()),
}


def flat_view(value,description,source_ids,target_ids):
    """Reorder declared integrated blocks on the last flat axis (also differences)."""
    order=uid_order(source_ids,target_ids)
    names=[name for name,_,_ in description]
    allowed={'positions','velocities','internalEnergies','crkMassDensities','crkVolumes'}
    if len(set(names)) != len(names) or not set(names)<=allowed:
        raise ValueError('solver layout contains undeclared/duplicate integrated fields')
    if any(not shape or any(type(i) is not int or i<=0 for i in shape) for _,shape,_ in description):
        raise ValueError('solver layout contains an invalid field shape')
    size=sum(math.prod(shape) for _,shape,_ in description)
    if not isinstance(value,torch.Tensor) or not value.ndim or value.shape[-1] != size:
        raise ValueError('solver vector does not match its declared integrated layout')
    pieces=[];offset=0
    for name,shape,dtype in description:
        if name not in PARTICLE_FIELDS or shape[0] != len(source_ids) or dtype != str(value.dtype):
            raise ValueError('solver layout contains an undeclared particle axis')
        width=math.prod(shape)
        part=value[...,offset:offset+width].reshape(*value.shape[:-1],*shape)
        pieces.append(part.index_select(value.ndim-1,order).reshape(*value.shape[:-1],width))
        offset+=width
    return torch.cat(pieces,dim=-1)


def context_view(context,target_ids,*,source_ids=None):
    if context is None:
        return None
    if type(context) is not dict or context.get('method') not in CONTEXT_KEYS:
        raise ValueError('history/reuse requires an audited method-owned context')
    if set(context)-CONTEXT_KEYS[context['method']]:
        raise ValueError('method-owned context contains undeclared fields')
    source=context.get('particle_uids') if source_ids is None else source_ids
    if source is None:
        raise ValueError('method-owned context needs an explicit particle UID owner')
    uid_order(source,target_ids)
    result=dict(context)
    description=context.get('layout')
    if not isinstance(description,(tuple,list)):
        raise ValueError('method-owned context is missing its integrated layout')
    for name in ('current','differences'):
        if name in context:
            result[name]=flat_view(context[name],description,source,target_ids)
    if 'order_defects' in context:
        defects=context['order_defects']
        if type(defects) is not dict or any(k!='upper' and k not in {f'lower_{i}' for i in range(1,13)} for k in defects):
            raise ValueError('unknown Adams defect descriptor')
        result['order_defects']={k:flat_view(v,description,source,target_ids) for k,v in defects.items()}
    if context.get('uid') is not None:
        result['uid']=context['uid'][uid_order(source,target_ids)]
    result['particle_uids']=target_ids.clone()
    return result


def integrated_description(state):
    from warpSPHIntegrators.state.fields import integrated_field_names,get_reference_state
    ref=get_reference_state(state)
    return tuple((n,tuple(getattr(ref,n).shape),str(getattr(ref,n).dtype)) for n in integrated_field_names(state))


def vector_view(value,state,source_ids,target_ids):
    from warpSPHIntegrators.state.fields import BlockState,flatten_integrated
    if isinstance(state,BlockState):
        parts=[];offset=0
        for substate in state.states:
            size=flatten_integrated(substate).numel()
            parts.append(vector_view(value[offset:offset+size],substate,source_ids,target_ids));offset+=size
        if offset != value.numel():raise ValueError('coupled vector size differs from its stage layout')
        return torch.cat(parts)
    return flat_view(value,integrated_description(state),source_ids,target_ids)


def system_view(state,target_ids):
    from warpSPHIntegrators.state.fields import BlockState
    from .sortedStep import _system_view
    if isinstance(state,BlockState):
        return BlockState(tuple(system_view(s,target_ids) for s in state.states))
    return _system_view(state,permute_state(state.state,uid_order(state.state.UIDs,target_ids)))


@dataclass(frozen=True)
class UIDPreconditioner:
    """A pure preconditioner declared in UID order, including coupled stage blocks.

    The supplied action receives flat vectors and physical states in this UID
    order. External particle coefficients belong to that same declared order.
    Raw native callback/context tensors remain rejected by the spatial stepper.
    """
    ids: torch.Tensor
    action: object

    def __post_init__(self):
        from .layout import ParticleLayout
        if not callable(self.action):raise TypeError('UID preconditioner action must be callable')
        ParticleLayout.create(self.ids,torch.arange(len(self.ids),device=self.ids.device))
        object.__setattr__(self,'ids',self.ids.clone())

    def bind(self,compute_ids):
        uid_order(self.ids,compute_ids)
        def apply(vector,state,context):
            public=system_view(state,self.ids)
            values=vector_view(vector,state,compute_ids,self.ids)
            metadata={k:v for k,v in context.items() if k not in ('state','evaluation_context')}
            if any(isinstance(v,torch.Tensor) and v.ndim for v in metadata.values()):
                raise ValueError('preconditioner context tensors need an explicit UID input adapter')
            metadata['state']=public
            # Resource tokens retain their actual compute owner rather than
            # masquerading as input-order geometry in a callback.
            metadata['compute_evaluation_context']=context.get('evaluation_context')
            result=self.action(values,public,metadata)
            if not isinstance(result,torch.Tensor) or result.shape != vector.shape or result.dtype != vector.dtype or result.device != vector.device:
                raise ValueError('UID preconditioner returned an incompatible vector')
            return vector_view(result,public,self.ids,compute_ids)
        return apply

    def __call__(self,vector,state,context):
        from warpSPHIntegrators.state.fields import BlockState
        first=state.states[0] if isinstance(state,BlockState) else state
        return self.bind(first.state.UIDs)(vector,state,context)


def solver_view(solver,options,compute_ids):
    options=dict(options or {})
    action=options.get('preconditioner',getattr(solver,'preconditioner',None))
    if action is None:return solver,options
    if type(action) is not UIDPreconditioner:
        raise ValueError('preconditioners need an explicit vector/input layout adapter')
    for context in (getattr(solver,'preconditioner_context',None),options.get('preconditioner_context')):
        if context is not None and any(isinstance(v,torch.Tensor) and v.ndim for v in context.values()):
            raise ValueError('preconditioner context tensors need an explicit UID input adapter')
    options['preconditioner']=action.bind(compute_ids)
    # Do not change the caller's solver configuration.
    return solver,options
