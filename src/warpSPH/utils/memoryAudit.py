"""Read-only tensor-storage inventory; views and aliases share one allocation.

Inspect stored fields only: computed tensor properties are deliberately ignored.
Group totals overlap; unionBytes and incrementalBytes form the additive accounting.
This accounts tensor storage, not allocator rounding/caches or foreign allocations.
"""
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
import torch


def tensorStorageInventory(roots, *, deviceType=None):
    storages={};members={}
    for group,root in roots.items():
        seen=set();found=set()
        def visit(value,path):
            if isinstance(value,torch.Tensor):
                if deviceType is not None and value.device.type!=deviceType:return
                storage=value.untyped_storage();key=f'{value.device}:{storage.data_ptr()}:{storage.nbytes()}'
                row=storages.setdefault(key,dict(address=storage.data_ptr(),bytes=storage.nbytes(),device=str(value.device),owners=[],examples=[]))
                if group not in row['owners']:row['owners'].append(group)
                if len(row['examples'])<12:row['examples'].append(dict(path=path,shape=list(value.shape),dtype=str(value.dtype),tensorBytes=value.numel()*value.element_size(),storageOffset=value.storage_offset()))
                found.add(key);return
            if value is None or id(value) in seen:return
            seen.add(id(value))
            if isinstance(value,dict):
                for k,v in value.items():visit(v,path+'.'+str(k))
            elif isinstance(value,(list,tuple)):
                for i,v in enumerate(value):visit(v,path+f'[{i}]')
            elif is_dataclass(value) and not isinstance(value,type):
                for f in fields(value):visit(getattr(value,f.name),path+'.'+f.name)
            elif isinstance(value,SimpleNamespace) or (type(value).__module__.startswith(('warpSPH','warpSPHCore','warpSPHIntegrators')) and hasattr(value,'__dict__')):
                for k,v in vars(value).items():visit(v,path+'.'+k)
        visit(root,group);members[group]=found
    claimed=set();groups={}
    for group,keys in members.items():
        new=keys-claimed;claimed|=keys
        groups[group]=dict(uniqueBytes=sum(storages[k]['bytes'] for k in keys),incrementalBytes=sum(storages[k]['bytes'] for k in new),storageCount=len(keys))
    return dict(unionBytes=sum(s['bytes'] for s in storages.values()),groups=groups,
                sharedBytes=sum(s['bytes'] for s in storages.values() if len(s['owners'])>1),storages=list(storages.values()))
