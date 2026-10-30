from dataclasses import dataclass
from types import SimpleNamespace
import torch
from warpSPH.utils.memoryAudit import tensorStorageInventory


def test_views_and_aliases_count_underlying_storage_once():
    x=torch.arange(32,dtype=torch.float32);view=x[4:8]
    r=tensorStorageInventory(dict(state=x,retained=[view,x]))
    assert r['unionBytes']==128 and r['sharedBytes']==128
    assert r['groups']['state']['incrementalBytes']==128
    assert r['groups']['retained']['incrementalBytes']==0
    assert len(r['storages'])==1


def test_stored_fields_exclude_computed_tensor_property_and_handle_cycles():
    @dataclass
    class State:
        x: object
        @property
        def Q(self):raise AssertionError('inventory must not materialize output properties')
    a=torch.ones(7);s=State(a);cycle=SimpleNamespace(state=s);cycle.self=cycle
    r=tensorStorageInventory(dict(state=s,holder=cycle),deviceType='cpu')
    assert r['unionBytes']==28 and r['groups']['holder']['incrementalBytes']==0


def test_independent_clones_and_device_filter_are_distinguished():
    x=torch.ones(7);r=tensorStorageInventory(dict(state=x,scratch=x.clone()))
    assert r['unionBytes']==56 and r['sharedBytes']==0
    assert tensorStorageInventory(dict(state=x),deviceType='cuda')['unionBytes']==0


def test_package_owned_container_exposes_stored_aliases_without_properties():
    class Holder:
        @property
        def computed(self):raise AssertionError('stored fields only')
    Holder.__module__='warpSPHCore.example'
    x=torch.ones(11);holder=Holder();holder.value=x;holder.again=x[2:]
    r=tensorStorageInventory(dict(holder=holder,state=x))
    assert r['unionBytes']==44 and r['sharedBytes']==44
    assert r['groups']['state']['incrementalBytes']==0
