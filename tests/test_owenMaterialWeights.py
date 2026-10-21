"""Dimension-generic material-volume weights, with unchanged Owen policy."""
import pytest
import torch
from warpSPHCore import ParticleState,DomainDescription,OperationProperties,KernelFunctions,SupportScheme,buildVerletList,get_torch_precision
from warpSPH.modules.adaptiveSupport.wp_psi0 import computePsi0Warp
from warpSPH.modules.adaptiveSupport.wp_materialPsi import computeMaterialPsi0Warp


@pytest.fixture(autouse=True)
def restore_neighbor_lanes():
    yield
    from warpSPHCore import setNeighborLanes
    setNeighborLanes(None)


@pytest.mark.parametrize('dim',[1,2,3])
@pytest.mark.parametrize('traversal,lanes',[('list',1),('list',32),('grid',32)])
def test_material_moment_weights_in_all_dimensions(dim,traversal,lanes):
    from warpSPHCore import setNeighborLanes,buildCompactHashMap
    setNeighborLanes(lanes)
    dtype=get_torch_precision();device='cuda' if torch.cuda.is_available() else 'cpu'
    T=lambda x:torch.tensor(x,dtype=dtype,device=device)
    x=torch.zeros((2,dim),dtype=dtype,device=device);x[1,0]=.2
    s=ParticleState(positions=x,supports=T([1.,1.]),masses=T([1.,1.]),
        kinds=torch.zeros(2,dtype=torch.int32,device=device),densities=T([1.,2.]))
    s.materials=torch.tensor([0,1],dtype=torch.int32,device=device)
    domain=DomainDescription(T([-2.]*dim),T([2.]*dim),torch.zeros(dim,dtype=torch.bool,device=device),dim)
    adj=buildVerletList(s,domain,verletScale=1.4,supportMode=SupportScheme.SuperSymmetric)
    if traversal=='grid':
        adj=buildCompactHashMap(s.positions,s.positions,s.supports,s.supports,domain.periodic,domain)
    p=OperationProperties(kernel=KernelFunctions.B8,supportMode=SupportScheme.Gather)
    plain=computePsi0Warp(s,p,domain,adjacency=adj);weighted=computeMaterialPsi0Warp(s,p,domain,adjacency=adj)
    # Only one nonzero gradient contribution per row; self gradient is zero.
    expected=plain[1]*T([.5,2.]).pow(1/dim)
    tol=3.e-14 if dtype==torch.float64 else 3.e-6
    torch.testing.assert_close(weighted[1],expected,rtol=tol,atol=tol)
    # Common mass/density scaling cancels in the cross-material volume ratio.
    s.masses*=7.;s.densities*=3.
    torch.testing.assert_close(computeMaterialPsi0Warp(s,p,domain,adjacency=adj)[1],weighted[1],rtol=tol,atol=tol)
    # A single physical material must retain the original unweighted result,
    # even with unequal particle volumes. Emitter IDs cannot affect this.
    s.materials[:]=0;s.emitters=torch.tensor([0,1],dtype=torch.int32,device=device)
    for a,b in zip(computeMaterialPsi0Warp(s,p,domain,adjacency=adj),plain):
        torch.testing.assert_close(a,b,rtol=tol,atol=tol)
    s.materials[:]=torch.tensor([0,1],dtype=torch.int32,device=device);s.densities=s.masses.clone()
    for a,b in zip(computeMaterialPsi0Warp(s,p,domain,adjacency=adj),plain):
        torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_material_support_option_roundtrip_and_explicit_opt_in():
    from warpSPH.configurations import CRKSPHConfig,MaterialConfig
    from warpSPH.configurations.compressibleConfig import compressibleConfigToDict,dictToCompressibleConfig
    c=CRKSPHConfig();assert not c.owenMaterialWeights
    c.owenMaterialWeights=True
    with pytest.raises(ValueError,match='materials table'):c.validate()
    c.materials=[MaterialConfig()];c.validate()
    assert dictToCompressibleConfig(compressibleConfigToDict(c)).owenMaterialWeights
    old=compressibleConfigToDict(c);old.pop('owenMaterialWeights')
    assert not dictToCompressibleConfig(old).owenMaterialWeights
