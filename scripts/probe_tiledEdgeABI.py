"""CPU-only layout feasibility, not a solver/storage implementation.

Wide logical arrays alias one small value with zero strides. They establish
shape/accessor/address-splitting support, not large physical memory access.
"""
from pathlib import Path
import json
import argparse
import torch
import warp as wp

BITS=16
TILE=1<<BITS

@wp.struct
class EdgeIndexView:
    values: wp.array2d(dtype=wp.int32)
    shift: wp.int32
    mask: wp.int64

@wp.func
def edge_index(view:EdgeIndexView,at:wp.int64):
    return view.values[wp.int32(at >> wp.int64(view.shift)),wp.int32(at & view.mask)]

@wp.kernel
def read_indices(view:EdgeIndexView,at:wp.array(dtype=wp.int64),out:wp.array(dtype=wp.int32)):
    i=wp.tid()
    out[i]=edge_index(view,at[i])

@wp.kernel
def read_vectors(a:wp.array2d(dtype=wp.vec3),at:wp.array(dtype=wp.int64),out:wp.array(dtype=wp.vec3)):
    i=wp.tid();p=at[i]
    out[i]=a[wp.int32(p >> wp.int64(BITS)),wp.int32(p & wp.int64(TILE-1))]

@wp.kernel
def update_vectors(a:wp.array2d(dtype=wp.vec3),at:wp.array(dtype=wp.int64)):
    p=at[wp.tid()]
    tile=wp.int32(p >> wp.int64(BITS));slot=wp.int32(p & wp.int64(TILE-1))
    a[tile,slot]=a[tile,slot]+wp.vec3(1.,2.,3.)


def index_view(tensor,shift):
    result=EdgeIndexView()
    result.values=wp.from_torch(tensor)
    result.shift=shift
    result.mask=(1<<shift)-1
    return result


def main():
    print('PHASE init',flush=True)
    wp.init()
    wp.config.verify_fp=True
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('.tmp/tiled-edge-abi'))
    root=parser.parse_args().out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    tests={}
    print('PHASE small allocation',flush=True)
    count=2*TILE
    indices=torch.tensor([0,TILE-1,TILE,TILE+1,count-1],dtype=torch.int64)
    src=torch.arange(count,dtype=torch.int32)
    tiled=src.view(2,TILE)
    assert tiled.data_ptr()==src.data_ptr()
    expected=src[indices].clone()
    for label,a in [('flat',index_view(src.as_strided((1,count),(0,1)),31)),('tiled',index_view(tiled,BITS))]:
        print('PHASE integer',label,flush=True)
        out=torch.zeros(len(indices),dtype=torch.int32)
        wp.launch(read_indices,dim=len(indices),inputs=[a,wp.from_torch(indices),wp.from_torch(out)],device='cpu')
        tests[label+'-integer-reads']=torch.equal(out,expected)
    print('PHASE vectors',flush=True)
    vectors=torch.arange(count*3,dtype=torch.float32).reshape(count,3)
    view=vectors.view(2,TILE,3)
    assert view.data_ptr()==vectors.data_ptr()
    wv=wp.from_torch(view,dtype=wp.vec3)
    before=vectors[indices].clone()
    wp.launch(update_vectors,dim=len(indices),inputs=[wv,wp.from_torch(indices)],device='cpu')
    tests['vector-stores-cross-tile']=torch.equal(vectors[indices],before+torch.tensor([1.,2.,3.]))
    print('PHASE high scalar',flush=True)
    high=2**31+7
    ids=torch.tensor([2**31-1,2**31,high],dtype=torch.int64)
    anchor=torch.tensor([37],dtype=torch.int32)
    wide=anchor.as_strided((high//TILE+1,TILE),(0,0))
    out=torch.zeros(len(ids),dtype=torch.int32)
    wp.launch(read_indices,dim=len(ids),inputs=[index_view(wide,BITS),wp.from_torch(ids),wp.from_torch(out)],device='cpu')
    tests['above-intmax-logical-reads']=torch.equal(out,torch.full_like(out,37))
    print('PHASE high vector',flush=True)
    vector_anchor=torch.tensor([1.,2.,3.])
    wide_v=vector_anchor.as_strided((high//TILE+1,TILE,3),(0,0,1))
    out_v=torch.zeros(len(ids),3)
    wp.launch(read_vectors,dim=len(ids),inputs=[wp.from_torch(wide_v,dtype=wp.vec3),wp.from_torch(ids),wp.from_torch(out_v,dtype=wp.vec3)],device='cpu')
    tests['above-intmax-logical-vector-reads']=torch.equal(out_v,vector_anchor.expand_as(out_v))
    print('PHASE flat rejection',flush=True)
    try:
        wp.from_torch(anchor.as_strided((high+1,),(0,)))
    except ValueError as error:
        flat_error=str(error);tests['flat-intmax-rejected']='signed 32-bit' in flat_error
    else:
        flat_error=None;tests['flat-intmax-rejected']=False
    assert all(tests.values()),tests
    result=dict(status='passed',checks=tests,warpVersion=wp.__version__,device='cpu',tileWidth=TILE,
                smallBackingBytes=src.untyped_storage().nbytes()+vectors.untyped_storage().nbytes(),
                wideLogicalEntries=wide.numel(),wideBackingBytes=anchor.untyped_storage().nbytes(),
                wideVectorBackingBytes=vector_anchor.untyped_storage().nbytes(),flatError=flat_error,
                limitation='Wide arrays use zero strides and do not prove physical >8GiB pointer access, CUDA performance, AD or solver correctness.')
    (root/'abi-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
