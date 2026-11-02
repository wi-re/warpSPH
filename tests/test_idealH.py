"""Independent scalar support and fresh-midpoint replacement contracts."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_gpu_translation_reduces_to_volume_ratio_and_independent_relaxation(tmp_path,monkeypatch):
    import warp as wp
    from warpSPHCore import get_torch_precision,ParticleState,DomainDescription
    if get_torch_precision()!=torch.float64:pytest.skip('IdealH comparison is float64')
    monkeypatch.setattr(wp.config,'kernel_cache_dir','/tmp/warpsph-idealh-test-kernels')
    from crkIdealH import ideal_support
    table=tmp_path/'tables.npz'
    # Manufactured gradient |W'|=1 and exact identity nPerh(root)=root.
    np.savez(table,gradMeta=[0.,4.,4.,3.],gradData=[1.,0.,0.],lookupMeta=[0.,10.,10.,2.],lookupData=[0.,10.,1.,1.])
    T=lambda a:torch.tensor(a,dtype=torch.float64)
    s=ParticleState(T([[0.,0.],[.1,0.]]),T([1.,1.]),T([1.,2.]),torch.zeros(2,dtype=torch.int32),T([1.,1.]))
    s.materials=torch.tensor([0,1],dtype=torch.int32)
    domain=DomainDescription(T([-5.,-5.]),T([5.,5.]),torch.tensor([False,False]),2)
    adj=NS(numNeighbors=torch.tensor([1,1],dtype=torch.int32),edgeOffsets=torch.tensor([0,1],dtype=torch.int32),paddedJ=torch.tensor([1,0],dtype=torch.int32))
    from warpSPHCore import offset_t,index_t
    adj.edgeOffsets=adj.edgeOffsets.to(torch.int64 if offset_t==wp.int64 else torch.int32)
    adj.paddedJ=adj.paddedJ.to(torch.int64 if index_t==wp.int64 else torch.int32)
    out,z,n=ideal_support(s,domain,adj,table)
    expected=np.sqrt([2.,.5]);ratio=1/expected;a=np.where(ratio>=1,.4*(1+ratio**-3),.4*(1+ratio**2))
    np.testing.assert_allclose(z.numpy(),expected,rtol=2.e-15)
    np.testing.assert_allclose(out.numpy(),1-a+a*ratio,rtol=2.e-15)
    s.masses*=17.;s.densities*=3.
    torch.testing.assert_close(ideal_support(s,domain,adj,table)[0],out,atol=1.e-15,rtol=0)
    s.materials[:]=0
    plain=ideal_support(s,domain,adj,table)[0]
    torch.testing.assert_close(plain,T([1.,1.]),atol=1.e-15,rtol=0)


def test_replacement_targets_follow_real_midpoint_updates_without_derivative_reuse(monkeypatch):
    import importlib
    from warpSPH.systems.compSPH import CompSPHState,CompSPHSystem
    from warpSPH.systems.compressibleMonaghan import CompressibleSystemUpdate
    from warpSPH.modules.compSPH.pairWork import beginPairWork
    from warpSPHIntegrators.drivers.runge_kutta import RungeKuttaB
    from study_idealH import install_support
    import warpSPHCore
    import crkIdealH
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    preparation=importlib.import_module('warpSPH.modules.compressibleStage')
    monkeypatch.setattr(preparation,'solveStageSupport',lambda *a,**k:None)
    monkeypatch.setattr(module,'computeCRKdudt',lambda state,*a:torch.zeros_like(state.densities))
    monkeypatch.setattr(warpSPHCore,'buildVerletList',lambda *a,**k:NS())
    monkeypatch.setattr(crkIdealH,'ideal_support',lambda state,*a,**k:(.5*state.supports+state.positions[:,0],None,None))
    original_apply=CompSPHSystem.apply_state_update
    monkeypatch.setattr(CompSPHSystem,'apply_state_update',original_apply)
    def ordinary_finalize(system,initial,dt,values,*a,**k):
        system.state.supports=values[-1][1].supports.clone();return system
    monkeypatch.setattr(CompSPHSystem,'finalize',ordinary_finalize)
    audit=install_support(dict(supportPolicy='replacement',steps=1,idealTable='unused'))
    class Toy(CompSPHSystem):
        def initialize(self,dt,*args,stageWeights=None,**kwargs):self.pairWork=beginPairWork(stageWeights);return self
        def initializeNewState(self,*args,**kwargs):return Toy(state=self.state.initializeNewState(),pairWork=self.pairWork,t=self.t)
    T=lambda a:torch.tensor(a,dtype=torch.float64);ids=torch.zeros(2,dtype=torch.int32);one=T([1.,1.]);R0=T([1.,2.]);x0=T([[.3,0.],[.6,0.]]);v=T([[.1,0.],[-.05,0.]])
    state=CompSPHState(positions=x0.clone(),velocities=v,supports=R0.clone(),masses=one,densities=one,kinds=ids,materials=ids,UIDs=torch.arange(2),UIDcounter=2,internalEnergies=one)
    sc=NS(adaptiveSupportIterations=1,crkEvolution='paper',compatibleEnergy=True,crkEnergyPartition='native',crkViscosityGradient='current',crkEnergyPredictor='raw')
    c=NS(domain=None,verletScale=1.4,compute=None);seen=[]
    # The static neighbor stand-in needs no search configuration.
    import warpSPH.configurations.compute as compute
    monkeypatch.setattr(compute,'neighborSearchOptions',lambda c:{})
    def rhs(system,dt,**kwargs):
        preparation.solveStageSupport(system,system.state,c,sc);s=system.state;seen.append(s.supports.clone())
        module.computeCRKdudt(s,c,sc,system.adjacency,None)
        return CompressibleSystemUpdate(dxdt=s.velocities,dvdt=torch.zeros_like(v),dudt=0*one,drhodt=0*one,dEdt=0*one),system.adjacency,s
    tableau=NS(a=np.array([[0.,0.],[.5,0.]]),b=np.array([0.,1.]),c=np.array([0.,.5]));dt=.2
    out=RungeKuttaB(Toy(state=state),dt,rhs,tableau,config=c,schemeConfig=sc)
    midtarget=.5*R0+x0[:,0];finaltarget=.5*midtarget+(x0+.5*dt*v)[:,0]
    torch.testing.assert_close(seen[0],R0);torch.testing.assert_close(seen[1],midtarget)
    torch.testing.assert_close(out.state.state.supports,finaltarget)
    assert audit['rhs']==2 and audit['replacements']==2 and audit['finalReplacements']==1
