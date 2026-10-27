"""Independent legacy Butcher sums and actual compiled Spheral stage energies."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
import torch
from warpSPH.configurations.crkSPH import CRKSPHConfig,crkSPHConfigToDict,dictToCRKSPHConfig
from warpSPH.configurations import MaterialConfig
from warpSPH.enumTypes import EnergyScheme
from warpSPH.modules.compSPH.pairWork import beginPairWork,accumulateStage,energyIncrement
from warpSPH.modules.crk.energyControl import EnergyStage,energyBeforeUpdate,spheralRate
from warpSPHIntegrators.state.specs import ComponentUpdateSpec


def directed(i,j,pair,n):
    order=np.argsort(np.r_[i,j],kind='stable');owner=np.r_[i,j][order];other=np.r_[j,i][order]
    return NS(paddedI=torch.tensor(owner),paddedJ=torch.tensor(other),
              numNeighbors=torch.tensor(np.bincount(owner,minlength=n))),order


@pytest.mark.parametrize('weights',[(1.,),(0.,1.),(.5,.5),(1/6,1/3,1/3,1/6),(1.,-2.,2.)])
def test_running_sum_and_list_rerun_match_independent_old_butcher_sum_and_conserve(weights):
    rng=np.random.default_rng(421);m=torch.tensor([1.,.25,2.,.5],dtype=torch.float64)
    i=np.array([0,0,1,2]);j=np.array([1,3,2,3]);adj,order=directed(i,j,None,4)
    I,J=adj.paddedI,adj.paddedJ;v=torch.tensor(rng.normal(size=(4,2)));dt=.03
    stages=[];updates=[];pw=beginPairWork(weights)
    for s,k in enumerate(weights):
        F=torch.tensor(rng.normal(size=(4,2)));ap=torch.cat((F/m[i,None],-F/m[j,None]))[order]
        f=torch.tensor(rng.random(4));fraction=torch.cat((f,1-f))[order]
        acc=torch.zeros_like(v).index_add(0,I,ap);updates.append(NS(dvdt=acc))
        state=NS(ap_ij=ap,av_ij=torch.zeros_like(ap),f_ij=fraction);stages.append(state)
        accumulateStage(pw,s,adj,state,EnergyScheme.CRK)
    initial=NS(velocities=v,internalEnergies=torch.ones(4,dtype=torch.float64))
    total=sum(k*u.dvdt for k,u in zip(weights,updates));half=v+.5*dt*total
    old=torch.zeros(4,dtype=torch.float64)
    for k,s in zip(weights,stages):
        old.index_add_(0,I,k*s.f_ij*torch.sum((half[J]-half[I])*s.ap_ij,dim=1))
    actual,request=energyIncrement(pw,initial,updates,weights,dt)
    assert request is None;torch.testing.assert_close(actual,old,rtol=1e-13,atol=1e-13)
    kinetic=torch.sum(.5*m*((v+dt*total).square().sum(1)-v.square().sum(1)))
    assert abs(float(kinetic+dt*torch.dot(m,actual)))<1e-13
    rerun=beginPairWork(weights);rerun.halfVelocity=half
    for s,state in enumerate(stages):accumulateStage(rerun,s,adj,state,EnergyScheme.CRK)
    again,request=energyIncrement(rerun,initial,updates,weights,dt)
    torch.testing.assert_close(again,old,rtol=1e-13,atol=1e-13)


def test_actual_compiled_half_step_is_reproduced_by_production_control():
    root=Path(__file__).resolve().parents[1]/'results/materialEOS/bubble_force_2026-10-19/n80-energy-validation/spheral/rhs'
    files=list(root.glob('*.npz'))
    if not files:pytest.skip('Compiled Spheral capture not installed')
    for path in files:
        a=dict(np.load(path));i,j=a['i'],a['j'];m=a['masses'];n=len(m)
        adj,order=directed(i,j,None,n)
        pair=torch.tensor(np.r_[a['nativePairAcceleration'],-a['nativePairAcceleration']*(m[i]/m[j])[:,None]][order])
        t=lambda key:torch.tensor(a[key])
        rate=spheralRate(adj,pair,t('nativeDudt'),t('velocities'),t('nativeAcceleration'),float(a['dt']))
        expected=t('internalEnergies')[a['real']]+float(a['dt'])*rate[a['real']]
        torch.testing.assert_close(expected,t('nativeNextEnergyReal'),rtol=0,atol=1e-12)


@pytest.mark.parametrize('index,predictor,compatible',[(0,'raw',False),(0,'compatible',True),(1,'raw',True),(1,'compatible',True)])
def test_midpoint_controls_consume_only_latest_rhs_and_do_not_use_rerun_half_velocity(index,predictor,compatible):
    adj=NS(paddedI=torch.tensor([0,1]),paddedJ=torch.tensor([1,0]),numNeighbors=torch.ones(2,dtype=torch.int64))
    vel=torch.tensor([[1.],[-1.]],dtype=torch.float64);acc=torch.tensor([[-2.],[4.]],dtype=torch.float64)
    raw=torch.tensor([3.,7.],dtype=torch.float64);u=torch.tensor([2.,8.],dtype=torch.float64)
    holder=beginPairWork((0.,1.));holder.halfVelocity=torch.full_like(vel,999.)
    holder.crkEnergyStage=EnergyStage(index,adj,acc,raw)
    system=NS(pairWork=holder,state=NS(velocities=vel,internalEnergies=u))
    config=NS(crkEnergyPartition='spheral',crkEnergyPredictor=predictor)
    got=energyBeforeUpdate(system,NS(dvdt=acc),ComponentUpdateSpec(.02),config)
    assert holder.crkEnergyStage is None
    if compatible:
        assert got is not None
        mass=torch.tensor([1.,.5],dtype=torch.float64)
        dKE=torch.sum(.5*mass*((vel+.02*acc).square().sum(1)-vel.square().sum(1)))
        assert abs(float(torch.dot(mass,got-u)+dKE))<1e-14
    else:assert got is None


def test_energy_configuration_defaults_roundtrip_and_scope():
    from warpSPH.runner.configuration import _validateSolver
    c=CRKSPHConfig(crkEvolution='spheral',crkDensityPolicy='sameMaterial',materials=[MaterialConfig()],crkEnergyPartition='spheral',crkEnergyPredictor='compatible')
    c.crkViscosityParams.oneSidedWork=True;c.validate()
    d=crkSPHConfigToDict(c);json.dumps(d);b=dictToCRKSPHConfig(d)
    assert b.crkEnergyPartition=='spheral' and b.crkEnergyPredictor=='compatible' and b.crkViscosityParams.oneSidedWork
    assert _validateSolver('CRKSPH',{'crkEnergyPartition':'spheral','crkEnergyPredictor':'compatible','crkViscosityParams':{'oneSidedWork':True}})
    for key in ('crkEnergyPartition','crkEnergyPredictor'):del d[key]
    del d['crkViscosityParams']['oneSidedWork'];b=dictToCRKSPHConfig(d)
    assert b.crkEnergyPartition=='native' and b.crkEnergyPredictor=='raw' and not b.crkViscosityParams.oneSidedWork
    c.crkEvolution='summation'
    c.validate() # independent energy controls no longer force a density/volume change
    with pytest.raises(ValueError,match='midpoint'):
        c.validate(NS(integrationScheme=NS(name='rungeKutta4')))


@pytest.mark.parametrize('predictor',['raw','compatible'])
def test_real_butcher_midpoint_restarts_from_initial_state_and_uses_thermal_predictor(predictor):
    from warpSPH.systems.compSPH import CompSPHSystem,CompSPHState
    from warpSPH.systems.compressibleMonaghan import CompressibleSystemUpdate
    from warpSPH.modules.crk.energyControl import retain
    from warpSPHIntegrators.drivers.runge_kutta import RungeKuttaB
    class Toy(CompSPHSystem):
        def initialize(self,dt,*args,stageWeights=None,**kwargs):
            self.pairWork=beginPairWork(stageWeights);return self
        def initializeNewState(self,*args,**kwargs):
            return Toy(state=self.state.initializeNewState(),pairWork=self.pairWork,t=self.t)
        def finalize(self,*args,**kwargs):return self
    mass=np.array([1.,.25]);v0=np.array([.4,-.6]);u0=np.array([2.,8.]);h=.02
    adj=NS(paddedI=torch.tensor([0,1]),paddedJ=torch.tensor([1,0]),numNeighbors=torch.ones(2,dtype=torch.int64))
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    ids=torch.zeros(2,dtype=torch.int32)
    state=CompSPHState(positions=t([[0.],[1.]]),velocities=t(v0[:,None]),masses=t(mass),supports=t([1.,1.]),
        densities=t([1.,1.]),kinds=ids,materials=ids,UIDs=torch.arange(2),UIDcounter=2,internalEnergies=t(u0))
    c=NS(crkEnergyPartition='spheral',crkEnergyPredictor=predictor,energyScheme=EnergyScheme.CRK)
    seen=[]
    def rhs(system,dt,stageIndex=None,**kwargs):
        s=system.state;F=-.2*s.internalEnergies.sum();acc=torch.stack((F/s.masses[0],-F/s.masses[1]))[:,None]
        raw=torch.stack((1+.2*s.internalEnergies[0],3+.1*s.internalEnergies[1]));seen.append(s.internalEnergies.clone())
        update=CompressibleSystemUpdate(dxdt=s.velocities,dvdt=acc,dudt=raw,drhodt=torch.zeros(2),dEdt=torch.zeros(2))
        s.ap_ij=acc;s.av_ij=torch.zeros_like(acc);s.f_ij=t([.5,.5])
        retain(system,s,update,adj,stageIndex,c);s.ap_ij=s.av_ij=s.f_ij=None
        return update,adj,s
    tableau=NS(a=np.array([[0.,0.],[.5,0.]]),b=np.array([0.,1.]),c=np.array([0.,.5]))
    result=RungeKuttaB(Toy(state=state),h,rhs,tableau,schemeConfig=c)
    def deriv(u):return -.2*u.sum()*np.array([1/mass[0],-1/mass[1]]),np.array([1+.2*u[0],3+.1*u[1]])
    def work(velocity,acc,raw,step):
        vh=velocity+.5*step*acc;w=(vh[1]-vh[0])*acc[0]
        weights=np.maximum(np.finfo(float).eps,raw*np.sign(w));f=weights[0]/weights.sum()
        return np.array([f*w,(1-f)*w*mass[0]/mass[1]])
    acc0,raw0=deriv(u0);mid=u0+.5*h*(raw0 if predictor=='raw' else work(v0,acc0,raw0,.5*h))
    acc1,raw1=deriv(mid);expected=u0+h*work(v0,acc1,raw1,h)
    torch.testing.assert_close(seen[1],t(mid),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(result.state.state.internalEnergies,t(expected),rtol=1e-14,atol=1e-14)
    assert result.state.pairWork.crkEnergyStage is None


def test_energy_conservation_alone_does_not_guarantee_midpoint_thermal_order(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_crkEnergyButcher import audit
    data=audit(tmp_path)
    raw=[r for r in data['rows'] if r['predictor']=='raw']
    compatible=[r for r in data['rows'] if r['predictor']=='compatible']
    assert .9<raw[-1]['observedOrder']<1.1
    assert 1.95<compatible[-1]['observedOrder']<2.05
    assert max(r['totalEnergyAbsDrift'] for r in data['rows'])<1e-12
