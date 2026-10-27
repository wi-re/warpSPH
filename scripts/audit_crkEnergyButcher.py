"""Temporal coupling counterexample, not an SPH fluid benchmark.

Two masses exchange equal/opposite energy-dependent forces. Raw dudt supplies
partition weights but differs from the compatible energy derivative. Compare
actual production midpoint stage construction to that compatible ODE.
No spatial solve, particle sampling or physical trajectory is involved.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(out):
    import torch
    from scipy.integrate import solve_ivp
    from types import SimpleNamespace as NS
    from warpSPH.systems.compSPH import CompSPHSystem,CompSPHState
    from warpSPH.systems.compressibleMonaghan import CompressibleSystemUpdate
    from warpSPH.modules.compSPH.pairWork import beginPairWork
    from warpSPH.modules.crk.energyControl import retain
    from warpSPH.enumTypes import EnergyScheme
    from warpSPHIntegrators.drivers.runge_kutta import RungeKuttaB
    class Toy(CompSPHSystem):
        def initialize(self,dt,*args,stageWeights=None,**kwargs):
            self.pairWork=beginPairWork(stageWeights);return self
        def initializeNewState(self,*args,**kwargs):
            return Toy(state=self.state.initializeNewState(),pairWork=self.pairWork,t=self.t)
        def finalize(self,*args,**kwargs):return self
    mass=np.array([1.,.25]);v0=np.array([.4,-.6]);u0=np.array([2.,8.]);end=.04
    def exact_rhs(t,y):
        v,u=y[:2],y[2:];F=-.2*u.sum();acc=F*np.array([1/mass[0],-1/mass[1]])
        raw=np.array([1+.2*u[0],3+.1*u[1]]);work=(v[1]-v[0])*acc[0]
        weights=np.maximum(np.finfo(float).eps,raw*np.sign(work));f=weights[0]/weights.sum()
        return np.r_[acc,[f*work,(1-f)*work*mass[0]/mass[1]]]
    reference=solve_ivp(exact_rhs,(0,end),np.r_[v0,u0],method='DOP853',rtol=2e-13,atol=2e-14).y[:,-1]
    adj=NS(paddedI=torch.tensor([0,1]),paddedJ=torch.tensor([1,0]),numNeighbors=torch.ones(2,dtype=torch.int64))
    tensor=lambda x:torch.tensor(x,dtype=torch.float64);ids=torch.zeros(2,dtype=torch.int32)
    tableau=NS(a=np.array([[0.,0.],[.5,0.]]),b=np.array([0.,1.]),c=np.array([0.,.5]))
    rows=[]
    for predictor in ('raw','compatible'):
        config=NS(crkEnergyPartition='spheral',crkEnergyPredictor=predictor,energyScheme=EnergyScheme.CRK)
        def rhs(system,dt,stageIndex=None,**kwargs):
            s=system.state;F=-.2*s.internalEnergies.sum();acc=torch.stack((F/s.masses[0],-F/s.masses[1]))[:,None]
            raw=torch.stack((1+.2*s.internalEnergies[0],3+.1*s.internalEnergies[1]))
            update=CompressibleSystemUpdate(dxdt=s.velocities,dvdt=acc,dudt=raw,drhodt=torch.zeros(2),dEdt=torch.zeros(2))
            s.ap_ij=acc;s.av_ij=torch.zeros_like(acc);s.f_ij=tensor([.5,.5])
            retain(system,s,update,adj,stageIndex,config);s.ap_ij=s.av_ij=s.f_ij=None
            return update,adj,s
        previous=None
        for steps in (4,8,16,32,64):
            state=CompSPHState(positions=tensor([[0.],[1.]]),velocities=tensor(v0[:,None]),masses=tensor(mass),
                supports=tensor([1.,1.]),densities=tensor([1.,1.]),kinds=ids,materials=ids,UIDs=torch.arange(2),
                UIDcounter=2,internalEnergies=tensor(u0))
            system=Toy(state=state)
            for step in range(steps):system=RungeKuttaB(system,end/steps,rhs,tableau,schemeConfig=config).state
            s=system.state;v=s.velocities.numpy().ravel();u=s.internalEnergies.numpy()
            error=float(np.linalg.norm(np.r_[v,u]-reference))
            row=dict(predictor=predictor,steps=steps,dt=end/steps,errorL2=error,
                observedOrder=float(np.log2(previous/error)) if previous else None,
                totalEnergyAbsDrift=float(abs(np.dot(mass,u+.5*v*v)-np.dot(mass,u0+.5*v0*v0))))
            rows.append(row);previous=error;print(row,flush=True)
    data=dict(rows=rows,reference=reference.tolist(),end=end,
        model='m=(1,.25), v0=(.4,-.6), u0=(2,8), F=-.2 sum(u), raw dudt=(1+.2u0,3+.1u1). Compatible ODE uses sign-filtered raw rates as pair-work weights. Actual Butcher midpoint and production energy control; no SPH spatial operator.',
        limits='Counterexample to inferring coupled thermodynamic order from Butcher order or energy conservation alone. Not a measured temporal order of shock-bubble or of arbitrary SPH flows; no historical regression attribution.')
    out.mkdir(parents=True,exist_ok=True);(out/'temporalCoupling.json').write_text(json.dumps(data,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for mode in ('raw','compatible'):
        r=[v for v in rows if v['predictor']==mode]
        ax.loglog([v['dt'] for v in r],[v['errorL2'] for v in r],'-o',label=mode+' predictor')
    ax.set(xlabel='Time step',ylabel='State L2 error versus compatible ODE',title='Two-mass temporal coupling counterexample')
    ax.grid(alpha=.2);ax.legend();fig.savefig(out/'temporalCoupling.png',dpi=180);plt.close(fig)
    return data


if __name__=='__main__':
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.output)
