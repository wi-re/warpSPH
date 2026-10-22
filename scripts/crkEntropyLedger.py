"""Read-only entropy accounting for fresh-midpoint CRK/paper density.

The accepted path is reset -> raw full-step candidate -> compatible endpoint.
The half-step predictor is a separate branch, never an additive budget term.
All snapshots use rho/u, avoiding stale EOS fields during state updates.
"""
import copy
import importlib
from pathlib import Path
import numpy as np


def entropy(rho,u,gamma=1.4):
    rho,u=np.asarray(rho),np.asarray(u)
    if np.any(rho<=0) or np.any(u<=0):
        raise ValueError('Entropy accounting requires positive density and thermal energy')
    return np.log((gamma-1)*u)+(1-gamma)*np.log(rho)


def raw_components(rho0,u0,rho1,u1,pressure_rate,dt,gamma=1.4):
    """Exact finite-increment split; logarithmic-mean allocation of thermal work.

    pressure+viscosity = log(u1/u0), density = (1-gamma)log(rho1/rho0).
    This bookkeeping allocation is not a separate physical trajectory.
    """
    entropy(rho0,u0,gamma);entropy(rho1,u1,gamma)
    z=(u1-u0)/u0
    ratio=np.ones_like(z,dtype=float)
    np.divide(np.log1p(z),z,out=ratio,where=z!=0)
    thermal=np.log1p(z)
    pressure=dt*np.asarray(pressure_rate)*ratio/u0
    return dict(density=(1-gamma)*np.log(np.asarray(rho1)/rho0),
        pressureHeating=pressure,viscousHeating=thermal-pressure)


def snapshot(state):
    fields=dict(rho='densities',u='internalEnergies',x='positions',v='velocities',m='masses',uid='UIDs')
    return {k:getattr(state,v).detach().cpu().numpy().copy() for k,v in fields.items()}


def pressure_only_config(config):
    # Warp structs cache a ctypes representation: shallow-copying the struct
    # shares that storage even when its Python attributes look independent.
    shadow=copy.copy(config)
    shadow.diffusionParams=copy.deepcopy(config.diffusionParams)
    shadow.diffusionParams.C_l=shadow.diffusionParams.C_q=0.
    return shadow


class EntropyLedger:
    def __init__(self,settings):
        self.s=settings;self.g=settings['gamma'];self.steps=[];self.rhs=[];self.arrays={};self.current=None
        self.counters=dict(resets=0,rhs=0,updates=0,finalizations=0,negativeEOSCalls=0,supportCalls=0)
    def average(self,a,mask=None):
        m=self.initial['m'];mask=np.ones(len(m),dtype=bool) if mask is None else mask
        return float(np.dot(m[mask],a[mask])/m[mask].sum())
    def begin(self,system):
        assert self.current is None,'Unexpected repeated step or unclosed ledger'
        before=snapshot(system.state)
        if not self.steps:
            self.initial=before
            x=before['x'][:,0]
            self.masks=dict(globalBox=np.ones(len(x),bool),centralInitial=abs(x)<.5,
                interfaceInitial=abs(x)<.05,seamInitial=abs(x)>1.45)
            if self.s['case']=='contact':
                self.masks.update(interfaceInitial=abs(abs(x)-.25)<.05,seamInitial=abs(x)>.45)
        np.testing.assert_array_equal(before['uid'],self.initial['uid'])
        self.current=dict(before=before,time=float(system.t),rhs=[])
    def reset(self,system):
        self.current['base']=snapshot(system.state);self.counters['resets']+=1
        np.testing.assert_array_equal(self.current['base']['u'],self.current['before']['u'])
    def derivative(self,state,rate,pressure_rate,gradient):
        a=snapshot(state);drho=-a['rho']*np.trace(gradient.detach().cpu().numpy(),axis1=-2,axis2=-1)
        du=rate.detach().cpu().numpy().copy();p=pressure_rate.detach().cpu().numpy().copy()
        d=dict(state=a,du=du,pressure=p,drho=drho)
        self.current['rhs'].append(d);self.counters['rhs']+=1
        stage=len(self.current['rhs'])-1
        row=dict(step=len(self.steps),stage=stage,time=self.current['time']+stage*.5*self.s['dt'],
            minDensity=float(a['rho'].min()),minEnergy=float(a['u'].min()))
        for name,mask in self.masks.items():
            row[name]=dict(pressureEntropyRate=self.average(p/a['u']+(1-self.g)*drho/a['rho'],mask),
                viscousEntropyRate=self.average((du-p)/a['u'],mask))
        self.rhs.append(row)
    def updated(self,system,spec):
        self.counters['updates']+=1
        n=len(self.current['rhs']);assert n in (1,2)
        a=snapshot(system.state);base=self.current['base'];rates=self.current['rhs'][-1]
        expected=.5*self.s['dt'] if n==1 else self.s['dt']
        assert abs(float(spec.derivative_dt)-expected)<1.e-14
        np.testing.assert_allclose(a['u'],base['u']+expected*rates['du'],rtol=0,atol=2.e-14)
        np.testing.assert_allclose(a['rho'],base['rho']+expected*rates['drho'],rtol=0,atol=2.e-14)
        self.current['predictor' if n==1 else 'raw']=a
    def finish(self,system,raw,dt):
        assert system.rerunRequest is None,'This audit does not fold rejected attempts into accepted budgets'
        self.counters['finalizations']+=1
        c=self.current;a,b,f=c['before'],c['base'],snapshot(system.state)
        assert len(c['rhs'])==2
        for k in ('rho','u'):
            np.testing.assert_array_equal(raw[k],c['raw'][k])
        d=raw_components(b['rho'],b['u'],raw['rho'],raw['u'],c['rhs'][-1]['pressure'],dt,self.g)
        d.update(reset=entropy(b['rho'],b['u'],self.g)-entropy(a['rho'],a['u'],self.g),
            compatible=np.log(f['u']/raw['u']),finalizationDensity=(1-self.g)*np.log(f['rho']/raw['rho']))
        np.testing.assert_array_equal(f['rho'],raw['rho'])
        net=entropy(f['rho'],f['u'],self.g)-entropy(a['rho'],a['u'],self.g)
        residual=net-sum(d.values())
        d.update(net=net,closure=residual,
            predictorBranch=entropy(c['predictor']['rho'],c['predictor']['u'],self.g)-entropy(b['rho'],b['u'],self.g))
        row=dict(step=len(self.steps),time=float(system.t),closureMax=float(abs(residual).max()))
        for name,mask in self.masks.items():
            row[name]={k:self.average(v,mask) for k,v in d.items()}
        self.steps.append(row)
        for key,value in d.items():self.arrays.setdefault(key,[]).append(value)
        self.final=f;self.current=None
    def summary(self):
        if not self.steps:return dict(counters=self.counters,steps=0)
        totals={k:np.sum(v,axis=0) for k,v in self.arrays.items()}
        true=entropy(self.final['rho'],self.final['u'],self.g)-entropy(self.initial['rho'],self.initial['u'],self.g)
        return dict(counters=self.counters,steps=len(self.steps),
            maximumStepClosure=max(r['closureMax'] for r in self.steps),
            maximumCumulativeClosure=float(abs(totals['net']-sum(totals[k] for k in
                ('reset','density','pressureHeating','viscousHeating','compatible','finalizationDensity'))).max()),
            maximumEndpointResidual=float(abs(totals['net']-true).max()),
            groups={name:{k:self.average(v,mask) for k,v in totals.items()} for name,mask in self.masks.items()},
            predictorPolicy='Separate off-path diagnostic, never included in the accepted-path sum')
    def save(self,folder):
        from compare_spheralMaterials import save
        save(folder/'entropySummary.json',self.summary());save(folder/'entropySteps.json',self.steps);save(folder/'entropyRHS.json',self.rhs)
        if self.steps:
            np.savez_compressed(folder/'entropyLedger.npz',**{k:np.stack(v) for k,v in self.arrays.items()},
                initialPositions=self.initial['x'],finalPositions=self.final['x'],masses=self.initial['m'],UIDs=self.initial['uid'])


def install(settings):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    ledger=EntropyLedger(settings)
    evolution=importlib.import_module('warpSPH.modules.crk.evolution')
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    system=importlib.import_module('warpSPH.systems.compSPH')
    eos=importlib.import_module('warpSPH.modules.eos.materials')
    prepare,du,apply,finalize,evaluate,solve=(evolution.prepareStep,module.computeCRKdudt,system.CompSPHSystem.apply_state_update,
        system.CompSPHSystem.finalize,eos.evaluateEOS,module.solveStageSupport)
    def reset(host,*args,**kwargs):
        ledger.begin(host);result=prepare(host,*args,**kwargs);ledger.reset(host);return result
    def derivative(state,c,sc,adj,stage,*args):
        assert sc.compatibleEnergy and sc.crkEvolution=='paper' and sc.crkEnergyPartition=='native'
        assert sc.crkEnergyPredictor=='raw' and sc.crkViscosityGradient=='current'
        result=du(state,c,sc,adj,stage,*args)
        shadow=pressure_only_config(sc)
        pressure=du(state,c,shadow,adj,stage,*args)
        ledger.derivative(state,result,pressure,stage.velocityGradient)
        return result
    def updated(host,update,spec,**kwargs):
        result=apply(host,update,spec,**kwargs);ledger.updated(host,spec);return result
    def finished(host,initial,dt,*args,**kwargs):
        raw=snapshot(host.state);result=finalize(host,initial,dt,*args,**kwargs);ledger.finish(host,raw,dt);return result
    def observed_eos(rho,*args,**kwargs):
        u=kwargs.get('u');ledger.counters['negativeEOSCalls']+=int(bool((rho<=0).any() or (u is not None and (u<=0).any())))
        return evaluate(rho,*args,**kwargs)
    def observed_support(host,state,c,sc,**kwargs):
        assert sc.owenTable=='hermiteShell' and sc.owenMaterialWeights and sc.adaptiveSupportIterations==1
        ledger.counters['supportCalls']+=1;return solve(host,state,c,sc,**kwargs)
    evolution.prepareStep=reset;module.computeCRKdudt=derivative
    system.CompSPHSystem.apply_state_update=updated;system.CompSPHSystem.finalize=finished
    eos.evaluateEOS=observed_eos;module.solveStageSupport=observed_support
    return ledger
