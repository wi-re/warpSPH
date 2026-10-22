"""Observation-only energy candidates and frozen RHS captures for CRK controls."""
import importlib
from types import SimpleNamespace as NS
from pathlib import Path
import numpy as np
from crkEntropyLedger import EntropyLedger,snapshot,pressure_only_config


def install(settings,folder):
    from warpSPHBootstrap import bootstrap
    bootstrap(precision='float64')
    import torch
    from crkThermalRate import thermal_entropy_rate
    from warpSPH.modules.compSPH.pairWork import beginPairWork,accumulateStage,energyIncrement
    from compare_spheralMaterials import host
    l=EntropyLedger(settings);l.shadowRows=[];l.frozen=[]
    evolution=importlib.import_module('warpSPH.modules.crk.evolution')
    module=importlib.import_module('warpSPH.schemes.compressible.crkSPH')
    system=importlib.import_module('warpSPH.systems.compSPH')
    eos=importlib.import_module('warpSPH.modules.eos.materials')
    prepare,du,apply,finalize,evaluate,solve,balance=(evolution.prepareStep,module.computeCRKdudt,system.CompSPHSystem.apply_state_update,
        system.CompSPHSystem.finalize,eos.evaluateEOS,module.solveStageSupport,module.computeCompSPHBalanceTerm)
    choice=settings['choice'];folder=Path(folder);(folder/'frozenStates').mkdir()
    def reset(h,*args,**kwargs):
        l.begin(h);result=prepare(h,*args,**kwargs);l.reset(h)
        l.current['shadowWork']=beginPairWork((0.,1.));l.current['seedSupport']=h.state.supports.detach().clone()
        return result
    def derivative(state,c,sc,adj,stage,*args):
        assert sc.compatibleEnergy==choice['compatible'] and sc.crkEvolution=='paper'
        assert sc.crkEnergyPartition=='native' and sc.crkEnergyPredictor=='raw' and sc.crkViscosityGradient=='current'
        base=du(state,c,sc,adj,stage,*args)
        if choice['thermal']=='base':
            selected=base;pressure=du(state,c,pressure_only_config(sc),adj,stage,*args)
        else:
            entropic=state.pressures/state.densities.pow(settings['gamma'])
            selected=thermal_entropy_rate(adj,state.ap_ij+state.av_ij,entropic,state.velocities)
            pressure=thermal_entropy_rate(adj,state.ap_ij,entropic,state.velocities)
        l.derivative(state,selected,pressure,stage.velocityGradient)
        count=len(l.current['rhs']);step=len(l.steps);target=step+1
        if count==2 and (target in {1,*[round(t/settings['dt']) for t in (.02,.04,.06,.1,.15,.2)]}):
            a=host(state);a.update(drhodt=l.current['rhs'][-1]['drho'],dudt=selected.detach().cpu().numpy(),
                interimDudt=base.detach().cpu().numpy(),pressureDudt=pressure.detach().cpu().numpy(),
                supportRate=((state.supports-l.current['seedSupport'])/settings['dt']).detach().cpu().numpy())
            path=folder/'frozenStates'/f'step_{target:04d}.npz';np.savez_compressed(path,**a)
            l.frozen.append(dict(path=str(path),step=target,time=l.current['time']+.5*settings['dt'],
                supportRateConvention='current midpoint H minus previous stored midpoint H, divided by full dt; first step excluded from dynamic-H claims'))
        return selected
    def observed_balance(state,c,sc,adj,*args):
        f=balance(state,c,sc,adj,*args)
        accumulateStage(l.current['shadowWork'],len(l.current['rhs'])-1,adj,
            NS(ap_ij=state.ap_ij,av_ij=state.av_ij,f_ij=f),sc.energyScheme)
        return f
    def updated(h,update,spec,**kwargs):
        result=apply(h,update,spec,**kwargs);l.updated(h,spec);return result
    def finished(h,initial,dt,returns,updates,weights,config,schemeConfig,*args,**kwargs):
        raw=snapshot(h.state)
        rate,rerun=energyIncrement(l.current['shadowWork'],initial.state,updates,weights,dt)
        assert rerun is None
        shadow=initial.state.internalEnergies+dt*rate
        shadow_u=shadow.detach().cpu().numpy().copy()
        assert np.all(shadow_u>0),'Compatible shadow candidate has nonpositive energy'
        base=l.current['base'];before=l.current['before']
        result=finalize(h,initial,dt,returns,updates,weights,config,schemeConfig,*args,**kwargs)
        if choice['compatible']:assert torch.equal(h.state.internalEnergies,shadow)
        else:np.testing.assert_array_equal(h.state.internalEnergies.detach().cpu().numpy(),raw['u'])
        l.finish(h,raw,dt)
        entropy_delta=np.log(shadow_u/raw['u'])
        l.arrays.setdefault('shadowCompatibleEntropy',[]).append(entropy_delta)
        l.arrays.setdefault('shadowEnergyDifference',[]).append(shadow_u-raw['u'])
        m=before['m'];mass=m.sum();E=np.dot(m,base['u']+.5*np.sum(base['v']**2,axis=1))
        shadowE=np.dot(m,shadow_u+.5*np.sum(raw['v']**2,axis=1));rawE=np.dot(m,raw['u']+.5*np.sum(raw['v']**2,axis=1))
        l.shadowRows.append(dict(step=len(l.steps),time=float(h.t),
            shadowEnergyRelativeClosure=float(shadowE/E-1),rawEnergyRelativeClosure=float(rawE/E-1),
            meanEntropyDifference=l.average(entropy_delta),maximumEnergyRelativeDifference=float(abs(shadow_u/raw['u']-1).max()),
            negativeShadowEntropyMassFraction=float(m[entropy_delta<-1.e-8].sum()/mass)))
        return result
    def observed_eos(rho,*args,**kwargs):
        u=kwargs.get('u');l.counters['negativeEOSCalls']+=int(bool((rho<=0).any() or (u is not None and (u<=0).any())))
        return evaluate(rho,*args,**kwargs)
    def observed_support(h,state,c,sc,**kwargs):
        assert sc.owenTable=='hermiteShell' and sc.owenMaterialWeights and sc.adaptiveSupportIterations==1
        l.counters['supportCalls']+=1;return solve(h,state,c,sc,**kwargs)
    evolution.prepareStep=reset;module.computeCRKdudt=derivative;module.computeCompSPHBalanceTerm=observed_balance
    system.CompSPHSystem.apply_state_update=updated;system.CompSPHSystem.finalize=finished
    eos.evaluateEOS=observed_eos;module.solveStageSupport=observed_support
    return l
